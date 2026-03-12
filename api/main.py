from __future__ import annotations

import hashlib
import hmac
import json
import multiprocessing
import os
import secrets
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Literal

# Make sure the engine is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import psycopg2
import psycopg2.extras
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from agent_stability_engine.adapters import (
    AnthropicChatAdapter,
    CustomEndpointAdapter,
    OpenAIChatAdapter,
)
from agent_stability_engine.contracts import (
    DEFAULT_JOB_TYPE,
    DEFAULT_SUITE,
    resolve_suite_path,
    validate_job_contract,
)
from agent_stability_engine.engine.embeddings import EmbeddingProvider
from agent_stability_engine.report.export import build_export_bundle
from agent_stability_engine.report.pdf_renderer import write_compliance_pdf
from agent_stability_engine.runners.agent_benchmark import run_agent_benchmark_suite
from agent_stability_engine.runners.benchmark import run_benchmark_suite
from agent_stability_engine.runners.conversation_benchmark import run_conversation_benchmark_suite
from agent_stability_engine.security import (
    build_otpauth_uri,
    generate_totp_secret,
    sanitize_error_message,
    validate_custom_endpoint_url,
    verify_totp,
)
from agent_stability_engine.traces.schema import AgentTrace

BASE_DIR = Path(__file__).parent.parent
DATABASE_URL = os.getenv("DATABASE_URL", "")
SESSION_TTL_HOURS = int(os.getenv("ASE_API_SESSION_TTL_HOURS", "168"))
WATCHDOG_TIMEOUT_SECONDS = int(os.getenv("ASE_WATCHDOG_TIMEOUT_SECONDS", "3600"))
_PASSWORD_ITERATIONS = 310_000
AUTH_RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("ASE_AUTH_RATE_LIMIT_WINDOW_SECONDS", "60"))
AUTH_RATE_LIMIT_MAX_REQUESTS = int(os.getenv("ASE_AUTH_RATE_LIMIT_MAX_REQUESTS", "10"))
JOB_RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("ASE_JOB_RATE_LIMIT_WINDOW_SECONDS", "60"))
JOB_RATE_LIMIT_MAX_REQUESTS = int(os.getenv("ASE_JOB_RATE_LIMIT_MAX_REQUESTS", "5"))
PUBLIC_EVAL_RATE_LIMIT_WINDOW_SECONDS = int(
    os.getenv("ASE_PUBLIC_EVAL_RATE_LIMIT_WINDOW_SECONDS", "60")
)
PUBLIC_EVAL_RATE_LIMIT_MAX_REQUESTS = int(os.getenv("ASE_PUBLIC_EVAL_RATE_LIMIT_MAX_REQUESTS", "5"))
MAX_CONCURRENT_JOBS_PER_USER = int(os.getenv("ASE_MAX_CONCURRENT_JOBS_PER_USER", "3"))
MAX_DAILY_JOBS_PER_USER = int(os.getenv("ASE_MAX_DAILY_JOBS_PER_USER", "100"))
LOGIN_FAILURE_WINDOW_SECONDS = int(os.getenv("ASE_LOGIN_FAILURE_WINDOW_SECONDS", "900"))
LOGIN_FAILURE_MAX_ATTEMPTS = int(os.getenv("ASE_LOGIN_FAILURE_MAX_ATTEMPTS", "8"))
LOGIN_LOCKOUT_SECONDS = int(os.getenv("ASE_LOGIN_LOCKOUT_SECONDS", "900"))
IP_BLOCK_WINDOW_SECONDS = int(os.getenv("ASE_IP_BLOCK_WINDOW_SECONDS", "900"))
IP_BLOCK_FAILURE_THRESHOLD = int(os.getenv("ASE_IP_BLOCK_FAILURE_THRESHOLD", "40"))
IP_BLOCK_SECONDS = int(os.getenv("ASE_IP_BLOCK_SECONDS", "1800"))
RATE_LIMIT_BACKEND = os.getenv("ASE_RATE_LIMIT_BACKEND", "memory").strip().lower()
TRUST_X_FORWARDED_FOR = os.getenv("ASE_TRUST_X_FORWARDED_FOR", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_auth_scheme = HTTPBearer(auto_error=False)

# Registry of active benchmark subprocesses, keyed by job_id.
# Populated by _run_job so that cancel_job can terminate them.
_running_processes: dict[str, multiprocessing.Process] = {}
_running_processes_lock = threading.Lock()


class _InMemorySlidingWindowRateLimiter:
    """Best-effort in-memory rate limiter.

    This protects single-instance deployments; multi-instance setups should
    enforce limits at an API gateway or shared store.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buckets: dict[str, deque[float]] = {}

    def allow(self, key: str, *, max_requests: int, window_seconds: int) -> bool:
        if max_requests <= 0:
            return False
        if window_seconds <= 0:
            return True

        now = time.monotonic()
        boundary = now - float(window_seconds)
        with self._lock:
            bucket = self._buckets.setdefault(key, deque())
            while bucket and bucket[0] <= boundary:
                bucket.popleft()
            if len(bucket) >= max_requests:
                return False
            bucket.append(now)
            return True


_rate_limiter = _InMemorySlidingWindowRateLimiter()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat().replace("+00:00", "Z")


def _parse_utc_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _validate_email(email: str) -> None:
    normalized = _normalize_email(email)
    if "@" not in normalized or normalized.startswith("@") or normalized.endswith("@"):
        raise HTTPException(status_code=400, detail="email must be valid")
    if "." not in normalized.split("@", 1)[1]:
        raise HTTPException(status_code=400, detail="email must be valid")


def _validate_password_strength(password: str) -> None:
    has_lower = any(ch.islower() for ch in password)
    has_upper = any(ch.isupper() for ch in password)
    has_digit = any(ch.isdigit() for ch in password)
    has_symbol = any(not ch.isalnum() for ch in password)
    has_space = any(ch.isspace() for ch in password)
    if not (has_lower and has_upper and has_digit and has_symbol) or has_space:
        raise HTTPException(
            status_code=400,
            detail=(
                "password must include upper/lowercase letters, a digit, " "a symbol, and no spaces"
            ),
        )


def _client_ip(request: Request) -> str:
    if TRUST_X_FORWARDED_FOR:
        xff = request.headers.get("x-forwarded-for", "").strip()
        if xff:
            first = xff.split(",")[0].strip()
            if first:
                return first
    if request.client is not None and request.client.host:
        return request.client.host
    return "unknown"


def _enforce_rate_limit(
    *,
    key: str,
    max_requests: int,
    window_seconds: int,
    detail: str,
) -> None:
    if RATE_LIMIT_BACKEND == "database":
        allowed = _rate_limit_allow_db(
            key=key,
            max_requests=max_requests,
            window_seconds=window_seconds,
        )
    else:
        allowed = _rate_limiter.allow(key, max_requests=max_requests, window_seconds=window_seconds)
    if not allowed:
        raise HTTPException(status_code=429, detail=detail)


def _login_identity_hash(email: str, ip: str) -> str:
    material = f"{_normalize_email(email)}|{ip}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _rate_limit_allow_db(*, key: str, max_requests: int, window_seconds: int) -> bool:
    if max_requests <= 0:
        return False
    if window_seconds <= 0:
        return True

    now = _utc_now()
    lower = now - timedelta(seconds=window_seconds)
    now_iso = now.isoformat().replace("+00:00", "Z")
    lower_iso = lower.isoformat().replace("+00:00", "Z")

    try:
        with _connect_db() as conn:
            conn.execute("DELETE FROM rate_limit_events WHERE created_at < %s", (lower_iso,))
            row = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM rate_limit_events
                WHERE rl_key = %s AND created_at >= %s
                """,
                (key, lower_iso),
            ).fetchone()
            count = int(row["count"]) if row is not None else 0
            if count >= max_requests:
                return False
            conn.execute(
                "INSERT INTO rate_limit_events (rl_key, created_at) VALUES (%s, %s)",
                (key, now_iso),
            )
            return True
    except Exception:  # noqa: BLE001
        # Fail-open to in-memory limiter if DB backend is unavailable.
        return _rate_limiter.allow(key, max_requests=max_requests, window_seconds=window_seconds)


def _is_ip_currently_blocked(ip: str) -> bool:
    if not ip or ip == "unknown":
        return False
    now_iso = _utc_now_iso()
    with _connect_db() as conn:
        row = conn.execute(
            """
            SELECT blocked_until
            FROM ip_blocks
            WHERE ip = %s
            """,
            (ip,),
        ).fetchone()
    if row is None:
        return False
    blocked_until = row["blocked_until"]
    return isinstance(blocked_until, str) and blocked_until > now_iso


def _enforce_ip_not_blocked(ip: str) -> None:
    if _is_ip_currently_blocked(ip):
        raise HTTPException(status_code=403, detail="request blocked for this IP")


def _hash_password(password: str, salt_hex: str) -> str:
    password_bytes = password.encode("utf-8")
    salt_bytes = bytes.fromhex(salt_hex)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password_bytes,
        salt_bytes,
        _PASSWORD_ITERATIONS,
    )
    return digest.hex()


def _verify_password(password: str, salt_hex: str, expected_hash_hex: str) -> bool:
    computed = _hash_password(password, salt_hex)
    return hmac.compare_digest(computed, expected_hash_hex)


class _DBWrapper:
    """Thin wrapper around a psycopg2 connection with sqlite3-compatible interface."""

    def __init__(self, conn: psycopg2.extensions.connection) -> None:
        self._conn = conn

    def execute(self, sql: str, params: tuple | None = None) -> psycopg2.extensions.cursor:
        cur = self._conn.cursor()
        cur.execute(sql, params)
        return cur

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def __enter__(self) -> _DBWrapper:
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        if exc_type:
            self._conn.rollback()
        else:
            self._conn.commit()
        self._conn.close()


def _connect_db() -> _DBWrapper:
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is not set. "
            "Add a PostgreSQL database plugin to your Railway project — "
            "Railway will then set DATABASE_URL automatically."
        )
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    return _DBWrapper(conn)


def _init_db() -> None:
    with _connect_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL DEFAULT '',
                business_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                suite TEXT NOT NULL DEFAULT 'examples/benchmarks/large_suite.json',
                job_type TEXT NOT NULL DEFAULT 'benchmark',
                fault_rate DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                workers INTEGER NOT NULL DEFAULT 1,
                run_count INTEGER NOT NULL,
                max_cases INTEGER NOT NULL,
                seed INTEGER NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                error_message TEXT,
                result_json TEXT,
                completed_cases INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_traces (
                id SERIAL PRIMARY KEY,
                job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                task_id TEXT NOT NULL,
                run_index INTEGER NOT NULL,
                trace_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS security_events (
                id SERIAL PRIMARY KEY,
                user_id TEXT REFERENCES users(id) ON DELETE SET NULL,
                event_type TEXT NOT NULL,
                ip TEXT NOT NULL,
                details_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_traces_job_id ON agent_traces(job_id)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_security_events_created_at "
            "ON security_events(created_at)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS auth_throttles (
                identity_hash TEXT PRIMARY KEY,
                failures INTEGER NOT NULL,
                window_started_at TEXT NOT NULL,
                locked_until TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_auth_throttles_locked_until "
            "ON auth_throttles(locked_until)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rate_limit_events (
                id SERIAL PRIMARY KEY,
                rl_key TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rate_limit_events_key_time "
            "ON rate_limit_events(rl_key, created_at)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ip_blocks (
                ip TEXT PRIMARY KEY,
                blocked_until TEXT NOT NULL,
                reason TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ip_blocks_blocked_until " "ON ip_blocks(blocked_until)"
        )
        # Idempotent migrations for pre-existing databases
        conn.execute(
            "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS completed_cases INTEGER NOT NULL DEFAULT 0"
        )
        conn.execute(
            "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS suite TEXT NOT NULL DEFAULT "
            "'examples/benchmarks/large_suite.json'"
        )
        conn.execute(
            "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS job_type TEXT NOT NULL DEFAULT 'benchmark'"
        )
        conn.execute(
            "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS fault_rate "
            "DOUBLE PRECISION NOT NULL DEFAULT 0.0"
        )
        conn.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS workers INTEGER NOT NULL DEFAULT 1")
        conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS name TEXT NOT NULL DEFAULT ''")
        conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS mfa_secret TEXT")
        conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS mfa_enabled BOOLEAN DEFAULT FALSE")


class UserPublic(BaseModel):
    id: str
    name: str
    business_name: str
    email: str
    created_at: str
    mfa_enabled: bool = False


class AuthResponse(BaseModel):
    token: str
    user: UserPublic


class RegisterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    business_name: str = Field(min_length=1, max_length=200)
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=8, max_length=128)
    mfa_code: str | None = Field(default=None, max_length=16)


class MFASetupResponse(BaseModel):
    secret: str
    otpauth_uri: str


class MFAEnableRequest(BaseModel):
    code: str = Field(min_length=6, max_length=16)


class MFADisableRequest(BaseModel):
    code: str = Field(min_length=6, max_length=16)


class JobCreateRequest(BaseModel):
    provider: Literal["openai", "anthropic", "custom"]
    model: str = Field(min_length=1, max_length=200)
    custom_endpoint: str | None = Field(default=None, max_length=2048)
    api_key: str = Field(min_length=1, max_length=2048)
    suite: str = Field(default=DEFAULT_SUITE, min_length=1, max_length=512)
    job_type: Literal["benchmark", "conversation_benchmark", "agent_benchmark"] = DEFAULT_JOB_TYPE
    fault_rate: float = Field(default=0.0, ge=0.0, le=0.5)
    run_count: int = Field(default=3, ge=2, le=10)
    max_cases: int = Field(default=5, ge=1, le=100)
    seed: int = 42
    workers: int = Field(default=3, ge=1, le=10)


class JobSummary(BaseModel):
    id: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    provider: str
    model: str
    suite: str
    job_type: str
    fault_rate: float
    workers: int
    run_count: int
    max_cases: int
    seed: int
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None
    error_message: str | None = None
    mean_asi: float | None = None
    num_cases: int | None = None
    completed_cases: int = 0


class JobListResponse(BaseModel):
    jobs: list[JobSummary]


class JobReportResponse(BaseModel):
    job_id: str
    report: dict[str, object]


class JobTraceRecord(BaseModel):
    task_id: str
    run_index: int
    trace: dict[str, object]


class JobTracesResponse(BaseModel):
    job_id: str
    traces: list[JobTraceRecord]


class EvaluateRequest(BaseModel):
    provider: Literal["openai", "anthropic", "custom"]
    model: str = Field(min_length=1, max_length=200)
    custom_endpoint: str | None = Field(default=None, max_length=2048)
    api_key: str = Field(min_length=1, max_length=2048)
    suite: str = Field(default=DEFAULT_SUITE, min_length=1, max_length=512)
    run_count: int = Field(default=3, ge=2, le=10)
    max_cases: int = Field(default=5, ge=1, le=100)
    seed: int = 42


class EvaluateResponse(BaseModel):
    model: str
    provider: str
    asi: float
    domain_scores: dict[str, float]
    num_cases: int
    run_count: int


def _row_to_user(row: psycopg2.extras.RealDictRow) -> UserPublic:
    raw_mfa = row.get("mfa_enabled")
    mfa_enabled = bool(raw_mfa) if isinstance(raw_mfa, (bool, int)) else False
    return UserPublic(
        id=str(row["id"]),
        name=str(row["name"]) if row["name"] else "",
        business_name=str(row["business_name"]),
        email=str(row["email"]),
        created_at=str(row["created_at"]),
        mfa_enabled=mfa_enabled,
    )


def _row_to_job_summary(row: psycopg2.extras.RealDictRow) -> JobSummary:
    mean_asi: float | None = None
    num_cases: int | None = None
    result_json = row["result_json"]
    if isinstance(result_json, str):
        try:
            parsed = json.loads(result_json)
            if isinstance(parsed, dict):
                raw_asi = parsed.get("mean_asi")
                raw_cases = parsed.get("num_cases")
                if isinstance(raw_asi, (int, float)):
                    mean_asi = float(raw_asi)
                if isinstance(raw_cases, int):
                    num_cases = raw_cases
        except json.JSONDecodeError:
            pass

    raw_completed = row["completed_cases"]
    completed_cases = int(raw_completed) if raw_completed is not None else 0
    raw_fault_rate = row.get("fault_rate", 0.0)
    fault_rate = float(raw_fault_rate) if isinstance(raw_fault_rate, (int, float)) else 0.0
    raw_workers = row.get("workers", 1)
    workers = int(raw_workers) if isinstance(raw_workers, (int, float)) else 1
    raw_suite = row.get("suite")
    suite = str(raw_suite) if isinstance(raw_suite, str) and raw_suite else DEFAULT_SUITE
    raw_job_type = row.get("job_type")
    job_type = (
        str(raw_job_type) if isinstance(raw_job_type, str) and raw_job_type else DEFAULT_JOB_TYPE
    )

    return JobSummary(
        id=str(row["id"]),
        status=str(row["status"]),
        provider=str(row["provider"]),
        model=str(row["model"]),
        suite=suite,
        job_type=job_type,
        fault_rate=fault_rate,
        workers=workers,
        run_count=int(row["run_count"]),
        max_cases=int(row["max_cases"]),
        seed=int(row["seed"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        started_at=str(row["started_at"]) if row["started_at"] else None,
        finished_at=str(row["finished_at"]) if row["finished_at"] else None,
        error_message=str(row["error_message"]) if row["error_message"] else None,
        mean_asi=mean_asi,
        num_cases=num_cases,
        completed_cases=completed_cases,
    )


def _create_session(conn: _DBWrapper, user_id: str) -> str:
    now = _utc_now()
    expires = now + timedelta(hours=SESSION_TTL_HOURS)
    now_iso = now.isoformat().replace("+00:00", "Z")
    token = f"ase_{secrets.token_urlsafe(32)}"
    conn.execute("DELETE FROM sessions WHERE expires_at <= %s", (now_iso,))
    conn.execute(
        """
        INSERT INTO sessions (token, user_id, created_at, expires_at)
        VALUES (%s, %s, %s, %s)
        """,
        (
            token,
            user_id,
            now_iso,
            expires.isoformat().replace("+00:00", "Z"),
        ),
    )
    return token


def _require_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_auth_scheme)],
) -> UserPublic:
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
        )

    token = credentials.credentials
    with _connect_db() as conn:
        row = conn.execute(
            """
            SELECT s.expires_at, u.id, u.name, u.business_name, u.email, u.created_at, u.mfa_enabled
            FROM sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.token = %s
            """,
            (token,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=401, detail="invalid session")
        expires_at = _parse_utc_iso(str(row["expires_at"]))
        if expires_at <= _utc_now():
            conn.execute("DELETE FROM sessions WHERE token = %s", (token,))
            conn.commit()
            raise HTTPException(status_code=401, detail="session expired")
        return _row_to_user(row)


def _build_agent(
    provider: str,
    model: str,
    api_key: str,
    custom_endpoint: str | None = None,
) -> OpenAIChatAdapter | AnthropicChatAdapter | CustomEndpointAdapter:
    if provider == "openai":
        return OpenAIChatAdapter(model=model, api_key=api_key)
    if provider == "anthropic":
        return AnthropicChatAdapter(model=model, api_key=api_key)
    if provider == "custom":
        if custom_endpoint is None:
            msg = "custom_endpoint is required when provider=custom"
            raise ValueError(msg)
        return CustomEndpointAdapter(endpoint_url=custom_endpoint, model=model, api_key=api_key)
    raise ValueError(f"unsupported provider: {provider}")


def _validate_custom_endpoint(provider: str, custom_endpoint: str | None) -> str | None:
    endpoint = custom_endpoint.strip() if isinstance(custom_endpoint, str) else None
    if provider == "custom":
        if not endpoint:
            raise HTTPException(status_code=400, detail="custom_endpoint is required for custom")
        try:
            return validate_custom_endpoint_url(endpoint)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    if endpoint:
        raise HTTPException(
            status_code=400,
            detail="custom_endpoint is only valid when provider is custom",
        )
    return None


def _sanitize_error_message(message: str, api_key: str) -> str:
    return sanitize_error_message(
        message,
        secrets=[api_key],
        max_length=2000,
    )


def _log_security_event(
    *,
    event_type: str,
    ip: str,
    user_id: str | None = None,
    details: dict[str, object] | None = None,
) -> None:
    payload = details or {}
    created_at = _utc_now_iso()
    try:
        with _connect_db() as conn:
            conn.execute(
                """
                INSERT INTO security_events (user_id, event_type, ip, details_json, created_at)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    user_id,
                    event_type,
                    ip,
                    json.dumps(payload),
                    created_at,
                ),
            )
    except Exception:  # noqa: BLE001
        # Best effort: never break request handling if audit logging fails.
        pass


def _enforce_job_limits(*, user_id: str) -> None:
    now = _utc_now()
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    day_floor = start_of_day.isoformat().replace("+00:00", "Z")

    with _connect_db() as conn:
        concurrent_row = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM jobs
            WHERE user_id = %s AND status IN ('queued', 'running')
            """,
            (user_id,),
        ).fetchone()
        daily_row = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM jobs
            WHERE user_id = %s AND created_at >= %s
            """,
            (user_id, day_floor),
        ).fetchone()

    concurrent_count = int(concurrent_row["count"]) if concurrent_row is not None else 0
    daily_count = int(daily_row["count"]) if daily_row is not None else 0

    if concurrent_count >= MAX_CONCURRENT_JOBS_PER_USER:
        raise HTTPException(
            status_code=429,
            detail=(
                "too many concurrent jobs for this account; "
                f"limit={MAX_CONCURRENT_JOBS_PER_USER}"
            ),
        )
    if daily_count >= MAX_DAILY_JOBS_PER_USER:
        raise HTTPException(
            status_code=429,
            detail=f"daily job limit reached; limit={MAX_DAILY_JOBS_PER_USER}",
        )


def _is_login_temporarily_blocked(*, email: str, ip: str) -> bool:
    identity = _login_identity_hash(email, ip)
    now = _utc_now()
    with _connect_db() as conn:
        row = conn.execute(
            """
            SELECT locked_until
            FROM auth_throttles
            WHERE identity_hash = %s
            """,
            (identity,),
        ).fetchone()
    if row is None:
        return False
    locked_until_raw = row["locked_until"]
    if not isinstance(locked_until_raw, str) or not locked_until_raw:
        return False
    return _parse_utc_iso(locked_until_raw) > now


def _record_login_failure(*, email: str, ip: str) -> tuple[int, bool]:
    identity = _login_identity_hash(email, ip)
    now = _utc_now()
    now_iso = now.isoformat().replace("+00:00", "Z")
    lockout_until_iso: str | None = None
    failures = 1
    lockout_applied = False

    with _connect_db() as conn:
        row = conn.execute(
            """
            SELECT failures, window_started_at
            FROM auth_throttles
            WHERE identity_hash = %s
            """,
            (identity,),
        ).fetchone()

        if row is None:
            conn.execute(
                """
                INSERT INTO auth_throttles (
                    identity_hash, failures, window_started_at, locked_until, updated_at
                )
                VALUES (%s, %s, %s, %s, %s)
                """,
                (identity, 1, now_iso, None, now_iso),
            )
            return (1, False)

        raw_failures = row["failures"]
        failures = int(raw_failures) if isinstance(raw_failures, (int, float)) else 0
        window_started_raw = row["window_started_at"]
        if not isinstance(window_started_raw, str):
            window_started_raw = now_iso
        window_started = _parse_utc_iso(window_started_raw)

        elapsed = (now - window_started).total_seconds()
        if elapsed > LOGIN_FAILURE_WINDOW_SECONDS:
            failures = 1
            window_started_raw = now_iso
        else:
            failures += 1

        if failures >= LOGIN_FAILURE_MAX_ATTEMPTS:
            lockout_until = now + timedelta(seconds=LOGIN_LOCKOUT_SECONDS)
            lockout_until_iso = lockout_until.isoformat().replace("+00:00", "Z")
            lockout_applied = True

        conn.execute(
            """
            UPDATE auth_throttles
            SET failures = %s, window_started_at = %s, locked_until = %s, updated_at = %s
            WHERE identity_hash = %s
            """,
            (failures, window_started_raw, lockout_until_iso, now_iso, identity),
        )
    return (failures, lockout_applied)


def _clear_login_failures(*, email: str, ip: str) -> None:
    identity = _login_identity_hash(email, ip)
    with _connect_db() as conn:
        conn.execute("DELETE FROM auth_throttles WHERE identity_hash = %s", (identity,))


def _block_ip(*, ip: str, reason: str, duration_seconds: int) -> None:
    if not ip or ip == "unknown":
        return
    now_iso = _utc_now_iso()
    blocked_until = (
        (_utc_now() + timedelta(seconds=max(duration_seconds, 1)))
        .isoformat()
        .replace("+00:00", "Z")
    )
    with _connect_db() as conn:
        conn.execute(
            """
            INSERT INTO ip_blocks (ip, blocked_until, reason, updated_at)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (ip)
            DO UPDATE SET blocked_until = EXCLUDED.blocked_until,
                          reason = EXCLUDED.reason,
                          updated_at = EXCLUDED.updated_at
            """,
            (ip, blocked_until, reason, now_iso),
        )


def _auto_block_if_abusive(*, ip: str) -> bool:
    if not ip or ip == "unknown":
        return False
    now = _utc_now()
    lower = (now - timedelta(seconds=IP_BLOCK_WINDOW_SECONDS)).isoformat().replace("+00:00", "Z")
    with _connect_db() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM security_events
            WHERE ip = %s
              AND event_type IN (
                  'auth_login_failed',
                  'auth_login_blocked',
                  'auth_register_conflict'
              )
              AND created_at >= %s
            """,
            (ip, lower),
        ).fetchone()
    failures = int(row["count"]) if row is not None else 0
    if failures >= IP_BLOCK_FAILURE_THRESHOLD:
        _block_ip(
            ip=ip,
            reason=f"auto_block_threshold_exceeded:{failures}",
            duration_seconds=IP_BLOCK_SECONDS,
        )
        return True
    return False


def _persist_agent_traces(*, job_id: str, traces: list[AgentTrace]) -> None:
    if not traces:
        return
    created_at = _utc_now_iso()
    with _connect_db() as conn:
        conn.execute("DELETE FROM agent_traces WHERE job_id = %s", (job_id,))
        for trace in traces:
            conn.execute(
                """
                INSERT INTO agent_traces (job_id, task_id, run_index, trace_json, created_at)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    job_id,
                    trace.task_id,
                    trace.run_index,
                    json.dumps(asdict(trace)),
                    created_at,
                ),
            )


def _run_benchmark_report(
    *,
    suite_path: Path,
    job_type: str,
    provider: str,
    model: str,
    api_key: str,
    custom_endpoint: str | None,
    run_count: int,
    max_cases: int,
    seed: int,
    fault_rate: float = 0.0,
    job_id: str | None = None,
    workers: int = 1,
) -> dict[str, object]:
    agent = _build_agent(
        provider=provider,
        model=model,
        api_key=api_key,
        custom_endpoint=custom_endpoint,
    )

    # When workers > 1, give each worker its own adapter instance so API calls
    # across cases happen truly in parallel (no shared-state lock needed).
    agent_factory: Callable[[], Callable[..., str]] | None = None
    if workers > 1:
        _p, _m, _k, _e = provider, model, api_key, custom_endpoint

        def agent_factory() -> Callable[..., str]:
            return _build_agent(provider=_p, model=_m, api_key=_k, custom_endpoint=_e)

    progress_callback = None
    if job_id is not None:
        _jid = job_id

        def progress_callback(completed: int, total: int, case_id: str) -> None:  # noqa: ARG001
            try:
                now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                with _connect_db() as _conn:
                    _conn.execute(
                        "UPDATE jobs SET completed_cases = %s, updated_at = %s WHERE id = %s",
                        (completed, now, _jid),
                    )
            except Exception:  # noqa: BLE001
                pass  # never let progress tracking crash the benchmark

    if job_type == "benchmark":
        result = run_benchmark_suite(
            suite_path=suite_path,
            agent_fn=agent,
            run_count=run_count,
            seed=seed,
            embedding_provider=EmbeddingProvider.HASH,
            max_cases=max_cases,
            workers=workers,
            agent_factory=agent_factory,
            progress_callback=progress_callback,
        )
        return result.report

    if job_type == "conversation_benchmark":
        result = run_conversation_benchmark_suite(
            suite_path=suite_path,
            adapter=agent,
            run_count=run_count,
            seed=seed,
            embedding_provider=EmbeddingProvider.HASH,
            max_cases=max_cases,
            workers=workers,
            agent_factory=agent_factory,
            progress_callback=progress_callback,
        )
        return result.report

    if job_type == "agent_benchmark":
        if provider == "custom":
            msg = "agent_benchmark currently supports only openai and anthropic providers"
            raise RuntimeError(msg)
        result = run_agent_benchmark_suite(
            suite_path=suite_path,
            adapter=agent,
            run_count=run_count,
            seed=seed,
            fault_rate=fault_rate,
            max_tasks=max_cases,
            workers=workers,
            progress_callback=progress_callback,
            agent_factory=agent_factory,
        )
        if job_id is not None:
            _persist_agent_traces(job_id=job_id, traces=result.traces)
        return result.report

    msg = f"job_type '{job_type}' is not enabled yet"
    raise RuntimeError(msg)


def _with_watchdog_metadata(
    *,
    report: dict[str, object],
    started_at: str,
    finished_at: str,
    triggered: bool,
    reason: str | None,
) -> dict[str, object]:
    elapsed = (_parse_utc_iso(finished_at) - _parse_utc_iso(started_at)).total_seconds()
    watchdog_block: dict[str, object] = {
        "enabled": True,
        "timeout_seconds": WATCHDOG_TIMEOUT_SECONDS,
        "triggered": triggered,
        "trigger_reason": reason,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": round(max(elapsed, 0.0), 3),
    }
    merged = dict(report)
    existing_notes = merged.get("notes")
    notes: list[str] = []
    if isinstance(existing_notes, list):
        notes = [str(note) for note in existing_notes]
    if triggered:
        notes.append(
            "watchdog_triggered=true " f"reason={reason} timeout_seconds={WATCHDOG_TIMEOUT_SECONDS}"
        )
    else:
        notes.append("watchdog_triggered=false " f"timeout_seconds={WATCHDOG_TIMEOUT_SECONDS}")
    merged["notes"] = notes
    merged["execution_watchdog"] = watchdog_block
    return merged


def _build_failure_report(
    *,
    provider: str,
    model: str,
    run_count: int,
    max_cases: int,
    seed: int,
    started_at: str,
    finished_at: str,
    reason: str,
    watchdog_triggered: bool,
) -> dict[str, object]:
    report: dict[str, object] = {
        "report_type": "job_failure",
        "timestamp_utc": finished_at,
        "status": "failed",
        "provider": provider,
        "model": model,
        "run_count": run_count,
        "max_cases": max_cases,
        "seed": seed,
        "failure_reason": reason,
        "notes": [
            "evaluation_failed=true",
            f"watchdog_triggered={str(watchdog_triggered).lower()}",
        ],
    }
    return _with_watchdog_metadata(
        report=report,
        started_at=started_at,
        finished_at=finished_at,
        triggered=watchdog_triggered,
        reason=reason if watchdog_triggered else None,
    )


def _run_job_worker(
    *,
    suite_path: str,
    provider: str,
    model: str,
    api_key: str,
    custom_endpoint: str | None,
    job_type: str,
    fault_rate: float,
    run_count: int,
    max_cases: int,
    seed: int,
    job_id: str,
    started_at: str,
    workers: int = 1,
) -> None:
    """Run the benchmark and write the result directly to PostgreSQL.

    Runs in a subprocess. DATABASE_URL is inherited from the parent environment.
    """
    try:
        report = _run_benchmark_report(
            suite_path=Path(suite_path),
            job_type=job_type,
            provider=provider,
            model=model,
            api_key=api_key,
            custom_endpoint=custom_endpoint,
            run_count=run_count,
            max_cases=max_cases,
            seed=seed,
            fault_rate=fault_rate,
            job_id=job_id,
            workers=workers,
        )
        finished_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        final_report = _with_watchdog_metadata(
            report=report,
            started_at=started_at,
            finished_at=finished_at,
            triggered=False,
            reason=None,
        )
        payload = json.dumps(final_report)
        with _connect_db() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = %s, result_json = %s, updated_at = %s, finished_at = %s
                WHERE id = %s
                """,
                ("completed", payload, finished_at, finished_at, job_id),
            )
    except Exception as exc:
        finished_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        raw_message = str(exc)[:8000] or "evaluation failed"
        message = _sanitize_error_message(raw_message, api_key)
        failure_report = _build_failure_report(
            provider=provider,
            model=model,
            run_count=run_count,
            max_cases=max_cases,
            seed=seed,
            started_at=started_at,
            finished_at=finished_at,
            reason=message,
            watchdog_triggered=False,
        )
        failure_report["job_type"] = job_type
        failure_report["fault_rate"] = fault_rate
        try:
            with _connect_db() as conn:
                conn.execute(
                    """
                    UPDATE jobs
                    SET status = %s, error_message = %s, result_json = %s,
                        updated_at = %s, finished_at = %s
                    WHERE id = %s
                    """,
                    (
                        "failed",
                        message,
                        json.dumps(failure_report),
                        finished_at,
                        finished_at,
                        job_id,
                    ),
                )
        except Exception:  # noqa: BLE001
            pass  # best-effort; watchdog will mark job as failed if DB write fails


def _run_job(
    *,
    job_id: str,
    suite_path: str,
    provider: str,
    model: str,
    api_key: str,
    custom_endpoint: str | None,
    job_type: str,
    fault_rate: float,
    run_count: int,
    max_cases: int,
    seed: int,
    workers: int = 1,
) -> None:
    started_at = _utc_now_iso()
    with _connect_db() as conn:
        conn.execute(
            """
            UPDATE jobs
            SET status = %s, started_at = %s, updated_at = %s
            WHERE id = %s
            """,
            ("running", started_at, started_at, job_id),
        )

    process = multiprocessing.Process(
        target=_run_job_worker,
        kwargs={
            "suite_path": suite_path,
            "provider": provider,
            "model": model,
            "api_key": api_key,
            "custom_endpoint": custom_endpoint,
            "job_type": job_type,
            "fault_rate": fault_rate,
            "run_count": run_count,
            "max_cases": max_cases,
            "seed": seed,
            "job_id": job_id,
            "started_at": started_at,
            "workers": workers,
        },
        daemon=True,
    )
    process.start()
    with _running_processes_lock:
        _running_processes[job_id] = process

    process.join(timeout=WATCHDOG_TIMEOUT_SECONDS)

    with _running_processes_lock:
        _running_processes.pop(job_id, None)

    if not process.is_alive():
        # Worker exited cleanly — it already wrote completed/failed to PostgreSQL.
        return

    # Watchdog: subprocess is still running past the timeout — kill it.
    process.terminate()
    process.join(timeout=5)
    finished_at = _utc_now_iso()
    reason = (
        "watchdog_timeout: evaluation considered stagnant and cancelled "
        f"after {WATCHDOG_TIMEOUT_SECONDS} seconds"
    )
    failure_report = _build_failure_report(
        provider=provider,
        model=model,
        run_count=run_count,
        max_cases=max_cases,
        seed=seed,
        started_at=started_at,
        finished_at=finished_at,
        reason=reason,
        watchdog_triggered=True,
    )
    failure_report["job_type"] = job_type
    failure_report["fault_rate"] = fault_rate
    with _connect_db() as conn:
        conn.execute(
            """
            UPDATE jobs
            SET status = %s, error_message = %s, result_json = %s,
                updated_at = %s, finished_at = %s
            WHERE id = %s
            """,
            ("failed", reason, json.dumps(failure_report), finished_at, finished_at, job_id),
        )


def _spawn_job(
    *,
    job_id: str,
    suite_path: str,
    provider: str,
    model: str,
    api_key: str,
    custom_endpoint: str | None,
    job_type: str,
    fault_rate: float,
    run_count: int,
    max_cases: int,
    seed: int,
    workers: int = 1,
) -> None:
    thread = threading.Thread(
        target=_run_job,
        kwargs={
            "job_id": job_id,
            "suite_path": suite_path,
            "provider": provider,
            "model": model,
            "api_key": api_key,
            "custom_endpoint": custom_endpoint,
            "job_type": job_type,
            "fault_rate": fault_rate,
            "run_count": run_count,
            "max_cases": max_cases,
            "seed": seed,
            "workers": workers,
        },
        daemon=True,
    )
    thread.start()


def _allowed_origins() -> list[str]:
    configured = os.getenv("ASE_ALLOWED_ORIGINS", "*").strip()
    if configured == "*":
        return ["*"]
    parts = [part.strip() for part in configured.split(",")]
    return [part for part in parts if part]


@asynccontextmanager
async def _lifespan(app: FastAPI):  # noqa: ARG001
    _init_db()
    yield


app = FastAPI(title="Stabilium API", version="0.2.0", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _add_security_headers(request: Request, call_next):  # type: ignore[no-untyped-def]
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/auth/register", response_model=AuthResponse)
def register(req: RegisterRequest, request: Request) -> AuthResponse:
    ip = _client_ip(request)
    _enforce_ip_not_blocked(ip)
    _enforce_rate_limit(
        key=f"auth:register:{ip}",
        max_requests=AUTH_RATE_LIMIT_MAX_REQUESTS,
        window_seconds=AUTH_RATE_LIMIT_WINDOW_SECONDS,
        detail="too many register attempts; please try again later",
    )
    _validate_email(req.email)
    _validate_password_strength(req.password)
    email = _normalize_email(req.email)
    now = _utc_now_iso()
    user_id = f"usr_{secrets.token_hex(12)}"
    salt = secrets.token_hex(16)
    password_hash = _hash_password(req.password, salt)

    with _connect_db() as conn:
        try:
            conn.execute(
                """
                INSERT INTO users (
                    id, name, business_name, email, password_salt, password_hash, created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    user_id,
                    req.name.strip(),
                    req.business_name.strip(),
                    email,
                    salt,
                    password_hash,
                    now,
                ),
            )
        except psycopg2.IntegrityError as exc:
            _log_security_event(
                event_type="auth_register_conflict",
                ip=ip,
                details={"email": email},
            )
            _auto_block_if_abusive(ip=ip)
            raise HTTPException(status_code=409, detail="email already registered") from exc

        token = _create_session(conn, user_id)
        conn.commit()
        row = conn.execute(
            """
            SELECT id, name, business_name, email, created_at, mfa_enabled
            FROM users
            WHERE id = %s
            """,
            (user_id,),
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=500, detail="failed to create user")
    _log_security_event(
        event_type="auth_register_success",
        ip=ip,
        user_id=user_id,
        details={"email": email},
    )
    return AuthResponse(token=token, user=_row_to_user(row))


@app.post("/auth/login", response_model=AuthResponse)
def login(req: LoginRequest, request: Request) -> AuthResponse:
    ip = _client_ip(request)
    _enforce_ip_not_blocked(ip)
    _enforce_rate_limit(
        key=f"auth:login:{ip}",
        max_requests=AUTH_RATE_LIMIT_MAX_REQUESTS,
        window_seconds=AUTH_RATE_LIMIT_WINDOW_SECONDS,
        detail="too many login attempts; please try again later",
    )
    _validate_email(req.email)
    email = _normalize_email(req.email)
    if _is_login_temporarily_blocked(email=email, ip=ip):
        _log_security_event(
            event_type="auth_login_blocked",
            ip=ip,
            details={"reason": "temporary_lockout"},
        )
        raise HTTPException(status_code=429, detail="too many login attempts; please try later")
    with _connect_db() as conn:
        row = conn.execute(
            """
            SELECT id, name, business_name, email, created_at, password_salt, password_hash,
                   mfa_enabled, mfa_secret
            FROM users
            WHERE email = %s
            """,
            (email,),
        ).fetchone()
        if row is None:
            failures, lockout_applied = _record_login_failure(email=email, ip=ip)
            _log_security_event(
                event_type="auth_login_failed",
                ip=ip,
                details={
                    "reason": "not_found",
                    "failure_count": failures,
                    "lockout_applied": lockout_applied,
                },
            )
            _auto_block_if_abusive(ip=ip)
            raise HTTPException(status_code=401, detail="invalid email or password")
        if not _verify_password(req.password, str(row["password_salt"]), str(row["password_hash"])):
            failures, lockout_applied = _record_login_failure(email=email, ip=ip)
            _log_security_event(
                event_type="auth_login_failed",
                ip=ip,
                details={
                    "reason": "password_mismatch",
                    "failure_count": failures,
                    "lockout_applied": lockout_applied,
                },
            )
            _auto_block_if_abusive(ip=ip)
            raise HTTPException(status_code=401, detail="invalid email or password")
        mfa_enabled_raw = row.get("mfa_enabled")
        mfa_enabled = bool(mfa_enabled_raw) if isinstance(mfa_enabled_raw, (bool, int)) else False
        mfa_secret_raw = row.get("mfa_secret")
        mfa_secret = str(mfa_secret_raw) if isinstance(mfa_secret_raw, str) else ""
        if mfa_enabled:
            if not req.mfa_code:
                _log_security_event(
                    event_type="auth_login_failed",
                    ip=ip,
                    details={"reason": "mfa_required_missing"},
                )
                raise HTTPException(status_code=401, detail="mfa code required")
            if not mfa_secret or not verify_totp(
                mfa_secret,
                req.mfa_code,
                timestamp=int(time.time()),
            ):
                failures, lockout_applied = _record_login_failure(email=email, ip=ip)
                _log_security_event(
                    event_type="auth_login_failed",
                    ip=ip,
                    details={
                        "reason": "mfa_code_invalid",
                        "failure_count": failures,
                        "lockout_applied": lockout_applied,
                    },
                )
                _auto_block_if_abusive(ip=ip)
                raise HTTPException(status_code=401, detail="invalid mfa code")
        token = _create_session(conn, str(row["id"]))
        conn.commit()
        user = _row_to_user(row)
    _clear_login_failures(email=email, ip=ip)
    _log_security_event(
        event_type="auth_login_success",
        ip=ip,
        user_id=user.id,
        details={"email": user.email},
    )
    return AuthResponse(token=token, user=user)


@app.get("/auth/me", response_model=UserPublic)
def auth_me(user: Annotated[UserPublic, Depends(_require_user)]) -> UserPublic:
    return user


@app.post("/auth/mfa/setup", response_model=MFASetupResponse)
def mfa_setup(
    request: Request,
    user: Annotated[UserPublic, Depends(_require_user)],
) -> MFASetupResponse:
    ip = _client_ip(request)
    _enforce_ip_not_blocked(ip)
    secret = generate_totp_secret()
    with _connect_db() as conn:
        conn.execute(
            "UPDATE users SET mfa_secret = %s, mfa_enabled = FALSE WHERE id = %s",
            (secret, user.id),
        )
    uri = build_otpauth_uri(secret=secret, account_name=user.email, issuer="Stabilium")
    _log_security_event(
        event_type="auth_mfa_setup",
        ip=ip,
        user_id=user.id,
        details={},
    )
    return MFASetupResponse(secret=secret, otpauth_uri=uri)


@app.post("/auth/mfa/enable", response_model=UserPublic)
def mfa_enable(
    req: MFAEnableRequest,
    request: Request,
    user: Annotated[UserPublic, Depends(_require_user)],
) -> UserPublic:
    ip = _client_ip(request)
    _enforce_ip_not_blocked(ip)
    with _connect_db() as conn:
        row = conn.execute(
            """
            SELECT id, name, business_name, email, created_at, mfa_secret, mfa_enabled
            FROM users
            WHERE id = %s
            """,
            (user.id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="user not found")
        secret_raw = row["mfa_secret"]
        secret = str(secret_raw) if isinstance(secret_raw, str) else ""
        if not secret:
            raise HTTPException(status_code=400, detail="mfa setup is required first")
        if not verify_totp(secret, req.code, timestamp=int(time.time())):
            raise HTTPException(status_code=400, detail="invalid mfa code")
        conn.execute("UPDATE users SET mfa_enabled = TRUE WHERE id = %s", (user.id,))
        updated = conn.execute(
            """
            SELECT id, name, business_name, email, created_at, mfa_enabled
            FROM users
            WHERE id = %s
            """,
            (user.id,),
        ).fetchone()
    _log_security_event(
        event_type="auth_mfa_enabled",
        ip=ip,
        user_id=user.id,
        details={},
    )
    if updated is None:
        raise HTTPException(status_code=500, detail="failed to enable mfa")
    return _row_to_user(updated)


@app.post("/auth/mfa/disable", response_model=UserPublic)
def mfa_disable(
    req: MFADisableRequest,
    request: Request,
    user: Annotated[UserPublic, Depends(_require_user)],
) -> UserPublic:
    ip = _client_ip(request)
    _enforce_ip_not_blocked(ip)
    with _connect_db() as conn:
        row = conn.execute(
            """
            SELECT id, name, business_name, email, created_at, mfa_secret, mfa_enabled
            FROM users
            WHERE id = %s
            """,
            (user.id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="user not found")
        mfa_enabled_raw = row["mfa_enabled"]
        mfa_enabled = bool(mfa_enabled_raw) if isinstance(mfa_enabled_raw, (bool, int)) else False
        secret_raw = row["mfa_secret"]
        secret = str(secret_raw) if isinstance(secret_raw, str) else ""
        if not mfa_enabled or not secret:
            raise HTTPException(status_code=400, detail="mfa is not enabled")
        if not verify_totp(secret, req.code, timestamp=int(time.time())):
            raise HTTPException(status_code=400, detail="invalid mfa code")
        conn.execute(
            "UPDATE users SET mfa_enabled = FALSE, mfa_secret = NULL WHERE id = %s",
            (user.id,),
        )
        updated = conn.execute(
            """
            SELECT id, name, business_name, email, created_at, mfa_enabled
            FROM users
            WHERE id = %s
            """,
            (user.id,),
        ).fetchone()
    _log_security_event(
        event_type="auth_mfa_disabled",
        ip=ip,
        user_id=user.id,
        details={},
    )
    if updated is None:
        raise HTTPException(status_code=500, detail="failed to disable mfa")
    return _row_to_user(updated)


@app.post("/auth/logout")
def logout(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_auth_scheme)],
    user: Annotated[UserPublic, Depends(_require_user)],
) -> dict[str, bool]:
    if credentials is not None and credentials.credentials:
        with _connect_db() as conn:
            conn.execute("DELETE FROM sessions WHERE token = %s", (credentials.credentials,))
    _log_security_event(
        event_type="auth_logout",
        ip=_client_ip(request),
        user_id=user.id,
        details={},
    )
    return {"ok": True}


@app.post("/jobs", response_model=JobSummary, status_code=202)
def create_job(
    request: Request,
    req: JobCreateRequest,
    user: Annotated[UserPublic, Depends(_require_user)],
) -> JobSummary:
    ip = _client_ip(request)
    _enforce_ip_not_blocked(ip)
    _enforce_rate_limit(
        key=f"job:create:{user.id}:{ip}",
        max_requests=JOB_RATE_LIMIT_MAX_REQUESTS,
        window_seconds=JOB_RATE_LIMIT_WINDOW_SECONDS,
        detail="too many job submissions; please slow down",
    )
    try:
        suite_path = resolve_suite_path(base_dir=BASE_DIR, suite=req.suite)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        validate_job_contract(job_type=req.job_type, fault_rate=req.fault_rate)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if req.job_type == "agent_benchmark" and req.provider == "custom":
        raise HTTPException(
            status_code=400,
            detail="job_type 'agent_benchmark' is supported only for openai and anthropic",
        )
    custom_endpoint = _validate_custom_endpoint(req.provider, req.custom_endpoint)
    api_key = req.api_key.strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="api_key is required")
    _enforce_job_limits(user_id=user.id)

    now = _utc_now_iso()
    job_id = f"job_{secrets.token_hex(12)}"
    with _connect_db() as conn:
        conn.execute(
            """
            INSERT INTO jobs (
                id, user_id, provider, model, suite, job_type, fault_rate, workers,
                run_count, max_cases, seed,
                status, created_at, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                job_id,
                user.id,
                req.provider,
                req.model.strip(),
                str(suite_path.relative_to(BASE_DIR)),
                req.job_type,
                req.fault_rate,
                req.workers,
                req.run_count,
                req.max_cases,
                req.seed,
                "queued",
                now,
                now,
            ),
        )
        row = conn.execute(
            "SELECT * FROM jobs WHERE id = %s AND user_id = %s", (job_id, user.id)
        ).fetchone()
        conn.commit()

    if row is None:
        raise HTTPException(status_code=500, detail="failed to create job")

    _spawn_job(
        job_id=job_id,
        suite_path=str(suite_path),
        provider=req.provider,
        model=req.model.strip(),
        api_key=api_key,
        custom_endpoint=custom_endpoint,
        job_type=req.job_type,
        fault_rate=req.fault_rate,
        run_count=req.run_count,
        max_cases=req.max_cases,
        seed=req.seed,
        workers=req.workers,
    )
    _log_security_event(
        event_type="job_created",
        ip=ip,
        user_id=user.id,
        details={
            "job_id": job_id,
            "provider": req.provider,
            "job_type": req.job_type,
            "max_cases": req.max_cases,
            "run_count": req.run_count,
        },
    )
    return _row_to_job_summary(row)


@app.get("/jobs", response_model=JobListResponse)
def list_jobs(user: Annotated[UserPublic, Depends(_require_user)]) -> JobListResponse:
    with _connect_db() as conn:
        rows = conn.execute(
            """
            SELECT * FROM jobs
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT 100
            """,
            (user.id,),
        ).fetchall()
    jobs = [_row_to_job_summary(row) for row in rows]
    return JobListResponse(jobs=jobs)


@app.get("/jobs/{job_id}", response_model=JobSummary)
def get_job(
    job_id: str,
    user: Annotated[UserPublic, Depends(_require_user)],
) -> JobSummary:
    with _connect_db() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE id = %s AND user_id = %s",
            (job_id, user.id),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="job not found")
    return _row_to_job_summary(row)


@app.delete("/jobs/{job_id}", status_code=200)
def cancel_job(
    request: Request,
    job_id: str,
    user: Annotated[UserPublic, Depends(_require_user)],
) -> dict[str, str]:
    with _connect_db() as conn:
        row = conn.execute(
            "SELECT status FROM jobs WHERE id = %s AND user_id = %s",
            (job_id, user.id),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="job not found")
    status_value = str(row["status"])
    if status_value not in {"queued", "running"}:
        raise HTTPException(status_code=409, detail=f"job is already {status_value}")

    # Terminate the subprocess if it is running.
    with _running_processes_lock:
        proc = _running_processes.pop(job_id, None)
    if proc is not None and proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)

    now = _utc_now_iso()
    with _connect_db() as conn:
        conn.execute(
            """
            UPDATE jobs
            SET status = %s, error_message = %s, updated_at = %s, finished_at = %s
            WHERE id = %s
            """,
            ("cancelled", "Cancelled by user", now, now, job_id),
        )
    _log_security_event(
        event_type="job_cancelled",
        ip=_client_ip(request),
        user_id=user.id,
        details={"job_id": job_id},
    )
    return {"status": "cancelled"}


def _load_job_report_payload(*, job_id: str, user_id: str) -> tuple[str, dict[str, object]]:
    with _connect_db() as conn:
        row = conn.execute(
            "SELECT status, result_json FROM jobs WHERE id = %s AND user_id = %s",
            (job_id, user_id),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="job not found")

    status_value = str(row["status"])
    if status_value not in {"completed", "failed"}:
        raise HTTPException(
            status_code=409,
            detail=f"report not available while job status is {status_value}",
        )

    raw_payload = row["result_json"]
    if not isinstance(raw_payload, str):
        raise HTTPException(status_code=500, detail="report payload missing")
    try:
        report_payload = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="report payload is invalid") from exc
    if not isinstance(report_payload, dict):
        raise HTTPException(status_code=500, detail="report payload has unexpected shape")
    return (status_value, report_payload)


@app.get("/jobs/{job_id}/report", response_model=JobReportResponse)
def get_job_report(
    job_id: str,
    user: Annotated[UserPublic, Depends(_require_user)],
) -> JobReportResponse:
    _, report_payload = _load_job_report_payload(job_id=job_id, user_id=user.id)
    return JobReportResponse(job_id=job_id, report=report_payload)


@app.get("/jobs/{job_id}/traces", response_model=JobTracesResponse)
def get_job_traces(
    job_id: str,
    user: Annotated[UserPublic, Depends(_require_user)],
) -> JobTracesResponse:
    with _connect_db() as conn:
        job_row = conn.execute(
            "SELECT job_type FROM jobs WHERE id = %s AND user_id = %s",
            (job_id, user.id),
        ).fetchone()
        if job_row is None:
            raise HTTPException(status_code=404, detail="job not found")
        job_type = str(job_row["job_type"])
        if job_type != "agent_benchmark":
            raise HTTPException(
                status_code=400,
                detail="traces are only available for job_type 'agent_benchmark'",
            )
        rows = conn.execute(
            """
            SELECT task_id, run_index, trace_json
            FROM agent_traces
            WHERE job_id = %s
            ORDER BY task_id ASC, run_index ASC, id ASC
            """,
            (job_id,),
        ).fetchall()

    traces: list[JobTraceRecord] = []
    for row in rows:
        raw_trace_json = row["trace_json"]
        if not isinstance(raw_trace_json, str):
            continue
        try:
            parsed = json.loads(raw_trace_json)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=500, detail="stored trace payload is invalid JSON"
            ) from exc
        if not isinstance(parsed, dict):
            raise HTTPException(
                status_code=500,
                detail="stored trace payload has unexpected shape",
            )
        traces.append(
            JobTraceRecord(
                task_id=str(row["task_id"]),
                run_index=int(row["run_index"]),
                trace=parsed,
            )
        )

    return JobTracesResponse(job_id=job_id, traces=traces)


@app.get("/jobs/{job_id}/report/pdf")
def get_job_report_pdf(
    job_id: str,
    user: Annotated[UserPublic, Depends(_require_user)],
) -> Response:
    _, report_payload = _load_job_report_payload(job_id=job_id, user_id=user.id)
    bundle = build_export_bundle(input_report=report_payload, history_reports=[])
    with NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        pdf_path = Path(tmp_file.name)
    try:
        write_compliance_pdf(bundle, pdf_path)
        pdf_bytes = pdf_path.read_bytes()
    finally:
        pdf_path.unlink(missing_ok=True)

    headers = {"Content-Disposition": f'attachment; filename="{job_id}-report.pdf"'}
    return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest, request: Request) -> EvaluateResponse:
    ip = _client_ip(request)
    _enforce_ip_not_blocked(ip)
    _enforce_rate_limit(
        key=f"public:evaluate:{ip}",
        max_requests=PUBLIC_EVAL_RATE_LIMIT_MAX_REQUESTS,
        window_seconds=PUBLIC_EVAL_RATE_LIMIT_WINDOW_SECONDS,
        detail="too many evaluate requests; please try again later",
    )
    try:
        suite_path = resolve_suite_path(base_dir=BASE_DIR, suite=req.suite)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    custom_endpoint = _validate_custom_endpoint(req.provider, req.custom_endpoint)
    api_key = req.api_key.strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="api_key is required")

    try:
        report = _run_benchmark_report(
            suite_path=suite_path,
            job_type="benchmark",
            provider=req.provider,
            model=req.model.strip(),
            api_key=api_key,
            custom_endpoint=custom_endpoint,
            run_count=req.run_count,
            max_cases=req.max_cases,
            seed=req.seed,
        )
    except Exception as exc:
        safe_error = _sanitize_error_message(str(exc), api_key)
        raise HTTPException(status_code=502, detail=f"evaluation failed: {safe_error}") from exc

    mean_asi = report.get("mean_asi")
    num_cases = report.get("num_cases")
    domain_scores = report.get("domain_scores")
    if not isinstance(mean_asi, (int, float)):
        raise HTTPException(status_code=500, detail="invalid benchmark output: mean_asi missing")
    if not isinstance(num_cases, int):
        raise HTTPException(status_code=500, detail="invalid benchmark output: num_cases missing")
    if not isinstance(domain_scores, dict):
        raise HTTPException(
            status_code=500, detail="invalid benchmark output: domain_scores missing"
        )

    parsed_domain_scores: dict[str, float] = {}
    for key, value in domain_scores.items():
        if isinstance(key, str) and isinstance(value, (int, float)):
            parsed_domain_scores[key] = float(value)

    return EvaluateResponse(
        model=req.model.strip(),
        provider=req.provider,
        asi=round(float(mean_asi), 2),
        domain_scores=parsed_domain_scores,
        num_cases=num_cases,
        run_count=req.run_count,
    )
