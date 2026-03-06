from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Literal

# Make sure the engine is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from agent_stability_engine.adapters import AnthropicChatAdapter, OpenAIChatAdapter
from agent_stability_engine.engine.embeddings import EmbeddingProvider
from agent_stability_engine.runners.benchmark import run_benchmark_suite

BASE_DIR = Path(__file__).parent.parent
SUITE_PATH = BASE_DIR / "examples" / "benchmarks" / "large_suite.json"
DB_PATH = Path(os.getenv("ASE_API_DB_PATH", str(Path(__file__).parent / "stabilium_api.db")))
SESSION_TTL_HOURS = int(os.getenv("ASE_API_SESSION_TTL_HOURS", "168"))
_PASSWORD_ITERATIONS = 310_000
_auth_scheme = HTTPBearer(auto_error=False)


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


def _connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def _init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect_db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                business_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
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
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            """
        )


class UserPublic(BaseModel):
    id: str
    business_name: str
    email: str
    created_at: str


class AuthResponse(BaseModel):
    token: str
    user: UserPublic


class RegisterRequest(BaseModel):
    business_name: str = Field(min_length=1, max_length=200)
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=8, max_length=128)


class JobCreateRequest(BaseModel):
    provider: Literal["openai", "anthropic"]
    model: str = Field(min_length=1, max_length=200)
    api_key: str = Field(min_length=1, max_length=2048)
    run_count: int = Field(default=3, ge=1, le=10)
    max_cases: int = Field(default=5, ge=1, le=100)
    seed: int = 42


class JobSummary(BaseModel):
    id: str
    status: Literal["queued", "running", "completed", "failed"]
    provider: str
    model: str
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


class JobListResponse(BaseModel):
    jobs: list[JobSummary]


class JobReportResponse(BaseModel):
    job_id: str
    report: dict[str, object]


class EvaluateRequest(BaseModel):
    provider: Literal["openai", "anthropic"]
    model: str = Field(min_length=1, max_length=200)
    api_key: str = Field(min_length=1, max_length=2048)
    run_count: int = Field(default=3, ge=1, le=10)
    max_cases: int = Field(default=5, ge=1, le=100)
    seed: int = 42


class EvaluateResponse(BaseModel):
    model: str
    provider: str
    asi: float
    domain_scores: dict[str, float]
    num_cases: int
    run_count: int


def _row_to_user(row: sqlite3.Row) -> UserPublic:
    return UserPublic(
        id=str(row["id"]),
        business_name=str(row["business_name"]),
        email=str(row["email"]),
        created_at=str(row["created_at"]),
    )


def _row_to_job_summary(row: sqlite3.Row) -> JobSummary:
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

    return JobSummary(
        id=str(row["id"]),
        status=str(row["status"]),
        provider=str(row["provider"]),
        model=str(row["model"]),
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
    )


def _create_session(conn: sqlite3.Connection, user_id: str) -> str:
    now = _utc_now()
    expires = now + timedelta(hours=SESSION_TTL_HOURS)
    token = f"ase_{secrets.token_urlsafe(32)}"
    conn.execute(
        """
        INSERT INTO sessions (token, user_id, created_at, expires_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            token,
            user_id,
            now.isoformat().replace("+00:00", "Z"),
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
            SELECT s.expires_at, u.id, u.business_name, u.email, u.created_at
            FROM sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.token = ?
            """,
            (token,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=401, detail="invalid session")
        expires_at = _parse_utc_iso(str(row["expires_at"]))
        if expires_at <= _utc_now():
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
            conn.commit()
            raise HTTPException(status_code=401, detail="session expired")
        return _row_to_user(row)


def _build_agent(
    provider: str, model: str, api_key: str
) -> OpenAIChatAdapter | AnthropicChatAdapter:
    if provider == "openai":
        return OpenAIChatAdapter(model=model, api_key=api_key)
    if provider == "anthropic":
        return AnthropicChatAdapter(model=model, api_key=api_key)
    raise ValueError(f"unsupported provider: {provider}")


def _run_job(
    *,
    job_id: str,
    provider: str,
    model: str,
    api_key: str,
    run_count: int,
    max_cases: int,
    seed: int,
) -> None:
    started_at = _utc_now_iso()
    with _connect_db() as conn:
        conn.execute(
            """
            UPDATE jobs
            SET status = ?, started_at = ?, updated_at = ?
            WHERE id = ?
            """,
            ("running", started_at, started_at, job_id),
        )
        conn.commit()

    try:
        agent = _build_agent(provider=provider, model=model, api_key=api_key)
        result = run_benchmark_suite(
            suite_path=SUITE_PATH,
            agent_fn=agent,
            run_count=run_count,
            seed=seed,
            embedding_provider=EmbeddingProvider.HASH,
            max_cases=max_cases,
            workers=1,
        )
        payload = json.dumps(result.report)
        finished_at = _utc_now_iso()
        with _connect_db() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, result_json = ?, updated_at = ?, finished_at = ?
                WHERE id = ?
                """,
                ("completed", payload, finished_at, finished_at, job_id),
            )
            conn.commit()
    except Exception as exc:
        finished_at = _utc_now_iso()
        message = str(exc)[:2000] or "evaluation failed"
        with _connect_db() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, error_message = ?, updated_at = ?, finished_at = ?
                WHERE id = ?
                """,
                ("failed", message, finished_at, finished_at, job_id),
            )
            conn.commit()


def _spawn_job(
    *,
    job_id: str,
    provider: str,
    model: str,
    api_key: str,
    run_count: int,
    max_cases: int,
    seed: int,
) -> None:
    thread = threading.Thread(
        target=_run_job,
        kwargs={
            "job_id": job_id,
            "provider": provider,
            "model": model,
            "api_key": api_key,
            "run_count": run_count,
            "max_cases": max_cases,
            "seed": seed,
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


_init_db()

app = FastAPI(title="Stabilium API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/auth/register", response_model=AuthResponse)
def register(req: RegisterRequest) -> AuthResponse:
    _validate_email(req.email)
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
                    id, business_name, email, password_salt, password_hash, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (user_id, req.business_name.strip(), email, salt, password_hash, now),
            )
        except sqlite3.IntegrityError as exc:
            raise HTTPException(status_code=409, detail="email already registered") from exc

        token = _create_session(conn, user_id)
        conn.commit()
        row = conn.execute(
            "SELECT id, business_name, email, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=500, detail="failed to create user")
    return AuthResponse(token=token, user=_row_to_user(row))


@app.post("/auth/login", response_model=AuthResponse)
def login(req: LoginRequest) -> AuthResponse:
    _validate_email(req.email)
    email = _normalize_email(req.email)
    with _connect_db() as conn:
        row = conn.execute(
            """
            SELECT id, business_name, email, created_at, password_salt, password_hash
            FROM users
            WHERE email = ?
            """,
            (email,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=401, detail="invalid email or password")
        if not _verify_password(req.password, str(row["password_salt"]), str(row["password_hash"])):
            raise HTTPException(status_code=401, detail="invalid email or password")
        token = _create_session(conn, str(row["id"]))
        conn.commit()
        user = _row_to_user(row)
    return AuthResponse(token=token, user=user)


@app.get("/auth/me", response_model=UserPublic)
def auth_me(user: Annotated[UserPublic, Depends(_require_user)]) -> UserPublic:
    return user


@app.post("/auth/logout")
def logout(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_auth_scheme)],
    _: Annotated[UserPublic, Depends(_require_user)],
) -> dict[str, bool]:
    if credentials is not None and credentials.credentials:
        with _connect_db() as conn:
            conn.execute("DELETE FROM sessions WHERE token = ?", (credentials.credentials,))
            conn.commit()
    return {"ok": True}


@app.post("/jobs", response_model=JobSummary, status_code=202)
def create_job(
    req: JobCreateRequest,
    user: Annotated[UserPublic, Depends(_require_user)],
) -> JobSummary:
    if not SUITE_PATH.exists():
        raise HTTPException(status_code=500, detail="benchmark suite missing on server")
    api_key = req.api_key.strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="api_key is required")

    now = _utc_now_iso()
    job_id = f"job_{secrets.token_hex(12)}"
    with _connect_db() as conn:
        conn.execute(
            """
            INSERT INTO jobs (
                id, user_id, provider, model, run_count, max_cases, seed,
                status, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                user.id,
                req.provider,
                req.model.strip(),
                req.run_count,
                req.max_cases,
                req.seed,
                "queued",
                now,
                now,
            ),
        )
        row = conn.execute(
            "SELECT * FROM jobs WHERE id = ? AND user_id = ?", (job_id, user.id)
        ).fetchone()
        conn.commit()

    if row is None:
        raise HTTPException(status_code=500, detail="failed to create job")

    _spawn_job(
        job_id=job_id,
        provider=req.provider,
        model=req.model.strip(),
        api_key=api_key,
        run_count=req.run_count,
        max_cases=req.max_cases,
        seed=req.seed,
    )
    return _row_to_job_summary(row)


@app.get("/jobs", response_model=JobListResponse)
def list_jobs(user: Annotated[UserPublic, Depends(_require_user)]) -> JobListResponse:
    with _connect_db() as conn:
        rows = conn.execute(
            """
            SELECT * FROM jobs
            WHERE user_id = ?
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
            "SELECT * FROM jobs WHERE id = ? AND user_id = ?",
            (job_id, user.id),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="job not found")
    return _row_to_job_summary(row)


@app.get("/jobs/{job_id}/report", response_model=JobReportResponse)
def get_job_report(
    job_id: str,
    user: Annotated[UserPublic, Depends(_require_user)],
) -> JobReportResponse:
    with _connect_db() as conn:
        row = conn.execute(
            "SELECT status, result_json FROM jobs WHERE id = ? AND user_id = ?",
            (job_id, user.id),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="job not found")
    status_value = str(row["status"])
    if status_value != "completed":
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
    return JobReportResponse(job_id=job_id, report=report_payload)


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest) -> EvaluateResponse:
    if not SUITE_PATH.exists():
        raise HTTPException(status_code=500, detail="benchmark suite missing on server")
    api_key = req.api_key.strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="api_key is required")

    try:
        agent = _build_agent(provider=req.provider, model=req.model.strip(), api_key=api_key)
        result = run_benchmark_suite(
            suite_path=SUITE_PATH,
            agent_fn=agent,
            run_count=req.run_count,
            seed=req.seed,
            embedding_provider=EmbeddingProvider.HASH,
            max_cases=req.max_cases,
            workers=1,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"evaluation failed: {exc}") from exc

    report = result.report
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
