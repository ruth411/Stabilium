from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import os
import re
import secrets
import socket
from collections.abc import Sequence
from urllib.parse import quote, urlsplit

_GENERIC_SECRET_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{10,}\b")
_BEARER_RE = re.compile(r"(Bearer\s+)([^\s,;]+)", re.IGNORECASE)
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _env_truthy(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def sanitize_error_message(
    message: str,
    *,
    secrets: Sequence[str] | None = None,
    max_length: int | None = None,
) -> str:
    """Redact API-key-like values and explicit secret strings from error text."""
    sanitized = message

    for secret in secrets or ():
        candidate = secret.strip()
        if candidate and candidate in sanitized:
            sanitized = sanitized.replace(candidate, "[REDACTED_API_KEY]")

    sanitized = _GENERIC_SECRET_RE.sub("[REDACTED_API_KEY]", sanitized)
    sanitized = _BEARER_RE.sub(r"\1[REDACTED_TOKEN]", sanitized)

    if max_length is not None and max_length >= 0:
        return sanitized[:max_length]
    return sanitized


def validate_custom_endpoint_url(endpoint_url: str) -> str:
    """Validate user-supplied custom endpoint URL for safety and consistency."""
    cleaned = endpoint_url.strip()
    parsed = urlsplit(cleaned)

    if parsed.scheme not in {"http", "https"}:
        msg = "endpoint_url must start with http:// or https://"
        raise ValueError(msg)
    if parsed.username or parsed.password:
        msg = "endpoint_url must not include URL-embedded credentials"
        raise ValueError(msg)
    host = (parsed.hostname or "").strip().lower()
    if not host:
        msg = "endpoint_url must include a hostname"
        raise ValueError(msg)

    # Default production posture: HTTPS only. Local/dev can explicitly opt in.
    if parsed.scheme != "https" and not _env_truthy("ASE_ALLOW_INSECURE_CUSTOM_ENDPOINTS"):
        msg = "endpoint_url must use https unless ASE_ALLOW_INSECURE_CUSTOM_ENDPOINTS=true"
        raise ValueError(msg)

    if host in _LOCAL_HOSTS or host.endswith(".local") or host.endswith(".internal"):
        msg = "endpoint_url host is not allowed"
        raise ValueError(msg)

    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None and not ip.is_global:
        msg = "endpoint_url must not target private or non-routable IP addresses"
        raise ValueError(msg)

    allowlist_raw = os.getenv("ASE_CUSTOM_ENDPOINT_ALLOWLIST", "").strip()
    if allowlist_raw:
        allowlist = {item.strip().lower() for item in allowlist_raw.split(",") if item.strip()}
        if host not in allowlist:
            msg = "endpoint_url host is not in ASE_CUSTOM_ENDPOINT_ALLOWLIST"
            raise ValueError(msg)

    return cleaned


def assert_public_endpoint_host(endpoint_url: str) -> None:
    """Resolve endpoint host and reject private/non-routable targets."""
    if _env_truthy("ASE_ALLOW_PRIVATE_DNS_TARGETS"):
        return

    parsed = urlsplit(endpoint_url)
    host = (parsed.hostname or "").strip()
    if not host:
        msg = "endpoint_url must include a hostname"
        raise ValueError(msg)

    # If host is already an IP literal, validate directly.
    try:
        direct_ip = ipaddress.ip_address(host)
    except ValueError:
        direct_ip = None
    if direct_ip is not None:
        if not direct_ip.is_global:
            msg = "endpoint_url resolved to a private or non-routable address"
            raise ValueError(msg)
        return

    try:
        infos = socket.getaddrinfo(host, parsed.port or 443, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        msg = "endpoint_url hostname could not be resolved"
        raise ValueError(msg) from exc

    resolved_any = False
    for info in infos:
        sockaddr = info[4]
        if not isinstance(sockaddr, tuple) or not sockaddr:
            continue
        ip_raw = sockaddr[0]
        try:
            resolved_ip = ipaddress.ip_address(ip_raw)
        except ValueError:
            continue
        resolved_any = True
        if not resolved_ip.is_global:
            msg = "endpoint_url resolved to a private or non-routable address"
            raise ValueError(msg)

    if not resolved_any:
        msg = "endpoint_url hostname did not resolve to a valid IP address"
        raise ValueError(msg)


def generate_totp_secret(*, bytes_length: int = 20) -> str:
    raw = secrets.token_bytes(bytes_length)
    return base64.b32encode(raw).decode("ascii").rstrip("=")


def build_otpauth_uri(*, secret: str, account_name: str, issuer: str = "Stabilium") -> str:
    label = quote(f"{issuer}:{account_name}")
    issuer_q = quote(issuer)
    secret_q = quote(secret)
    return f"otpauth://totp/{label}?secret={secret_q}&issuer={issuer_q}"


def totp_code(
    secret: str,
    *,
    timestamp: int,
    period_seconds: int = 30,
    digits: int = 6,
) -> str:
    normalized = secret.strip().upper()
    if not normalized:
        raise ValueError("empty TOTP secret")
    padded = normalized + "=" * ((8 - len(normalized) % 8) % 8)
    key = base64.b32decode(padded, casefold=True)
    counter = int(timestamp // period_seconds)
    msg = counter.to_bytes(8, byteorder="big", signed=False)
    digest = hmac.new(key, msg, hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    code_int = int.from_bytes(digest[offset : offset + 4], byteorder="big") & 0x7FFFFFFF
    code = code_int % (10**digits)
    return f"{code:0{digits}d}"


def verify_totp(
    secret: str,
    code: str,
    *,
    timestamp: int,
    period_seconds: int = 30,
    digits: int = 6,
    window: int = 1,
) -> bool:
    trimmed = "".join(ch for ch in code if ch.isdigit())
    if len(trimmed) != digits:
        return False
    for delta in range(-window, window + 1):
        candidate_ts = timestamp + delta * period_seconds
        if candidate_ts < 0:
            continue
        candidate = totp_code(
            secret,
            timestamp=candidate_ts,
            period_seconds=period_seconds,
            digits=digits,
        )
        if hmac.compare_digest(candidate, trimmed):
            return True
    return False
