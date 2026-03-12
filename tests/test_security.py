from __future__ import annotations

import socket

import pytest

from agent_stability_engine.security import (
    assert_public_endpoint_host,
    sanitize_error_message,
    validate_custom_endpoint_url,
)


def test_sanitize_error_message_redacts_explicit_secret() -> None:
    message = "OpenAI key failed: sk-proj-abc123SECRET"
    sanitized = sanitize_error_message(message, secrets=["sk-proj-abc123SECRET"])
    assert "sk-proj-abc123SECRET" not in sanitized
    assert "[REDACTED_API_KEY]" in sanitized


def test_sanitize_error_message_redacts_bearer_tokens() -> None:
    message = "Authorization: Bearer sk-ant-api03-abcdef0123456789"
    sanitized = sanitize_error_message(message)
    assert "sk-ant-api03-abcdef0123456789" not in sanitized
    assert "Bearer [REDACTED_TOKEN]" in sanitized


def test_sanitize_error_message_respects_max_length() -> None:
    sanitized = sanitize_error_message("x" * 50, max_length=12)
    assert sanitized == "x" * 12


def test_validate_custom_endpoint_url_accepts_https() -> None:
    assert validate_custom_endpoint_url("https://example.com/v1/infer") == (
        "https://example.com/v1/infer"
    )


def test_validate_custom_endpoint_url_rejects_private_ip() -> None:
    with pytest.raises(ValueError, match="private or non-routable"):
        validate_custom_endpoint_url("https://10.0.0.8/infer")


def test_validate_custom_endpoint_url_rejects_non_allowlisted_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ASE_CUSTOM_ENDPOINT_ALLOWLIST", "allowed.example.com")
    with pytest.raises(ValueError, match="ALLOWLIST"):
        validate_custom_endpoint_url("https://example.com/infer")


def test_assert_public_endpoint_host_rejects_private_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_getaddrinfo(*_args: object, **_kwargs: object) -> list[tuple[object, ...]]:
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                ("10.1.2.3", 443),
            )
        ]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(ValueError, match="private or non-routable"):
        assert_public_endpoint_host("https://example.com/infer")


def test_assert_public_endpoint_host_accepts_public_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_getaddrinfo(*_args: object, **_kwargs: object) -> list[tuple[object, ...]]:
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                ("93.184.216.34", 443),
            )
        ]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    assert_public_endpoint_host("https://example.com/infer")
