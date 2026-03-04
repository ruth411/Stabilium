from __future__ import annotations

from pathlib import Path


def write_compliance_pdf(bundle: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = _bundle_lines(bundle)
    pdf_data = _render_single_page_pdf(lines)
    output_path.write_bytes(pdf_data)


def _bundle_lines(bundle: dict[str, object]) -> list[str]:
    summary = bundle.get("summary")
    metrics = bundle.get("metrics")
    methodology = bundle.get("methodology")
    attestation = bundle.get("attestation")

    summary_obj = summary if isinstance(summary, dict) else {}
    metrics_obj = metrics if isinstance(metrics, dict) else {}
    methodology_obj = methodology if isinstance(methodology, dict) else {}
    attestation_obj = attestation if isinstance(attestation, dict) else {}

    asi = summary_obj.get("asi_score")
    confidence = summary_obj.get("asi_confidence")
    confidence_obj = confidence if isinstance(confidence, dict) else {}
    ci_low = confidence_obj.get("ci_low")
    ci_high = confidence_obj.get("ci_high")
    sample_size = confidence_obj.get("sample_size")

    trend = metrics_obj.get("trend")
    trend_obj = trend if isinstance(trend, dict) else {}
    delta = trend_obj.get("delta_vs_previous")
    direction = trend_obj.get("direction")

    report_sha = attestation_obj.get("report_sha256")
    signature = attestation_obj.get("signature_hmac_sha256")
    signed = attestation_obj.get("signed")

    lines = [
        "Stabilium Compliance Export",
        "",
        f"Created At: {bundle.get('created_at_utc', 'n/a')}",
        f"Report Type: {summary_obj.get('report_type', 'n/a')}",
        f"Model Provider: {summary_obj.get('model_provider', 'n/a')}",
        f"Model Name: {summary_obj.get('model_name', 'n/a')}",
        f"Suite Name: {summary_obj.get('suite_name', 'n/a')}",
        "",
        "Reliability Summary",
        f"ASI Score: {asi}",
        f"ASI 95% CI: [{ci_low}, {ci_high}]",
        f"Sample Size (n): {sample_size}",
        "",
        "Trend",
        f"Direction: {direction}",
        f"Delta vs Previous: {delta}",
        "",
        "Methodology",
        f"Version: {methodology_obj.get('version', 'n/a')}",
        f"Confidence Method: {methodology_obj.get('confidence_method', 'n/a')}",
        f"Significance Method: {methodology_obj.get('significance_method', 'n/a')}",
        "",
        "Attestation",
        f"Report SHA256: {report_sha}",
        f"Signed: {signed}",
        f"Signature (HMAC-SHA256): {signature}",
    ]
    # Keep single-page output deterministic and readable.
    return lines[:44]


def _render_single_page_pdf(lines: list[str]) -> bytes:
    escaped_lines = [_escape_pdf_text(line) for line in lines]
    content_parts = ["BT", "/F1 11 Tf", "50 760 Td"]
    for line in escaped_lines:
        content_parts.append(f"({line}) Tj")
        content_parts.append("T*")
    content_parts.append("ET")
    content_stream = "\n".join(content_parts).encode("latin-1", errors="replace")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Count 1 /Kids [3 0 R] >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length "
        + str(len(content_stream)).encode("ascii")
        + b" >>\nstream\n"
        + content_stream
        + b"\nendstream",
    ]

    result = bytearray()
    result.extend(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(result))
        result.extend(f"{index} 0 obj\n".encode("ascii"))
        result.extend(obj)
        result.extend(b"\nendobj\n")

    xref_offset = len(result)
    result.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    result.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        result.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    result.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(result)


def _escape_pdf_text(value: str) -> str:
    escaped = value.replace("\\", "\\\\")
    escaped = escaped.replace("(", "\\(").replace(")", "\\)")
    return escaped
