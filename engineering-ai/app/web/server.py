"""Stage 2b web UI server (stdlib only, no new dependencies).

Serves the interactive extraction page and runs the deterministic Stage 2/2b
pipeline on uploaded files or pasted text.

Run from the repo root (engineering-ai/):

    python -m app.web.server [--port 8000]

Then open http://localhost:8000
"""

from __future__ import annotations

import errno
import json
import sys
import tempfile
from argparse import ArgumentParser
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from app.config import load_project_inputs, project_inputs_provenance
from app.extraction.document_loader import ExtractedDocument, TextChunk, load_document
from app.extraction.field_extractor import extract_fields
from app.pdf_builder import CHART_CONFLICT_MIN_SIDES, executive_summary_filename
from app.report.executive_summary import build_executive_summary_narrative_pdf
from app.report.key_points import extract_key_points, is_key_point_parameter
from app.report.verification_table import append_verification_table_pdf
from app.report.pdf_report import ExtractionPayload, generate_pdf
from app.report.summary import describe_field_coverage, generate_clean_summary
from app.ai.fallback import maybe_build_ai_suggestions

WEB_DIR = Path(__file__).resolve().parent
INDEX_HTML = WEB_DIR / "index.html"

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB


def _json_response(handler: BaseHTTPRequestHandler, payload, status: int = 200) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _pdf_response(handler: BaseHTTPRequestHandler, body: bytes, filename: str) -> None:
    """Send PDF bytes as a download (never as JSON)."""
    handler.send_response(200)
    handler.send_header("Content-Type", "application/pdf")
    handler.send_header("Content-Disposition", f'attachment; filename="{filename}"')
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _render_pdf(fields: list[dict], conflicts: list[dict], warnings: list[str]) -> bytes:
    """Build the fixed-layout PDF and return its bytes.

    The browser posts the fields it currently holds, so any value the user
    corrected on screen is what gets printed. This module is the only place
    the PDF is assembled -- the UI never builds a report itself.
    """
    payload = ExtractionPayload.from_dict(
        {"fields": fields, "conflicts": conflicts, "warnings": warnings}
    )
    with tempfile.TemporaryDirectory(prefix="engai-pdf-") as tmp:
        out = generate_pdf(payload, Path(tmp) / "review.pdf")
        return out.read_bytes()


def _read_body(handler: BaseHTTPRequestHandler) -> bytes:
    length = int(handler.headers.get("Content-Length") or 0)
    if length <= 0:
        return b""
    if length > MAX_UPLOAD_BYTES:
        raise ValueError("upload too large (limit 20 MB)")
    return handler.rfile.read(length)


def _parse_multipart(
    content_type: str, body: bytes
) -> tuple[list[tuple[str, bytes]], dict[str, str]]:
    """Return (files, fields) pairs from a multipart/form-data body."""
    msg = BytesParser(policy=email_policy).parsebytes(
        b"Content-Type: " + content_type.encode("utf-8") + b"\r\n\r\n" + body
    )
    files: list[tuple[str, bytes]] = []
    fields: dict[str, str] = {}
    for part in msg.iter_parts():
        name = part.get_filename()
        if name:
            files.append((name, part.get_payload(decode=True) or b""))
            continue
        field_name = part.get_param("name", header="content-disposition")
        if field_name:
            payload = part.get_payload(decode=True) or b""
            fields[field_name] = payload.decode("utf-8", "replace")
    return files, fields


def get_val(fields, param_name, context_contains=None):
    needle = str(param_name).lower().rstrip(".")
    context_needle = (
        str(context_contains).lower() if context_contains is not None else None
    )
    for field in fields:
        parameter = str(field.get("Parameter", field.get("parameter", ""))).lower()
        context = str(field.get("Context", field.get("context", ""))).lower()
        if needle not in parameter.rstrip("."):
            continue
        if context_needle is not None and context_needle not in context:
            continue
        value = field.get("Value", field.get("value"))
        unit = field.get("Unit", field.get("unit")) or ""
        if value is None:
            value = "Not specified"
        return f"{value} {unit}".strip()
    return "Not specified"


def build_clean_summary(fields, project=None):
    return generate_clean_summary(fields, project=project)


def _inputs_from_form(fields: dict[str, str]) -> object:
    """Read the optional ``project_inputs`` JSON form field from the upload."""
    raw = fields.get("project_inputs")
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _run_extraction(
    documents: list[ExtractedDocument], project_inputs: object = None
) -> dict:
    result = extract_fields(documents)
    payload = result.to_dict()
    key_points = extract_key_points(result, documents).to_dict()
    payload["key_points"] = key_points
    payload["clean_engineering_summary"] = build_clean_summary(key_points)
    payload["field_coverage"] = describe_field_coverage(payload)
    payload["document_text"] = "\n\n".join(doc.full_text() for doc in documents)[:60_000]
    payload["project_inputs"] = load_project_inputs(overrides=project_inputs)
    payload["provenance"] = {"project_inputs": project_inputs_provenance()}
    payload["warnings"] = _conflict_warnings(payload) + list(
        payload.get("warnings") or []
    )
    # Optional, flag-gated, and never merged into `fields`: suggestions are
    # separate AI-generated entries for missing required key points. When the
    # flag is off this returns None and the payload is byte-for-byte unchanged.
    ai_suggestions = maybe_build_ai_suggestions(
        documents, result, payload["key_points"]
    )
    if ai_suggestions is not None:
        payload["ai_suggestions"] = ai_suggestions
    return payload


def _conflict_warnings(payload: dict) -> list[str]:
    """Restate each *meaningful* unresolved conflict as a warning.

    Conflicts are still reported separately (and never resolved); this makes the
    ones a summary would actually surface visible in the warning list. The field
    extractor also flags every chart/coordinate column and long plotted data
    series (see ``is_key_point_parameter``); those are drawing data, not
    specification disagreements, so they are omitted here -- exactly as the
    Executive Summary omits them.
    """
    warnings: list[str] = []
    for conflict in payload.get("conflicts") or []:
        if not isinstance(conflict, dict):
            continue
        parameter = conflict.get("parameter") or "value"
        if not is_key_point_parameter(parameter):
            continue
        sides = sorted(
            {
                f"{field.get('value')} {field.get('unit') or ''}".strip()
                for field in conflict.get("fields") or []
                if isinstance(field, dict)
            }
        )
        if len(sides) >= CHART_CONFLICT_MIN_SIDES:
            continue  # chart/analysis data series, not a conflict
        warnings.append(f"Unresolved conflict — {parameter}: {'; '.join(sides)}")
    return warnings


def _pasted_document(text: str) -> ExtractedDocument:
    stripped = text.strip()
    return ExtractedDocument(
        path=Path("pasted-text.txt"),
        file_type="txt",
        text_chunks=[TextChunk("pasted text", stripped)] if stripped else [],
    )


class _Handler(BaseHTTPRequestHandler):
    server_version = "EngineeringAI/2b"

    def log_message(self, fmt: str, *args) -> None:
        pass  # keep console quiet; errors are returned to the client

    # -- routes -------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] not in ("/", "/index.html"):
            _json_response(self, {"error": "not found"}, status=404)
            return
        body = INDEX_HTML.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        # Always serve the current UI: a stale cached page hides newly added
        # form fields until the user does a hard refresh.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        try:
            result = self._handle_post()
        except Exception as exc:  # never crash the thread, report to the client
            _json_response(self, {"error": str(exc)}, status=400)
            return
        if isinstance(result, tuple):
            body, filename = result
            _pdf_response(self, body, filename)
            return
        _json_response(self, result)

    def _handle_post(self):
        route = self.path.split("?", 1)[0]
        if route == "/api/extract":
            return self._api_extract()
        if route == "/api/pdf":
            return self._api_pdf()
        if route == "/api/executive-summary.pdf":
            return self._api_executive_summary_pdf()
        raise ValueError(f"unknown route {route}")

    def _api_executive_summary_pdf(self) -> tuple[bytes, str]:
        """Render the five-paragraph Executive Summary PDF.

        Stateless like every other route here: the browser sends the data it
        already holds, so any value corrected on screen is the value printed,
        and no model is loaded to rebuild it.
        """
        data = json.loads(_read_body(self).decode("utf-8") or "{}")
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        try:
            body = build_executive_summary_narrative_pdf(data)
        except Exception as exc:
            print(f"[PDF] ERROR building Executive Summary: {exc}")
            import traceback
            traceback.print_exc()
            # Try to produce something minimal
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas as pdfcanvas
            import io
            buf = io.BytesIO()
            c = pdfcanvas.Canvas(buf, pagesize=A4)
            c.drawString(72, 750, "Executive Summary")
            c.showPage()
            c.save()
            body = buf.getvalue()
        try:
            body = append_verification_table_pdf(body, data)
        except Exception as exc:
            print(f"[PDF] ERROR appending verification table: {exc}")
            import traceback
            traceback.print_exc()
        # Try to add cover page if available
        try:
            from app.report.cover_page_v2 import build_cover_page_pdf
            import pymupdf
            cover_fields = data.get("cover_page") or {}
            # merge in some fields if passed differently, but don't auto-extract from report
            cover_pdf = build_cover_page_pdf(cover_fields)
            # prepend
            base = pymupdf.open(stream=cover_pdf, filetype="pdf")
            rest = pymupdf.open(stream=body, filetype="pdf")
            base.insert_pdf(rest)
            body = base.tobytes(garbage=4, deflate=True)
            base.close()
            rest.close()
        except Exception as exc:
            print(f"[PDF] ERROR adding cover page: {exc}", flush=True)
            import traceback
            traceback.print_exc()
            # Still generate rest of PDF even if cover fails
        return body, executive_summary_filename(data)

    def _api_pdf(self) -> tuple[bytes, str]:
        """Render the fixed-layout PDF from the fields the browser sends."""
        data = json.loads(_read_body(self).decode("utf-8") or "{}")
        fields = [item for item in data.get("fields") or [] if isinstance(item, dict)]
        conflicts = [item for item in data.get("conflicts") or [] if isinstance(item, dict)]
        warnings = [str(item) for item in data.get("warnings") or []]
        try:
            body = _render_pdf(fields, conflicts, warnings)
        except Exception as exc:
            print(f"[PDF] ERROR rendering PDF: {exc}")
            import traceback
            traceback.print_exc()
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas as pdfcanvas
            import io
            buf = io.BytesIO()
            c = pdfcanvas.Canvas(buf, pagesize=A4)
            c.drawString(72, 750, "Helical Pile Review Report")
            c.showPage()
            c.save()
            body = buf.getvalue()
        try:
            from app.report.cover_page_v2 import build_cover_page_pdf
            import pymupdf
            cover_fields = data.get("cover_page") or {}
            cover_pdf = build_cover_page_pdf(cover_fields)
            base = pymupdf.open(stream=cover_pdf, filetype="pdf")
            rest = pymupdf.open(stream=body, filetype="pdf")
            base.insert_pdf(rest)
            body = base.tobytes(garbage=4, deflate=True)
            base.close()
            rest.close()
        except Exception as exc:
            print(f"[PDF] ERROR adding cover page: {exc}", flush=True)
            import traceback
            traceback.print_exc()
        return body, "helical_pile_review_report.pdf"

    def _api_extract(self) -> dict:
        content_type = self.headers.get("Content-Type", "")
        body = _read_body(self)
        documents: list[ExtractedDocument] = []

        if "multipart/form-data" in content_type:
            files, fields = _parse_multipart(content_type, body)
            if not files:
                raise ValueError("no files received")
            inputs = _inputs_from_form(fields)
            with tempfile.TemporaryDirectory(prefix="engai-upload-") as tmp:
                for name, data in files:
                    path = Path(tmp) / (Path(name).name or "upload")
                    path.write_bytes(data)
                    documents.append(load_document(path))
                return _run_extraction(documents, inputs)

        if "application/json" in content_type:
            data = json.loads(body.decode("utf-8") or "{}")
            text = (data.get("text") or "").strip()
            if not text:
                raise ValueError("no text received")
            return _run_extraction([_pasted_document(text)], data.get("project_inputs"))

        raise ValueError("unsupported content type; send multipart/form-data or JSON")


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Stage 2b extraction UI server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    server = None
    try:
        server = ThreadingHTTPServer((args.host, args.port), _Handler)
    except OSError as exc:
        if getattr(exc, "errno", None) == errno.EADDRINUSE:
            print(
                f"Port {args.port} is already in use (another server is running).",
                file=sys.stderr,
            )
            print(
                "Stop that server, or start this one on a different port:",
                file=sys.stderr,
            )
            print(
                f"    python -m app.web.server --port {args.port + 1}",
                file=sys.stderr,
            )
            return 1
        raise

    print(f"Serving on http://{args.host}:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

