"""Tests for kb_fetch: PDF/HTML extraction, user agent, SSRF guard. No network."""

import signal
import time
from pathlib import Path

import kb_fetch
import pytest


def _minimal_pdf(text: str | None) -> bytes:
    """A minimal one-page PDF, with or without a text operator."""
    if text is None:
        content = b""
    else:
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        content = f"BT /F1 24 Tf 72 700 Td ({escaped}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream",
    ]
    chunks = [b"%PDF-1.4\n"]
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(sum(map(len, chunks)))
        chunks.append(f"{number} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref_offset = sum(map(len, chunks))
    xref = [f"xref\n0 {len(objects) + 1}\n".encode(), b"0000000000 65535 f \n"]
    xref.extend(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:])
    trailer = f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode()
    return b"".join([*chunks, *xref, trailer])


ARTICLE_HTML = b"""<html><head><title>HTML Title</title></head><body><article><h1>HTML Title</h1>
<p>This real paragraph describes a helpful and detailed source for the knowledge base harvester.</p>
<p>A second paragraph adds enough connected content for the extraction to keep it.</p></article></body></html>"""


def test_extract_pdf_reads_minimal_pdf_without_metadata():
    title, date, body = kb_fetch._extract_pdf(_minimal_pdf("Hello PDF"), "https://example.com/test.pdf")
    assert title == "" and date == "" and "Hello PDF" in body


def test_extract_pdf_rejects_pdf_without_text_layer():
    with pytest.raises(RuntimeError, match="without a text layer"):
        kb_fetch._extract_pdf(_minimal_pdf(None), "https://example.com/scan.pdf")


def test_fetch_article_routes_pdf_by_magic_bytes(monkeypatch):
    pdf = _minimal_pdf("Magic PDF")
    monkeypatch.setattr(kb_fetch, "safe_fetch", lambda url: (pdf, "utf-8", "application/octet-stream"))
    assert "Magic PDF" in kb_fetch.fetch_article("https://example.com/generic")[2]


def test_fetch_article_routes_html_to_trafilatura(monkeypatch):
    monkeypatch.setattr(kb_fetch, "safe_fetch", lambda url: (ARTICLE_HTML, "utf-8", "text/html"))
    assert "This real paragraph" in kb_fetch.fetch_article("https://example.com/article")[2]


def test_fetch_article_without_main_text_fails_loudly(monkeypatch):
    monkeypatch.setattr(kb_fetch, "safe_fetch", lambda url: (b"<html><body></body></html>", "utf-8", "text/html"))
    with pytest.raises(RuntimeError, match="no main text"):
        kb_fetch.fetch_article("https://example.com/app")


class _Headers(dict):
    def get_content_charset(self):
        return "utf-8"

    def get_content_type(self):
        return self.get("Content-Type", "text/html")


class _Resp:
    def __init__(self, status: int, body: bytes = b"", headers: dict | None = None):
        self.status, self.body, self.headers = status, body, _Headers(headers or {})

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def getcode(self):
        return self.status

    def read(self, n: int = -1):
        out, self.body = self.body, b""  # a stream: the body once, then EOF
        return out


class _Opener:
    """Records every request; answers from a {url: _Resp} map."""

    def __init__(self, answers: dict):
        self.answers, self.requests = answers, []

    def open(self, req, timeout=None):
        self.requests.append(req)
        return self.answers[req.full_url]


@pytest.fixture
def opener(monkeypatch):
    def install(answers: dict) -> _Opener:
        fake = _Opener(answers)
        monkeypatch.setattr(kb_fetch, "_OPENER", fake)
        return fake

    monkeypatch.setattr(kb_fetch, "host_is_public", lambda host: host.endswith(".example"))
    return install


def test_requests_use_browser_user_agent_by_default(opener, monkeypatch):
    monkeypatch.delenv("KB_USER_AGENT", raising=False)
    fake = opener({"https://docs.example/a": _Resp(200, b"<html></html>")})
    kb_fetch.safe_fetch("https://docs.example/a")
    assert fake.requests[0].get_header("User-agent") == kb_fetch.DEFAULT_USER_AGENT
    assert "de" not in fake.requests[0].get_header("Accept-language")


def test_kb_user_agent_overrides_the_default(opener, monkeypatch):
    monkeypatch.setenv("KB_USER_AGENT", "my-crawler/1.0 (+mailto:me@example.org)")
    fake = opener({"https://docs.example/a": _Resp(200, b"x")})
    kb_fetch.safe_fetch("https://docs.example/a")
    assert fake.requests[0].get_header("User-agent") == "my-crawler/1.0 (+mailto:me@example.org)"


def test_redirects_are_followed_and_checked_per_hop(opener):
    fake = opener(
        {
            "https://docs.example/old": _Resp(301, headers={"Location": "/new"}),
            "https://docs.example/new": _Resp(200, b"moved here"),
        }
    )
    assert kb_fetch.safe_fetch("https://docs.example/old")[0] == b"moved here"
    assert [r.full_url for r in fake.requests] == ["https://docs.example/old", "https://docs.example/new"]


def test_redirect_to_internal_host_is_refused(opener):
    opener({"https://docs.example/x": _Resp(302, headers={"Location": "https://169.254.169.254/latest/meta-data"})})
    with pytest.raises(RuntimeError, match="non-public"):
        kb_fetch.safe_fetch("https://docs.example/x")


def test_redirect_to_plain_http_is_refused(opener):
    opener({"https://docs.example/x": _Resp(302, headers={"Location": "http://docs.example/x"})})
    with pytest.raises(RuntimeError, match="non-https"):
        kb_fetch.safe_fetch("https://docs.example/x")


def test_non_https_is_refused_before_any_request(opener):
    fake = opener({})
    with pytest.raises(RuntimeError, match="non-https"):
        kb_fetch.safe_fetch("http://docs.example/a")
    assert fake.requests == []


def test_oversized_body_is_refused(opener, monkeypatch):
    monkeypatch.setattr(kb_fetch, "MAX_FETCH_BYTES", 10)
    opener({"https://docs.example/big": _Resp(200, b"x" * 11)})
    with pytest.raises(RuntimeError, match="exceeds"):
        kb_fetch.safe_fetch("https://docs.example/big")


class _Stall(_Resp):
    """A server that sends a byte every few seconds: each socket read succeeds, so the socket timeout
    never fires and the read just keeps blocking (in the body, a header or a chunk size line)."""

    def read(self, n: int = -1):
        time.sleep(5)
        return b"x"


def test_a_stalling_download_hits_the_total_time_limit(opener, monkeypatch):
    monkeypatch.setattr(kb_fetch, "FETCH_TOTAL_S", 0.2)
    opener({"https://docs.example/slow": _Stall(200)})
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="took longer"):
        kb_fetch.safe_fetch("https://docs.example/slow")
    assert time.monotonic() - started < 2


def test_time_limit_is_cleared_after_a_fetch(opener):
    before = signal.getsignal(signal.SIGALRM)
    opener({"https://docs.example/a": _Resp(200, b"<html></html>")})
    kb_fetch.safe_fetch("https://docs.example/a")
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0 and signal.getsignal(signal.SIGALRM) is before


@pytest.mark.parametrize("host", ["127.0.0.1", "10.0.0.1", "192.168.1.10", "169.254.169.254", "::1", "0.0.0.0"])
def test_internal_addresses_are_not_public(host):
    assert kb_fetch.host_is_public(host) is False


def test_no_browser_automation_in_the_fetcher():
    """The public copy must never ship code that drives a browser to get past bot protection."""
    source = Path(kb_fetch.__file__).read_text(encoding="utf-8").lower()
    for word in ("playwright", "selenium", "puppeteer", "patchright", "cloudflare challenge"):
        assert word not in source, word
