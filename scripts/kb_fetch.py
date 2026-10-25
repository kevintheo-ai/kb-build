"""
kb_fetch.py - download helpers for kb_harvest.py.

- fetch_article(url): HTML (main text via trafilatura) or PDF (text layer via pypdf).
- fetch_youtube(video_id): English transcript via youtube-transcript-api. kb_harvest only calls
  it for KBs with `allow_youtube: true`, see README "YouTube transcripts".

Requests go out with a regular browser user agent, because many documentation sites turn away
unknown clients. Set KB_USER_AGENT to identify your crawler instead. kb-build does not read
robots.txt: check the terms of every site before you add it as a source.

SSRF guard: the source URLs come from your own config, but the pages are served by third parties.
A compromised listed site or an open redirect could point a fetch at localhost, a private network
or a cloud metadata address. So the fetcher follows redirects itself and checks every hop:
https only, the host must resolve to public addresses only, bounded size and time, fail closed.

Fetched text is stored as data. Nothing in it is ever executed.
"""

from __future__ import annotations

import contextlib
import ipaddress
import os
import re
import signal
import socket
import threading
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
FETCH_TIMEOUT_S = 20  # per socket operation
FETCH_TOTAL_S = 120  # whole download of one URL
MAX_FETCH_BYTES = 8_000_000
MAX_REDIRECTS = 5
PDF_MAX_PAGES = 200  # time bound for very long PDFs
_REDIRECT_CODES = {301, 302, 303, 307, 308}


def user_agent() -> str:
    return os.environ.get("KB_USER_AGENT") or DEFAULT_USER_AGENT


def slugify(text: str, maxlen: int = 60) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text[:maxlen].strip("-") or "untitled"


# Characters YAML does not allow anywhere in a document. PDF metadata titles can carry them, and
# one of them makes the whole front matter unreadable (source and trust of the doc are lost).
_YAML_FORBIDDEN = re.compile("[^\t\n\r\x20-\x7e\x85\xa0-\ud7ff\ue000-\ufffd\U00010000-\U0010ffff]")


def yaml_escape(value: str) -> str:
    """Safe single-line YAML scalar (double-quoted, escaped)."""
    # Every line break YAML knows becomes a space: reading the file back turns a lone CR into a line
    # break ("\r---\r" in a title would end the front matter early), NEL/LS/PS would be folded silently.
    v = re.sub("[\r\n\x85\u2028\u2029]+", " ", _YAML_FORBIDDEN.sub("", value or ""))
    v = v.replace("\\", "\\\\").replace('"', '\\"').strip()
    return f'"{v}"'


def yaml_value(value) -> str:
    """Front matter value of manual enrichment: strings via yaml_escape, anything else dumped by the
    same YAML library that reads it back, so booleans, numbers, dates and lists keep their type."""
    if isinstance(value, str):
        return yaml_escape(value)
    import yaml

    class NoAliases(yaml.SafeDumper):
        """Each field is dumped on its own, so every dump would name its anchor &id001: in one front
        matter that is a duplicate anchor and the whole block becomes unreadable. Write values out."""

        def ignore_aliases(self, data):
            return True

    dumped = yaml.dump(value, Dumper=NoAliases, default_flow_style=True, allow_unicode=True, width=1 << 30)
    return dumped.removesuffix("\n...\n").strip()  # a scalar comes with a document-end marker


class _NoAutoRedirect(urllib.request.HTTPRedirectHandler):
    """Hand the 3xx response back so the caller validates the next hop itself."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = urllib.request.build_opener(_NoAutoRedirect)


def host_is_public(host: str) -> bool:
    """True only if the host resolves AND every address is globally routable (no loopback,
    private, link-local, multicast or reserved range). Any error counts as not public."""
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    if not infos:
        return False
    for info in infos:
        raw = info[4][0].split("%", 1)[0]  # strip an IPv6 zone id
        try:
            addr = ipaddress.ip_address(raw)
        except ValueError:
            return False
        if not addr.is_global or addr.is_multicast or addr.is_reserved:
            return False
    return True


def safe_fetch(url: str) -> tuple[bytes, str, str]:
    """SSRF-guarded GET -> (raw bytes, charset, content type), all hops within FETCH_TOTAL_S."""
    with _time_limit(FETCH_TOTAL_S, url):
        return _fetch(url)


def _fetch(url: str) -> tuple[bytes, str, str]:
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        parsed = urllib.parse.urlparse(current)
        if parsed.scheme != "https":
            raise RuntimeError(f"refusing non-https URL: {current!r}")
        host = parsed.hostname or ""
        if not host or not host_is_public(host):
            raise RuntimeError(f"refusing URL to a non-public or unresolvable host {host!r}: {current!r}")
        # No Accept-Encoding: urllib does not decompress on its own.
        req = urllib.request.Request(
            current,
            headers={
                "User-Agent": user_agent(),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        try:
            resp = _OPENER.open(req, timeout=FETCH_TIMEOUT_S)
        except urllib.error.HTTPError as exc:
            # Some stacks surface a 3xx as HTTPError: follow it by hand like any other hop.
            if exc.code in _REDIRECT_CODES and exc.headers.get("Location"):
                current = urllib.parse.urljoin(current, exc.headers["Location"])
                continue
            raise RuntimeError(f"HTTP {exc.code} for {current}") from exc
        with resp:
            status = resp.getcode()
            if status in _REDIRECT_CODES:
                loc = resp.headers.get("Location")
                if not loc:
                    raise RuntimeError(f"redirect without Location for {current}")
                current = urllib.parse.urljoin(current, loc)
                continue
            if status != 200:
                raise RuntimeError(f"HTTP {status} for {current}")
            data = resp.read(MAX_FETCH_BYTES + 1)
            charset = resp.headers.get_content_charset() or "utf-8"
            content_type = (resp.headers.get_content_type() or "").lower()
        if len(data) > MAX_FETCH_BYTES:
            raise RuntimeError(f"response body exceeds {MAX_FETCH_BYTES} bytes for {current}")
        return data, charset, content_type
    raise RuntimeError(f"too many redirects (>{MAX_REDIRECTS}) starting at {url}")


@contextlib.contextmanager
def _time_limit(seconds: float, url: str):
    """Hard limit for one whole fetch. The socket timeout only bounds each single read: a server that
    sends a byte every few seconds (in the headers, a chunk size or the body) would hold the harvest
    and its lock for hours. SIGALRM interrupts any blocking read, but only in the main thread."""
    if threading.current_thread() is not threading.main_thread():
        yield  # ponytail: no hard limit off the main thread; kb_harvest fetches in the main thread
        return

    def _expired(signum, frame):
        raise RuntimeError(f"download took longer than {seconds}s for {url}")

    previous = signal.signal(signal.SIGALRM, _expired)
    signal.setitimer(signal.ITIMER_REAL, max(seconds, 0.001))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)  # ponytail: an outer ITIMER_REAL is not restored
        signal.signal(signal.SIGALRM, previous)


def _extract_pdf(data: bytes, url: str) -> tuple[str, str, str]:
    """(title, creation date, text) from a PDF byte stream."""
    import io

    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        with contextlib.suppress(Exception):  # an empty owner password must not reject the PDF
            reader.decrypt("")
    pages: list[str] = []
    for page in reader.pages[:PDF_MAX_PAGES]:
        try:
            pages.append(page.extract_text() or "")
        except Exception:  # noqa: S112 - skip a single broken page
            continue
    body = re.sub(r"\n{3,}", "\n\n", "\n\n".join(pages)).strip()
    if not body:
        raise RuntimeError(f"PDF without a text layer (probably a scan, no OCR): {url}")
    try:
        metadata = reader.metadata
    except Exception:
        metadata = None
    try:
        title = (metadata.title if metadata else "") or ""
    except Exception:
        title = ""
    date = ""
    with contextlib.suppress(Exception):  # a broken date field only costs the date
        created = metadata.creation_date if metadata else None
        if created:
            date = created.date().isoformat()
    return title.strip(), date.strip(), body


def _extract_html(html: str) -> tuple[str, str, str]:
    """(title, date, main text) via trafilatura."""
    import trafilatura

    meta = trafilatura.extract_metadata(html)
    title = (meta.title if meta and meta.title else "") or ""
    date = (meta.date if meta and meta.date else "") or ""
    body = trafilatura.extract(html, include_comments=False, include_tables=True, favor_recall=True) or ""
    return title.strip(), str(date).strip(), body.strip()


def fetch_article(url: str) -> tuple[str, str, str]:
    """(title, date, body text) from an HTML page or a PDF. Raises on any hard failure."""
    data, charset, content_type = safe_fetch(url)
    if not data:
        raise RuntimeError(f"empty response for {url}")
    if data[:5] == b"%PDF-" or "application/pdf" in content_type:
        return _extract_pdf(data, url)
    title, date, body = _extract_html(data.decode(charset, errors="replace"))
    if not body:
        # Typical for pages that render their text with JavaScript: kb-build does not run a
        # browser. Save the page as a document under kb/<slug>/docs/ instead.
        raise RuntimeError(f"no main text extracted from {url} (JavaScript-rendered page?)")
    return title, date, body


def fetch_youtube(video_id: str) -> str:
    """English transcript text of one YouTube video (unofficial endpoint, see README)."""
    from youtube_transcript_api import YouTubeTranscriptApi

    fetched = YouTubeTranscriptApi().fetch(video_id, languages=["en", "en-US", "en-GB"])
    text = " ".join(s.text for s in fetched if s.text and s.text.strip())
    if not text.strip():
        raise RuntimeError(f"empty transcript for YouTube video {video_id}")
    return text.strip()
