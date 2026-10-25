"""
kb_common.py - shared core of kb-build.

Reads the registry and turns one entry into a KBConfig. Every knowledge base (KB) lives in its
own Qdrant collection, so searches never mix domains and a KB can be rebuilt or deleted alone.

Locations (environment variables):
    KB_HOME         base folder for data and the default registry (default: this repo)
    KB_REGISTRY     registry file (default: $KB_HOME/config/kb_registry.yaml)
    QDRANT_URL      Qdrant server (default: http://127.0.0.1:6333)
    QDRANT_API_KEY  optional

Used by kb_harvest.py, kb_ingest.py, kb_search.py, kb_refresh.py and by the generated advisor
agents (through kb_search.py).
"""

from __future__ import annotations

import os
import re
import sys
import time
import uuid as _uuid
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info < (3, 11):  # noqa: UP036 - a clear message for people on older interpreters
    sys.exit(f"kb-build needs Python 3.11+, this is {sys.version_info[0]}.{sys.version_info[1]}")

# Load BGE-M3 from the local cache only: no online check on every start and no silent model
# update (new weights would make the stored vectors incompatible). The one-time download is
# `python scripts/bge_embedder.py`; set HF_HUB_OFFLINE=0 to override. This must run before any
# third-party import: huggingface_hub reads the variable once, when it is first imported.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from qdrant_client import models

KB_HOME = Path(os.environ.get("KB_HOME") or Path(__file__).resolve().parent.parent).expanduser().resolve()
REGISTRY_PATH = Path(os.environ.get("KB_REGISTRY") or KB_HOME / "config" / "kb_registry.yaml").expanduser().resolve()
ASSETS_ROOT = KB_HOME / "kb"  # source documents per KB: kb/<slug>/harvest and kb/<slug>/docs
STATE_ROOT = KB_HOME / "state"  # derived state per KB: ingest manifest, locks, curation marker
QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")

VECTOR_SIZE = 1024  # BGE-M3 dense
DEFAULT_CHUNK_SIZE = 1000  # characters; per KB via `chunk_size`
DEFAULT_CHUNK_OVERLAP = 120

# Payload fields every KB carries (filters + the advisors' duty to cite their sources).
STANDARD_FRONTMATTER_KEYS = (
    "author",
    "author_slug",
    "affiliation",
    "title",
    "date",
    "topic",
    "content_type",
    "source_url",
    "trust",
    "tags",
    "harvested_at",  # last (re)fetch: the freshness signal for living docs without a publication date
)

VALID_TRUST = {"high", "medium", "low"}
VALID_STATUS = {"active", "paused", "archived"}


class KBError(RuntimeError):
    """Registry or configuration problem. Always raised with a message that says what to do."""


@dataclass
class KBConfig:
    slug: str
    title: str = ""
    description: str = ""
    collection: str = ""
    status: str = "active"  # active | paused | archived
    created: str = ""
    content_dirs: list[Path] = field(default_factory=list)
    include: list[str] = field(default_factory=lambda: ["*.md"])
    sources_config: Path | None = None  # web sources for kb_harvest.py
    allow_youtube: bool = False  # YouTube transcripts are opt-in, see README
    advisor: str = ""  # name of the generated advisor agent (empty = none yet)
    default_trust: str = "medium"
    context_prefix: bool = True  # breadcrumb (title > H1 > H2) in front of the embedded text
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    golden_queries: list[dict] = field(default_factory=list)  # [{q, expect, expect_file}]
    control_queries: list[str] = field(default_factory=list)  # empty = built-in off-topic questions
    # Refresh behaviour (kb_refresh.py). refresh_enabled None = auto: harvest refresh only if
    # the KB has a sources_config; ingest and eval run for every active KB.
    refresh_enabled: bool | None = None
    refetch_days: int = 90  # docs older than this are fetched again (content drift at the same URL)
    curation_days: int = 90  # after this, the refresh summary reports curation_due
    # Recency weighting in the ranking:
    # final = (1-w)*RRF + w*exp(-ln2 * age_days / half_life). w=0 turns it off.
    recency_weight: float = 0.3
    half_life_days: int = 365

    def harvest_refresh_enabled(self) -> bool:
        if self.refresh_enabled is not None:
            return self.refresh_enabled and self.sources_config is not None
        return self.sources_config is not None

    def state_dir(self) -> Path:
        return STATE_ROOT / self.slug

    def manifest_path(self) -> Path:
        return self.state_dir() / "ingest_manifest.json"


def default_collection(slug: str) -> str:
    """`support-handbook` -> `support_handbook_kb`."""
    base = re.sub(r"[^a-z0-9]+", "_", slug.lower()).strip("_")
    if not base:
        raise KBError(f"cannot derive a collection name from slug {slug!r}")
    return f"{base}_kb"


def _expand(p: str | Path) -> Path:
    """Paths inside the registry: `~` expands, relative paths resolve against KB_HOME."""
    path = Path(str(p)).expanduser()
    return path if path.is_absolute() else KB_HOME / path


def _num(raw: dict, key: str, default, cast, slug: str):
    """Numeric registry values: a typo like `chunk_size: big` gives a KBError, not a traceback."""
    val = raw.get(key, default)
    try:
        return cast(val)
    except (TypeError, ValueError) as exc:
        raise KBError(f"KB {slug!r}: `{key}: {val!r}` is not a number ({cast.__name__} expected)") from exc


def load_registry(registry_path: Path | None = None) -> dict:
    """The raw registry dict. A missing file is a hard, explained error."""
    path = Path(registry_path).expanduser() if registry_path else REGISTRY_PATH
    if not path.exists():
        raise KBError(
            f"registry not found: {path}\n"
            "  -> cp config/kb_registry.example.yaml config/kb_registry.yaml (or set KB_REGISTRY)"
        )
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise KBError(f"registry {path} is not a YAML mapping")
    data.setdefault("kbs", {})
    if not isinstance(data["kbs"], dict):
        raise KBError(f"registry {path}: `kbs:` must be a mapping")
    # Effective collection names must be unique, otherwise two KBs mix their points and the
    # ingest of one deletes the other's.
    seen: dict[str, str] = {}
    for slug, raw in data["kbs"].items():
        coll = str((raw or {}).get("collection") or default_collection(str(slug)))
        other = seen.get(coll)
        if other is not None:
            raise KBError(f"collection collision: KBs {other!r} and {slug!r} both map to {coll!r}")
        seen[coll] = str(slug)
    return data


def list_kbs(registry_path: Path | None = None) -> list[str]:
    return sorted(load_registry(registry_path)["kbs"].keys())


def get_kb(slug: str, registry_path: Path | None = None) -> KBConfig:
    """KBConfig for `slug`, strictly validated."""
    reg = load_registry(registry_path)
    raw = reg["kbs"].get(slug)
    if raw is None:
        avail = ", ".join(sorted(reg["kbs"])) or "(none)"
        raise KBError(f"unknown KB {slug!r}. Registry KBs: {avail}")
    if not isinstance(raw, dict):
        raise KBError(f"registry entry {slug!r} is not a mapping")
    # The slug becomes a path component (data and state folders): validate it strictly instead
    # of allowing traversal.
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", slug):
        raise KBError(f"invalid KB slug {slug!r}: allowed is [a-z0-9][a-z0-9_-]{{0,63}}")
    # Another registry (--registry, tests, staging) does NOT isolate the per-slug folders and the
    # collection. A slug or a collection of the main registry in a test registry would change the
    # real KB (`demo_web` and `demo-web` share `demo_web_kb`, so slugs alone are not enough).
    main = Path(REGISTRY_PATH).resolve()
    if registry_path is not None and Path(registry_path).expanduser().resolve() != main and main.exists():
        try:
            main_kbs = load_registry(main).get("kbs") or {}
        except KBError:
            main_kbs = {}  # broken main registry: the guard cannot be enforced, the main path stays usable
        main_colls = {str((r or {}).get("collection") or default_collection(str(s))) for s, r in main_kbs.items()}
        coll = str(raw.get("collection") or default_collection(slug))
        if slug in main_kbs or coll in main_colls:
            raise KBError(
                f"KB {slug!r} (collection {coll!r}) collides with the main registry ({main}). Runs with another "
                f"registry share data, state and collection: use your own slug (for example `{slug}-test`)."
            )

    cfg = KBConfig(slug=slug)
    cfg.title = str(raw.get("title", slug))
    cfg.description = str(raw.get("description", ""))
    cfg.collection = str(raw.get("collection") or default_collection(slug))
    cfg.status = str(raw.get("status", "active"))
    # Exact enum: `status: Active` would otherwise silently drop out of the refresh.
    if cfg.status not in VALID_STATUS:
        raise KBError(f"KB {slug!r}: status {cfg.status!r} is not one of {sorted(VALID_STATUS)} (exact, lowercase)")
    cfg.created = str(raw.get("created", ""))
    # A YAML scalar instead of a list becomes a one-element list (otherwise the loop below
    # would iterate over characters). An empty key (YAML null) counts as not set.
    raw_dirs = raw.get("content_dirs") or []
    raw_dirs = [raw_dirs] if isinstance(raw_dirs, str) else raw_dirs
    raw_inc = raw.get("include")
    raw_inc = ["*.md"] if raw_inc is None else [raw_inc] if isinstance(raw_inc, str) else raw_inc
    # Check before expanding: an empty entry would expand to KB_HOME itself and pull in every file.
    if not raw_dirs or any(not isinstance(d, str) or not d.strip() for d in raw_dirs):
        raise KBError(f"KB {slug!r}: content_dirs is empty or invalid")
    if not raw_inc or any(not isinstance(g, str) or not g.strip() for g in raw_inc):  # switch off: status: paused
        raise KBError(f"KB {slug!r}: include is empty or invalid")
    cfg.content_dirs = [_expand(d) for d in raw_dirs]
    cfg.include = [str(g) for g in raw_inc]
    sc = raw.get("sources_config")
    cfg.sources_config = _expand(sc) if sc else None
    allow_yt = raw.get("allow_youtube", False)
    if not isinstance(allow_yt, bool):
        raise KBError(f"KB {slug!r}: `allow_youtube: {allow_yt!r}` must be a native YAML boolean (true/false)")
    cfg.allow_youtube = allow_yt
    cfg.advisor = str(raw.get("advisor", ""))
    cfg.default_trust = str(raw.get("default_trust", "medium"))
    ctx = raw.get("context_prefix", True)
    if not isinstance(ctx, bool):  # the string "false" would be truthy
        raise KBError(f"KB {slug!r}: `context_prefix: {ctx!r}` must be a native YAML boolean (true/false)")
    cfg.context_prefix = ctx
    cfg.chunk_size = _num(raw, "chunk_size", DEFAULT_CHUNK_SIZE, int, slug)
    cfg.chunk_overlap = _num(raw, "chunk_overlap", DEFAULT_CHUNK_OVERLAP, int, slug)
    gq = raw.get("golden_queries", [])
    cfg.golden_queries = [g for g in gq if isinstance(g, dict) and g.get("q")]
    cq = raw.get("control_queries", [])
    if not isinstance(cq, list) or not all(isinstance(q, str) and q.strip() for q in cq):
        raise KBError(f"KB {slug!r}: `control_queries` must be a list of non-empty strings")
    cfg.control_queries = cq
    refresh = raw.get("refresh") or {}
    if isinstance(refresh, dict):
        if "enabled" in refresh:
            # Native YAML booleans only: `enabled: "false"` (a quoted string) is truthy and would
            # switch the network harvest ON.
            if not isinstance(refresh["enabled"], bool):
                raise KBError(
                    f"KB {slug!r}: `refresh.enabled: {refresh['enabled']!r}` must be a native YAML boolean "
                    "(true/false without quotes)"
                )
            cfg.refresh_enabled = refresh["enabled"]
        cfg.refetch_days = _num(refresh, "refetch_days", cfg.refetch_days, int, slug)
        cfg.curation_days = _num(refresh, "curation_days", cfg.curation_days, int, slug)
        if cfg.refetch_days < 1 or cfg.curation_days < 1:
            raise KBError(
                f"KB {slug!r}: refresh.refetch_days and curation_days must be >= 1 "
                f"(got {cfg.refetch_days}/{cfg.curation_days})"
            )
    cfg.recency_weight = _num(raw, "recency_weight", cfg.recency_weight, float, slug)
    cfg.half_life_days = _num(raw, "half_life_days", cfg.half_life_days, int, slug)
    if not (0.0 <= cfg.recency_weight <= 0.8):
        raise KBError(f"KB {slug!r}: recency_weight {cfg.recency_weight} outside [0, 0.8]")
    if cfg.half_life_days < 30:
        raise KBError(f"KB {slug!r}: half_life_days {cfg.half_life_days} < 30 (too aggressive)")
    if cfg.chunk_size <= 0 or not (0 <= cfg.chunk_overlap < cfg.chunk_size):
        raise KBError(
            f"KB {slug!r}: chunk_size {cfg.chunk_size}/chunk_overlap {cfg.chunk_overlap} invalid "
            "(chunk_size > 0 and 0 <= overlap < chunk_size)"
        )
    if cfg.default_trust not in VALID_TRUST:
        raise KBError(f"KB {slug!r}: default_trust {cfg.default_trust!r} is not one of {sorted(VALID_TRUST)}")
    return cfg


# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------
_clients: dict[int, object] = {}


def get_client(timeout: int = 10):
    """Qdrant client for QDRANT_URL (optional QDRANT_API_KEY). One retry against a short blip,
    then a KBError that says how to start Qdrant. No silent fallback to a second store."""
    if timeout in _clients:
        return _clients[timeout]
    from qdrant_client import QdrantClient

    api_key = os.environ.get("QDRANT_API_KEY") or None
    last: Exception | None = None
    for attempt in range(2):
        try:
            client = QdrantClient(url=QDRANT_URL, api_key=api_key, timeout=timeout)
            client.get_collections()
            _clients[timeout] = client
            return client
        except Exception as exc:
            last = exc
            if attempt == 0:
                time.sleep(0.5)
    raise KBError(
        f"Qdrant is not reachable at {QDRANT_URL} ({type(last).__name__}: {last}).\n"
        "  Start it from the repo folder: docker compose up -d\n"
        "  Or point QDRANT_URL (and QDRANT_API_KEY) at your Qdrant server."
    )


# ---------------------------------------------------------------------------
# Deterministic point IDs, one UUID namespace per collection.
# Never change this: every stored point would become an orphan.
# ---------------------------------------------------------------------------
def kb_namespace(collection: str) -> _uuid.UUID:
    return _uuid.uuid5(_uuid.NAMESPACE_URL, f"kb-build://{collection}")


def point_id(collection: str, file_path: str, chunk_index: int, content_hash: str) -> str:
    return str(_uuid.uuid5(kb_namespace(collection), f"{collection}:{file_path}:{chunk_index}:{content_hash}"))


def ensure_collection(client, collection: str) -> bool:
    """BGE-M3 collection (dense 1024 cosine + sparse lexical weights). Idempotent; True if created."""
    if client.collection_exists(collection):
        return False
    client.create_collection(
        collection_name=collection,
        vectors_config={
            "dense": models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False)),
        },
    )
    return True


def ensure_payload_indexes(client, collection: str) -> None:
    """Payload indexes for the standard filters (idempotent)."""
    from qdrant_client.http.exceptions import UnexpectedResponse

    indexes = {
        "kb": models.PayloadSchemaType.KEYWORD,
        "content_hash": models.PayloadSchemaType.KEYWORD,
        "file_path": models.PayloadSchemaType.KEYWORD,
        "author": models.PayloadSchemaType.KEYWORD,
        "author_slug": models.PayloadSchemaType.KEYWORD,
        "topic": models.PayloadSchemaType.KEYWORD,
        "content_type": models.PayloadSchemaType.KEYWORD,
        "trust": models.PayloadSchemaType.KEYWORD,
        "source_url": models.PayloadSchemaType.KEYWORD,
        "year": models.PayloadSchemaType.INTEGER,
    }
    for fieldname, schema_type in indexes.items():
        try:
            client.create_payload_index(collection, fieldname, schema_type)
        except UnexpectedResponse as exc:
            # Only "already exists" (409) is harmless. Anything else (e.g. a 400 type conflict)
            # would silently disable a filter (--trust, --author, --since-year).
            if getattr(exc, "status_code", None) == 409:
                continue
            raise


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
def parse_frontmatter(raw: str) -> tuple[dict, str]:
    """YAML front matter -> (fields, body). Without front matter: ({}, raw). Malformed front
    matter is reported loudly: the document would otherwise lose its source and trust silently."""
    raw = raw.lstrip("\ufeff")  # a UTF-8 BOM would hide the opening ---
    if not raw.startswith("---"):
        return {}, raw
    # The end marker is a line of its own: a value such as "----" or "---text" is not.
    m = re.compile(r"\n---[ \t]*(?:\n|$)").search(raw, 3)
    if m is None:
        return {}, raw
    block = raw[3 : m.start()]
    body = raw[m.end() :].lstrip("\n")
    try:
        import yaml

        data = yaml.safe_load(block)
        if not isinstance(data, dict):
            print("WARN: front matter present but not a mapping - source attribution is lost", file=sys.stderr)
            return {}, body
        return data, body
    except Exception as exc:
        print(f"WARN: front matter not parseable ({exc}) - source attribution is lost", file=sys.stderr)
        return {}, body


# A `date` before this year counts as an extraction artefact: page-date guessers read section
# numbers or copyright lines as dates. The year stays in the payload (it is real for laws and
# court rulings); only the freshness falls back to the fetch date.
PLAUSIBLE_MIN_YEAR = 2015


def freshness_anchor(payload: dict):
    """(datetime, basis) for freshness: `date` (publication) when plausible -> "date", else
    `harvested_at` (a living doc without a date was current when fetched) -> "fetched", else None."""
    from datetime import UTC, datetime, timedelta

    def _parse(s) -> datetime | None:
        txt = str(s or "")
        # Strict ISO first. Impossible calendar dates are dropped, not clamped: "2026-99-99"
        # must not get top freshness.
        m = re.search(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", txt)
        if m:
            try:
                return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=UTC)
            except ValueError:
                return None
        # Only a year: mid-year is precise enough for a half-life measured in months.
        m = re.search(r"\b(20\d{2})\b", txt)
        if m:
            return datetime(int(m.group(1)), 7, 1, tzinfo=UTC)
        return None

    horizon = datetime.now(UTC) + timedelta(days=2)  # time-zone slack for "today/tomorrow"
    dt = _parse(payload.get("date"))
    # Plausibility in both directions: too old and in the future are both artefacts. A typo
    # like "2099" must not push genuinely fresh documents down.
    if dt is not None and dt.year >= PLAUSIBLE_MIN_YEAR and dt <= horizon:
        return dt, "date"
    dt = _parse(payload.get("harvested_at"))
    if dt is None or dt > horizon:
        return None
    return dt, "fetched"


def recency_score(payload: dict, half_life_days: int) -> tuple[float, int] | None:
    """Freshness factor for the ranking -> (exp(-ln2 * age / half_life), age_days). None (no
    anchor) is treated as neutral by the caller: undated docs are neither punished nor preferred."""
    import math
    from datetime import UTC, datetime

    anchor = freshness_anchor(payload)
    if anchor is None:
        return None
    age_days = max(0, (datetime.now(UTC) - anchor[0]).days)
    return math.exp(-math.log(2) * age_days / max(half_life_days, 1)), age_days


def derive_year(date_str: str | None) -> int | None:
    """'2026-05-17' / '17.05.2026' / '2026' -> 2026 (best effort)."""
    if not date_str:
        return None
    m = re.search(r"\b(20\d{2}|19\d{2})\b", str(date_str))
    return int(m.group(1)) if m else None


def build_context_prefix(title: str, h1: str, h2: str, h3: str) -> str:
    """Breadcrumb in front of the embedded text, so a chunk stays findable without its document
    (a cheap version of contextual retrieval)."""
    parts = [p.strip() for p in (title, h1, h2, h3) if p and p.strip()]
    dedup: list[str] = []
    for p in parts:
        if not dedup or dedup[-1].lower() != p.lower():
            dedup.append(p)
    return " > ".join(dedup)
