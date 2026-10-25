"""
kb_harvest.py - download the web sources of a registry KB as Markdown files.

Sources file (JSON, path = `sources_config` in the registry):
    {
      "sources": [
        {
          "name": "Example Docs", "slug": "example-docs",
          "affiliation": "Example Inc.", "default_topic": "basics",
          "trust": "medium",
          "living_doc": true,
          "urls": [{"url": "https://docs.example.com/page", "title": "Curated title", "topic": "setup"}],
          "youtube": [{"id": "VIDEO_ID", "title": "Talk title"}]
        }
      ]
    }

`living_doc` (optional JSON boolean): help centers and policy pages. Their page date is only a
guess by the extractor, so none is written; freshness comes from `harvested_at` instead.
`youtube` entries need `allow_youtube: true` for the KB in the registry (see README).

Output: $KB_HOME/kb/<kb-slug>/harvest/<source-slug>/NNN_<title>.md

    python scripts/kb_harvest.py --kb <slug>             # fetch everything new
    python scripts/kb_harvest.py --kb <slug> --source <source-slug>
    python scripts/kb_harvest.py --kb <slug> --smoke     # offline: validate the sources file
    python scripts/kb_harvest.py --kb <slug> --probe     # one new URL per source
    python scripts/kb_harvest.py --kb <slug> --dry-run   # plan without fetching
    python scripts/kb_harvest.py --kb <slug> --stats
    python scripts/kb_harvest.py --kb <slug> --missing   # configured but never fetched (exit 1 = gaps)

Fetched content is data, never instructions: text is extracted and stored, nothing is executed.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from kb_common import ASSETS_ROOT, VALID_TRUST, KBConfig, KBError, get_kb, parse_frontmatter
from kb_fetch import fetch_article, fetch_youtube, slugify, yaml_escape, yaml_value

MIN_CONTENT_BYTES = 400  # below this the extraction failed (loud fail)
FETCH_SLEEP_S = 1.5  # politeness pause between network calls
FAIL_RATIO_MAX = 0.30  # more than 30 % hard failures = the run fails
# Rate limits (429, YouTube IP blocks) are expected and recover on the next run, so they do not
# count towards the fail ratio. "429" only as a standalone number ("14290" characters is not a
# status code), and no bare "rate" (it matches "generated" in transcript error messages).
RATE_LIMIT_RE = re.compile(r"\b429\b|too ?many ?requests|rate[ _-]?limit|ipblocked|requestblocked")
# Front matter keys that write_doc regenerates from the sources file. Every other key is manual
# enrichment (e.g. `superseded: true`) and must survive a refetch.
GENERATED_FM_KEYS = frozenset(
    {
        "author",
        "author_slug",
        "affiliation",
        "source_url",
        "title",
        "date",
        "topic",
        "content_type",
        "trust",
        "harvested_at",
    }
)


def is_rate_limited(exc: BaseException) -> bool:
    """Rate limit (harmless, recovers next run) rather than a hard failure. URLs are removed
    before matching: "429" or "rate-limit" inside a path is not a status code."""
    blob = re.sub(r"https?://\S+", " ", f"{type(exc).__name__} {exc}").lower()
    return bool(RATE_LIMIT_RE.search(blob))


@dataclass
class Item:
    source_name: str
    source_slug: str
    affiliation: str
    content_type: str  # "article" | "transcript"
    topic: str
    trust: str
    url: str
    youtube_id: str | None = None
    title_hint: str | None = None
    living_doc: bool = False  # page date is a guess: write none, freshness via harvested_at


def harvest_root(kb: KBConfig) -> Path:
    return ASSETS_ROOT / kb.slug / "harvest"


def load_sources(kb: KBConfig) -> list[Item]:
    if not kb.sources_config:
        raise KBError(
            f"KB {kb.slug!r} has no `sources_config` in the registry. For local folders run kb_ingest.py directly."
        )
    if not kb.sources_config.exists():
        raise KBError(f"sources_config not found: {kb.sources_config}")
    raw = json.loads(kb.sources_config.read_text(encoding="utf-8"))
    items: list[Item] = []
    for i, src in enumerate(raw.get("sources", [])):
        name = src.get("name")
        if not name:
            raise KBError(f"sources file: source #{i} without 'name' ({kb.sources_config})")
        # Explicit slugs are slugified too: they become path components, "../" must never pass.
        slug = slugify(src.get("slug") or name)
        affiliation = src.get("affiliation", "")
        default_topic = src.get("default_topic", kb.slug)
        trust = str(src.get("trust", kb.default_trust)).lower()
        if trust not in VALID_TRUST:
            raise KBError(f"source {slug!r}: trust {trust!r} is not one of {sorted(VALID_TRUST)}")
        living = src.get("living_doc", False)
        if not isinstance(living, bool):  # the string "false" would be truthy
            raise KBError(f"source {slug!r}: living_doc must be a JSON boolean (true/false), not {living!r}")
        for art in src.get("urls", []):
            if isinstance(art, str):
                url, topic, title = art, default_topic, None
            else:
                url = art.get("url")
                topic, title = art.get("topic", default_topic), art.get("title")
            if not url:
                raise KBError(f"source {slug!r}: URL entry without 'url' ({kb.sources_config})")
            items.append(
                Item(name, slug, affiliation, "article", topic, trust, url, title_hint=title, living_doc=living)
            )
        videos = src.get("youtube", [])
        if videos and not kb.allow_youtube:
            raise KBError(
                f"source {slug!r} lists YouTube videos, but KB {kb.slug!r} does not allow them. YouTube transcripts "
                "come from an unofficial endpoint that YouTube's robots.txt disallows and that may break YouTube's "
                "terms. If you accept that, set `allow_youtube: true` for this KB (README: YouTube transcripts)."
            )
        for yt in videos:
            if isinstance(yt, str):
                vid, topic, title = yt, default_topic, None
            else:
                vid = yt.get("id")
                topic, title = yt.get("topic", default_topic), yt.get("title")
            if not vid:
                raise KBError(f"source {slug!r}: YouTube entry without 'id' ({kb.sources_config})")
            url = f"https://www.youtube.com/watch?v={vid}"
            items.append(
                Item(name, slug, affiliation, "transcript", topic, trust, url, youtube_id=vid, title_hint=title)
            )
    if not items:
        raise KBError(f"sources file {kb.sources_config} contains no URLs or videos")
    return items


def write_doc(
    root: Path,
    item: Item,
    idx: int,
    title: str,
    date: str,
    body: str,
    *,
    out_path: Path | None = None,
    extra: dict | None = None,
) -> Path:
    # A curated title_hint beats the scraped <title> (redirects inherit the target's title).
    # out_path set = refresh: overwrite the same file so the ingest updates it in place.
    title = item.title_hint or title or f"{item.source_name} {item.content_type}"
    if out_path is None:
        out_dir = root / item.source_slug
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{idx:03d}_{slugify(title)}.md"
        if out_path.exists():  # never overwrite on an index collision
            raise RuntimeError(f"target file already exists (index collision): {out_path}")
    from datetime import UTC, datetime

    frontmatter = (
        "---\n"
        f"author: {yaml_escape(item.source_name)}\n"
        f"author_slug: {yaml_escape(item.source_slug)}\n"
        f"affiliation: {yaml_escape(item.affiliation)}\n"
        f"source_url: {yaml_escape(item.url)}\n"
        f"title: {yaml_escape(title)}\n"
        f"date: {yaml_escape(date)}\n"
        f"topic: {yaml_escape(item.topic)}\n"
        f"content_type: {yaml_escape(item.content_type)}\n"
        f"trust: {item.trust}\n"
        f'harvested_at: "{datetime.now(UTC).date().isoformat()}"\n'
    )
    # Keep manual enrichment across a refetch: well-formed keys only, values keep their type
    # (`superseded: false` must not come back as the truthy string "False").
    for k, v in (extra or {}).items():
        if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", str(k)) and k not in GENERATED_FM_KEYS:
            frontmatter += f"{k}: {yaml_value(v)}\n"
    frontmatter += "---\n\n"
    # Atomic write: a crash or a full disk in the middle of a refetch must never half-overwrite
    # the only good copy.
    tmp = out_path.with_suffix(".md.tmp")
    tmp.write_text(frontmatter + f"# {title}\n\n" + body + "\n", encoding="utf-8")
    os.replace(tmp, out_path)
    return out_path


def _harvested_age_days(f: Path) -> int:
    """Age of a harvested doc in days: `harvested_at`, falling back to the file mtime."""
    from datetime import UTC, datetime

    head = f.read_text(encoding="utf-8", errors="replace")[:700]
    m = re.search(r'^harvested_at: "?(\d{4}-\d{2}-\d{2})"?', head, re.MULTILINE)
    if m:
        try:
            dt = datetime.strptime(m.group(1), "%Y-%m-%d").replace(tzinfo=UTC)
            return max(0, int((datetime.now(UTC) - dt).days))
        except ValueError:
            pass
    return max(0, int((time.time() - f.stat().st_mtime) / 86400))


def _fetch_youtube_backoff(video_id: str, *, attempts: int = 3, base_wait: float = 45.0, spacing: float = 15.0) -> str:
    """Transcript fetches in a row trigger YouTube's 429 limit quickly: keep a fixed spacing
    between calls and back off exponentially on rate limits. Other errors (no transcript,
    video gone) are raised at once."""
    for attempt in range(attempts):
        try:
            text = fetch_youtube(video_id)
            time.sleep(spacing)
            return text
        except Exception as exc:
            if not is_rate_limited(exc) or attempt == attempts - 1:
                raise
            time.sleep(base_wait * (2**attempt))
    raise RuntimeError(f"unreachable: YouTube backoff exhausted for {video_id}")


def harvest(
    kb: KBConfig,
    items: list[Item],
    *,
    probe: bool,
    dry_run: bool,
    source_filter: str | None,
    refetch_days: int | None = None,
) -> int:
    root = harvest_root(kb)
    if source_filter:
        items = [it for it in items if it.source_slug == source_filter]
        if not items:
            raise KBError(f"no items for --source {source_filter!r}")

    # Exclusive lock: a manual harvest and the scheduled refresh of the same KB would compete
    # for file indexes and refetches.
    if not dry_run:
        kb.state_dir().mkdir(parents=True, exist_ok=True)
        lock_f = open(kb.state_dir() / "harvest.lock", "w")  # noqa: SIM115 - held for the whole harvest
        try:
            fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"ERROR: another harvest for {kb.slug!r} is already running (harvest.lock).", file=sys.stderr)
            return 2

    counts = {"fetched": 0, "refreshed": 0, "skipped": 0, "failed": 0, "failed_ratelimit": 0}
    per_src_idx: dict[str, int] = {}
    failures: list[str] = []
    # URL -> file index, built once per source (re-reading every doc per URL is quadratic).
    url_index: dict[str, dict[str, Path]] = {}

    def _prior_for(slug: str, url: str) -> Path | None:
        if slug not in url_index:
            idx: dict[str, Path] = {}
            d = root / slug
            if d.exists():
                for f in sorted(d.glob("*.md")):
                    fm, _ = parse_frontmatter(f.read_text(encoding="utf-8", errors="replace"))
                    u = fm.get("source_url")
                    if u and str(u) not in idx:
                        idx[str(u)] = f
            url_index[slug] = idx
        return url_index[slug].get(url)

    if probe:
        # One URL per source that is not harvested yet: an already fetched one would be skipped
        # and the probe would check nothing.
        picked: dict[str, Item] = {}
        for it in items:
            if it.source_slug not in picked and not _prior_for(it.source_slug, it.url):
                picked[it.source_slug] = it
        items = list(picked.values())
        if not items:
            print("[probe] every source is harvested already, nothing to probe")

    for it in items:
        if it.source_slug not in per_src_idx:
            d = root / it.source_slug
            # Highest number prefix + 1, not the file count: gaps must never hand out an
            # existing number again.
            nums = [
                int(m.group(1)) for f in (d.glob("*.md") if d.exists() else []) if (m := re.match(r"(\d+)_", f.name))
            ]
            per_src_idx[it.source_slug] = (max(nums) + 1) if nums else 0
        prior = None if dry_run else _prior_for(it.source_slug, it.url)
        refresh_due = bool(prior and refetch_days is not None and _harvested_age_days(prior) >= refetch_days)
        if dry_run:
            print(f"[plan] {it.source_slug} {it.content_type} topic={it.topic} :: {it.url}")
            counts["fetched"] += 1
            continue
        if prior and not refresh_due:
            counts["skipped"] += 1
            continue
        old_fm: dict = {}
        if prior and refresh_due:
            old_fm, _ = parse_frontmatter(prior.read_text(encoding="utf-8", errors="replace"))
            # Docs marked as superseded are not fetched again (the marker would be lost).
            if old_fm.get("superseded"):
                counts["skipped"] += 1
                print(f"[skip] superseded: {prior.name}")
                continue
        try:
            if it.content_type == "transcript":
                body = _fetch_youtube_backoff(it.youtube_id or "")
                title, date = it.title_hint or "", ""
            else:
                title, date, body = fetch_article(it.url)
            if it.living_doc:
                date = ""
            if len(body.encode("utf-8")) < MIN_CONTENT_BYTES:
                raise RuntimeError(f"extracted body < {MIN_CONTENT_BYTES} bytes")
            if prior and refresh_due:
                # A block or error page above the minimum size must not silently replace the
                # good old content.
                old_raw = prior.read_text(encoding="utf-8", errors="replace")
                old_end = old_raw.find("\n---", 3)
                old_body_len = len(old_raw[old_end + 4 :] if old_end != -1 else old_raw)
                if len(body) < 0.5 * old_body_len:
                    raise RuntimeError(
                        f"refetched body suspiciously short ({len(body)} vs. {old_body_len} characters before), "
                        "the old doc stays"
                    )
                # A metadata hiccup must not erase a known date (living docs carry none on purpose).
                if not date and not it.living_doc:
                    m = re.search(r'^date: "([^"\n]*)"', old_raw[: old_end if old_end != -1 else 1200], re.MULTILINE)
                    date = m.group(1) if m else ""
                extra = {k: v for k, v in old_fm.items() if k not in GENERATED_FM_KEYS}
                out = write_doc(root, it, 0, title, date, body, out_path=prior, extra=extra)
                counts["refreshed"] += 1
                print(f"[refresh] {out.relative_to(root)}")
            else:
                idx = per_src_idx[it.source_slug]
                out = write_doc(root, it, idx, title, date, body)
                per_src_idx[it.source_slug] = idx + 1
                # Update the index at once: the same URL listed twice must find this new file.
                url_index.setdefault(it.source_slug, {})[it.url] = out
                counts["fetched"] += 1
                print(f"[ok] {out.relative_to(root)}")
        except Exception as exc:
            counts["failed"] += 1
            if is_rate_limited(exc):
                counts["failed_ratelimit"] += 1
            failures.append(f"{it.url} :: {exc}")
            print(f"[FAIL] {it.url} :: {exc}", file=sys.stderr)
        time.sleep(FETCH_SLEEP_S)

    # kb_refresh.py parses this line (failed=N), keep its shape.
    print(
        f"\n=== HARVEST DONE [{kb.slug}] === fetched={counts['fetched']} refreshed={counts['refreshed']} "
        f"skipped={counts['skipped']} failed={counts['failed']} (rate-limited: {counts['failed_ratelimit']})"
    )
    # The fail ratio counts hard failures only, over ALL items in scope (including skipped ones
    # that are already fresh on disk). Counting only the touched items would make a complete KB
    # report ~100 % on the few dead URLs that are left. Trade-off: a small refetch window that
    # fails completely can stay below the threshold; `--missing` makes such gaps visible.
    hard_failed = counts["failed"] - counts["failed_ratelimit"]
    configured = len(items)
    if configured and hard_failed / configured > FAIL_RATIO_MAX:
        print(
            f"ERROR: {hard_failed}/{configured} sources in scope failed hard (> {FAIL_RATIO_MAX:.0%}, "
            f"rate-limited not counted: {counts['failed_ratelimit']}). Check the sources (dead URLs? bot protection?).",
            file=sys.stderr,
        )
        return 1
    if failures:
        print("Failed (retried on the next run):", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
    return 0


def smoke(kb: KBConfig) -> int:
    """Offline: load and validate the sources file, count the plan. No network."""
    items = load_sources(kb)
    by_src: dict[str, int] = {}
    for it in items:
        by_src[it.source_slug] = by_src.get(it.source_slug, 0) + 1
    print(f"=== kb_harvest smoke [{kb.slug}] (offline) ===")
    print(f"config: {kb.sources_config}")
    for s, n in sorted(by_src.items()):
        print(f"  {s}: {n} items")
    print(f"total: {len(items)} items -> {harvest_root(kb)}")
    return 0


def missing(kb: KBConfig) -> int:
    """Configured sources without a fetched file (dead URLs, permanent failures). Returns the count."""
    root = harvest_root(kb)
    have: set[str] = set()
    for f in root.rglob("*.md") if root.exists() else []:
        url = parse_frontmatter(f.read_text(encoding="utf-8", errors="replace"))[0].get("source_url")
        if url:
            have.add(str(url))
    items = load_sources(kb)
    gone = [it for it in items if it.url not in have]
    for it in gone:
        print(f"  missing: [{it.source_slug}] {it.content_type} {it.url}")
    print(f"{len(gone)} of {len(items)} configured sources without a file ({kb.slug})")
    return len(gone)


def stats(kb: KBConfig) -> int:
    root = harvest_root(kb)
    if not root.exists():
        print(f"(nothing harvested yet: {root})")
        return 0
    total = 0
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        n = len(list(d.glob("*.md")))
        total += n
        print(f"  {d.name}: {n} docs")
    print(f"total: {total} docs in {root}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Download the web sources of a registry KB", allow_abbrev=False)
    ap.add_argument("--kb", required=True, help="KB slug from the registry")
    ap.add_argument("--source", help="only this source (slug)")
    ap.add_argument("--smoke", action="store_true", help="offline: validate the sources file, no fetch")
    ap.add_argument("--probe", action="store_true", help="fetch one URL per source that is not harvested yet")
    ap.add_argument("--dry-run", action="store_true", help="plan without fetching or writing")
    ap.add_argument("--stats", action="store_true", help="count harvested docs")
    ap.add_argument("--missing", action="store_true", help="list configured sources without a file")
    ap.add_argument(
        "--refetch-older-than",
        type=int,
        default=None,
        metavar="DAYS",
        help="fetch docs again whose harvest is older than DAYS (in place)",
    )
    ap.add_argument("--registry", type=Path, default=None, help="another registry file (tests, staging)")
    args = ap.parse_args()

    try:
        kb = get_kb(args.kb, args.registry)
        if args.stats:
            return stats(kb)
        if args.missing:
            return 1 if missing(kb) else 0  # like --eval: exit 1 = gaps found
        if args.smoke:
            return smoke(kb)
        items = load_sources(kb)
        return harvest(
            kb,
            items,
            probe=args.probe,
            dry_run=args.dry_run,
            source_filter=args.source,
            refetch_days=args.refetch_older_than,
        )
    except KBError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
