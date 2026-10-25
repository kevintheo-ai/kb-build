"""
kb_ingest.py - Markdown sources -> the Qdrant collection of a registry KB.

Chunks every file by headings, embeds it with BGE-M3 (dense + sparse) and upserts the chunks into
the KB's own collection. Incremental via a manifest ($KB_HOME/state/<slug>/): unchanged files are
skipped, deleted files are removed, orphaned chunks of changed files are cleaned up.

    python scripts/kb_ingest.py <slug>            # incremental
    python scripts/kb_ingest.py <slug> --full     # re-embed everything
    python scripts/kb_ingest.py <slug> --dry-run  # show what would happen
    python scripts/kb_ingest.py <slug> --stats    # collection stats
    python scripts/kb_ingest.py <slug> --smoke    # offline: chunk one file, no Qdrant, no model
    python scripts/kb_ingest.py --list            # all registry KBs

Convert PDF/Office/HTML files to Markdown first (e.g. with markitdown): the ingester reads
.md/.txt only, on purpose.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import sys
from pathlib import Path

from kb_common import (
    STANDARD_FRONTMATTER_KEYS,
    VALID_TRUST,
    VECTOR_SIZE,
    KBConfig,
    KBError,
    build_context_prefix,
    derive_year,
    ensure_collection,
    ensure_payload_indexes,
    get_client,
    get_kb,
    list_kbs,
    parse_frontmatter,
    point_id,
)
from markdown_chunker import _content_hash, chunk_markdown_file
from qdrant_client import models

# Instruction-like phrases in harvested content are FLAGGED, never blocked (courses about
# prompting would be false positives). kb_search shows the flag; the chunk stays data like
# everything else. A narrow list of explicit jailbreak phrases (English and German), not a
# classifier: "always recommend X" is not caught.
INJECTION_PATTERNS = re.compile(
    r"ignore (all |any )?(previous|prior|above) (instructions|rules|prompts)"
    r"|disregard (all |the )?(previous|prior|system)"
    r"|begin system prompt|<system>|\[system\]"
    r"|do not (follow|obey) (the |your )?(rules|instructions)"
    r"|you are now (in )?(developer|jailbreak|dan) mode"
    r"|ignoriere (alle |s(?:ae|ä)mtliche )?(vorherigen|bisherigen|obigen) (anweisungen|regeln|instruktionen)"
    r"|vergiss (alles|alle (vorherigen |bisherigen )?(anweisungen|regeln))"
    r"|neue anweisungen:|du bist (jetzt|ab jetzt|von nun an)"
    r"|end of instructions|new instructions:",
    re.IGNORECASE,
)


def scan_files(kb: KBConfig) -> list[Path]:
    """All source files of the KB (content_dirs x include globs), deduplicated. Folders whose
    name starts with "." or "_" are skipped (park rejected or draft files there)."""
    files: set[Path] = set()
    for root in kb.content_dirs:
        if not root.is_dir():
            continue
        for pattern in kb.include:
            for f in root.rglob(pattern):
                if f.is_file() and not any(part.startswith((".", "_")) for part in f.relative_to(root).parts[:-1]):
                    files.add(f.resolve())
    return sorted(files)


def _load_manifest(kb: KBConfig) -> dict:
    p = kb.manifest_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        # A silent {} would disable the orphan cleanup and leave stale points behind.
        raise KBError(
            f"ingest manifest is corrupt: {p} ({exc}). Check or delete it, then run `kb_ingest.py {kb.slug} --full`."
        ) from exc


def _save_manifest(kb: KBConfig, manifest: dict) -> None:
    """Atomic (temp file + os.replace): a crash mid-write never leaves half a manifest."""
    kb.state_dir().mkdir(parents=True, exist_ok=True)
    tmp = kb.manifest_path().with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, kb.manifest_path())


def _fingerprint(kb: KBConfig) -> str:
    """Everything that shapes the stored points. If it changes, an unchanged content hash is no
    longer a reason to skip a file: a new model revision must not leave old vectors behind, a new
    default_trust not the old trust in the points of files without their own."""
    from bge_embedder import MODEL_REVISION  # lazy like the model: kb_common must be imported first

    return (
        f"{kb.collection}|{kb.chunk_size}|{kb.chunk_overlap}|{int(kb.context_prefix)}|{VECTOR_SIZE}|"
        f"{MODEL_REVISION}|{kb.default_trust}"
    )


def _file_payload_base(kb: KBConfig, f: Path, fm: dict) -> dict:
    """Front matter -> standard payload. Values are passed through as data, never interpreted."""
    base: dict = {k: fm.get(k) for k in STANDARD_FRONTMATTER_KEYS if fm.get(k) not in (None, "")}
    trust = str(base.get("trust") or kb.default_trust).lower()
    base["trust"] = trust if trust in VALID_TRUST else kb.default_trust
    base["title"] = str(base.get("title") or f.stem)
    if base.get("date"):
        base["date"] = str(base["date"])
    base["year"] = derive_year(base.get("date"))
    base["kb"] = kb.slug
    return base


def build_points(kb: KBConfig, f: Path, embedder) -> tuple[list, list[str]]:
    """Chunk + embed one file -> (PointStructs, point_ids)."""
    raw = f.read_text(encoding="utf-8", errors="replace")
    fm, _body = parse_frontmatter(raw)
    base = _file_payload_base(kb, f, fm)
    if INJECTION_PATTERNS.search(raw):
        base["injection_suspect"] = True
        print(f"  WARN: instruction-like phrase in {f.name}: its chunks are flagged injection_suspect", file=sys.stderr)

    chunks = chunk_markdown_file(f, max_chunk_size=kb.chunk_size, chunk_overlap=kb.chunk_overlap)
    if not chunks:
        return [], []

    embed_texts: list[str] = []
    for c in chunks:
        meta = c["metadata"]
        crumb = (
            build_context_prefix(base["title"], meta.get("h1", ""), meta.get("h2", ""), meta.get("h3", ""))
            if kb.context_prefix
            else ""
        )
        embed_texts.append(f"{crumb}\n\n{c['text']}" if crumb else c["text"])

    embeddings = embedder.embed_batch_dense_and_sparse(embed_texts)

    from datetime import UTC, datetime

    now = datetime.now(UTC).isoformat()
    points: list = []
    ids: list[str] = []
    for c, embed_text, (dense, sparse) in zip(chunks, embed_texts, embeddings, strict=True):
        meta = c["metadata"]
        pid = point_id(kb.collection, str(f), meta["chunk_index"], meta["content_hash"])
        payload = {
            **base,
            "text": c["text"],
            "context": embed_text[: len(embed_text) - len(c["text"])].strip() or None,
            "file_path": str(f),
            "h1": meta.get("h1", ""),
            "h2": meta.get("h2", ""),
            "h3": meta.get("h3", ""),
            "h4": meta.get("h4", ""),
            "section": meta.get("h2") or meta.get("h1") or "",
            "chunk_index": meta["chunk_index"],
            "content_hash": meta["content_hash"],
            "created_at": now,
        }
        points.append(models.PointStruct(id=pid, vector={"dense": dense, "sparse": sparse}, payload=payload))
        ids.append(pid)
    return points, ids


def ingest(kb: KBConfig, *, full: bool, dry_run: bool, allow_prune: bool = False) -> dict:
    missing = [d for d in kb.content_dirs if not d.is_dir()]
    if len(missing) == len(kb.content_dirs):
        print(f"ERROR: no content_dir of KB {kb.slug!r} exists: {[str(d) for d in missing]}", file=sys.stderr)
        print("  -> add sources first (kb_harvest.py, markitdown, or by hand).", file=sys.stderr)
        sys.exit(2)
    for d in missing:
        print(f"WARN: content_dir missing (skipped): {d}", file=sys.stderr)

    # Exclusive lock: a manual run and the scheduled refresh must never touch the manifest at
    # the same time (read-modify-write).
    kb.state_dir().mkdir(parents=True, exist_ok=True)
    lock_f = open(kb.state_dir() / "ingest.lock", "w")  # noqa: SIM115 - held for the whole ingest
    try:
        fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(
            f"ERROR: another ingest for {kb.slug!r} is already running (ingest.lock), try again later.", file=sys.stderr
        )
        sys.exit(4)

    files = scan_files(kb)
    # --full re-embeds every file but works on the same manifest: file by file the old points are
    # replaced and the checkpoint is saved, so a crash at any point loses neither old nor new IDs.
    manifest = _load_manifest(kb)
    manifest.pop("_meta", None)  # a global fingerprint of older versions; the pipeline is tracked per file now
    current = {str(f): _content_hash(f.read_text(encoding="utf-8", errors="replace")) for f in files}

    summary = {
        "kb": kb.slug,
        "collection": kb.collection,
        "files_total": len(files),
        "indexed": 0,
        "skipped": 0,
        "deleted_files": 0,
        "points_upserted": 0,
        "points_deleted": 0,
    }
    if not files:
        print(f"WARN: no source files in {[str(d) for d in kb.content_dirs]} (include={kb.include})", file=sys.stderr)
        if not manifest:
            return summary
        # 0 source files with a non-empty manifest would mean "everything was deleted" and wipe
        # the whole collection with exit 0. Typical causes: an include typo, an emptied or moved
        # folder. The scheduled refresh runs unattended, so stop hard unless --allow-prune.
        if not allow_prune:
            print(
                f"ERROR: 0 source files, but the manifest tracks {len(manifest)} files. "
                "Refusing to empty the collection. If that is intended: "
                f"`kb_ingest.py {kb.slug} --allow-prune`.",
                file=sys.stderr,
            )
            sys.exit(3)

    client = None
    if not dry_run:  # a dry run writes nothing, not even an empty collection
        client = get_client(timeout=30)
        if ensure_collection(client, kb.collection):
            print(f"collection '{kb.collection}' created.")
        ensure_payload_indexes(client, kb.collection)

    # Every manifest entry records the pipeline its points were embedded with. After a change (model,
    # chunking, default trust) each file is re-embedded once; an interrupted run resumes where it
    # stopped, and the files of an unmounted folder catch up when it is back.
    fp_now = _fingerprint(kb)
    outdated = sum(1 for k, v in manifest.items() if k in current and v.get("pipeline") != fp_now)
    if outdated and not full:
        print(
            f"WARN: {outdated} file(s) not embedded with the current pipeline ({fp_now}): re-embedding them.",
            file=sys.stderr,
        )

    embedder = None  # lazy: load the model only when something needs embedding

    for f in files:
        fp = str(f)
        entry = manifest.get(fp, {})
        if not full and entry.get("content_hash") == current[fp] and entry.get("pipeline") == fp_now:
            summary["skipped"] += 1
            continue
        if dry_run:
            print(f"  [dry-run] would index: {f.name}")
            summary["indexed"] += 1
            continue
        if embedder is None:
            from bge_embedder import BGEEmbedder

            embedder = BGEEmbedder()
        points, new_ids = build_points(kb, f, embedder)
        old_ids = entry.get("point_ids", [])
        # Write-ahead: track old and new IDs before Qdrant changes, without a content hash so the next
        # run takes this file again. A crash between upsert and checkpoint never leaves unknown points.
        manifest[fp] = {"content_hash": None, "point_ids": sorted(set(old_ids) | set(new_ids))}
        _save_manifest(kb, manifest)
        if points:
            client.upsert(collection_name=kb.collection, points=points, wait=True)
            summary["points_upserted"] += len(points)
        # Also with 0 chunks (file emptied, front matter only): remove the old points and track
        # the new state, otherwise the old chunks live on.
        stale = list(set(old_ids) - set(new_ids))
        if stale:
            client.delete(collection_name=kb.collection, points_selector=models.PointIdsList(points=stale))
            summary["points_deleted"] += len(stale)

        manifest[fp] = {"content_hash": current[fp], "point_ids": new_ids, "pipeline": fp_now}
        _save_manifest(kb, manifest)  # checkpoint per file: a crash keeps the progress
        summary["indexed"] += 1
        print(f"  indexed: {f.name} ({len(points)} chunks)")

    # Deleted files: remove their points. Files under a content_dir that is MISSING right now do
    # not count as deleted (unmounted volume, folder being moved): that would be a silent mass delete.
    # Resolved like the manifest keys: a symlinked folder whose drive is gone still matches.
    missing_prefixes = tuple(str(d.resolve()) + os.sep for d in missing)
    prune = [
        fp
        for fp in manifest
        if fp != "_meta" and fp not in current and not (missing_prefixes and fp.startswith(missing_prefixes))
    ]
    tracked = sum(1 for k in manifest if k != "_meta")
    # Prune ratio guard: if one incremental run would delete more than half of the tracked
    # files (include typo, half-moved folder), stop before deleting unless --allow-prune.
    # --full stays an explicit operator command.
    if prune and not allow_prune and not full and len(prune) > 0.5 * tracked:
        print(
            f"ERROR: this run would delete {len(prune)}/{tracked} tracked files (>50%). Refused. "
            f"Check the sources, or on purpose: `kb_ingest.py {kb.slug} --allow-prune`.",
            file=sys.stderr,
        )
        sys.exit(3)
    for fp in prune:
        ids = manifest[fp].get("point_ids", [])
        if dry_run:
            # A preview must change neither the manifest nor the counters.
            print(f"  [dry-run] would delete: {Path(fp).name} ({len(ids)} points)")
            continue
        if ids:
            client.delete(collection_name=kb.collection, points_selector=models.PointIdsList(points=ids))
            summary["points_deleted"] += len(ids)
        del manifest[fp]
        summary["deleted_files"] += 1

    if not dry_run:
        # Count check BEFORE the final save: if it fails, the prune deletions stay uncommitted
        # and a transient error heals on the next run instead of sticking in exit 3.
        total_manifest = sum(len(v.get("point_ids", [])) for v in manifest.values())
        total_live = client.count(collection_name=kb.collection, exact=True).count
        summary["points_live"] = total_live
        summary["points_manifest"] = total_manifest
        if total_live < total_manifest:
            print(
                f"ERROR: count check failed: collection {kb.collection} has {total_live} points, the manifest "
                f"expects >= {total_manifest}. Manifest NOT saved; if it repeats: `kb_ingest.py {kb.slug} --full`.",
                file=sys.stderr,
            )
            sys.exit(3)
        _save_manifest(kb, manifest)
    return summary


def show_stats(kb: KBConfig) -> None:
    client = get_client()
    if not client.collection_exists(kb.collection):
        print(f"collection '{kb.collection}' does not exist yet.")
        return
    n = client.count(collection_name=kb.collection, exact=True).count
    print(f"KB '{kb.slug}' -> collection '{kb.collection}': {n} points")
    if kb.manifest_path().exists():
        m = _load_manifest(kb)  # through the corruption guard
        print(f"manifest: {sum(1 for k in m if k != '_meta')} files tracked ({kb.manifest_path()})")


def smoke(kb: KBConfig) -> int:
    """Offline: chunk the first file of every content_dir. No Qdrant, no model."""
    print(f"=== kb_ingest smoke [{kb.slug}] (offline) ===")
    any_file = False
    for root in kb.content_dirs:
        # resolve() on both sides: scan_files returns resolved paths; a symlinked content_dir
        # would otherwise find nothing.
        files = [f for f in scan_files(kb) if f.is_relative_to(root.expanduser().resolve())][:1]
        if not files:
            print(f"[smoke] {root}: no files")
            continue
        any_file = True
        f = files[0]
        fm, _ = parse_frontmatter(f.read_text(encoding="utf-8", errors="replace"))
        base = _file_payload_base(kb, f, fm)
        chunks = chunk_markdown_file(f, max_chunk_size=kb.chunk_size, chunk_overlap=kb.chunk_overlap)
        crumb = build_context_prefix(
            base["title"],
            chunks[0]["metadata"].get("h1", "") if chunks else "",
            chunks[0]["metadata"].get("h2", "") if chunks else "",
            "",
        )
        print(f"[smoke] {f.name}: chunks={len(chunks)} trust={base['trust']} year={base['year']} breadcrumb={crumb!r}")
    if not any_file:
        print("[smoke] FAIL: not a single source file found", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Index a registry KB into Qdrant (BGE-M3 hybrid)", allow_abbrev=False)
    ap.add_argument("slug", nargs="?", help="KB slug from the registry")
    ap.add_argument("--full", action="store_true", help="re-embed every file (its old points are replaced)")
    ap.add_argument("--dry-run", action="store_true", help="show, do not write")
    ap.add_argument(
        "--allow-prune",
        action="store_true",
        help="allow a mass delete (0 source files, or more than half of the tracked files gone)",
    )
    ap.add_argument("--stats", action="store_true", help="collection stats")
    ap.add_argument("--smoke", action="store_true", help="offline smoke (chunk, no Qdrant)")
    ap.add_argument("--list", action="store_true", help="list all registry KBs")
    ap.add_argument("--registry", type=Path, default=None, help="another registry file (tests, staging)")
    args = ap.parse_args()

    try:
        if args.list:
            for s in list_kbs(args.registry):
                print(s)
            return 0
        if not args.slug:
            ap.error("slug missing (or use --list)")
        kb = get_kb(args.slug, args.registry)
        if args.stats:
            show_stats(kb)
            return 0
        if args.smoke:
            return smoke(kb)
        summary = ingest(kb, full=args.full, dry_run=args.dry_run, allow_prune=args.allow_prune)
    except (KBError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print("\n=== INGEST DONE ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
