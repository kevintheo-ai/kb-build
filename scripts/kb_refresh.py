"""
kb_refresh.py - keep every active registry KB fresh, without any LLM.

Per active KB: (1) harvest refresh - new URLs from the sources file, retries of earlier
failures, and a refetch of docs older than `refetch_days` (content drift at the same URL);
(2) incremental ingest; (3) golden-query eval. Problems are printed as `ALERT[level] ...` lines
on stderr. Exit code 2 = an ingest or eval run failed, a KB crashed or the registry is unusable;
warnings (failed sources, an eval MISS) keep exit code 0, their ALERT lines are the signal. After
`curation_days`, the summary marks a KB `curation_due`: time to look for NEW sources, which
needs research and a human eye, so it is deliberately not automated.

Schedule it with cron, launchd or systemd, e.g. daily at 04:30 (cron mails stderr output):

    30 4 * * *  cd /path/to/kb-build && .venv/bin/python scripts/kb_refresh.py

    python scripts/kb_refresh.py                     # all active KBs
    python scripts/kb_refresh.py --kb <slug>         # one KB
    python scripts/kb_refresh.py --dry-run           # plan only
    python scripts/kb_refresh.py --smoke             # offline self-check
    python scripts/kb_refresh.py --mark-curated <slug>
    python scripts/kb_refresh.py --mark-curated-all  # end of a maintenance pass
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from kb_common import KBConfig, KBError, get_kb, list_kbs

SCRIPTS = Path(__file__).resolve().parent
SUBPROC_TIMEOUT_S = 3600  # large harvests take a while (politeness pauses)


def _alert(level: str, message: str) -> None:
    print(f"ALERT[{level}] kb-refresh: {message}", file=sys.stderr)


def _run(cmd: list[str]) -> tuple[int, str]:
    """Run a script with the same interpreter -> (exit code, complete output). Never truncate
    before parsing: with many [FAIL] lines on stderr the summary line would fall out of a window."""
    try:
        p = subprocess.run(
            [sys.executable, *cmd], capture_output=True, text=True, timeout=SUBPROC_TIMEOUT_S, check=False
        )
        return p.returncode, ((p.stdout or "") + ("\n" + p.stderr if p.stderr else "")).strip()
    except subprocess.TimeoutExpired:
        return 124, f"TIMEOUT after {SUBPROC_TIMEOUT_S}s: {' '.join(cmd)}"


def _curation_marker(kb: KBConfig) -> Path:
    return kb.state_dir() / "last_curation"


def mark_curated(kb: KBConfig) -> None:
    kb.state_dir().mkdir(parents=True, exist_ok=True)
    _curation_marker(kb).write_text(datetime.now(UTC).date().isoformat() + "\n", encoding="utf-8")


def _curation_age_days(kb: KBConfig) -> int | None:
    """Days since the last source curation; None = no marker yet (gets seeded)."""
    p = _curation_marker(kb)
    if not p.exists():
        return None
    try:
        dt = datetime.strptime(p.read_text(encoding="utf-8").strip()[:10], "%Y-%m-%d").replace(tzinfo=UTC)
        return max(0, (datetime.now(UTC) - dt).days)
    except ValueError:
        # An unreadable marker would silence the reminder for good (the seeder skips existing
        # files): report it and reset it to today.
        _alert("warn", f"{kb.slug}: curation marker unreadable ({p}), reset to today")
        mark_curated(kb)
        return 0


def _seed_curation_marker(kb: KBConfig) -> None:
    """First run: set the marker to the KB's `created` date (or today), the cadence starts there."""
    if _curation_marker(kb).exists():
        return
    kb.state_dir().mkdir(parents=True, exist_ok=True)
    seed = kb.created[:10] if kb.created else datetime.now(UTC).date().isoformat()
    _curation_marker(kb).write_text(seed + "\n", encoding="utf-8")


def refresh_kb(kb: KBConfig, *, dry_run: bool, registry: Path | None = None) -> dict:
    result: dict = {"kb": kb.slug, "harvest": None, "ingest": None, "eval": None, "curation_due": False, "errors": 0}

    # Pass --registry on to EVERY child: otherwise a staging run plans against staging but
    # changes the real KB.
    reg_args = ["--registry", str(registry)] if registry else []
    steps: list[tuple[str, list[str]]] = []
    if kb.harvest_refresh_enabled():
        harvest_cmd = [str(SCRIPTS / "kb_harvest.py"), "--kb", kb.slug, "--refetch-older-than", str(kb.refetch_days)]
        steps.append(("harvest", [*harvest_cmd, *reg_args]))
    steps.append(("ingest", [str(SCRIPTS / "kb_ingest.py"), kb.slug, *reg_args]))
    steps.append(("eval", [str(SCRIPTS / "kb_search.py"), "--kb", kb.slug, "--eval", *reg_args]))

    if dry_run:
        for name, cmd in steps:
            print(f"[plan:{kb.slug}] {name}: {' '.join(cmd)}")
        result["planned"] = [s[0] for s in steps]
        return result

    for name, cmd in steps:
        code, out = _run(cmd)
        result[name] = code
        if name == "harvest" and code == 2:
            # Lock or config problem: do NOT ingest a half-finished state. If another harvest
            # runs in parallel, its own pipeline takes over.
            _alert("warn", f"{kb.slug}: harvest skipped (exit 2: lock/config), rest of the run left out")
            print(f"[{kb.slug}] {name}: exit={code} (lock/config) -> ingest/eval skipped")
            break
        if name == "harvest" and code != 0:
            _alert("warn", f"{kb.slug}: harvest refresh exit {code} (fail ratio/timeout) - {out[-200:]}")
        elif name == "harvest":
            # Exit 0 with some failures (<= 30 %): report them instead of dropping them.
            # Permanent gaps show up in the maintenance pass (`kb_harvest.py --missing`).
            m = re.search(r"HARVEST DONE.*?failed=(\d+)", out)
            if m and int(m.group(1)) > 0:
                _alert("info", f"{kb.slug}: harvest ok, {m.group(1)} source(s) failed (retried next run)")
        elif name == "ingest" and code != 0:
            _alert("error", f"{kb.slug}: ingest FAILED (exit {code}) - {out[-200:]}")
            result["errors"] += 1
            break  # no eval without a clean ingest
        elif name == "eval" and code == 1 and re.search(r"^result: \d+/\d+ golden queries hit@", out, re.MULTILINE):
            # A MISS only when the eval ran to its result line: a traceback also exits 1.
            _alert("warn", f"{kb.slug}: golden-query eval MISS (exit 1) - check chunking and sources")
        elif name == "eval" and code != 0:
            # Not a MISS: the eval could not run (model missing, Qdrant down, no golden queries).
            _alert("error", f"{kb.slug}: eval FAILED (exit {code}) - {out[-200:]}")
            result["errors"] += 1
        print(f"[{kb.slug}] {name}: exit={code}")

    _seed_curation_marker(kb)
    age = _curation_age_days(kb)
    if age is not None and age >= kb.curation_days:
        result["curation_due"] = True
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Refresh all registry KBs", allow_abbrev=False)
    ap.add_argument("--kb", help="only this KB (default: all active)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    ap.add_argument("--smoke", action="store_true", help="offline self-check (registry + markers + plan)")
    ap.add_argument("--mark-curated", metavar="SLUG", help="set the curation marker of one KB to today")
    ap.add_argument("--mark-curated-all", action="store_true", help="end of a maintenance pass: mark all active KBs")
    ap.add_argument("--registry", type=Path, default=None, help="another registry file (tests, staging)")
    args = ap.parse_args()
    if args.mark_curated_all and (args.kb or args.mark_curated):
        # Otherwise "--kb x --mark-curated-all" would silently mark ALL KBs.
        ap.error("--mark-curated-all covers all active KBs; for a single KB use --mark-curated <slug>")

    try:
        if args.mark_curated_all:
            active = [k for k in (get_kb(s, args.registry) for s in list_kbs(args.registry)) if k.status == "active"]
            if args.dry_run:
                print(f"[plan] would set {len(active)} curation markers (--dry-run: nothing written)")
                return 0
            for kb in active:
                mark_curated(kb)
            print(f"maintenance pass recorded: {', '.join(k.slug for k in active)}")
            return 0
        if args.mark_curated:
            kb = get_kb(args.mark_curated, args.registry)
            if args.dry_run:
                # A preview must not set the marker (that would silence the reminder).
                print(f"[plan:{kb.slug}] would set the curation marker to today (--dry-run: nothing written)")
                return 0
            mark_curated(kb)
            print(f"[{kb.slug}] curation marker set to today.")
            return 0

        slugs = [args.kb] if args.kb else list_kbs(args.registry)
        kbs = [k for k in (get_kb(s, args.registry) for s in slugs) if k.status == "active"]
        if not kbs:
            print("No active registry KB, nothing to do.")
            return 0

        if args.smoke:
            print("=== kb_refresh smoke (offline) ===")
            for kb in kbs:
                age = _curation_age_days(kb)
                age_txt = f"{age}d ago" if age is not None else "not seeded yet"
                print(
                    f"  {kb.slug}: harvest_refresh={'yes' if kb.harvest_refresh_enabled() else 'no'} "
                    f"refetch>{kb.refetch_days}d curation>{kb.curation_days}d (last curation: {age_txt})"
                )
            return 0

        results = []
        for kb in kbs:
            try:
                results.append(refresh_kb(kb, dry_run=args.dry_run, registry=args.registry))
            except Exception as exc:
                # A crash in KB #1 must not cost KB #2..N their refresh: isolate, alert, go on.
                _alert("error", f"{kb.slug}: refresh CRASH {type(exc).__name__}: {exc} - the other KBs continue")
                results.append({"kb": kb.slug, "errors": 1, "crash": f"{type(exc).__name__}: {exc}"})
    except KBError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print("\n=== KB-REFRESH SUMMARY ===")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 2 if any(r["errors"] for r in results) else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:  # scheduled runs: never crash silently
        _alert("error", f"kb_refresh CRASH: {type(exc).__name__}: {exc}")
        raise SystemExit(3) from exc
