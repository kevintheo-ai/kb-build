"""
advisor_bash_guard.py - PreToolUse hook that limits an advisor agent to ONE command.

An advisor (skills/kb-build/references/advisor-template.md) only needs to search its own knowledge
base. Its frontmatter registers this script as a PreToolUse hook for Bash, so Claude Code checks
every Bash call of that agent here before running it:

    allowed:  <python> <repo>/scripts/kb_search.py --kb <slug> "<query>" [search flags]
    blocked:  anything else, any other KB or registry, and shell syntax that could chain,
              substitute or redirect commands (; & | < > ( ) $ ` \\ and line breaks)

Exit 0 lets the call through (Claude Code's normal permission rules still apply on top).
Exit 2 blocks it; the reason on stderr goes back to the agent. Any error inside the guard
blocks as well (fail closed). Standard library only, so it starts fast.

    python <repo>/scripts/advisor_bash_guard.py --python <python> --repo <repo> --kb <slug>
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys

# Characters that let a single line turn into more than one command or read/write files,
# even inside double quotes ($ and backticks still expand there).
FORBIDDEN = re.compile(r"[;&|<>()$`\\\n\r\x00]")
VALUE_FLAGS = {
    "--limit",
    "--top",
    "--topk",
    "--author",
    "--topic",
    "--content-type",
    "--trust",
    "--year",
    "--since-year",
}
NUMERIC_FLAGS = {"--limit", "--top", "--topk", "--year", "--since-year"}
SWITCH_FLAGS = {"--full", "--no-recency", "--json"}
# Outside quotes the shell expands these into file names or extra words, which could add
# arguments after this check (a file called "--registry" in the working directory, say).
EXPANDING = set("*?[]{}~#")


def _unquoted_expansion(command: str) -> str | None:
    """First character outside quotes that the shell would expand, else None. Backslashes are
    already forbidden, so quote tracking needs no escape handling."""
    single = double = False
    for ch in command:
        if ch == "'" and not double:
            single = not single
        elif ch == '"' and not single:
            double = not double
        elif ch in EXPANDING and not (single or double):
            return ch
    return None


def check(command: str, python: str, repo: str, slug: str) -> str | None:
    """None if `command` is the one allowed shape, otherwise the reason it is blocked."""
    if FORBIDDEN.search(command):
        return "shell syntax is not allowed (; & | < > ( ) $ ` backslash, line breaks)"
    ch = _unquoted_expansion(command)
    if ch:
        return f"unquoted {ch!r} would be expanded by the shell: put the query in quotes"
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        return f"command cannot be parsed ({exc})"
    script = f"{repo.rstrip('/')}/scripts/kb_search.py"
    if argv[:4] != [python, script, "--kb", slug]:
        return "only the knowledge-base search of this advisor may run"
    queries = 0
    i = 4
    while i < len(argv):
        arg = argv[i]
        if arg in SWITCH_FLAGS:
            i += 1
        elif arg in VALUE_FLAGS:
            if i + 1 >= len(argv):
                return f"{arg} needs a value"
            value = argv[i + 1]
            if arg in NUMERIC_FLAGS and not value.isdigit():
                return f"{arg} needs a number"
            if value.startswith("-"):
                return f"{arg} needs a value, not another flag"
            i += 2
        elif arg.startswith("-"):
            return f"flag {arg} is not allowed (e.g. no --kb, --registry, --eval)"
        else:
            queries += 1
            i += 1
    if queries != 1:
        return "exactly one search query is required"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="PreToolUse guard for kb-build advisors")
    ap.add_argument("--python", required=True, help="the exact interpreter path the advisor uses")
    ap.add_argument("--repo", required=True, help="absolute path of the kb-build repo")
    ap.add_argument("--kb", required=True, help="the advisor's KB slug")
    args = ap.parse_args()

    event = json.load(sys.stdin)
    if event.get("tool_name") != "Bash":
        return 0
    command = str((event.get("tool_input") or {}).get("command") or "")
    reason = check(command, args.python, args.repo, args.kb)
    if reason is None:
        return 0
    allowed = (
        f'{args.python} {args.repo.rstrip("/")}/scripts/kb_search.py --kb {args.kb} "<query>" [--limit N] [--full]'
    )
    print(f"advisor guard blocked this command: {reason}. The only allowed command is: {allowed}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:  # fail closed: a broken guard blocks, it never waves a command through
        print(f"advisor guard error, command blocked: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
