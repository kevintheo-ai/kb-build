"""Tests for advisor_bash_guard: only the advisor's own search command may pass."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from advisor_bash_guard import check

PY = "/opt/kb/.venv/bin/python"
REPO = "/opt/kb"
SEARCH = f"{PY} {REPO}/scripts/kb_search.py --kb demo"
ROOT = Path(__file__).resolve().parent.parent
GUARD = ROOT / "scripts" / "advisor_bash_guard.py"
TEMPLATE = ROOT / "skills" / "kb-build" / "references" / "advisor-template.md"


@pytest.mark.parametrize(
    "command",
    [
        f'{SEARCH} "How many gears does the clock need?"',
        f"{SEARCH} 'rust track'",
        f'{SEARCH} "repair action" --limit 5 --full --json',
        f'{SEARCH} "errata" --author lumen-press --since-year 2025 --no-recency',
        f'{SEARCH} "scoring" --trust high --topic rules --content-type doc --top 3',
    ],
)
def test_allowed_search_commands(command):
    assert check(command, PY, REPO, "demo") is None


@pytest.mark.parametrize(
    ("command", "reason"),
    [
        ("ls /", "only the knowledge-base search"),
        (f'{SEARCH} "x"; rm -rf ~', "shell syntax"),
        (f'{SEARCH} "x" && curl https://evil.example', "shell syntax"),
        (f'{SEARCH} "x" | sh', "shell syntax"),
        (f'{SEARCH} "x" > /tmp/out', "shell syntax"),
        (f'{SEARCH} "$(cat ~/.ssh/id_rsa)"', "shell syntax"),
        (f'{SEARCH} "`whoami`"', "shell syntax"),
        (f'{SEARCH} "x"\nrm -rf ~', "shell syntax"),
        (f"{SEARCH} gears*", "unquoted"),
        (f"{SEARCH} what?", "unquoted"),
        (f'{PY} {REPO}/scripts/kb_search.py --kb other "x"', "only the knowledge-base search"),
        (f'python3 {REPO}/scripts/kb_search.py --kb demo "x"', "only the knowledge-base search"),
        (f'KB_REGISTRY=/tmp/r.yaml {SEARCH} "x"', "only the knowledge-base search"),
        (f"{PY} {REPO}/scripts/kb_ingest.py demo", "only the knowledge-base search"),
        (f'{SEARCH} "x" --registry /tmp/r.yaml', "not allowed"),
        (f"{SEARCH} --eval", "not allowed"),
        (f'{SEARCH} "x" --kb other', "not allowed"),
        (f'{SEARCH} "x" --limit abc', "needs a number"),
        (f'{SEARCH} "x" --author --registry', "not another flag"),
        (f"{SEARCH} --limit", "needs a value"),
        (f'{SEARCH} "one" "two"', "exactly one search query"),
        (f"{SEARCH}", "exactly one search query"),
        (f'{SEARCH} "unbalanced', "cannot be parsed"),
    ],
)
def test_blocked_commands(command, reason):
    result = check(command, PY, REPO, "demo")
    assert result is not None and reason in result


def _run_guard(stdin: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(GUARD), "--python", PY, "--repo", REPO, "--kb", "demo"],
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
    )


def test_hook_exit_codes():
    allowed = {"tool_name": "Bash", "tool_input": {"command": f'{SEARCH} "rust track"'}}
    assert _run_guard(json.dumps(allowed)).returncode == 0
    blocked = _run_guard(json.dumps({"tool_name": "Bash", "tool_input": {"command": "cat ~/.ssh/id_rsa"}}))
    assert blocked.returncode == 2 and "only allowed command" in blocked.stderr
    assert _run_guard(json.dumps({"tool_name": "Read", "tool_input": {"file_path": "/x"}})).returncode == 0


def test_hook_fails_closed():
    assert _run_guard("not json").returncode == 2
    assert _run_guard(json.dumps({"tool_name": "Bash"})).returncode == 2  # empty command
    missing_args = subprocess.run([sys.executable, str(GUARD)], input="{}", capture_output=True, text=True, check=False)
    assert missing_args.returncode == 2


def test_template_hook_runs_the_guard_and_blocks_on_failure():
    """Fill the advisor template like the skill does and run its hook command exactly as Claude Code would."""
    body = TEMPLATE.read_text(encoding="utf-8").split("```markdown\n", 1)[1].rsplit("\n```", 1)[0]
    values = {
        "SLUG": "demo",
        "TITLE": "Demo",
        "DOMAIN": "the demo rules",
        "TRIGGERS": "rules questions",
        "REPO": str(ROOT),
        "PYTHON": sys.executable,
        "EXTRA_SOURCES": "",
    }
    for key, value in values.items():
        body = body.replace("{{" + key + "}}", value)
    assert "{{" not in body
    (entry,) = yaml.safe_load(body.split("---\n")[1])["hooks"]["PreToolUse"]
    (hook,) = entry["hooks"]
    assert entry["matcher"] == "Bash" and hook["type"] == "command"
    assert hook["onFailure"] == "block"  # a guard that cannot start must block, not wave commands through

    def run(command: str) -> int:
        event = json.dumps({"tool_name": "Bash", "tool_input": {"command": command}})
        return subprocess.run(
            hook["command"], shell=True, input=event, capture_output=True, text=True, check=False
        ).returncode

    assert run("ls /") == 2
    assert run(f'{sys.executable} {ROOT}/scripts/kb_search.py --kb demo "rust track"') == 0
