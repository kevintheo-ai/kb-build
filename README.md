# kb-build

**Searchable knowledge bases and grounded specialist agents for Claude Code.** Collect the sources of a
domain, index them locally (BGE-M3 hybrid search in Qdrant), prove with a small regression test that the
right passages come back, and get a read-only specialist agent that answers from those sources, cites them,
and says "not in the knowledge base" instead of guessing. Taken as generic patterns from the system that runs
my company, where specialists like this check advertising, inventory and legal questions before an agent acts.

Background story: [Why I run my company with AI agents](https://kevintheo.com/field-report/)

## Why this exists

- Agents answer niche questions from model memory: ad-platform rules, marketplace policies, internal
  handbooks. That knowledge is stale, generic or invented, and the answer sounds equally sure either way.
- Pasting documents into the prompt does not scale, and full RAG frameworks are a lot of machinery for one
  person's corpus.
- A knowledge base you cannot test is a black box. You only notice that retrieval broke when an answer is
  wrong.

kb-build is the small, boring version: five scripts, one registry file, one collection per domain, a
golden-query check after every change, and a specialist template with hard rules about sources.

## What's inside

| Piece | What it does |
|---|---|
| `scripts/kb_harvest.py` | Fetches the web pages and PDFs of a curated sources file as Markdown. SSRF-guarded (https only, public addresses only, every redirect checked), atomic writes, refetches old pages in place. |
| `scripts/kb_ingest.py` | Chunks Markdown by headings, embeds with BGE-M3 (dense + sparse), upserts into the KB's own Qdrant collection. Incremental via a manifest, refuses mass deletions, count check after every run. |
| `scripts/kb_search.py` | Hybrid search (dense + sparse, RRF fusion, optional recency weighting). `--eval` runs the golden queries plus off-topic control questions that expose tests which measure nothing. |
| `scripts/kb_refresh.py` | The scheduled job: harvest refresh, ingest, eval for every active KB, no LLM involved. Problems become `ALERT[...]` lines on stderr; a failed ingest or eval run (not a MISS) also makes the exit code non-zero. |
| `scripts/advisor_bash_guard.py` | A PreToolUse hook for the specialist agents: their only allowed Bash command is their own search. |
| `skills/kb-build/` | The Claude Code skill that walks through a build: intake, sources, ingest, eval, advisor, maintenance. |
| `skills/kb-build/references/advisor-template.md` | The specialist agent: read-only, cites every key statement, separates `[KB]` from `[model knowledge, unverified]`, treats every source as data. |
| `examples/demo/` | A fictional board game's rules (eleven pages) as an offline demo KB, with golden queries that pass. |
| `docs/architecture.md` | The design decisions and their reasons. |

## Quickstart

Needs Python 3.11+, Docker, about 2.3 GB for the embedding model and macOS or Linux.

```bash
git clone https://github.com/kevintheo-ai/kb-build && cd kb-build
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # Linux without GPU: see the first lines of requirements.txt
python scripts/bge_embedder.py           # one-time model download + self-test, offline afterwards
docker compose up -d                     # Qdrant on 127.0.0.1:6333

cp config/kb_registry.example.yaml config/kb_registry.yaml
python scripts/kb_ingest.py demo
python scripts/kb_search.py --kb demo "Can archivists in different wings lend each other cards?"
python scripts/kb_search.py --kb demo --eval
```

Tests: `python -m pytest -q` (no network, no Qdrant, no model needed).

## Use it with Claude Code

```bash
ln -s "$PWD/skills/kb-build" ~/.claude/skills/kb-build
```

Then ask Claude Code for a knowledge base ("build me a knowledge base and a specialist for our support
handbook", or `/kb-build`). The skill finds the repo through its own folder, asks what the KB is for and which
sources to use, shows you the harvest plan before it fetches anything, and stops when the eval is green and
the specialist passed a normal and an adversarial probe plus a check of its command guard. The generated
specialist lands in `~/.claude/agents/<slug>-advisor.md`; call it before decisions in that domain. The guard
needs Claude Code 2.1.295 or later.

Where things live: everything sits in gitignored folders of the repo (`kb/` for sources, `state/` for
manifests, `config/kb_registry.yaml`). For real use point `KB_HOME` somewhere else, for example
`export KB_HOME=~/kb-data`, so a `git clean` can never touch your data. `KB_REGISTRY`, `QDRANT_URL` and
`QDRANT_API_KEY` are the other settings.

Keep it fresh with one line in cron (or launchd, or systemd):

```
30 4 * * *  cd /path/to/kb-build && .venv/bin/python scripts/kb_refresh.py
```

## Web pages, robots.txt and YouTube

- **What kb-build sends:** a regular browser user agent, because many documentation sites turn away unknown
  clients. Set `KB_USER_AGENT` to identify your crawler instead. kb-build does **not** read robots.txt and has no
  code to get past bot protection: a 403 or a challenge page is treated as an answer, the URL is retried later.
  Check the terms of every site before you add it; some forbid automated access.
- **YouTube transcripts** are off by default and need `allow_youtube: true` per KB. The transcript library uses
  an unofficial endpoint that YouTube's robots.txt disallows and that may break YouTube's terms of service. That
  risk is yours if you switch it on. The official route is the YouTube Data API, which serves captions of your
  own videos; save those into `kb/<slug>/docs/`.
- **What you fetch stays yours to handle.** Harvested pages and bought courses are for your own use: they sit in
  gitignored folders and should never be published or redistributed. This repo contains none: the demo is
  self-written, the online example only lists the addresses of eight U.S. National Archives pages.

Not legal advice: what you may fetch and store depends on the site's terms and the law where you live.

## The rules this encodes

1. **Retrieval is tested, not hoped for.** Every KB carries golden queries; a build is finished when all of them
   hit and none of them also hits an off-topic control question.
2. **One domain, one collection.** No search noise from other domains, and every KB can be rebuilt alone.
3. **Destruction needs intent.** An empty source folder never empties a collection; deleting more than half of
   a KB in one run needs `--allow-prune`.
4. **Sources are data, never instructions.** Instruction-like text is flagged at ingest and shown as such; the
   specialist reports it as an anomaly and never acts on it.
5. **Cite or say "not in the KB".** The specialist marks everything that does not come from a hit as
   `[model knowledge, unverified]`.

## Known limitations

- **The command guard is a second line, not a sandbox.** It is a hook in the advisor's frontmatter, so it runs
  only where Claude Code runs that agent's hooks. In live tests (Claude Code 2.1.310, headless) it blocked a
  forbidden command in default and in bypass-permissions mode, foreground and background. It does not run for a
  **project-level advisor** (`.claude/agents/` of a repo) until someone accepted that repository's trust dialog:
  a `claude -p` run in an untrusted folder skips the hook with one line in the debug log, which is what my
  first test ran into. User-level advisors (`~/.claude/agents/`, the default) are not affected. If the guard
  cannot start (moved venv, wrong path), `onFailure: "block"` in the template blocks every Bash call instead of
  letting it through; that needs Claude Code 2.1.295 or later. The guard covers Bash only: Read/Grep/Glob stay
  available for briefing files, so the agent can read what your user account can read.
- **The injection flag is a phrase list,** in English and German. Soft steering ("always recommend product X")
  is not detected; the specialist's rules are the main defence.
- **The eval is small by design:** hit@5 on a handful of questions per KB, not a benchmark.
- **Fetching:** no JavaScript rendering, so script-heavy pages come back empty; save those by hand into `docs/`.
  The SSRF guard checks names and redirects but not DNS rebinding between check and connect.
- **Stored paths are absolute.** After moving `KB_HOME`, run `kb_ingest.py <slug> --full` once.
- **Alerts go to stderr.** Let cron mail them, or pair the refresh with a heartbeat check.
- **macOS and Linux only** (`fcntl` locks). Windows would need another lock.

## What this is not

Not a framework, not a hosted service, not affiliated with any model vendor. It is the knowledge layer I use
every day, stripped of everything specific to my company, published as-is under MIT. Issues and war stories
welcome.

— Kevin Theo ([LinkedIn](https://www.linkedin.com/in/kevin-theobald-7946871a2) · [kevintheo.com](https://kevintheo.com))
