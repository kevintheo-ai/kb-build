# Architecture and design decisions

## One toolset, one registry, one collection per KB

All knowledge bases share the same five scripts. A new KB is a registry entry plus a few runs, not a new
project. Each KB gets its own Qdrant collection instead of one shared collection with a `kb` filter:

- searches never pick up noise from another domain,
- a KB can be rebuilt, backed up or deleted on its own,
- the blast radius of a bad ingest is one KB.

Qdrant's guidance favours one collection with payload multitenancy for many homogeneous tenants. kb-build
targets a handful of domains, not hundreds of tenants, so the trade-off goes the other way. Questions across
domains are answered by calling `kb_search.py` once per KB.

## Stack

- **Embeddings:** BGE-M3 through FlagEmbedding, locally. One forward pass yields a dense vector (1024
  dimensions, cosine) and sparse lexical weights. The model is pinned to one Hugging Face revision and loaded
  offline after the first download, so vectors stay comparable.
- **Store:** a Qdrant server (Docker), named vectors `dense` and `sparse`. No embedded fallback: a second,
  silently diverging store would be worse than a clear error.
- **Retrieval:** hybrid. A dense prefetch and a sparse prefetch are fused with Reciprocal Rank Fusion in Qdrant.
  The sparse side matters for exact terms (product names, feature names, paragraph numbers). Before results are
  shown, heading-only chunks and word-for-word duplicates (navigation lists, author boxes) are dropped.
- **Chunking:** heading-aware (H1-H4 become metadata), then a recursive character split. Default 1000
  characters with 120 overlap, per KB via `chunk_size` / `chunk_overlap`; the golden-query eval tells you
  whether it fits.
- **Context prefix:** a breadcrumb `Title > H1 > H2 > H3` is prepended to the text that gets embedded, while the
  payload keeps the raw text. It is a cheap stand-in for LLM-written chunk contexts.

## Idempotence and lifecycle

- **Deterministic point IDs:** UUID5 in a namespace per collection, over `file : chunk index : content hash`.
  Never rename a collection; all points would become orphans.
- **Manifest** (`state/<slug>/ingest_manifest.json`): content hash, point IDs and pipeline fingerprint per file
  (collection, chunk size, overlap, context prefix, vector size, model revision, default trust). Unchanged files
  are skipped, changed files replace their chunks, deleted files lose their points, and a file embedded with
  another pipeline is embedded again: a change re-embeds every file once, an interrupted run resumes where it
  stopped, files of an unmounted folder catch up when it is back. Before Qdrant changes, a file's entry lists its
  old and new point IDs (write-ahead), so a crash never leaves points that no manifest knows.
- **Guards:** a count check after every ingest (live points must be at least what the manifest tracks), a
  refusal to empty a collection when no source file is found, a refusal to delete more than half of the tracked
  files in one incremental run, exclusive locks for harvest and ingest, atomic writes of documents and
  manifests, a fail-ratio limit on harvests (rate limits excluded).

## Payload schema (extend by adding fields only)

`kb, text, context, file_path, h1-h4, section, chunk_index, content_hash, author, author_slug, affiliation,
title, date, year, topic, content_type, source_url, trust, tags, harvested_at, created_at, injection_suspect`.
Keyword indexes on kb, author, author_slug, topic, content_type, trust, source_url, content_hash and file_path,
an integer index on year.

## Eval instead of flying blind

Each KB carries 3-5 golden queries in the registry. `kb_search.py --eval` is a hit@5 regression after every
ingest: the expected phrase (from the answering section) must appear in the top hits, in the expected file.
There is deliberately no LLM-judged evaluation harness: for single-user corpora of a few hundred to a few
thousand chunks, the golden-query regression catches chunking and coverage regressions with little upkeep.

**Cross-check against blind questions:** in a small corpus almost every document lands in the top five
sometimes, so a question can "pass" without measuring anything. The eval therefore also runs off-topic control
questions and prints `WARN not discriminative` when an expectation matches there as well. The warning does not
change the exit code (so a scheduled run stays quiet), but it should be zero after a build or maintenance pass.

The eval runs without recency weighting and without deduplication: an ageing expected document must not flip a
question from OK to MISS without any change in the corpus.

## Freshness

- **Recency weighting at query time:** `final = (1 - w) * RRF + w * exp(-ln2 * age / half_life)`, with RRF
  normalised to a maximum of 1. Defaults: `recency_weight` 0.3 (max 0.8), `half_life_days` 365. The server
  returns `max(3 * limit, 24)` candidates and the client re-sorts them, so strongly relevant old documents stay in
  the pool and only move down.
- **Age basis:** the publication `date` if it is plausible (2015 or later, not in the future), otherwise the
  fetch date `harvested_at`, otherwise no anchor: then the blend is a no-op and pure relevance counts. Undated
  documents are neither punished nor preferred.
- **Date artefacts at the source:** page-date guessers read section numbers and copyright lines as dates. Help
  centers and policy pages therefore get `living_doc: true` in the sources file: no page date is written,
  freshness comes from the fetch date.
- **Content refresh:** `kb_refresh.py` refetches documents older than `refetch_days` in place (stable file
  names, the ingest replaces their chunks) and retries earlier failures. Finding NEW sources is deliberately not
  automated: it needs research and a human decision. `curation_due` in the refresh summary is the reminder.

## Prompt-injection hardening, end to end

1. **Harvest:** text extraction only (trafilatura, pypdf, transcript text), nothing is executed. The fetcher
   follows redirects itself and checks every hop: https only, public addresses only, bounded size and time.
2. **Ingest:** instruction-like phrases (a narrow English and German list) mark a document `injection_suspect`.
   It is flagged, not blocked, because documents about prompting would be false positives.
3. **Search output:** flagged hits carry an `INJECTION-SUSPECT` banner; every hit names its source and trust.
4. **Advisor:** read-only brief with an explicit rule that external content is data, a duty to cite, permission
   to say "not in the knowledge base", and a PreToolUse guard (`scripts/advisor_bash_guard.py`) that blocks every
   Bash command except the advisor's own search.

Honest limits: the phrase list does not catch soft steering ("always recommend X"); the guard restricts Bash,
while Read/Grep/Glob stay available for reading briefing files; and the guard runs only where Claude Code runs
the agent's hooks: always for user-level agents, for project-level agents only in a repository whose trust
dialog was accepted (a `claude -p` run in an untrusted folder skips them). The hook sets `onFailure: "block"`, so
a guard that cannot start blocks every command instead of passing them. The advisor's own rules are the first
line, the guard is the second (see README, Known limitations).

## Deliberate upgrade paths (not built)

| Upgrade | When it makes sense |
|---|---|
| LLM-written chunk contexts (full contextual retrieval) | the eval shows systematic misses despite chunk tuning |
| A reranker (for example bge-reranker-v2-m3 on the top 20-25 candidates) | corpora beyond roughly 20k chunks, or top-3 precision is critical |
| Near-duplicate and relevance filtering at harvest time | harvests with lots of boilerplate per source |
| Change detection on official changelog pages | domains where a changelog announces every rule change |

## Deleting a KB

Set `status: archived` in the registry, delete the collection in Qdrant, delete `kb/<slug>/` and `state/<slug>/`,
remove the advisor file. Keep the registry entry as a tombstone so the slug is not reused by accident.
