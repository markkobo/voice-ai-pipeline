# Multi-commit Staff Review — 2026-05-17

Four parallel staff-engineer reviews against substantive commits in the
recent stack:

| Commit | What it shipped |
|---|---|
| `8161535` | Slice 2A — text ingestion engine + `/ingest` endpoint |
| `a0e7b8e` | Slice 1 — corpus storage layer + REST endpoints |
| `6c9a87a` | Training UI grouping + SFT dropdown + LLM-cancel diagnostics |
| `00df3a5` | Jinja2 pilot (dev UI refactor) |

Total surface: ~28 numbered comments. Triage below.

---

## Addressed in immediate follow-up commit

| Review | # | Severity | Fix |
|---|---|---|---|
| 8161535 | 1 | BLOCKER | Encoding cascade rewrite: Big5 tried before gb18030 (gb18030 was permissively eating Big5 → mojibake), plus `_looks_like_real_text` sanity check on each candidate. UTF-16 BE/LE BOM fast-paths. U+2028 / U+2029 + embedded NUL handling. |
| 8161535 | 3 | MAJOR | `_find_original` no longer globs `original.*` — uses recorded `item.filename` extension. |
| 8161535 | 6 | MAJOR | Concurrent `/ingest` race fixed: `extracted.txt` and `chunks.jsonl` both written via `_atomic_write_text` (tempfile + fsync + os.replace). |
| 8161535 | 12 | MINOR | Embedded NUL bytes stripped in `_normalize_newlines`. |
| 8161535 | 13 | MINOR | U+2028 / U+2029 normalized to `\n` so paragraph regex sees the breaks. |
| a0e7b8e | 11 | BLOCKER | `persona_id` and `item_id` validated against strict regex on every read/write/ingest path. `^[a-z][a-z0-9_]*$` for persona, UUID-style for item. New `InvalidCorpusIdError` → 400. Five path-traversal test shapes parametrized. |
| a0e7b8e | 12 | BLOCKER | `delete` now holds the metadata exclusive lock across rmtree + index-remove. `update` re-checks `meta_path.exists()` inside the lock and raises `CorpusItemNotFound` if a concurrent delete won. |
| a0e7b8e | 13 | BLOCKER | `CorpusService.upload` rmtree's the item dir on save failure so original bytes don't orphan. |
| a0e7b8e | 16 | MAJOR | Filename sanitization: control chars + `/`/`\` stripped before persistence (anti log-injection, anti malformed-JSON consumer). |
| 6c9a87a | 21 | BLOCKER | Stale-cancel race fixed via utterance-seq stamping. `begin_utterance` increments seq; `cancel_llm_task` stamps the latch with current seq; `set_llm_task` honors the latch iff seqs match. A cancel arriving between utterance N's `clear_llm_task` and N+1's `set_llm_task` no longer kills N+1. |
| 6c9a87a | 22 | MAJOR | `WebSocketDisconnect` now triggers a `cancel_llm_task(origin="ws_disconnect")` — distinct origin in the log instead of `remove_session` confusion. |

Test count added: 30+ across `test_corpus_encoding.py`, `test_state_manager_cancel_seq.py`, plus parametric path-traversal cases in `test_corpus_contract.py`.

---

## Deferred to task #62 (data-model + extractor-protocol commit, BEFORE slice 2C)

These are structural changes that PDF/audio extractors will require
anyway — doing them now would conflate "review fix" with "next feature."

| Review | # | Severity | Plan |
|---|---|---|---|
| 8161535 | 2 | BLOCKER | Atomic write of `extracted.txt` + `chunks.jsonl` + status flip as a single unit. Land via the `Extractor` protocol (each extractor returns an `ExtractResult`; service does the persist transaction). |
| 8161535 | 4 | MAJOR | `ingesting` status state machine + startup-sweep that resets stranded `ingesting` to `failed("interrupted")`. |
| 8161535 | 5 | MAJOR | Idempotency vs chunker changes: add `chunker_version` field to chunk records + bump on chunker tweaks. |
| 8161535 | 7 | MAJOR | Manifest cache invalidation hook from `IngestionService.ingest()` once denormalized manifest lands. |
| 8161535 | 8 | MAJOR | Chunker mid-stream runt folding (currently only folds the trailing fragment). |
| 8161535 | 9 | MINOR | Strip trailing whitespace from chunk text after slicing. |
| 8161535 | 10 | MAJOR | UTF-16 endian autodetect without BOM (counts non-printables on both endians, picks the cleaner). |
| a0e7b8e | 14 | MAJOR | Lock-file cleanup (lazy unlink under exclusive lock after every successful write). |
| a0e7b8e | 17 | MAJOR | Per-kind size caps (50 MB CSV, 20 MB text/md). |
| a0e7b8e | 18 | MAJOR | Denormalized manifest cache (`manifest.json` updated on every save/delete inside the index lock). |
| a0e7b8e | 19 | MAJOR | Pydantic on-disk-vs-API posture: `extra="ignore"` for the on-disk shape, `extra="forbid"` only for the API response. |
| 6c9a87a | 15 | MAJOR | `persona_speaker_alias: Optional[str]` field on `CorpusItem` (Phase 3 LoRA needs it to filter persona-side turns from chat exports). |
| 6c9a87a | 16 | MAJOR | ISO-8601 timestamp normalization on chat parsers — for Phase 2 RAG date-filtering. |
| 6c9a87a | 18 | MAJOR | `content_sha256` field on `CorpusItem` for upload-time dedup. |
| 6c9a87a | 19 | MAJOR | `(kind, ext) → callable` registry → `Extractor` protocol with `supports / extract / is_async / needs_gpu`. Hard prerequisite for slice 2C/2D. |
| 6c9a87a | 25 | MAJOR | Per-chunk provenance (time range, speakers covered). |
| 00df3a5 | 5 | MAJOR | Asset-hashing + `Cache-Control: public, immutable` for production (family UI specifically). |
| 00df3a5 | 4 | MAJOR | Dev-mode `Cache-Control: no-cache` so browsers always revalidate static assets. |

---

## Deferred to task #64 (status-bar partial refactor, next commit)

| Review | # | Severity | Plan |
|---|---|---|---|
| 00df3a5 | 7 | MAJOR | Extract `templates/_status_bar.html` (`{% include %}`), `static/css/_status_bar.css`, `static/js/_status_bar.js`. Fixes the recordings.html drift (missing `title=` attrs and IDs the JS expects). |

---

## Deferred to a streaming-code-tests commit

| Review | # | Severity | Plan |
|---|---|---|---|
| 6c9a87a | 20 | BLOCKER per CLAUDE.md | Streaming-code-changes-without-tests violates the project's own policy. Test scaffolding mostly landed in this commit (`test_state_manager_cancel_seq.py`); still missing: (a) integration test asserting `WebSocketDisconnect` cancel-origin shows up in the log, (b) test that `openai_client` distinct CANCELLED-path messages fire correctly, (c) test for the function-calling `stream_with_function` CANCELLED logging (parallel patch missed). |
| 6c9a87a | 23 | MAJOR | Defensive `if not chunk.choices: continue` in `openai_client.stream` — and corresponding test. |
| 6c9a87a | 24 | MAJOR | Patch the same diagnostic logging through `stream_with_function` (parallel sibling). |
| 00df3a5 | 6 | MAJOR | HTML-link-validation smoke test: parse the response, extract every `<link>` and `<script>` URL, GET each, assert 200. Replaces the current string-match-only smoke tests. |

---

## Notes on items NOT actioned

| Review | # | Severity | Reason |
|---|---|---|---|
| a0e7b8e | 15 | MAJOR | NFS flock semantics — documented as a known limitation but not changing the lock strategy. Production target (DGX Spark / single box) makes this academic. |
| 6c9a87a | 1 | n/a | Reviewer challenged the strong-ref-for-create_task pattern as chasing a phantom. Kept the set + done_callback (cheap insurance), updated comment to drop the "GC was the cause" claim. *(Comment is correct in this commit — already pruned in the latest version.)* |
| 8161535 | 11 (MINOR) | MINOR | Grapheme-cluster splitting on hard cut — emoji-heavy chat exports. Will revisit if observed. |
| 8161535 | 15 (MINOR) | MINOR | `listener_tag=None` explicit test — already implicit in existing tests. Skip. |
| a0e7b8e | NIT | NIT | All NITs noted but not fixed (commit body lifecycle, exception aliasing cleanup, etc.) — too churny for net code-clarity gain. |
| 00df3a5 | 1 (MINOR) | MINOR | `templates.env.auto_reload = True` is a no-op default, but documents intent. Keep as documentation. |
| 00df3a5 | 3 (MINOR) | MINOR | Autoescape note added to follow-up doc, not code. Will note in family-UI work when `{{ }}` actually appears. |

---

## Final test count

After this commit: **~395 passed**, 1 skipped (was 364/1). Net new tests:
~30 across encoding, state-manager cancel race, path traversal.
