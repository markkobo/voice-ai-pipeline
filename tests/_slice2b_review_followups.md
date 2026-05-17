# Slice 2B Review — Follow-ups Deferred to Later Commits

Staff-engineer review of commit `3f2f55e` produced 25 numbered comments
(3 BLOCKERs, 13 MAJORs, several MINORs / NITs). The follow-up commit
addresses the BLOCKERs + parser-level MAJORs that are pure
regex/dispatch fixes. The remaining items live here so they aren't
lost.

## Addressed in the immediate follow-up commit

| # | Severity | Comment | Resolution |
|---|---|---|---|
| 1 | BLOCKER | WhatsApp body starting with date-shape misparses as new msg | Split into Android/iOS regexes with mandatory `-` / `]` separator |
| 2 | BLOCKER | Line parser requires literal `\t`, real exports lose to email/zip | Accept `\t` OR runs of 2+ spaces |
| 3 | BLOCKER | Upload allowlist > ingestion capability ("accept then 415") | Tightened `ALLOWED_EXTENSIONS_BY_KIND` to slice-2B-supported only |
| 4 | MAJOR | CSV column-fuzz collisions ("lifetime" → "time", "senderid" → "sender", "message_id" → "body") | Exact-match-first, then substring-with-negative-token-exclusion |
| 5 | MAJOR | Headerless CSV silently returns `[]` | Sniff row 0 col 0 for timestamp shape; if found, positional `(time, sender, body)` |
| 6 | MAJOR | WhatsApp system lines fold into previous speaker's message | `_WHATSAPP_SYSTEM_HEADER_RE` recognizes the header-without-sender shape and drops it |
| 7 | MAJOR | Prose containing "wechat" / date-shapes misclassified | WeChat sniff requires CSV-shape too; WhatsApp sniff requires mandatory separator |
| 8 | MAJOR | IsSender truthy missing Chinese "是" and lowercase variants | `_WECHAT_TRUE_VALUES` set with case-insensitive Chinese + English forms |
| 9 | MAJOR | WeChat→WhatsApp fallback chain produces garbage on misdetect | Removed; if sniff says wechat, only run wechat parser |
| 11 | MAJOR | O(N²) continuation string concat | Accumulate `current_lines: list[str]`, join at flush — both parsers |
| 13 | MINOR | Line date regex too loose on parens | Tightened: closing paren required when opener present |
| 21 | MAJOR | Missing tests for real-world failure modes | 15 new test cases covering each of the above |

Final test count after follow-up: 37 chat-parser unit tests (was 22) +
3 ingest contract tests updated.

## Deferred to "before slice 2C" commit (task #62)

These are data-model / abstraction changes — not pure parser fixes.
Doing them in the same follow-up would balloon the diff and conflate
correctness fixes with structural refactors.

| # | Severity | Comment | Plan |
|---|---|---|---|
| 15 | MAJOR | `ChatMessage` has no persona-speaker info; Phase 3 LoRA can't filter | Add `persona_speaker_alias: Optional[str]` to `CorpusItem`. Renderer/harvester uses it to split persona-side turns. |
| 16 | MAJOR | No timestamp normalization; Phase 2 RAG date-filter has nothing to filter on | Parse to ISO-8601 at ingest. Best-effort with locale ambiguity (DD/MM vs MM/DD record-the-assumption). |
| 18 | MAJOR | No content-hash dedup; double-upload double-counts toward LoRA thresholds | `content_sha256` field on `CorpusItem`; check at upload time. |
| 19 | MAJOR | `(kind, ext) → callable` registry won't scale to PDF + audio | Refactor to `Extractor` protocol with `supports / extract / is_async / needs_gpu`. |
| 25 | MAJOR | Per-chunk provenance lost when chunker re-parses canonical text | Chunker takes optional `message_metadata: list[ChunkMessageMeta]` from the parser; chunks.jsonl records the time-range + speakers covered. |

## Deferred to a later "polish" commit

| # | Severity | Comment | Notes |
|---|---|---|---|
| 6b | MAJOR | WhatsApp sender names containing `:` (rare; `Bot::Alerts`) break sender capture | Real-world frequency low; revisit when seen. |
| 10 | MAJOR | No parser-level size cap; 200 MB upload → OOM on Mac Mini fallback | Add 50 MB per-conversation ceiling and `MAX_LINE_BYTES = 10_000` early-skip. Applies to all extractors so belongs in the protocol refactor (#19) layer. |
| 12 | MINOR | ReDoS-ish on huge single lines | Same fix as #10 — line-length skip. |
| 14 | MINOR | Double-pass on misdetected text | Cosmetic; trivial after #19. |
| 17 | MAJOR | Sticker / media references pollute chunks (`<sticker>`, `[貼圖]`) | Strip/normalize after we have a real export to point at; locale matters. |
| 20 | MINOR | `ChatFormatRouter` should be its own class | Will fall out of #19 refactor. |
| 22 | MINOR | Test name `test_text_kind_no_longer_accepts_csv_unsupported` misleading | Already deleted in this commit (replaced with the upload-rejection test). |
| 23 | NIT | Internal kind-comparison consistency | Verified — `item.kind` is enum throughout. |
| 24 | NIT | CSV-injection if someone opens `extracted.txt` in a spreadsheet | Document in `app/services/corpus/__init__.py` docstring. Deferred. |
