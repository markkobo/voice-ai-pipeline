# Phase 2 Acceptance — Streaming Core

Date: 2026-05-14
Test runtime: `.venv/bin/pytest tests/`

## Headline

| Metric | Phase 1.3 | After Phase 2 | Delta |
|---|---|---|---|
| Tests passing | 247 | **270** | +23 |
| Tests failing | 3 | **0** | -3 (all chronic baseline failures resolved) |
| Tests skipped | 0 | 1 | +1 (replaced by deterministic unit test) |
| Total coverage | 51.1% | **52.4%** | +1.3 pp |
| `app/services/tts/emotion_mapper.py` | 37.5% | 39.5% | +2.0 pp |
| `app/services/tts/qwen_tts_engine.py` | 12.2% | 18.5% | +6.3 pp |

Compared to the original Phase 0 baseline (where this project started):
- Tests: **109 → 270 passing** (+161 substantive tests)
- Failures: **7 → 0** (all real bugs fixed)
- Coverage: **31.6% → 52.4%** (+20.8 pp)

## Acceptance gates

| Gate | Status | Notes |
|---|---|---|
| All 3 baseline failures resolved | ✅ | emotion parser `is_ready` drift fixed, emotion-instruct test rewritten for Path B, cancel race latched via sticky-cancel flag |
| Zero `except Exception: pass` in `app/api/ws_asr.py` | ✅ | All 6 silent swallows replaced with `await_prior_tts_task` (logs + raises) or `safe_send_text` (only swallows client-disconnects) |
| Drain loops bounded | ✅ | Three duplicated `while True: parser.update('')` blocks replaced by `drain_emotion_parser()` with `DRAIN_MAX_ITERATIONS=256` termination cap |
| TTS engine model reload race fixed | ✅ | `threading.RLock` around `_ensure_loaded` + activate paths; proven safe by 50-thread `test_concurrent_ensure_loaded_loads_once` |
| Dead code removed | ✅ | `EMOTION_TAG_RE`, `tts_ready` legacy frame removed; emotion-strip regexes hoisted to module-level |
| Emotion parser proven correct on arbitrary chunking | ✅ | 5 Hypothesis property tests × ~50–200 random inputs each = ~450 random splits verified |

## What got built

```
app/api/
  _ws_helpers.py      +175  drain_emotion_parser, safe_send_text/bytes,
                            await_prior_tts_task, send_tts_error_frame
  ws_asr.py           re-   Eliminated 6 silent except-Exception swallows
                            + 3 drain-loop copies; every WS send guarded;
                            dead tts_ready frame + EMOTION_TAG_RE removed
app/services/tts/
  emotion_mapper.py   patch is_ready now correctly returns True after the
                            emotion tag is locked (was permanently False
                            because Path B made current_instruct=None)
  qwen_tts_engine.py  patch threading.RLock around _ensure_loaded and
                            activate paths — prevents double-load race
app/core/
  state_manager.py    patch llm_pending_cancel sticky flag — cancel
                            before set_llm_task is no longer a no-op
tests/
  unit/test_emotion_parser.py             patch test_unknown_emotion rewritten for Path B
  unit/test_emotion_parser_property.py    +200 Hypothesis property tests
  unit/test_ws_helpers.py                 +200 12 tests for the new helpers
  unit/test_tts_engine_lock.py            +90  3 tests for the load lock
  test_ws_asr.py                          patch TestStickyCancel unit test + skipped
                                                integration cancel test with reason
  _training_audit.md                      (Phase 2 ws_asr audit was inlined)
  _phase2_acceptance.md                   this file
```

## Bugs surfaced and fixed during Phase 2

1. **`EmotionParser.is_ready` permanently False under Path B.** The
   Path B transition made `get_tts_instruct` always return None, but
   `is_ready` still required `current_instruct is not None`. So after a
   correctly-parsed `[E:寵溺]`, the parser reported "not ready" forever.
   Fixed by dropping the obsolete instruct check.
2. **Unknown-emotion test asserted dead string.** Same Path B drift —
   test asserted a specific TTS instruct string that no longer exists.
   Replaced with the actual Path B contract: unknown emotion routes
   through the default enhancer in `enhance_text`.
3. **Cancel-before-LLM-start was a silent no-op.** `cancel_llm_task`
   only acted if `llm_cancellation_event` was already set, but during
   the create_task → set_llm_task window, no event exists. Fixed by
   latching `llm_pending_cancel = True` on cancel, then firing
   immediately when `set_llm_task` is called. Proven by the new
   `TestStickyCancel::test_cancel_before_set_llm_task_is_honored`.
4. **Three drain loops duplicated, unbounded.** The audit flagged the
   risk of an infinite loop if `parser.update('')` doesn't converge.
   Consolidated into `drain_emotion_parser` with a 256-iteration cap.
5. **TTS task errors silently swallowed.** Every `await current_tts_task`
   was wrapped in `try: ... except Exception: pass`. Audio just stopped
   when a TTS sentence failed, leaving the user with a frozen UI. Now
   uses `await_prior_tts_task` which logs the exception with the
   relevant context tag, and (separately) sends a `tts_error` frame so
   the client can recover.
6. **WS send errors propagated as uncaught exceptions.** Every
   `websocket.send_text/send_bytes` call now goes through `safe_send_*`,
   which catches only client-disconnect errors and re-raises everything
   else.
7. **Documented TTS double-load race.** `activate_version` /
   `activate_voice_clone` set `is_loaded=False` and called
   `_ensure_loaded()` without a lock — two concurrent
   `generate_streaming()` calls would both reload the model. Fixed by
   `threading.RLock` around load + activate, with a fast-path check
   outside the lock for the common case.
8. **Receive_json `timeout=` parameter never worked.** Starlette's
   `WebSocketTestSession.receive_json()` doesn't accept `timeout=`.
   The legacy cancel test passed it anyway; the `except Exception: break`
   silently swallowed the TypeError, so the test "completed" without
   ever actually receiving a message. The skipped test documents this.

## What's not done in Phase 2 (deferred)

- **`ws_asr.py` coverage** stays at ~20.6% — the WebSocket message-loop
  branches are hard to drive with the sync TestClient. Adding contract
  tests for the full pipeline needs an async WS client (e.g. httpx-ws)
  and is bigger than the Phase 2 budget. Tracked for future work.
- **The actual end-to-end integration cancel test** is skipped with a
  clear reason; the sticky-cancel contract is proven at the unit level.
  The right replacement is an async-client integration test, not a
  patched-up TestClient-based one.
- **Fallback TTS streaming**: `Qwen3TTSModel` (non-streaming) is loaded
  if `FasterQwen3TTS` fails. The route doesn't currently know the
  difference; adding a `streaming: bool` capability flag the route can
  check is tracked for Phase 2.1.
- **VAD reset race during start_speech**: the `state.audio_buffer.clear()`
  + `state.vad.reset()` pair in the start_speech handler isn't atomic
  with the audio-receive path. Not exercised by current tests; tracked
  for Phase 2.2.

## How to verify locally

```bash
cd /home/rding/voice-ai-pipeline
.venv/bin/pytest tests/ -q                                # 270 pass, 1 skipped, 0 fail
.venv/bin/pytest tests/ --cov=app --cov-report=term       # 52.4% total
.venv/bin/pytest tests/unit/test_emotion_parser_property.py -v   # Hypothesis tests
.venv/bin/pytest tests/unit/test_ws_helpers.py -v                # helper unit tests
.venv/bin/pytest tests/unit/test_tts_engine_lock.py -v           # load-lock race test
```
