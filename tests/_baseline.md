# Test Baseline — Phase 0, Task 5

Captured: 2026-05-13 against commit `f1c9213` + Phase 0 conftest changes.

Runtime: host `.venv` at `/home/rding/voice-ai-pipeline/.venv` (Python 3.12.3,
torch 2.6.0+cu124). Host does **not** have `ffmpeg`/`ffprobe` installed — some
upload tests can only pass in the project container.

## Headline numbers

| Metric | Value |
|---|---|
| Tests collected | 116 |
| Passing | 109 |
| Failing | 7 |
| Errors | 0 |
| Walltime | ~9s |
| **Coverage (total)** | **31.6%** |

This 31.6% is the floor — no PR may drop below it without explicit acceptance.

## Coverage by module (current state)

| Module | Coverage | Notes |
|---|---|---|
| `app/api/listeners.py` | 42.4% | No happy-path tests for create/update/delete |
| `app/api/personas.py` | 36.7% | Same — only schema imports exercised |
| `app/api/recordings.py` | **20.0%** | 17 endpoints, ~4 tested; Phase 1.1 target ≥85% |
| `app/api/training.py` | 13.6% | 13 endpoints, 0 tested |
| `app/api/tts_stream.py` | 15.3% | 0 tests for TTS HTTP |
| `app/api/ws_asr.py` | 20.6% | 2 happy-path WS tests, no barge-in/partial-tag |
| `app/services/recordings/pipeline.py` | 6.8% | All pipeline tests deleted (Phase 0 audit) |
| `app/services/training_service/training_job.py` | 10.3% | Script-gen anti-pattern tests deleted |
| `app/services/tts/emotion_mapper.py` | 37.5% | Parser tests cover the streaming path but not all emit edge cases |
| `app/services/recordings/file_storage.py` | 56.2% | Good for unit-level |
| `app/services/recordings/metadata.py` | 57.8% | Good for unit-level |
| `app/services/recordings/quality.py` | 48.8% | Real numpy tests |
| `app/services/training.py` | 61.7% | `test_training.py` exercises VersionManager CRUD |
| `app/services/asr/vad_engine.py` | 78.0% | `TestEnergyVAD` is solid |
| `app/services/asr/silero_vad.py` | 61.5% | |
| `app/main.py` | 78.7% | |
| `app/logging_config.py` | 76.8% | |
| `telemetry/` | not measured | (omitted by coverage config) |
| **TOTAL** | **31.6%** | |

## Failing tests at baseline

These will be addressed in Phase 1.1+ or are environment-only. The plan does
not block on them.

| Test | Category | Action |
|---|---|---|
| `test_upload_invalid_listener_id` | **env** (missing `ffprobe`) | Will pass in container; replaced by contract test in Task #13 |
| `test_upload_invalid_persona_id` | **env** | Same |
| `test_upload_and_get_recording` | **env** | Same |
| `test_websocket_cancel_stops_llm` | **flaky timing** (mock LLM streams completes before cancel) | Address in Phase 2 streaming work |
| `test_is_ready_after_emotion` | **real test/code mismatch** — `parser.is_ready` returns False after `[E:調皮]` lock | Production drift; either parser or test is wrong; fix in Phase 2 |
| `test_unknown_emotion_uses_default_instruct` | **real test/code mismatch** | Fix in Phase 2 |
| `test_training_version_creation` | **production drift** — `base_model` default changed from `VoiceDesign` to `Base`, test wasn't updated | Update test or production in Phase 1.2 |

None of the 7 are introduced by Phase 0 work. The metadata.py errors from the
first pytest run (14 errors due to hardcoded `/workspace` paths) were
resolved by making `isolated_data` an `autouse=True` fixture.

## Warnings noted

- `DeprecationWarning: on_event is deprecated, use lifespan event handlers` —
  app/main.py uses the old FastAPI startup pattern; Phase 1.3 cross-cutting.
- `SyntaxWarning: invalid escape sequence '\['` in standalone_ui.py and
  recordings_ui.py — raw strings missing `r` prefix; trivial fix in Phase 1.3.
- `RuntimeWarning: divide by zero encountered in log10` in
  app/services/recordings/quality.py:100 — SNR calc on silent input; needs a
  guard, real bug.
- `Couldn't find ffmpeg or avconv` — host environment.

## How to reproduce

```bash
cd /home/rding/voice-ai-pipeline
.venv/bin/pytest tests/ --tb=short --cov=app --cov-report=term --cov-report=xml
```

For full-fidelity (with `ffprobe`), run inside the project container:
```bash
bash setup_container.sh start
docker exec voice-ai-pipeline pytest tests/ --tb=short --cov=app
```

## Acceptance floor for Phase 1.1

- Coverage on `app/api/recordings.py` must rise from **20.0%** → **≥85%**
- Coverage on `app/services/recordings/*` must rise from current avg → **≥85%**
- Zero new failing tests on host (excluding the 7 listed above, which fall
  under Phase 1.2 / Phase 2 scope)
- Total coverage must rise from **31.6%** → **≥40%** (recordings module alone
  carries roughly this much weight)
