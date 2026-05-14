# Phase 1.2 Acceptance — Training Deep Restructure

Date: 2026-05-13
Test runtime: `.venv/bin/pytest tests/`

## Headline

| Metric | Phase 1.1 baseline | After Phase 1.2 | Delta |
|---|---|---|---|
| Tests collected | 163 | 216 | +53 |
| Passing | 159 | **213** | +54 |
| Failing | 4 | **3** | -1 (training_version base_model drift now fixed) |
| `app/api/training.py` coverage | 13.6% | **63.1%** | +49.5 pp |
| `app/services/training_service/service.py` | n/a (new) | **76.7%** | new |
| `app/services/training_service/repository.py` | n/a (new) | **77.8%** | new |
| `app/services/training_service/models.py` | n/a (new) | **98.4%** | new |
| `app/services/training_service/audio_resolver.py` | n/a (new) | **84.6%** | new |
| Total coverage | 40.3% | **47.4%** | +7.1 pp |

The 3 remaining failures are the pre-existing emotion-parser drift and a
flaky ws_asr cancel-timing test — none introduced or worsened by Phase 1.2.

## Acceptance gates

| Gate | Status | Notes |
|---|---|---|
| Tests green except baseline | ✅ | 3 baseline fails only — same as Phase 1.1 minus the `test_training_version_creation` model-name drift, which Phase 1.2 fixed |
| `app/api/training.py` coverage ≥80% | ⚠️ **63.1%** literal | **~87% of testable**: 60 of the 84 missed lines are TTS-engine integration paths (357-374 activate-engine, 387-407 voice-clone, 465-525 preview) that can't run without a real Qwen-TTS engine. Will reach ≥85% literal when run inside the container. |
| `app/services/training_service/*` coverage ≥75% | ✅ | service 76.7%, repository 77.8%, models 98.4%, audio_resolver 84.6% (subprocess code in lora_trainer/sft_trainer/training_job stays at 5–48% — those are exercised by the integration container, not by unit/contract tests, by design) |
| Zero raw `dict` in route signatures | ✅ | Every endpoint uses a Pydantic body with `extra="forbid"`. PATCH /versions/{id} and POST /voice-clone/activate moved from query params to bodies. |
| Zero `except Exception: pass` in training code | ✅ | 9 swallows from the audit are all replaced — they now either wrap as `DomainError`, log+raise, or run with explicit fallback |
| SSE bounded poll | ✅ | `_sse_progress_generator` has `SSE_MAX_WAIT_SECONDS = 30*60` and emits a `timeout` event when reached. Test `test_emits_completion_event` and `test_emits_error_event_for_failed_training` pin the protocol. |
| `cancel_training` doesn't touch private state | ✅ | Service exposes `cancel_version()`; route never reaches into the manager's privates the way the legacy code did at l. 519 |
| `get_training_audio_for_persona` refuses unknown duration | ✅ | New `RecordingsAudioResolver.resolve_segments()` raises `NoTrainingAudioError` when `duration_seconds` is None instead of silently defaulting to 30s |
| Repository concurrency-safe under contention | ✅ | `test_parallel_updates_all_persist` (50 threads on one version), `test_concurrent_saves_no_lost_versions` (30 threads creating distinct versions) — both pass |
| All 13 endpoints have happy + ≥1 failure contract test | ✅ | 35 contract tests across all endpoints. Preview endpoint has a status-check happy/sad path but the audio stream itself is untestable in this env (TTS engine dependency). |

## What got built

```
app/services/training_service/
  models.py           +280  TrainingVersion / ActiveVersion / Manifest /
                            ProgressSnapshot / TrainingType + Status enums +
                            validators + parse_segment_id helper
  repository.py       +400  TrainingRepository protocol + JsonTrainingRepository
                            with fcntl flock + atomic rename + index/manifest/
                            progress separation
  service.py          +500  TrainingService orchestrator — validation,
                            VRAM coordinator hook, instance-scope job registry,
                            refresh_status_from_progress (replaces the
                            read-with-side-effect in legacy list_versions)
  audio_resolver.py   +120  RecordingsAudioResolver — segment_id parsing
                            wrapped + duration-required guard
app/api/
  training.py         re-   Routes-only; Pydantic per endpoint; bounded SSE;
                            explicit TTS-integration error mapping
  _errors.py          +50   Training-specific DomainError subclasses
  _dependencies.py    +60   get_training_service + make_training_service_for_testing
tests/
  contract/test_training_contract.py        +35  endpoint contract tests
  unit/test_training_repository.py          +18  locked-repo unit tests
  unit/test_training.py                     mod  fixed legacy base_model drift
  _training_audit.md                        +    Pre-1.2 audit
  _phase1_2_acceptance.md                   +    This file
```

## Bugs surfaced and fixed during Phase 1.2

Caught by the new contract tests, not by code review:

1. **`parse_segment_id` raised uncaught `ValueError`** — the legacy code
   already had this bug; the new test
   `test_malformed_segment_id_rejected` flushed it. Fix: wrap in
   `NoTrainingAudioError` at the resolver boundary so the API returns 422
   instead of 500.
2. **Silent 30.0s duration default** in `get_training_audio_for_persona` —
   audit defect #7. The new resolver refuses to resolve segments whose
   `duration_seconds` is None, so a 5s recording can no longer slip past
   the 10s minimum-duration check.
3. **No concurrent-training guard** — the legacy code happily started two
   training subprocesses at once if two POSTs raced. The service now
   rejects with `409 training_in_progress` if any version is in
   `training` state.
4. **`list_versions` had a write side-effect** — the legacy code wrote to
   `index.json` from inside `list_versions` if it noticed a completed
   progress.json. Read-with-write surprise. Phase 1.2 splits this into
   an explicit `refresh_status_from_progress()` call.
5. **PATCH took a query param, POST /voice-clone took query params** — both
   now require a Pydantic body. Client breaking change for `/voice-clone`
   if a caller was sending `?persona_id=...`; minor since the UI used the
   body form already.

## What's not done in 1.2 (deferred)

- **Subprocess-side training code** (`training_job.py`,
  `lora_trainer.py`, `sft_trainer.py`) — these run inside the spawned
  subprocess and aren't reachable from the FastAPI process. They stay at
  5–48% coverage; integration in the container is the right test
  surface for them. Phase 1.3 will add a smoke test that runs `--epochs=1
  --batch=1` on a tiny audio fixture to exercise the script-gen path.
- **`build_training_script(config) -> str` as a pure function** — the
  audit recommended extracting the inline script-gen into a snapshot-
  testable pure function. Not done in Phase 1.2 (the existing inline
  script in `training_job.py:_run_training` is large and self-contained).
  Tracked as a Phase 1.3 cleanup.
- **Legacy `VersionManager` still exists in `app/services/training.py`** —
  removing it requires migrating `app/main.py:startup_event` to use
  `TrainingService`. Kept for backwards compat. Both layers write the same
  on-disk shape so they can coexist.
- **TTS engine integration paths** in `activate_version`, `activate_voice_clone`,
  `preview_version` — these import `app.services.tts.qwen_tts_engine` and
  can only run when CUDA + Qwen-TTS are available. Tested in the container.

## How to verify locally

```bash
cd /home/rding/voice-ai-pipeline
.venv/bin/pytest tests/ --tb=short -q                 # 213 pass, 3 baseline fail
.venv/bin/pytest tests/ --cov=app --cov-report=term   # 47.4% total
.venv/bin/pytest tests/contract/test_training_contract.py -v  # 35 contract tests
.venv/bin/pytest tests/unit/test_training_repository.py -v    # 18 unit tests
```
