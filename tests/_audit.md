# Test Audit — Phase 0, Task 1

Date: 2026-05-13
Total: ~3,200 lines across 12 test files.

Classification: **substantive** (asserts behavior, catches regressions), **superficial**
(asserts keys exist or trivial language semantics, would not catch a real
regression), **infrastructure** (fixtures/helpers).

Per user decision, superficial tests are **deleted**, not tagged. Replacements
land as contract tests in Phase 1.1+.

---

## File-by-file classification

### tests/conftest.py — infrastructure (rewrite, not delete)
Has fixtures but they're thin and not used consistently. The `event_loop`
session fixture is deprecated under pytest-asyncio 0.21+. The `test_client`
fixture has no isolation between tests — state from one test leaks to the next.
Replaced wholesale in Task #3.

### tests/test_ws_asr.py (308 lines)

| Test class / function | Verdict | Reason |
|---|---|---|
| `TestStateManager` (3 tests) | **keep** | Real session lifecycle assertions |
| `TestEnergyVAD` (3 tests) | **keep** | Real VAD math + calibration |
| `TestMockASR.test_recognize_returns_result` | **keep** | Minimal but asserts shape |
| `TestWebSocketIntegration.test_full_websocket_flow_with_test_client` | **keep** | Real WS round-trip |
| `TestWebSocketIntegration.test_websocket_tts_streaming_flow` | **DELETE** | Swallows exceptions in `while True` loop, only counts message types, has 30s wall-clock timeout. Wouldn't catch missing emotion, wrong PCM, or any content regression. Replace with focused contract test. |
| `TestWebSocketCancel.test_websocket_cancel_stops_llm` | **keep** | Asserts cancel produces `llm_cancelled` |
| `health_check` fixture at EOF | **DELETE** | Looks like a test but is a fixture nobody uses. Dead code. |

### tests/unit/test_emotion_parser.py (178 lines)
**All 14 tests substantive — keep entire file.** Gold-standard: asserts
state-machine transitions, character-by-character emit timing, partial-tag
behavior, multibyte chars. Will get additional Hypothesis-based property tests in
Phase 2.

### tests/unit/test_file_storage.py (111 lines)
**All 12 tests substantive — keep.** Real path-parsing assertions, validation
errors raised correctly.

### tests/unit/test_metadata.py (173 lines)
**All 16 tests substantive — keep.** Mutates real metadata, asserts round-trip
via JSON file. **Note**: these tests will need updating in Phase 1.1 when
`RecordingMetadata` is replaced by Pydantic `Recording` model — but the
behavioral contracts they encode (training_ready auto-calc, expiry on
processed, error state) must survive the refactor.

### tests/unit/test_pipeline.py (95 lines) — **DELETE ENTIRE FILE**
Every test mocks `list_all_recordings` and asserts that MagicMock returned what
was passed in. Tests wiring of mocks, not pipeline behavior. Never runs the
actual denoise/enhance/diarize/transcribe path. Replace with integration tests
against a real audio fixture in Phase 1.1.

### tests/unit/test_quality.py (184 lines)
**All 13 tests substantive — keep.** Uses real numpy arrays and exercises the
actual SNR/RMS/clarity calculations. Threshold tests at end could be tightened
but are not harmful.

### tests/unit/test_regressions.py (97 lines) — **DELETE ENTIRE FILE**
- 5 of 7 tests are `pass` — pure documentation comments masquerading as tests
- `test_segment_parsing_no_leading_underscore` tests Python string slicing as a
  stand-in for JS frontend code — wrong language, wrong layer
- `test_audio_resampling_for_24k` greps a hardcoded path
  `/workspace/voice-ai-pipeline/data/models/persona_mocdl6af_v9_…/train_lora.py`
  — broken on any other environment, also tests the wrong artifact (a generated
  script in a specific old training run)

The intent (preserving past fixes) is valid; the implementation is not. Replace
with proper regression tests inside each module's test file (e.g.,
`test_metadata.py` gets a "speaker_labels syncs with speaker_segments" test
that actually runs the code).

### tests/unit/test_training.py (256 lines)
**Keep, but reshape.** Tests real `VersionManager` CRUD across temp dirs. The
repeated `with patch(...): with patch(...):` pattern is ugly — replace with a
single `version_manager` fixture in conftest. No deletions.

### tests/unit/test_training_job.py (548 lines) — **MIXED**

| Class | Verdict | Reason |
|---|---|---|
| `TestMergeLoraFunction` | **keep** | Real file system: creates/doesn't-create adapter dir, asserts `merge_lora()` return value |
| `TestSFTModelSaving.test_sft_config_should_not_have_dtype_key` | **keep** | Real JSON round-trip |
| `TestSFTAutoActivation` | **keep** | Imports real `TrainingJob`, asserts attribute |
| `TestTrainingJobScriptGeneration` (7 tests) | **DELETE** | Anti-pattern: tests re-implement the training script generation inline (`_generate_actual_script`) and assert against that copy. So they test the test's copy of the logic, not the real `_run_training()` method. The actual script generator could break and these tests would still pass. |
| `TestTrainingJobUseLoraLogic` (3 tests) | **DELETE** | Asserts `("sft" == "lora") is False`. This is testing Python's `==` operator. Trivial. |
| `TestTrainingJobIntegration.test_training_job_initialization` | **keep** | Real `TrainingJob.__init__` |
| `TestTrainingJobIntegration.test_script_file_is_valid_python` | **DELETE** | Same anti-pattern — `ast.parse`s a minimal script reconstructed in the test, not the real generated script |

Replacement plan (Phase 1.2 or later): extract script-generation into a pure
function `build_training_script(config, training_type) -> str`, then test the
real function output (golden file snapshot or AST-walking it).

### tests/unit/test_sft_chunking.py (672 lines)
**Substantive — keep.** Tests chunking arithmetic against the parameters used in
`SpeechDataset`. Some tests (e.g. validating that 6sec chunk + 3sec hop is
"reasonable") are tautologies. But the core math (chunks produced from N
audio files of given lengths) is real and the validation is meaningful. No
deletions; minor cleanup as we go.

### tests/unit/test_sft_training_validation.py (265 lines) — **DELETE ENTIRE FILE**
Revised after closer inspection: the file imports **nothing** from `app/`. Every
test exercises its own f-string or Python's `<` operator (e.g. `log_msg = f"[VALIDATION] ..."`
then `assert "[VALIDATION]" in log_msg`). It is structurally incapable of
catching any production-code regression. Replace with real validation tests
attached to the actual training entry-point in Phase 1.2.

### tests/integration/test_recordings_api.py (196 lines)

| Test | Verdict | Reason |
|---|---|---|
| `test_list_recordings_empty` | **keep** | Real endpoint hit, asserts paginated shape |
| `test_get_personas` | **DELETE** | Only `len(data) > 0` + one membership. Doesn't validate schema (id, label, description), doesn't catch missing fields. Replace with a contract test that asserts full Pydantic schema. |
| `test_get_listeners` | **DELETE** | Same as above |
| `test_get_recording_stats` | **DELETE** | `assert "raw_size_bytes" in data` — exists check only |
| `test_upload_invalid_listener_id` | **keep** | Asserts 400 |
| `test_upload_invalid_persona_id` | **keep** | Asserts 400 |
| `test_get_nonexistent_recording` | **keep** | Asserts 404 |
| `test_delete_nonexistent_recording` | **keep** | Asserts 404 |
| `test_update_nonexistent_recording` | **keep** | Asserts 404 |
| `test_upload_and_get_recording` | **keep, will strengthen** | Round-trips, but doesn't verify the audio bytes survived or that duration was extracted. Strengthen in Phase 1.1. |
| `test_pagination_params` | **DELETE** | Only asserts keys exist. Doesn't upload N=25 recordings and verify page 2/3 returns the right slice. |
| `TestRecordingsUI` (6 tests) | **DELETE ALL** | Greps HTML for element IDs like `id="recBtn"`. Breaks on every UI refactor, value=0. UI is tested by hand or with Playwright, not by string-matching HTML. |

---

## Summary

| Verdict | Count | Lines deleted (approx) |
|---|---|---|
| **Delete entire file** | 2 (test_pipeline.py, test_regressions.py) | ~192 |
| **Delete tests within file** | ~21 tests across 4 files | ~600 |
| **Keep substantive** | ~85 tests | ~2,400 |
| **Keep with refactor** | ~30 tests (test_training.py, test_metadata.py) | (no line delta now, transformed in Phase 1.1) |

**Net effect**: ~800 lines removed in Task #6. Coverage will drop temporarily —
contract tests in Task #13 + concurrency test in Task #14 will recover it.

---

## Anti-patterns to avoid going forward

These showed up repeatedly and should not return in new tests:

1. **Reimplementing the SUT in the test** — `_generate_actual_script` copies
   the script-generation logic into the test file. Tests must call the real
   code path.
2. **`assert <python operator> works`** — `assert 29 < 30`, `assert ("a" == "b")
   is False`. These test the language, not the project.
3. **`pass` as a test body** — if a regression is documented, it must execute
   the code; otherwise it goes in a comment in the relevant module.
4. **HTML element ID grepping** — fragile to any UI change.
5. **Wall-clock timeouts with `except Exception: break`** — flaky and hides
   real errors.
6. **Mocking the layer under test** — `test_pipeline.py` mocks
   `list_all_recordings` then asserts the mock was called. Mock the layer
   *below* the SUT, not the SUT itself.
