# Bug Fix Patterns for voice-ai-pipeline

When fixing bugs, check this file first. Add new patterns after each fix.

---

## Training Progress Not Updating (2026-04-25/27)
**Symptom:** Training starts but UI shows no progress, appears stalled
**Root Cause:** progress.json was never created - code only updated if file exists
**Fix:** training_job.py - Add init_progress() function to create progress.json with status=training before training starts
**Detection:** `curl /api/training/versions/{id}/progress` returns error or progress.json doesn't exist
**Regression Test:** `tests/unit/test_regressions.py::TestTrainingBugs::test_progress_json_created_on_start`

---

## FastAPI Query Param vs Body (2026-04-25)
**Symptom:** PATCH updates don't persist, API returns null
**Root Cause:** FastAPI Optional[str] params are query params, not body
**Fix:** recordings_ui.py - Use URLSearchParams, not JSON body
**Detection:** API returns {"persona_id": null} despite correct payload
**Regression Test:** `tests/unit/test_regressions.py::TestAPIBugs::test_fastapi_query_vs_body`

---

## Segment Parsing Wrong Speaker ID (2026-04-26)
**Symptom:** Training preview shows no segment details, "Segment not found"
**Root Cause:** segId.substring(speakerIndex + '_SPEAKER_'.length) gives "00" but speaker_segments has "SPEAKER_00"
**Fix:** training_ui.py - Use + 1 instead of + '_SPEAKER_'.length to get full "SPEAKER_00"
**Detection:** "Segment not found: 00 in {recording_id}"
**Regression Test:** `tests/unit/test_regressions.py::TestUIBugs::test_segment_parsing_no_leading_underscore`

---

## Listener Filter Persists on Persona Change (2026-04-25)
**Symptom:** Empty state after switching personas in training UI
**Root Cause:** listenerFilter not reset when persona changes
**Fix:** training_ui.py - Reset listener filter on persona change
**Detection:** UI shows empty recordings after persona dropdown change
**Regression Test:** `tests/unit/test_regressions.py::TestUIBugs::test_listener_filter_resets_on_persona_change`

---

## Auto-Refresh Disrupts Modal Interaction (2026-04-25)
**Symptom:** Dropdown/modal state lost during 5-second polling
**Root Cause:** loadRecordings() rebuilds entire DOM unconditionally
**Fix:** recordings_ui.py - Skip refresh when modal open or select focused
**Detection:** UI state resets mid-interaction
**Regression Test:** `tests/unit/test_regressions.py::TestUIBugs::test_auto_refresh_skips_modal`

---

## Training Persona Filter Not Filtering (2026-04-25)
**Symptom:** Selected persona "牛哥" but all recordings shown including 小S
**Root Cause:** renderTree() only filtered by listenerId, not personaId
**Fix:** training_ui.js - Add persona filter check before listener filter
**Detection:** Recording list shows recordings from all personas despite filter
**Regression Test:** `tests/unit/test_regressions.py::TestUIBugs::test_persona_filter_in_training_ui`

---

## Training Fails "Only Support 24kHz Audio" (2026-04-27)
**Symptom:** Training fails immediately with error "Only support 24kHz audio"
**Root Cause:** Audio files are 48kHz but Qwen3-TTS requires 24kHz for speaker embedding extraction
**Fix:** training_job.py - Add scipy resampling to 24kHz in both:
  1. `speech_tokenizer.encode()` path (line ~185)
  2. `model.extract_speaker_embedding()` path (line ~264)
**Detection:** training_result.json shows success=false with "Only support 24kHz audio"
**Regression Test:** `tests/unit/test_regressions.py::TestTrainingBugs::test_audio_resampling_for_24k`

---

## LoRA Training Fails "device not defined" (2026-04-27)
**Symptom:** Training fails with "cannot access local variable 'device' where it is not associated with a value"
**Root Cause:** `device` used in fallback code block before it was defined (lines 325, 331 vs 345)
**Fix:** training_job.py - Move `device = next(model.parameters()).device` to BEFORE the if/else block at line 307
**Detection:** training_result.json shows success=false with device error
**Regression Test:** `tests/unit/test_regressions.py::TestTrainingBugs::test_lora_device_variable`

---

## SFT Preview Fails with KeyError: True (2026-04-27)
**Symptom:** Previewing SFT model produces no audio, server log shows "Chunk producer error: True"
**Root Cause:** training_job.py sets spk_is_dialect[speaker]=True during SFT merge, but inference code expects a dialect STRING (like "chinese"), not boolean True
**Fix:** training_job.py - Comment out the spk_is_dialect assignment (line 542-545), and fix existing model configs to use spk_is_dialect[speaker]=False
**Detection:** Preview API returns empty audio, log shows KeyError: True
**Regression Test:** `tests/unit/test_regressions.py::TestTrainingBugs::test_sft_preview_no_keyerror`

---

## Adding New Patterns

After fixing a bug:
1. Add pattern above with: Date, Symptom, Root Cause, Fix, Detection, Regression Test
2. Add regression test to `tests/unit/test_regressions.py`
3. Run: `pytest tests/unit/test_regressions.py -v --tb=short`

## Quick Test Commands

```bash
# Run regression tests only
pytest tests/unit/test_regressions.py -v --tb=short

# Run related tests (filter by keyword)
pytest tests/unit/ -k "training" --tb=short

# Show only failures
pytest tests/unit/ --tb=short 2>&1 | grep -E "FAIL|ERROR|passed|failed"
```
