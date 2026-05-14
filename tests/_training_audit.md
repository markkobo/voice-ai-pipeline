# Training Audit — Phase 1.2, Task 16

Date: 2026-05-13
Scope: `app/api/training.py` + `app/services/training.py` + `app/services/training_service/`

13 endpoints, 0 tests today. `app/api/training.py` coverage at baseline is 13.6%.

---

## Endpoint inventory

| # | Method + path | Body type | Returns | Defects |
|---|---|---|---|---|
| 1 | `GET /api/training/versions/{id}/progress` | — | SSE event stream | Tight-poll `while True` with no max-wait; no timeout event; line 96 swallows `Exception` |
| 2 | `GET /api/training/versions` | — | `{versions, count}` | Side effect: `list_versions` writes to disk via `update_version_status` if it sees a completed `progress.json` — read-with-write surprise |
| 3 | `GET /api/training/versions/{id}` | — | TrainingVersion dict | OK shape, but no Pydantic; nullable progress shape inconsistent |
| 4 | `GET /api/training/status` | — | `{is_training, version_id?, persona_id?}` | Implicit union shape (different keys when not training) — not Pydantic |
| 5 | `POST /api/training/versions` | TrainingRequest | dict | Validation gaps: `rank` not in {4,8,16,32}; `num_epochs` unbounded; `batch_size` unbounded; `training_type` not in {lora, sft}; `learning_rate` not bounded. Segment-id parse (l. 222-227) is fragile string split — no test |
| 6 | `PATCH /api/training/versions/{id}` | **query param** `nickname` | dict | Inconsistent: should be body. No Pydantic |
| 7 | `POST /api/training/versions/{id}/activate` | — | dict | Line 430 silently swallows TTS activation failure — caller sees 200 even though activation may have failed |
| 8 | `POST /api/training/voice-clone/activate` | **query params** persona_id, ref_audio_path | dict | Should be a body. Lines 466-468 catch+re-raise as HTTPException(500, str(e)) — leaks internal error text |
| 9 | `DELETE /api/training/versions/{id}` | — | dict | OK shape; refuses active version via 404 (should be 409) |
| 10 | `GET /api/training/active` | query `persona_id` | dict | Implicit union shape based on `active` boolean |
| 11 | `GET /api/training/versions/{id}/manifest` | — | manifest dict | No Pydantic |
| 12 | `POST /api/training/versions/{id}/cancel` | — | dict | Cancel reaches into `manager._save_index()` (private method) on line 519. Manipulates `version.status = "failed"` directly. Sets `"cancelled"` regardless of whether cancellation actually succeeded |
| 13 | `POST /api/training/versions/{id}/preview` | PreviewRequest | StreamingResponse (WAV) | Buffers all audio chunks before yielding — defeats streaming; `audio_stream()` blocks on TTS engine; no error path if engine errors mid-generation |

---

## Cross-cutting defects in `app/services/training.py`

### 1. JSON corruption race (CRITICAL — same as recordings had)

`VersionManager._save_index()` (l. 101-108) writes `VERSION_INDEX_FILE` with no locking and no atomic rename:

```python
with open(VERSION_INDEX_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
```

Concurrent writes (e.g., a training thread completing while a user creates a new version) race here. The recordings fix in Phase 1.1 used `fcntl.flock` + atomic rename — we'll do the same.

### 2. In-memory `self._versions` cache → stale reads

`VersionManager.__init__` loads `self._versions` once. Subsequent reads return cached state. If the file changes from another process (the training subprocess updates progress, the API reads), the cached list is stale until the next `_load_index()` call. Only `__init__` ever calls `_load_index()` — so the cache is **never refreshed**.

This explains why `list_versions()` re-reads `progress.json` and writes back via `update_version_status` — it's papering over a stale-cache bug.

### 3. Module-level singletons

- `_version_manager: Optional[VersionManager] = None` at line 318
- `_training_jobs: dict[str, TrainingJob] = {}` in `app/api/training.py:39`

Both leak state across tests. The recordings fix moved this to `app.state` + FastAPI `dependency_overrides`. Same pattern here.

### 4. Read-with-write surprise

`list_versions()` mutates the version index. Reading the version list is supposed to be idempotent. Move the "sync status from progress.json" logic to an explicit `refresh_status(version_id)` call so the read path is clean.

### 5. Silent exception swallows

| Location | What it swallows | Impact |
|---|---|---|
| `api/training.py:96-97` | progress.json parse error | SSE stream silently continues with stale data |
| `api/training.py:315-316` | TTS unload failure during SFT setup | VRAM not freed → next training OOMs |
| `api/training.py:324-325` | ASR unload failure during SFT setup | Same |
| `api/training.py:430-431` | LoRA activation on TTS engine failure | Caller gets 200 but TTS is still on the old model |
| `api/training.py:466-468` | Voice-clone activation failure | Leaks `str(e)` of internal error to API client |
| `training_service/training_job.py:955-956,961-962` | Lock release failure | Locks stay set forever, blocking future requests |

### 6. Segment-ID parsing fragility

`api/training.py:218-227` and `services/training.py:354-365` both do `rfind("_SPEAKER_")` to split a segment_id. This is duplicated, untested, and silently drops malformed IDs. Should be a single helper in the service with explicit failure on parse error.

### 7. `get_training_audio_for_persona` returns silent defaults

`services/training.py:388` — if segment metadata is missing a `duration_seconds`, falls back to `30.0` seconds. This silently lies about training data: a recording that's actually 5s gets reported as 30s and passes the 10s minimum check. Bug — should refuse training if duration is unknown.

### 8. Manifest path naming is brittle

`api/training.py:289` builds the manifest from `(audio_files, selected_recordings)` zipped together. But `audio_files` is built via `get_training_audio_for_persona` which can return fewer entries than `selected_recordings` (it silently skips missing audio). The zip then mis-aligns recording_id with audio_path.

### 9. Hardcoded base_model in two places

`TrainingVersion.base_model = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"` (l. 34) and `TrainingConfig.base_model = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"` (`lora_trainer.py:29`). Will drift. Should reference a single constant in a config module.

### 10. `cancel_training` reaches into private state

`api/training.py:518-519`:
```python
version.status = "failed"
manager._save_index()
```

Mutates a TrainingVersion directly and calls a `_private` method on the manager. The service should expose a clean `mark_cancelled(version_id)` API.

---

## Test gaps

- **Zero** integration tests for any of the 13 endpoints.
- The legacy unit tests in `tests/unit/test_training.py` cover only `VersionManager` CRUD (13 tests, real fs but heavy `with patch(...)` nesting). One of them fails at baseline because `base_model` default drifted.
- No tests for `get_training_audio_for_persona`, the segment-id parser, the manifest builder, the SSE generator, or the activate→TTS flow.
- The deleted `tests/unit/test_training_job.py` superficial script-generation tests are gone; what should replace them is a real `build_training_script(config) -> str` pure function with a snapshot test. That's Phase 1.2 work tracked in Task #24.

---

## Acceptance bar for Phase 1.2

(Locking in the same gates we hit for recordings + a few training-specific ones.)

| Gate | Target |
|---|---|
| `app/api/training.py` coverage | ≥80% (from 13.6%) |
| `app/services/training_service/*` coverage | ≥75% (currently mixed; lora_trainer/sft_trainer are subprocess-bound and won't reach this — service.py + repository.py are the targets) |
| Zero raw `dict` in route signatures | required |
| Zero `except Exception: pass` in training modules | required |
| SSE generator uses bounded poll (max_wait) | required |
| `cancel_training` doesn't touch private state | required |
| `get_training_audio_for_persona` refuses unknown duration | required |
| All 13 endpoints have happy + ≥1 failure-mode contract test | required |
| Repository concurrency test (parallel create + update) | required |
