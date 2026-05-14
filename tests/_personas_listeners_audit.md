# Personas + Listeners Audit — Phase 1.3, Task 26

Date: 2026-05-13
Scope: `app/api/personas.py`, `app/api/listeners.py`, `app/services/personas.py`, `app/services/listeners.py`

Coverage at baseline (before Phase 1.3):
- `app/api/personas.py`: 36.7%
- `app/api/listeners.py`: 42.4%
- `app/services/personas.py`: 38.7%
- `app/services/listeners.py`: 41.1%

These modules are **less broken than recordings/training were**, but the
gaps are the same shape: list-of-dicts state, no atomic write, no domain
models, no tests, inconsistent error handling vs Phase 1.1/1.2.

---

## Strengths (don't regress)

- **Pydantic input models exist** — `PersonaCreate`, `PersonaUpdate`,
  `ListenerCreate`, `ListenerUpdate`. Routes already use `extra` defaults
  (Pydantic accepts unknown by default — Phase 1.3 will switch to `forbid`).
- **fcntl locking exists** — both services use a `_with_lock` helper around
  the index file. Concurrent writes can't interleave at the byte level.
- **Fixed vs dynamic distinction** — personas are seeded with 4 fixed
  family members (`xiao_s`, `caregiver`, `elder_gentle`, `elder_playful`).
  Fixed personas can't be deleted; the API guards this.

## Defects

### 1. No atomic-rename writes

`app/services/personas.py:69-72`:
```python
def _save_personas_unlocked(personas: list[dict]) -> None:
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(personas, f, ensure_ascii=False, indent=2)
```

The flock prevents concurrent writers from interleaving, but if the
process is killed mid-write, the file ends up truncated/partial. The
recordings fix used `tempfile + fsync + os.replace`; Phase 1.3 will too.
Same defect in `listeners.py`.

### 2. Hardcoded relative paths

```python
DATA_DIR = Path("data/personas")
DATA_FILE = DATA_DIR / "personas.json"
```

`Path("data/personas")` is relative to CWD. Works if uvicorn is run from
the repo root; breaks otherwise. No env-var override. Phase 1.3 will
inject `data_root` via constructor like the other repos.

### 3. List-of-dicts state, no domain model

`list_personas() -> list[dict]` with no schema. Callers (e.g. the
PersonaValidator in `app/api/_dependencies.py`) reach into the dicts with
string keys. A typo in a key fails silently.

### 4. Inconsistent error handling vs Phase 1.1/1.2

```python
# personas.py routes:
if not get_persona(persona_id):
    raise HTTPException(404, f"Persona not found: {persona_id}")
```

vs the recordings/training pattern:
```python
raise PersonaNotFoundError(...)  # mapped to 404 by the DomainError handler
```

The legacy 404 messages are plain strings, not the
`{error, message, details}` envelope the rest of the API uses. Client
code that switches on `error` codes can't distinguish a persona-404 from
a recording-404.

### 5. Duplicated VALID_EMOTIONS

```python
# listeners.py:22
VALID_EMOTIONS = {"寵溺", "撒嬌", "幽默", "毒舌", "溫和", "開心", "認真", "默認"}
```

And `app/services/tts/emotion_mapper.py` has its own emotion list. The
two can drift. Phase 1.3 will keep the listener-side set as the source
of truth for the API but flag it for a future audit.

### 6. No tests

Zero integration tests for personas or listeners endpoints. The 4
superficial tests deleted in Phase 0 (`test_get_personas`,
`test_get_listeners`) only asserted `len(data) > 0`. Phase 1.3 adds real
contract tests covering CRUD + validation.

### 7. Mixed sync/init semantics

`_load_personas_unlocked()` does:
```python
if not DATA_FILE.exists():
    _save_personas_unlocked(FIXED_PERSONAS.copy())
    return FIXED_PERSONAS.copy()

# Else merge fixed personas that aren't in stored:
stored_ids = {p["persona_id"] for p in stored}
for fp in FIXED_PERSONAS:
    if fp["persona_id"] not in stored_ids:
        stored.append(fp)
```

This is a "self-healing seed" — if the file is missing a fixed persona,
it gets re-added on the next read. But it's a **read-with-write** pattern
(audit defect #4 from Phase 1.2). Phase 1.3 will explicitly seed on first
init and stop merging on every read.

### 8. Listeners has no fixed/dynamic distinction

Personas have `type ∈ {fixed, dynamic}`. Listeners have only
`is_family ∈ {true, false}` and there's no protection against deleting
a seeded listener like `child` (which other code paths assume exists).
This is a real bug that the existing UI papers over but the test will
flush.

---

## Out of scope

- **The dual persona system** — `app/resources/personas/*.json` is a
  separate code path used by `app/services/llm/prompt_manager.py` to
  load persona text. Phase 1.3 only touches the CRUD list at
  `data/personas/personas.json`. Unifying them is a Phase 2+ concern.
- **The persona-resource JSON schema validator** mentioned in the
  Phase 1.1 plan — same reason.

---

## Acceptance bar for Phase 1.3

| Gate | Target |
|---|---|
| `app/api/personas.py` coverage | ≥80% (from 36.7%) |
| `app/api/listeners.py` coverage | ≥80% (from 42.4%) |
| `app/services/personas/*` coverage | ≥80% |
| `app/services/listeners/*` coverage | ≥80% |
| Zero raw `dict` in route signatures | required (currently none) |
| Zero HTTPException raised in personas/listeners API code | required — use DomainError instead |
| Atomic-rename writes in both repos | required |
| Both CRUD endpoints have happy + ≥1 failure contract test | required |
| Concurrent create test (no lost writes) | required |
