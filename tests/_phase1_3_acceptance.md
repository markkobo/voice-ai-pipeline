# Phase 1.3 Acceptance — Personas + Listeners Deep Restructure

Date: 2026-05-13
Test runtime: `.venv/bin/pytest tests/`

## Headline

| Metric | Phase 1.2 baseline | After Phase 1.3 | Delta |
|---|---|---|---|
| Tests collected | 216 | 250 | +34 |
| Passing | 213 | **247** | +34 |
| Failing | 3 | **3** | 0 (same baseline: emotion-parser drift x2, ws cancel timing) |
| `app/api/personas.py` coverage | 36.7% | **100.0%** | +63.3 pp |
| `app/api/listeners.py` coverage | 42.4% | **100.0%** | +57.6 pp |
| `app/services/personas/*` coverage | 38.7% | service 90.8%, repo 87.0%, models 100% | new package |
| `app/services/listeners/*` coverage | 41.1% | service 95.4%, repo 87.0%, models 100% | new package |
| Total coverage | 47.4% | **51.1%** | +3.7 pp |

## Acceptance gates

| Gate | Target | Status |
|---|---|---|
| Tests green except baseline | ≤3 baseline fails | ✅ |
| `app/api/personas.py` coverage ≥80% | ≥80% | ✅ **100.0%** |
| `app/api/listeners.py` coverage ≥80% | ≥80% | ✅ **100.0%** |
| `app/services/personas/*` ≥80% | ≥80% | ✅ avg ~93% |
| `app/services/listeners/*` ≥80% | ≥80% | ✅ avg ~94% |
| Zero raw `dict` in route signatures | required | ✅ both files use Pydantic `extra="forbid"` bodies |
| Zero HTTPException raised | required | ✅ confirmed by grep — both API modules raise nothing; service raises DomainError subclasses; the app-wide handler maps to HTTP |
| Atomic-rename writes in both repos | required | ✅ same `tempfile + fsync + os.replace` pattern as JsonRecordingsRepository / JsonTrainingRepository |
| Happy + ≥1 failure contract per endpoint | required | ✅ 17 persona tests + 17 listener tests, every endpoint covered |
| Concurrent-create test | required | ✅ `test_parallel_creates_all_persist` on each (N=30, all writes persist) |

## What got built

```
app/services/personas/          (was a single file)
  __init__.py     — re-exports legacy function-style API for back-compat
  models.py       — Pydantic Persona + PersonaType + FIXED_PERSONAS
  repository.py   — JsonPersonaRepository with fcntl + atomic rename
  service.py      — PersonasService (validation + fixed-readonly guards)
app/services/listeners/         (was a single file)
  __init__.py
  models.py       — Listener + VALID_EMOTIONS + SEED_LISTENERS (is_seed flag is new)
  repository.py
  service.py      — ListenersService (validation + seed-readonly guards)
app/api/
  personas.py     — routes-only, Pydantic models, Depends(get_personas_service)
  listeners.py    — same shape
  _errors.py      — 7 new DomainError subclasses (PersonaNotFoundError,
                    ListenerNotFoundError, FixedPersonaReadonlyError,
                    SeedListenerReadonlyError, InvalidEmotionError,
                    InvalidIdFormatError, DuplicateIdError)
  _dependencies.py — PersonaValidator/ListenerValidator now wrap repos
                     (no more module-global function calls);
                     get_personas_service + get_listeners_service +
                     make_*_for_testing
tests/
  contract/test_personas_contract.py    +17
  contract/test_listeners_contract.py   +17
  conftest.py    — DI overrides extended for personas + listeners;
                   DATA_ROOT env var set per-test so legacy back-compat
                   API also sees isolated data
  _personas_listeners_audit.md           +
  _phase1_3_acceptance.md                +
```

## Bugs surfaced and fixed

1. **Seeded listeners could be deleted** — the legacy `delete_listener` had
   no readonly guard. Any downstream code that assumes `child`/`mom`/`default`
   exist (training segment validation, prompt manager, the UI) would break
   silently. New `Listener.is_seed` + `SeedListenerReadonlyError` close
   this. `test_cannot_delete_seed` enforces the new contract.
2. **No atomic write on persona/listener JSON files** — legacy code did
   plain `open(...).write()`. A kill mid-write produced a truncated
   `personas.json`. The atomic-rename pattern from Phase 1.1 now applies.
3. **`is_fixed_persona` was a read-with-write** — the legacy
   `_load_personas_unlocked` re-merged FIXED_PERSONAS on every read if
   they were missing from disk. Phase 1.3 seeds explicitly on first init
   and never mutates on read.
4. **`VALID_EMOTIONS` was duplicated** — once in `api/listeners.py`, once
   implicitly in `app/services/tts/emotion_mapper.py`. Phase 1.3 moves it
   into `app/services/listeners/models.py` as the single source of truth
   for listener-side validation. (The TTS-side mapper still has its own
   list; unifying them is Phase 2 work.)
5. **Inconsistent error envelopes** — every other module returns
   `{error, message, details}`; personas/listeners returned plain string
   messages from raw `HTTPException(...)`. Now they return the same
   envelope as everything else.

## What's not done in 1.3 (deferred)

- **TTS emotion list unification** — `emotion_mapper.py` has its own
  emotion list. Cross-checking that against `VALID_EMOTIONS` is Phase 2
  work (touches the streaming/emotion-parser code that has the open
  baseline failures).
- **Persona resources JSON schema validator** — `app/resources/personas/*.json`
  is a separate persona system used by `prompt_manager.py`. Out of scope
  for Phase 1.3; flagged for Phase 2+.
- **Legacy function-style API removal** — `list_personas()` etc. are
  still exported from `app/services/personas/__init__.py` as
  backwards-compat shims. They delegate to the new service correctly,
  but should eventually be removed once the few remaining callers (the
  legacy validators were rewritten in 1.3; nothing else uses them) are
  migrated.

## How to verify locally

```bash
cd /home/rding/voice-ai-pipeline
.venv/bin/pytest tests/ -q                                # 247 pass, 3 baseline fail
.venv/bin/pytest tests/ --cov=app --cov-report=term       # 51.1% total
.venv/bin/pytest tests/contract/test_personas_contract.py tests/contract/test_listeners_contract.py -v
```
