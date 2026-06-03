"""
Locks the empty-ASR signal contract shipped during 06/02 demo prep.

Before the fix, ws_asr.py would `continue` past the asr_result send when
the ASR returned empty text. The client's FSM (which entered THINKING
on vad_commit) had no signal to transition out of THINKING — mic stayed
closed, conversation appeared frozen.

After the fix (commit 3703f20):
1. state_manager.commit_utterance ALWAYS returns a well-formed
   asr_result dict (even when audio buffer is empty or ASR returns
   empty text).
2. ws_asr.py ALWAYS sends that dict to the client, then skips the LLM
   when text is empty.
3. standalone.js asr_result handler treats empty text + listen-only as
   the same "transition out of THINKING" case.

These tests cover layer (1) at the state_manager level. Layers (2) and
(3) are integration-tested via the full WS flow in test_ws_asr.py.
"""
import pytest

from app.core.state_manager import StateManager
from app.services.asr.engine import MockASR


class _EmptyASR:
    """Mock ASR that always returns empty text. Simulates the production
    case where the real Qwen3-ASR hallucination filter zeros the text."""

    latency_ms = 0
    model_name = "empty-mock"

    async def recognize(self, audio_bytes: bytes):
        return {"text": "", "asr_inference_ms": 0}


@pytest.fixture
def manager_with_empty_asr():
    """StateManager with a session pre-configured to use _EmptyASR."""
    manager = StateManager()
    sid = "empty-asr-session"
    manager.create_session(sid)
    # Replace the auto-created ASR with our empty mock
    state = manager.get_session(sid)
    state.asr = _EmptyASR()
    # Mark configured so commit_utterance doesn't bail on the is_configured check
    state.is_configured = True
    return manager, sid


@pytest.fixture
def manager_with_mock_asr():
    """StateManager with a session pre-configured to use MockASR (returns
    a fixed non-empty string)."""
    manager = StateManager()
    sid = "mock-asr-session"
    manager.create_session(sid)
    state = manager.get_session(sid)
    state.asr = MockASR(latency_ms=0)
    state.is_configured = True
    return manager, sid


@pytest.mark.asyncio
async def test_commit_with_empty_buffer_returns_asr_result(manager_with_empty_asr):
    """Empty audio buffer must still return an asr_result envelope.
    Without this, the client gets nothing → stuck in THINKING."""
    manager, sid = manager_with_empty_asr
    # Buffer is empty by default — don't push any audio
    result = await manager.commit_utterance(sid)

    assert result["type"] == "asr_result"
    assert result["is_final"] is True
    assert result["text"] == ""
    assert "utterance_id" in result
    assert "telemetry" in result


@pytest.mark.asyncio
async def test_commit_with_audio_but_empty_asr_returns_asr_result(manager_with_empty_asr):
    """ASR returns empty text on non-empty buffer (hallucination filter
    fired). Server still must send asr_result so client unsticks."""
    manager, sid = manager_with_empty_asr
    state = manager.get_session(sid)
    state.audio_buffer.extend(b"\x00\x01" * 5000)  # nonzero buffer

    result = await manager.commit_utterance(sid)

    assert result["type"] == "asr_result"
    assert result["text"] == ""
    assert result["is_final"] is True


@pytest.mark.asyncio
async def test_commit_with_real_text_returns_text(manager_with_mock_asr):
    """Sanity check: when ASR returns text, that text is in the
    asr_result. We didn't accidentally clear the success path."""
    manager, sid = manager_with_mock_asr
    state = manager.get_session(sid)
    state.audio_buffer.extend(b"\x00\x01" * 5000)

    result = await manager.commit_utterance(sid)

    assert result["type"] == "asr_result"
    assert result["text"] == "模擬語音辨識結果..."
    assert result["is_final"] is True


@pytest.mark.asyncio
async def test_commit_resets_utterance_state(manager_with_empty_asr):
    """After commit, the audio buffer is cleared and utterance_id rolls
    over so the NEXT utterance gets a fresh id. Without this, repeated
    empty commits would re-process the same buffer or collide ids."""
    manager, sid = manager_with_empty_asr
    state = manager.get_session(sid)
    state.audio_buffer.extend(b"\x00\x01" * 5000)
    first_utterance_id = state.utterance_id

    await manager.commit_utterance(sid)

    # Buffer cleared
    assert len(state.audio_buffer) == 0
    # Utterance id rolled
    assert state.utterance_id != first_utterance_id


@pytest.mark.asyncio
async def test_commit_envelope_has_telemetry_even_when_empty(manager_with_empty_asr):
    """Telemetry fields present on the empty-text path. Downstream
    metrics collection assumes the schema is stable."""
    manager, sid = manager_with_empty_asr
    result = await manager.commit_utterance(sid)

    assert "telemetry" in result
    tel = result["telemetry"]
    assert "vad_latency_ms" in tel
    assert "asr_inference_ms" in tel
