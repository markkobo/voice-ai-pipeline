"""Unit tests for app/api/_ws_helpers.py."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.api._ws_helpers import (
    DRAIN_MAX_ITERATIONS,
    await_prior_tts_task,
    drain_emotion_parser,
    safe_send_bytes,
    safe_send_text,
    send_tts_error_frame,
)


# ---------------------------------------------------------------------------
# drain_emotion_parser
# ---------------------------------------------------------------------------
class FakeParser:
    """Yields a fixed list of results then None forever."""

    def __init__(self, results: list):
        self.results = list(results)

    def update(self, _):
        if self.results:
            return self.results.pop(0)
        return None


@pytest.mark.asyncio
async def test_drain_yields_each_buffered_item():
    parser = FakeParser([(None, "a"), (None, "b"), (None, "c"), None])
    out: list[tuple] = []
    async for item in drain_emotion_parser(parser):
        out.append(item)
    assert out == [(None, "a"), (None, "b"), (None, "c")]


@pytest.mark.asyncio
async def test_drain_stops_on_empty_content():
    # The legacy code's exit condition was `(None, '')`. Helper honors it.
    parser = FakeParser([(None, "a"), (None, ""), (None, "should-not-emit")])
    out = [x async for x in drain_emotion_parser(parser)]
    assert out == [(None, "a")]


@pytest.mark.asyncio
async def test_drain_stops_on_none():
    parser = FakeParser([(None, "a"), None, (None, "should-not-emit")])
    out = [x async for x in drain_emotion_parser(parser)]
    assert out == [(None, "a")]


@pytest.mark.asyncio
async def test_drain_bounded_to_max_iterations():
    """A buggy parser that never converges must NOT infinite-loop."""

    class NeverConvergingParser:
        def update(self, _):
            return (None, "x")  # always returns content

    parser = NeverConvergingParser()
    count = 0
    async for _ in drain_emotion_parser(parser):
        count += 1
    assert count == DRAIN_MAX_ITERATIONS


# ---------------------------------------------------------------------------
# safe_send_text / safe_send_bytes
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_safe_send_text_happy_path():
    ws = MagicMock()
    ws.send_text = AsyncMock()
    ok = await safe_send_text(ws, {"hello": "world"})
    assert ok is True
    ws.send_text.assert_awaited_once()
    sent = ws.send_text.await_args.args[0]
    assert json.loads(sent) == {"hello": "world"}


@pytest.mark.asyncio
async def test_safe_send_text_swallows_disconnect():
    ws = MagicMock()
    ws.send_text = AsyncMock(side_effect=RuntimeError("WebSocket is not connected. Need to call \"accept\" first."))
    # "connected" doesn't match our close/disconnect filter, so this should raise.
    with pytest.raises(RuntimeError):
        await safe_send_text(ws, {"x": 1})

    ws.send_text = AsyncMock(side_effect=RuntimeError("Cannot call 'send' once a close message has been sent."))
    ok = await safe_send_text(ws, {"x": 1})
    assert ok is False


@pytest.mark.asyncio
async def test_safe_send_bytes_happy_path_and_disconnect():
    ws = MagicMock()
    ws.send_bytes = AsyncMock()
    assert await safe_send_bytes(ws, b"\x00\x01") is True

    ws.send_bytes = AsyncMock(side_effect=RuntimeError("Unexpected disconnect"))
    assert await safe_send_bytes(ws, b"\x00\x01") is False


# ---------------------------------------------------------------------------
# await_prior_tts_task
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_await_prior_tts_task_none_is_noop():
    await await_prior_tts_task(None, "test-context")  # no exception


@pytest.mark.asyncio
async def test_await_prior_tts_task_awaits_running():
    async def runner():
        await asyncio.sleep(0)
        return "done"

    t = asyncio.create_task(runner())
    await await_prior_tts_task(t, "ctx")
    assert t.done()


@pytest.mark.asyncio
async def test_await_prior_tts_task_logs_exception(caplog):
    async def boom():
        raise RuntimeError("kaboom")

    t = asyncio.create_task(boom())
    await asyncio.sleep(0.01)  # let it run + fail
    with caplog.at_level("ERROR"):
        await await_prior_tts_task(t, "boom-ctx")
    # No exception escapes; log records the failure.
    assert any("boom-ctx" in r.message or "kaboom" in str(r) for r in caplog.records)


@pytest.mark.asyncio
async def test_await_prior_tts_task_propagates_cancellation():
    async def runner():
        await asyncio.sleep(10)

    t = asyncio.create_task(runner())
    await asyncio.sleep(0)
    t.cancel()
    # When awaiter is cancelled while waiting, propagate.
    with pytest.raises(asyncio.CancelledError):
        await await_prior_tts_task(t, "ctx")


# ---------------------------------------------------------------------------
# send_tts_error_frame
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_send_tts_error_frame_shape():
    ws = MagicMock()
    ws.send_text = AsyncMock()
    await send_tts_error_frame(ws, sentence_idx=2, error="OOM")
    sent = json.loads(ws.send_text.await_args.args[0])
    assert sent == {"type": "tts_error", "sentence_idx": 2, "error": "OOM"}
