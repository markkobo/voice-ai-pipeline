"""
Shared helpers for the WebSocket pipeline.

These extract the repeated patterns from `ws_asr.py` so the message loop
isn't littered with copies of the same drain-loop / TTS-task-await /
guarded-send code.

Three primitives:
- `drain_emotion_parser(parser)` — bounded async generator that yields
  every buffered content chunk left in an EmotionMapper / EmotionParser
  after a fresh `update()` call. Replaces three near-identical
  `while True: parser.update('')` blocks in ws_asr.
- `safe_send_text(ws, payload)` — guards a JSON send so client disconnects
  don't propagate as uncaught exceptions and don't kill the LLM stream.
- `await_prior_tts_task(task)` — awaits a previous TTS task before starting
  a new one, logging any error instead of silently swallowing.

All three have unit tests in `tests/unit/test_ws_helpers.py`.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Optional

log = logging.getLogger(__name__)


# Bounded drain limit. The parser's `update('')` is supposed to converge to
# None in O(1) once the buffer is empty — the property test in
# test_emotion_parser_property.py pins that. This cap is defense-in-depth
# in case a future change introduces a regression.
DRAIN_MAX_ITERATIONS = 256


async def drain_emotion_parser(parser) -> AsyncIterator[tuple[Optional[str], str]]:
    """
    Yield all buffered (emotion, content) tuples left in `parser`.

    Drives `parser.update('')` until it returns None or yields empty content.
    Bounded by `DRAIN_MAX_ITERATIONS` to guarantee termination even if the
    parser's invariant breaks.
    """
    for _ in range(DRAIN_MAX_ITERATIONS):
        result = parser.update("")
        if result is None:
            return
        emotion, content = result
        if not content:
            # (None, '') or (emotion, '') — parser signaling "nothing more".
            return
        yield emotion, content
    log.warning(
        "drain_emotion_parser hit %d-iteration cap — likely a parser bug",
        DRAIN_MAX_ITERATIONS,
    )


async def safe_send_text(ws, payload: dict) -> bool:
    """
    Send a JSON message, swallowing only client-disconnect errors.

    Returns True if the message was sent, False if the client has gone away.
    Other exceptions are logged and re-raised — they indicate a real bug.
    """
    try:
        await ws.send_text(json.dumps(payload, ensure_ascii=False))
        return True
    except RuntimeError as e:
        # Starlette raises RuntimeError("Cannot call 'send' once a close message
        # has been sent.") on attempted writes after close. That's a normal
        # disconnect, not a bug.
        msg = str(e).lower()
        if "close" in msg or "disconnect" in msg:
            log.debug("safe_send_text: client gone (%s)", e)
            return False
        raise


async def safe_send_bytes(ws, data: bytes) -> bool:
    """Bytes equivalent of safe_send_text."""
    try:
        await ws.send_bytes(data)
        return True
    except RuntimeError as e:
        msg = str(e).lower()
        if "close" in msg or "disconnect" in msg:
            log.debug("safe_send_bytes: client gone (%s)", e)
            return False
        raise


async def await_prior_tts_task(task: Optional[asyncio.Task], context: str) -> None:
    """
    Await `task` if present. Logs and continues on exception instead of
    silently swallowing — the legacy `except Exception: pass` hid real
    TTS errors from operators.

    `context` is a short string used in the log message so the operator
    knows where the error came from (e.g. "before-first-emotion" vs
    "post-emotion-token").
    """
    if task is None:
        return
    if task.done() and not task.cancelled():
        # Surface any exception the task already finished with.
        exc = task.exception()
        if exc is not None:
            log.exception("Prior TTS task (%s) failed: %s", context, exc, exc_info=exc)
        return
    try:
        await task
    except asyncio.CancelledError:
        # Cancellation is normal (barge-in path) — re-raise so the awaiter
        # also gets cancelled if appropriate.
        raise
    except Exception:
        log.exception("Prior TTS task (%s) raised", context)


async def send_tts_error_frame(
    ws,
    *,
    sentence_idx: int,
    error: str,
) -> None:
    """Tell the client a TTS sentence failed so its audio worklet can recover.

    The legacy code silently swallowed TTS errors — audio just stopped and
    the user saw a frozen UI. Sending this frame lets the client log,
    retry, or show a placeholder.
    """
    await safe_send_text(
        ws,
        {"type": "tts_error", "sentence_idx": sentence_idx, "error": error},
    )
