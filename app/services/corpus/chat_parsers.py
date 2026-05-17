"""
Chat-export parsers — WhatsApp / Line / WeChat → canonical text stream.

Per RFC_M6 Phase 0 slice 2B. Each parser reads a platform-specific
export, emits a list of ChatMessage records, and the
`messages_to_text()` helper formats them as a canonical multi-line
string that the existing chunker (chunker.py) consumes downstream.

Per-platform notes:

- **WhatsApp** — two .txt formats in the wild:
    "DD/MM/YY, HH:MM - Sender: Msg" (Android-export style)
    "[DD/MM/YY, HH:MM:SS] Sender: Msg" (iOS-export style)
  Locale variants (MM/DD/YY, comma vs period, 12h vs 24h) handled
  best-effort — we extract sender + message, store raw timestamp string.

- **Line** — Traditional Chinese export header `[LINE] 聊天記錄 ...`,
  followed by date sections `YYYY/MM/DD（週X）\n` and rows
  `HH:MM\\tSender\\tMessage`. We carry the date section forward.

- **WeChat** — no native export; users go through third-party tools
  that emit CSV. Common schema columns: `StrTime`, `IsSender`,
  `Message`, `Type` (or Chinese equivalents 時間 / 發送者 / 內容).
  We detect headers and map flexibly. WeChat HTML deferred.

Format detection (`detect_chat_format`) is best-effort sniff over the
first ~2 KB. Falls back to None when nothing matches — the caller
should treat that as "not a chat export, ingest as plain text".

This module is pure: no IO, no filesystem, no dependencies on the
service layer. Easy to unit-test.
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class ChatMessage:
    """One parsed chat message — platform-agnostic shape."""
    timestamp: str      # raw timestamp text, normalization deferred
    sender: str         # display name as found in export
    text: str           # message body (sticker/media references kept inline)
    platform: str       # "whatsapp" | "line" | "wechat"


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------
_LINE_HEADER_RE = re.compile(r"^\s*\[LINE\]", re.IGNORECASE)
_WHATSAPP_LINE_RE = re.compile(
    r"^\[?\d{1,4}[./-]\d{1,2}[./-]\d{1,4},?\s+\d{1,2}:\d{2}"
)
_WECHAT_CSV_HINT_RE = re.compile(
    r"(StrTime|IsSender|時間.*發送者|時間.*訊息|wechat)",
    re.IGNORECASE,
)


def detect_chat_format(text: str) -> Optional[str]:
    """Sniff the first ~2 KB for a chat-export signature.

    Returns one of "whatsapp" | "line" | "wechat" | None.
    """
    head = text[:2048]
    if _LINE_HEADER_RE.search(head):
        return "line"
    if _WECHAT_CSV_HINT_RE.search(head.splitlines()[0] if head else ""):
        return "wechat"
    # WhatsApp check: any of the first 20 non-empty lines starts with a
    # timestamp-prefixed message?
    nonblank = [ln for ln in head.splitlines() if ln.strip()][:20]
    if any(_WHATSAPP_LINE_RE.match(ln) for ln in nonblank):
        return "whatsapp"
    return None


# ---------------------------------------------------------------------------
# WhatsApp
# ---------------------------------------------------------------------------
# Two leading-line patterns. Both have an optional `[ ]` wrap and an
# `AM/PM` suffix tolerated. Sender is captured up to the first `:` that
# isn't part of the timestamp.
_WHATSAPP_MSG_RE = re.compile(
    r"""
    ^\[?(?P<ts>\d{1,4}[./-]\d{1,2}[./-]\d{1,4}        # date
        [,\s]+\d{1,2}:\d{2}(?::\d{2})?                  # time
        (?:\s?[AaPp][Mm])?)\]?                          # AM/PM
    \s*[-—]?\s*                                          # separator (- or em-dash)
    (?P<sender>[^:]{1,80}?):\ ?                          # sender up to :
    (?P<msg>.*)                                          # message body
    """,
    re.VERBOSE,
)


def parse_whatsapp(text: str) -> list[ChatMessage]:
    """Parse a WhatsApp .txt export.

    Multi-line messages: subsequent lines that don't match the
    timestamp-prefix get appended to the previous message's text.
    """
    out: list[ChatMessage] = []
    current: Optional[ChatMessage] = None

    for raw_line in text.splitlines():
        m = _WHATSAPP_MSG_RE.match(raw_line)
        if m:
            if current:
                out.append(current)
            current = ChatMessage(
                timestamp=m.group("ts").strip(),
                sender=m.group("sender").strip(),
                text=m.group("msg").rstrip(),
                platform="whatsapp",
            )
        elif current is not None and raw_line.strip():
            # Continuation line of the prior message.
            current = ChatMessage(
                timestamp=current.timestamp,
                sender=current.sender,
                text=current.text + "\n" + raw_line.rstrip(),
                platform="whatsapp",
            )
        # else: stray blank line — ignore.

    if current:
        out.append(current)
    return out


# ---------------------------------------------------------------------------
# Line (Traditional Chinese export style)
# ---------------------------------------------------------------------------
# Date section header: "2024/03/05（二）" or "2024/03/05(Tue)"
_LINE_DATE_RE = re.compile(
    r"^(?P<date>\d{4}/\d{1,2}/\d{1,2})[\s（(].*?[)）]?\s*$"
)
# Message row: tab-separated, "HH:MM\tSender\tMessage"
_LINE_ROW_RE = re.compile(
    r"^(?P<time>\d{1,2}:\d{2})\t(?P<sender>[^\t]+?)\t(?P<msg>.*)$"
)


def parse_line(text: str) -> list[ChatMessage]:
    """Parse a Line .txt export.

    Carries the current date forward across rows. Continuation lines
    (no leading timestamp) attach to the previous message.
    """
    out: list[ChatMessage] = []
    current_date: Optional[str] = None
    current_msg: Optional[ChatMessage] = None

    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue

        date_m = _LINE_DATE_RE.match(raw_line)
        if date_m:
            if current_msg:
                out.append(current_msg)
                current_msg = None
            current_date = date_m.group("date")
            continue

        row_m = _LINE_ROW_RE.match(raw_line)
        if row_m:
            if current_msg:
                out.append(current_msg)
            ts = f"{current_date} {row_m.group('time')}" if current_date else row_m.group("time")
            current_msg = ChatMessage(
                timestamp=ts,
                sender=row_m.group("sender").strip(),
                text=row_m.group("msg").rstrip(),
                platform="line",
            )
        elif current_msg is not None:
            # Continuation — only when we're inside a message.
            current_msg = ChatMessage(
                timestamp=current_msg.timestamp,
                sender=current_msg.sender,
                text=current_msg.text + "\n" + raw_line.rstrip(),
                platform="line",
            )
        # else: header lines like "[LINE] ..." or "儲存日期：..." get ignored.

    if current_msg:
        out.append(current_msg)
    return out


# ---------------------------------------------------------------------------
# WeChat (CSV — third-party export tool output)
# ---------------------------------------------------------------------------
# Column name candidates for each field. Lowercased and matched
# case-insensitively. First match wins.
_WECHAT_TIME_KEYS = ("strtime", "time", "timestamp", "時間", "日期時間")
_WECHAT_SENDER_KEYS = (
    "sender", "from", "username", "nickname",
    "發送者", "暱稱", "用戶", "talker",
)
_WECHAT_BODY_KEYS = (
    "message", "msg", "content", "text", "body",
    "內容", "訊息", "消息",
)
# Some tools emit "IsSender" (0/1) without a name column — fall back to
# "me" / "them" labels then.
_WECHAT_ISSELF_KEYS = ("issender", "isself", "is_me", "issend", "issendmsg")


def parse_wechat_csv(text: str) -> list[ChatMessage]:
    """Parse a WeChat CSV export emitted by common third-party tools.

    Flexible header mapping — we look at the first row of the CSV and
    map columns to (timestamp, sender, body) by name fuzz.
    """
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        return []

    header = [h.strip().lower() for h in rows[0]]

    def _find(keys: Iterable[str], exclude_idx: set[int] = frozenset()) -> Optional[int]:
        for i, name in enumerate(header):
            if i in exclude_idx:
                continue
            for key in keys:
                if key in name:
                    return i
        return None

    # Find issender FIRST so it's excluded from sender-column lookup
    # (otherwise "IsSender" greedy-matches the "sender" substring).
    isself_col = _find(_WECHAT_ISSELF_KEYS)
    excluded = {isself_col} if isself_col is not None else set()

    ts_col = _find(_WECHAT_TIME_KEYS)
    sender_col = _find(_WECHAT_SENDER_KEYS, exclude_idx=excluded)
    body_col = _find(_WECHAT_BODY_KEYS)

    # Body column is mandatory — without it we have nothing.
    if body_col is None:
        return []

    out: list[ChatMessage] = []
    for row in rows[1:]:
        if not row or len(row) <= body_col:
            continue
        msg = row[body_col].strip() if body_col < len(row) else ""
        if not msg:
            continue

        ts = row[ts_col].strip() if ts_col is not None and ts_col < len(row) else ""

        if sender_col is not None and sender_col < len(row) and row[sender_col].strip():
            sender = row[sender_col].strip()
        elif isself_col is not None and isself_col < len(row):
            sender = "me" if row[isself_col].strip() in ("1", "true", "True", "Y") else "them"
        else:
            sender = "?"

        out.append(ChatMessage(
            timestamp=ts,
            sender=sender,
            text=msg,
            platform="wechat",
        ))

    return out


# ---------------------------------------------------------------------------
# Canonical formatting
# ---------------------------------------------------------------------------
def messages_to_text(messages: list[ChatMessage]) -> str:
    """Render parsed messages as canonical multi-line text.

    Format: `[<timestamp>] <sender>: <msg>` per line, one blank line
    between messages with different senders (visual paragraph break →
    paragraph-aware chunker will prefer breaks there).
    """
    lines: list[str] = []
    prev_sender: Optional[str] = None
    for m in messages:
        if prev_sender is not None and m.sender != prev_sender and lines:
            lines.append("")  # blank line → paragraph boundary for chunker
        prefix = f"[{m.timestamp}] " if m.timestamp else ""
        lines.append(f"{prefix}{m.sender}: {m.text}")
        prev_sender = m.sender
    return "\n".join(lines)
