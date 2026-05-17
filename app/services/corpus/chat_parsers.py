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
# WhatsApp sniff uses the MANDATORY separator (same as the parser, review
# #1) so prose containing date-shapes doesn't trip it.
_WHATSAPP_SNIFF_RE = re.compile(
    r"^(\[?)\d{1,4}[./-]\d{1,2}[./-]\d{1,4},?\s+\d{1,2}:\d{2}.{0,15}"
    r"(\1\s+-\s+|\]\s+)"  # close-bracket + space, OR " - "
    r"\S"
)
# WeChat CSV sniff: must look like a CSV header — at least one comma AND
# at least one column-name token. Loose tokens like "wechat" alone are NOT
# enough (review #7 — "Notes on WeChat backup" was misclassified).
_WECHAT_CSV_HINT_RE = re.compile(
    r"\b(strtime|issender|talker|時間|發送者|nickname|message|msg|content|內容|訊息)\b",
    re.IGNORECASE,
)


def detect_chat_format(text: str) -> Optional[str]:
    """Sniff the first ~2 KB for a chat-export signature.

    Returns one of "whatsapp" | "line" | "wechat" | None.

    The WhatsApp sniff requires the mandatory " - " or "]" separator
    after the timestamp (review #1). The WeChat sniff requires both
    "looks like a CSV header row" AND a known chat column-name token
    (review #7) — a stray mention of "wechat" in prose is not enough.
    """
    head = text[:2048]
    if _LINE_HEADER_RE.search(head):
        return "line"

    # WeChat CSV sniff: first non-blank line must be a CSV-shaped header
    # (commas present + chat column-name token + no double quotes around
    # an entire sentence that would suggest prose).
    first_line = next((ln for ln in head.splitlines() if ln.strip()), "")
    if (
        "," in first_line
        and _WECHAT_CSV_HINT_RE.search(first_line)
        # Reject lines that look like prose: contain ". " or end with .!?
        # which a CSV header doesn't.
        and not re.search(r"[.!?]\s+\S|[.!?]$", first_line)
    ):
        return "wechat"

    # WhatsApp check: any of the first 20 non-empty lines is a real
    # header (timestamp + mandatory separator).
    nonblank = [ln for ln in head.splitlines() if ln.strip()][:20]
    if any(_WHATSAPP_SNIFF_RE.match(ln) for ln in nonblank):
        return "whatsapp"
    return None


# ---------------------------------------------------------------------------
# WhatsApp
# ---------------------------------------------------------------------------
# Two leading-line patterns with the MANDATORY separator that distinguishes
# a real header from a body line that just happens to start with a date:
#   Android: "DD/MM/YY, HH:MM - Sender: Msg"  → requires " - " after timestamp
#   iOS:     "[DD/MM/YY, HH:MM:SS] Sender: Msg"  → requires "]" wrap
# Earlier version used `\s*[-—]?\s*` (optional dash) which let lines like
# "12/03/24, 14:32 was when we met" misparse as a new header. Review #1.
_WHATSAPP_ANDROID_RE = re.compile(
    r"""
    ^(?P<ts>\d{1,4}[./-]\d{1,2}[./-]\d{1,4}             # date
        [,\s]+\d{1,2}:\d{2}(?::\d{2})?                  # time
        (?:\s?[AaPp][Mm])?)                             # AM/PM
    \s+[-—]\s+                                          # MANDATORY " - " separator
    (?P<sender>[^:]{1,80}?):\ ?                         # sender up to :
    (?P<msg>.*)                                         # message body
    """,
    re.VERBOSE,
)
_WHATSAPP_IOS_RE = re.compile(
    r"""
    ^\[(?P<ts>\d{1,4}[./-]\d{1,2}[./-]\d{1,4}           # date
        [,\s]+\d{1,2}:\d{2}(?::\d{2})?                  # time
        (?:\s?[AaPp][Mm])?)\]                           # MANDATORY ] wrap
    \s+
    (?P<sender>[^:]{1,80}?):\ ?
    (?P<msg>.*)
    """,
    re.VERBOSE,
)

# System lines have a header (timestamp + separator) but NO sender:msg
# split — they're a single sentence like "Messages and calls are end-to-end
# encrypted." or "<name> added <name>". Match the header part only so we
# can recognize them and not slot them into the previous speaker's message
# (review #6).
_WHATSAPP_SYSTEM_HEADER_RE = re.compile(
    r"""
    ^\[?\d{1,4}[./-]\d{1,2}[./-]\d{1,4}
    [,\s]+\d{1,2}:\d{2}(?::\d{2})?
    (?:\s?[AaPp][Mm])?\]?
    \s+([-—]\s+)?
    """,
    re.VERBOSE,
)


def parse_whatsapp(text: str) -> list[ChatMessage]:
    """Parse a WhatsApp .txt export.

    Header detection requires the mandatory separator (" - " for Android,
    "]" wrap for iOS) — review #1 fix. Lines that have the timestamp
    pattern but no sender (system notices like "Messages and calls are
    end-to-end encrypted.") are recognized and dropped, not appended to
    the previous speaker's message — review #6 fix.

    Multi-line continuations append O(N) without rebuilding the
    ChatMessage tuple every time (review #11).
    """
    out: list[ChatMessage] = []
    # current_lines accumulates continuation lines; flushed into a
    # ChatMessage when we hit the next header or end of input.
    current_ts: Optional[str] = None
    current_sender: Optional[str] = None
    current_lines: list[str] = []

    def _flush() -> None:
        if current_ts is not None and current_sender is not None:
            out.append(ChatMessage(
                timestamp=current_ts,
                sender=current_sender,
                text="\n".join(current_lines).rstrip(),
                platform="whatsapp",
            ))

    for raw_line in text.splitlines():
        m = _WHATSAPP_ANDROID_RE.match(raw_line) or _WHATSAPP_IOS_RE.match(raw_line)
        if m:
            _flush()
            current_ts = m.group("ts").strip()
            current_sender = m.group("sender").strip()
            current_lines = [m.group("msg").rstrip()]
            continue

        # Recognize system-line headers (timestamp + separator but no
        # `sender:` part) — drop them, don't fold into previous message.
        if _WHATSAPP_SYSTEM_HEADER_RE.match(raw_line):
            _flush()
            current_ts = None
            current_sender = None
            current_lines = []
            continue

        # Continuation of the current message — must currently have one.
        if current_sender is not None and raw_line.strip():
            current_lines.append(raw_line.rstrip())
        # else: stray blank line before any header, or after a system line —
        # ignore.

    _flush()
    return out


# ---------------------------------------------------------------------------
# Line (Traditional Chinese export style)
# ---------------------------------------------------------------------------
# Date section header: "2024/03/05（二）" or "2024/03/05(Tue)" or plain "2024/03/05".
# Closing paren is required when an opening one is present (review #13).
_LINE_DATE_RE = re.compile(
    r"^(?P<date>\d{4}/\d{1,2}/\d{1,2})(?:\s*(?:\(.+?\)|（.+?）))?\s*$"
)
# Message row separator: real Line exports use \t between fields, but
# zip-roundtripping and email forwarding regularly convert tabs to runs of
# spaces (review #2). Accept either: tab OR two-or-more spaces.
_LINE_ROW_RE = re.compile(
    r"^(?P<time>\d{1,2}:\d{2})[\t]+(?P<sender>[^\t]+?)[\t]+(?P<msg>.*)$"
)
_LINE_ROW_SPACES_RE = re.compile(
    r"^(?P<time>\d{1,2}:\d{2})\s{2,}(?P<sender>\S(?:[^\s].*?\S)?)\s{2,}(?P<msg>.*)$"
)


def parse_line(text: str) -> list[ChatMessage]:
    """Parse a Line .txt export.

    Accepts either tabs OR runs of 2+ spaces between fields (review #2 —
    real-world exports lose tabs to email/zip round-trips).

    Carries the current date forward across rows. Continuation lines
    append in O(N) via a `current_lines` list, not by rebuilding the
    frozen ChatMessage every time (review #11).
    """
    out: list[ChatMessage] = []
    current_date: Optional[str] = None
    current_ts: Optional[str] = None
    current_sender: Optional[str] = None
    current_lines: list[str] = []

    def _flush() -> None:
        if current_ts is not None and current_sender is not None:
            out.append(ChatMessage(
                timestamp=current_ts,
                sender=current_sender,
                text="\n".join(current_lines).rstrip(),
                platform="line",
            ))

    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue

        date_m = _LINE_DATE_RE.match(raw_line)
        if date_m:
            _flush()
            current_ts = None
            current_sender = None
            current_lines = []
            current_date = date_m.group("date")
            continue

        row_m = _LINE_ROW_RE.match(raw_line) or _LINE_ROW_SPACES_RE.match(raw_line)
        if row_m:
            _flush()
            current_ts = (
                f"{current_date} {row_m.group('time')}"
                if current_date else row_m.group("time")
            )
            current_sender = row_m.group("sender").strip()
            current_lines = [row_m.group("msg").rstrip()]
        elif current_sender is not None:
            # Continuation — only when we're inside a message.
            current_lines.append(raw_line.rstrip())
        # else: header lines like "[LINE] ..." or "儲存日期：..." get ignored.

    _flush()
    return out


# ---------------------------------------------------------------------------
# WeChat (CSV — third-party export tool output)
# ---------------------------------------------------------------------------
# Column name candidates for each field. Listed in priority order; exact
# match wins over substring match (review #4 — "IsSender" greedy-matched
# "sender", "lifetime" matched "time", "senderid" matched "sender").
_WECHAT_TIME_KEYS = (
    "strtime", "timestamp", "time", "datetime", "date",
    "時間", "日期時間", "日期",
)
_WECHAT_SENDER_KEYS = (
    "sender", "from", "username", "nickname", "talker",
    "發送者", "發送人", "暱稱", "用戶", "用户名", "名字",
)
_WECHAT_BODY_KEYS = (
    "message", "msg", "content", "text", "body",
    "內容", "訊息", "消息",
)
# Some tools emit "IsSender" (0/1) without a name column — fall back to
# "me" / "them" labels then. Match exact-only to avoid false positives.
_WECHAT_ISSELF_KEYS = (
    "issender", "isself", "is_me", "is_self", "issend", "issendmsg",
)

# Tokens that, if present in a column name, mean it is NOT that field
# (review #4). Used to skip false-positive substring matches.
_NEGATIVE_TOKENS_FOR_SENDER = ("id", "name", "type", "len")  # senderid, sendername, sendertype, senderlen
_NEGATIVE_TOKENS_FOR_TIME = ("lifetime", "meantime", "runtime", "downtime", "uptime")
_NEGATIVE_TOKENS_FOR_BODY = ("len", "id", "type", "seq", "size", "area")

# Truthy values for IsSender — review #8 (was missing Chinese "是",
# lowercase yes/y, etc.).
_WECHAT_TRUE_VALUES = frozenset({
    "1", "true", "yes", "y", "t", "是", "true.", "self",
})


def _find_column(
    header: list[str],
    keys: Iterable[str],
    negative_tokens: Iterable[str] = (),
    exclude_idx: set[int] = frozenset(),
) -> Optional[int]:
    """Find best matching column.

    Strategy (review #4):
      1. Exact match (e.g. column literally named "time")
      2. Substring match, BUT skip names containing any negative_tokens
    """
    neg = tuple(negative_tokens)
    # Pass 1: exact match.
    for i, name in enumerate(header):
        if i in exclude_idx:
            continue
        if name in keys:
            return i
    # Pass 2: substring match with negative-token exclusion.
    for i, name in enumerate(header):
        if i in exclude_idx:
            continue
        if any(nt in name for nt in neg):
            continue
        for key in keys:
            if key in name:
                return i
    return None


def parse_wechat_csv(text: str) -> list[ChatMessage]:
    """Parse a WeChat CSV export emitted by common third-party tools.

    Flexible header mapping — looks at the first row of the CSV and maps
    columns to (timestamp, sender, body) by exact-match-then-substring
    fuzz with negative-token exclusion (review #4).

    Headerless fallback (review #5): if no body column can be found by
    name, AND every cell in row 0 looks like a data row (e.g. row 0
    contains a timestamp-shaped value), treat the file as headerless
    with positional `(time, sender, body)` columns.
    """
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        return []

    header = [h.strip().lower() for h in rows[0]]

    # Find issender FIRST so it's excluded from sender-column lookup.
    isself_col = _find_column(header, _WECHAT_ISSELF_KEYS)
    excluded = {isself_col} if isself_col is not None else set()

    ts_col = _find_column(header, _WECHAT_TIME_KEYS, _NEGATIVE_TOKENS_FOR_TIME)
    sender_col = _find_column(
        header, _WECHAT_SENDER_KEYS,
        _NEGATIVE_TOKENS_FOR_SENDER, excluded,
    )
    body_col = _find_column(header, _WECHAT_BODY_KEYS, _NEGATIVE_TOKENS_FOR_BODY)

    data_rows = rows[1:]

    # Headerless fallback (review #5).
    if body_col is None:
        # Heuristic: if row 0 column 0 looks like a timestamp (digit-heavy
        # with separators) AND there are ≥3 columns, assume positional
        # (time, sender, body) and treat all rows as data.
        if rows and len(rows[0]) >= 3 and _looks_like_timestamp(rows[0][0]):
            ts_col, sender_col, body_col = 0, 1, 2
            isself_col = None
            data_rows = rows
        else:
            return []

    out: list[ChatMessage] = []
    for row in data_rows:
        if not row or len(row) <= body_col:
            continue
        msg = row[body_col].strip() if body_col < len(row) else ""
        if not msg:
            continue

        ts = row[ts_col].strip() if ts_col is not None and ts_col < len(row) else ""

        if sender_col is not None and sender_col < len(row) and row[sender_col].strip():
            sender = row[sender_col].strip()
        elif isself_col is not None and isself_col < len(row):
            flag = row[isself_col].strip().lower()
            sender = "me" if flag in _WECHAT_TRUE_VALUES else "them"
        else:
            sender = "?"

        out.append(ChatMessage(
            timestamp=ts,
            sender=sender,
            text=msg,
            platform="wechat",
        ))

    return out


def _looks_like_timestamp(s: str) -> bool:
    """Heuristic for headerless-CSV detection: cell looks like a date or
    datetime."""
    s = s.strip()
    if len(s) < 8:
        return False
    digits = sum(1 for c in s if c.isdigit())
    seps = sum(1 for c in s if c in "/-: .")
    # At least 6 digits AND at least 2 separators → "2024-03-05 08:30".
    return digits >= 6 and seps >= 2


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
