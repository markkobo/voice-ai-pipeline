"""
Unit tests for chat-export parsers (slice 2B).

Each parser is pure (string in, list out) → easy to test without
filesystem fixtures. Covers format detection, multi-line continuations,
header/footer trimming, edge cases.
"""
from __future__ import annotations

import pytest

from app.services.corpus.chat_parsers import (
    detect_chat_format,
    messages_to_text,
    parse_line,
    parse_whatsapp,
    parse_wechat_csv,
)


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------
class TestFormatDetection:
    def test_detect_line_header(self):
        text = "[LINE] 與媽的聊天記錄\n儲存日期：2024/03/12 14:32\n"
        assert detect_chat_format(text) == "line"

    def test_detect_whatsapp_basic(self):
        text = "12/03/24, 14:32 - John: Hello\n"
        assert detect_chat_format(text) == "whatsapp"

    def test_detect_whatsapp_ios_brackets(self):
        text = "[12/03/24, 14:32:01] John: Hello\n"
        assert detect_chat_format(text) == "whatsapp"

    def test_detect_wechat_csv_header(self):
        text = "StrTime,IsSender,Message,Type\n2024-03-05 08:30,0,早安,text\n"
        assert detect_chat_format(text) == "wechat"

    def test_detect_freeform_text_returns_none(self):
        text = "Just some random notes about cooking.\nNothing chat-like here.\n"
        assert detect_chat_format(text) is None

    def test_detect_empty_returns_none(self):
        assert detect_chat_format("") is None


# ---------------------------------------------------------------------------
# WhatsApp parser
# ---------------------------------------------------------------------------
class TestWhatsAppParser:
    def test_android_style_basic(self):
        text = (
            "12/03/24, 14:32 - John: Hello there\n"
            "12/03/24, 14:33 - Mary: Hi John!\n"
        )
        msgs = parse_whatsapp(text)
        assert len(msgs) == 2
        assert msgs[0].sender == "John"
        assert msgs[0].text == "Hello there"
        assert msgs[1].sender == "Mary"
        assert msgs[1].text == "Hi John!"
        assert all(m.platform == "whatsapp" for m in msgs)

    def test_ios_style_brackets(self):
        text = (
            "[12/03/24, 14:32:01] John: Hello\n"
            "[12/03/24, 14:32:45] Mary: Hi\n"
        )
        msgs = parse_whatsapp(text)
        assert len(msgs) == 2
        assert msgs[0].sender == "John"
        assert "14:32:01" in msgs[0].timestamp

    def test_multiline_continuation(self):
        text = (
            "12/03/24, 14:32 - John: Line one\n"
            "Line two of John's msg\n"
            "Line three\n"
            "12/03/24, 14:33 - Mary: Reply\n"
        )
        msgs = parse_whatsapp(text)
        assert len(msgs) == 2
        assert "Line one\nLine two of John's msg\nLine three" == msgs[0].text
        assert msgs[1].sender == "Mary"

    def test_empty_input(self):
        assert parse_whatsapp("") == []

    def test_only_garbage_lines(self):
        # No timestamp-prefixed line, nothing to parse.
        assert parse_whatsapp("just\nrandom\ntext\n") == []

    # ----- Review-driven robustness tests (commit 3f2f55e follow-up) -----

    def test_body_starting_with_date_pattern_does_not_open_new_msg(self):
        """Review #1 BLOCKER: a continuation line whose body starts with
        a date-shape must NOT be parsed as a new message header."""
        text = (
            "12/03/24, 14:32 - John: 我們在 2024-01-01 14:32 那天遇到\n"
            "12/03/24, 14:33 - Mary: 真巧\n"
        )
        msgs = parse_whatsapp(text)
        assert len(msgs) == 2
        assert msgs[0].sender == "John"
        # Body should NOT have been split despite the embedded date.
        assert "2024-01-01" in msgs[0].text
        assert msgs[1].sender == "Mary"

    def test_system_lines_are_dropped_not_appended(self):
        """Review #6 MAJOR: system notices (timestamp + no `sender:` part)
        must not get folded into the previous speaker's message."""
        text = (
            "12/03/24, 14:32 - John: Hello\n"
            "12/03/24, 14:33 - Messages and calls are end-to-end encrypted.\n"
            "12/03/24, 14:34 - Mary: Hi\n"
        )
        msgs = parse_whatsapp(text)
        # John's message should NOT contain the encryption notice.
        assert all("end-to-end encrypted" not in m.text for m in msgs)
        # And the encryption notice should NOT have been treated as
        # a sender called "Messages and calls are end".
        assert all(
            "Messages and calls" not in m.sender for m in msgs
        )

    def test_bom_prefixed_first_line(self):
        """Windows Notepad save prepends a UTF-8 BOM. The parser should
        still match the first message."""
        text = "﻿12/03/24, 14:32 - John: Hello\n"
        msgs = parse_whatsapp(text)
        # BOM in front of digit still parses — re's `^` lets it through.
        # We at least don't crash.
        assert len(msgs) <= 1


class TestLineParserRobustness:
    """Review #2 + #13 — real-world Line export robustness."""

    def test_accepts_runs_of_spaces_when_tabs_are_lost(self):
        """Review #2 BLOCKER: zip-roundtrip / email forwarding often
        converts tabs in Line exports to runs of spaces. Parser must
        still produce messages."""
        text = (
            "2024/03/05（二）\n"
            "08:30   媽   早安\n"        # 3 spaces between fields
            "08:32  我  早\n"             # 2 spaces between fields
        )
        msgs = parse_line(text)
        assert len(msgs) == 2
        assert msgs[0].sender == "媽"
        assert msgs[0].text == "早安"
        assert msgs[1].sender == "我"

    def test_date_regex_requires_matched_parens(self):
        """Review #13: bare '2024/03/05 (memo' (unmatched paren in a
        continuation line) must not be eaten as a new date section."""
        text = (
            "2024/03/05（二）\n"
            "08:30\t媽\t早安\n"
            "2024/03/05 (memo\n"      # continuation, NOT a new date header
            "08:32\t我\t嗨\n"
        )
        msgs = parse_line(text)
        # The "(memo" line should attach to the previous message, not
        # start a new section.
        assert len(msgs) == 2
        assert "(memo" in msgs[0].text


class TestWeChatCsvParserRobustness:
    """Review #4 / #5 / #8 — collision-resistant column detection."""

    def test_exact_match_wins_over_substring(self):
        """Review #4 MAJOR: 'lifetime' must NOT win over 'time' for the
        timestamp column."""
        text = (
            "lifetime,time,sender,message\n"
            "long,2024-03-05 08:30,媽,早安\n"
        )
        msgs = parse_wechat_csv(text)
        assert len(msgs) == 1
        # Timestamp should be the 'time' column, not the 'lifetime' one.
        assert msgs[0].timestamp == "2024-03-05 08:30"

    def test_senderid_does_not_win_over_sender(self):
        """Review #4: 'SenderId' must not match the sender lookup when
        a real 'Sender' column is also present."""
        text = (
            "time,senderid,sender,message\n"
            "2024-03-05,7782,媽,早安\n"
        )
        msgs = parse_wechat_csv(text)
        assert len(msgs) == 1
        assert msgs[0].sender == "媽"

    def test_message_id_does_not_become_body(self):
        """Review #4: 'message_id' must not match the body lookup."""
        text = (
            "time,sender,message_id,message\n"
            "2024-03-05,媽,abc-123,早安\n"
        )
        msgs = parse_wechat_csv(text)
        assert len(msgs) == 1
        assert msgs[0].text == "早安"

    def test_headerless_csv_fallback(self):
        """Review #5: when no body column can be identified by name AND
        row 0 column 0 looks like a timestamp, treat as headerless with
        positional (time, sender, body)."""
        text = (
            "2024-03-05 08:30,媽,早安\n"
            "2024-03-05 08:32,我,早\n"
        )
        msgs = parse_wechat_csv(text)
        assert len(msgs) == 2
        assert msgs[0].sender == "媽"
        assert msgs[0].text == "早安"

    def test_issender_chinese_true_value(self):
        """Review #8: '是' must be treated as truthy for IsSender."""
        text = (
            "time,IsSender,message\n"
            "2024-03-05,是,早\n"
            "2024-03-05,否,早安\n"
        )
        msgs = parse_wechat_csv(text)
        assert len(msgs) == 2
        assert msgs[0].sender == "me"
        assert msgs[1].sender == "them"

    def test_issender_lowercase_truthy(self):
        """Review #8: 'yes' / 'y' / lowercase 'true' must all be truthy."""
        text = (
            "time,IsSender,message\n"
            "2024-03-05,yes,a\n"
            "2024-03-05,Y,b\n"
            "2024-03-05,true,c\n"
            "2024-03-05,no,d\n"
        )
        msgs = parse_wechat_csv(text)
        senders = [m.sender for m in msgs]
        assert senders == ["me", "me", "me", "them"]


class TestFormatDetectionRobustness:
    """Review #7 — file content containing 'wechat' or 'sender' benignly
    must not be misclassified."""

    def test_prose_containing_word_wechat_not_misclassified(self):
        text = (
            "Notes on WeChat backup process.\n"
            "Step 1: install Memotrace.\n"
            "Step 2: export.\n"
        )
        # First line contains "WeChat" but is clearly prose. Must NOT
        # detect as wechat-csv.
        assert detect_chat_format(text) != "wechat"

    def test_prose_with_date_pattern_not_misclassified_whatsapp(self):
        """A prose line containing '14:32, ' or '12/03/24' must not
        trip WhatsApp sniff because there's no mandatory " - " separator."""
        text = (
            "On 12/03/24, 14:32 we met for lunch.\n"
            "It was great.\n"
        )
        assert detect_chat_format(text) != "whatsapp"

    def test_real_wechat_csv_still_detected(self):
        """Sanity: tightening must not break the happy path."""
        text = (
            "StrTime,IsSender,Message,Type\n"
            "2024-03-05 08:30,0,早安,text\n"
        )
        assert detect_chat_format(text) == "wechat"


# ---------------------------------------------------------------------------
# Line parser
# ---------------------------------------------------------------------------
class TestLineParser:
    def test_basic_tab_separated(self):
        text = (
            "[LINE] 聊天記錄\n"
            "儲存日期：2024/03/12 14:32\n"
            "\n"
            "2024/03/05（二）\n"
            "08:30\t媽\t早安\n"
            "08:32\t我\t早\n"
        )
        msgs = parse_line(text)
        assert len(msgs) == 2
        assert msgs[0].sender == "媽"
        assert msgs[0].text == "早安"
        assert msgs[0].timestamp == "2024/03/05 08:30"
        assert msgs[1].sender == "我"

    def test_carries_date_across_sections(self):
        text = (
            "[LINE] 聊天記錄\n"
            "\n"
            "2024/03/05（二）\n"
            "08:30\t媽\t早安\n"
            "\n"
            "2024/03/06（三）\n"
            "10:00\t我\t嗨\n"
        )
        msgs = parse_line(text)
        assert len(msgs) == 2
        assert msgs[0].timestamp.startswith("2024/03/05")
        assert msgs[1].timestamp.startswith("2024/03/06")

    def test_multiline_continuation(self):
        text = (
            "2024/03/05（二）\n"
            "08:30\t媽\t第一行\n"
            "第二行\n"
            "08:32\t我\t回覆\n"
        )
        msgs = parse_line(text)
        assert len(msgs) == 2
        assert msgs[0].text == "第一行\n第二行"

    def test_empty_input(self):
        assert parse_line("") == []


# ---------------------------------------------------------------------------
# WeChat CSV parser
# ---------------------------------------------------------------------------
class TestWeChatCsvParser:
    def test_strtime_issender_message_schema(self):
        text = (
            "StrTime,IsSender,Message,Type\n"
            "2024-03-05 08:30:00,0,早安,text\n"
            "2024-03-05 08:32:00,1,早,text\n"
        )
        msgs = parse_wechat_csv(text)
        assert len(msgs) == 2
        assert msgs[0].sender == "them"
        assert msgs[0].text == "早安"
        assert msgs[0].timestamp == "2024-03-05 08:30:00"
        assert msgs[1].sender == "me"

    def test_named_sender_schema(self):
        text = (
            "time,sender,message\n"
            "2024-03-05 08:30,媽,早安\n"
            "2024-03-05 08:32,我,早\n"
        )
        msgs = parse_wechat_csv(text)
        assert len(msgs) == 2
        assert msgs[0].sender == "媽"
        assert msgs[1].sender == "我"

    def test_chinese_column_names(self):
        text = (
            "時間,發送者,內容\n"
            "2024-03-05 08:30,媽,早安\n"
        )
        msgs = parse_wechat_csv(text)
        assert len(msgs) == 1
        assert msgs[0].sender == "媽"
        assert msgs[0].text == "早安"

    def test_missing_body_column_returns_empty(self):
        # No identifiable body column → nothing to parse.
        text = "a,b,c\n1,2,3\n"
        assert parse_wechat_csv(text) == []

    def test_skips_empty_body_rows(self):
        text = (
            "time,sender,message\n"
            "2024-03-05 08:30,媽,早安\n"
            "2024-03-05 08:31,媽,\n"   # empty body
            "2024-03-05 08:32,我,早\n"
        )
        msgs = parse_wechat_csv(text)
        assert len(msgs) == 2


# ---------------------------------------------------------------------------
# Canonical text formatting
# ---------------------------------------------------------------------------
class TestMessagesToText:
    def test_paragraph_break_between_speakers(self):
        from app.services.corpus.chat_parsers import ChatMessage
        msgs = [
            ChatMessage("2024-03-05 08:30", "媽", "早安", "wechat"),
            ChatMessage("2024-03-05 08:31", "媽", "今天天氣好", "wechat"),
            ChatMessage("2024-03-05 08:32", "我", "嗨", "wechat"),
        ]
        text = messages_to_text(msgs)
        # Same speaker stays adjacent; speaker change gets a blank line.
        assert "[2024-03-05 08:30] 媽: 早安\n[2024-03-05 08:31] 媽: 今天天氣好\n\n[2024-03-05 08:32] 我: 嗨" == text

    def test_no_timestamp_drops_prefix(self):
        from app.services.corpus.chat_parsers import ChatMessage
        msgs = [ChatMessage("", "alice", "hi", "wechat")]
        assert messages_to_text(msgs) == "alice: hi"

    def test_empty_list(self):
        assert messages_to_text([]) == ""
