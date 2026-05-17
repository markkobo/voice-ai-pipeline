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
