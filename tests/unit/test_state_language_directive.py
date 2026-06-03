"""
Locks the language directive plumbing shipped during 06/02 demo prep.

UI dropdown sends `language: "chinese" | "english" | "auto"` in the
config payload. StateManager normalizes this onto `state.language`
(string lowercased, or None when absent / "auto"). ws_asr.run_llm_stream
reads `state.language` and appends a LANGUAGE directive to the system
prompt before calling the LLM.

These tests cover the state_manager normalization layer. The end-to-end
LLM-injection path is covered separately at the ws_asr integration level
once we add a real LLM mock there.
"""
import pytest

from app.core.state_manager import StateManager


@pytest.fixture
def manager():
    return StateManager()


@pytest.fixture
def session(manager):
    sid = "lang-test"
    manager.create_session(sid)
    return sid, manager.get_session(sid)


def _config_with_language(value):
    """Minimal config payload — only language is the field under test."""
    return {
        "audio": {"sample_rate": 24000, "channels": 1, "format": "pcm"},
        "language": value,
    }


class TestLanguageDirectiveNormalization:
    """state.language is normalized to lowercase concrete strings or None.
    'auto' and '' are treated as no-override (None)."""

    def test_default_is_none(self, session):
        sid, st = session
        assert st.language is None

    def test_chinese_lowercased(self, manager, session):
        sid, _ = session
        manager.update_config(sid, _config_with_language("chinese"))
        assert manager.get_session(sid).language == "chinese"

    def test_english_lowercased(self, manager, session):
        sid, _ = session
        manager.update_config(sid, _config_with_language("english"))
        assert manager.get_session(sid).language == "english"

    def test_uppercase_normalized(self, manager, session):
        """UI value comes through whatever case; server lowercases it."""
        sid, _ = session
        manager.update_config(sid, _config_with_language("CHINESE"))
        assert manager.get_session(sid).language == "chinese"

    def test_auto_treated_as_none(self, manager, session):
        """'auto' from the dropdown means no override — store as None so
        the LLM injection branch is skipped."""
        sid, _ = session
        manager.update_config(sid, _config_with_language("auto"))
        assert manager.get_session(sid).language is None

    def test_empty_string_treated_as_none(self, manager, session):
        sid, _ = session
        manager.update_config(sid, _config_with_language(""))
        assert manager.get_session(sid).language is None

    def test_absent_field_keeps_prior_value(self, manager, session):
        """Per the update_config contract: 'language' absent = no change.
        Important for resends — losing language on every commit would
        cause repeated drops to None mid-session."""
        sid, _ = session
        manager.update_config(sid, _config_with_language("chinese"))
        # Now send a config WITHOUT language
        manager.update_config(
            sid,
            {"audio": {"sample_rate": 24000, "channels": 1, "format": "pcm"}},
        )
        assert manager.get_session(sid).language == "chinese"

    def test_switching_between_languages(self, manager, session):
        """Language can flip back and forth — each new value overrides."""
        sid, _ = session
        manager.update_config(sid, _config_with_language("chinese"))
        assert manager.get_session(sid).language == "chinese"
        manager.update_config(sid, _config_with_language("english"))
        assert manager.get_session(sid).language == "english"
        manager.update_config(sid, _config_with_language("auto"))
        assert manager.get_session(sid).language is None

    def test_null_explicit_is_none(self, manager, session):
        """Explicit JSON null comes through as None and clears state."""
        sid, _ = session
        manager.update_config(sid, _config_with_language("chinese"))
        manager.update_config(sid, _config_with_language(None))
        assert manager.get_session(sid).language is None


class TestLanguageWithListenOnly:
    """Language and listen_only are independent fields — setting one
    must not clear the other. This regression caught us during demo
    prep when each fix touched the config schema."""

    def test_setting_language_preserves_listen_only(self, manager, session):
        sid, _ = session
        manager.update_config(
            sid,
            {
                "audio": {"sample_rate": 24000, "channels": 1, "format": "pcm"},
                "listen_only": True,
            },
        )
        assert manager.get_session(sid).listen_only is True
        manager.update_config(sid, _config_with_language("chinese"))
        # listen_only was not in the second config — must remain True
        st = manager.get_session(sid)
        assert st.listen_only is True
        assert st.language == "chinese"

    def test_setting_listen_only_preserves_language(self, manager, session):
        sid, _ = session
        manager.update_config(sid, _config_with_language("english"))
        manager.update_config(
            sid,
            {
                "audio": {"sample_rate": 24000, "channels": 1, "format": "pcm"},
                "listen_only": True,
            },
        )
        st = manager.get_session(sid)
        assert st.language == "english"
        assert st.listen_only is True
