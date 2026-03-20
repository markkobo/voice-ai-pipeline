"""ASR (Automatic Speech Recognition) engine module.

.. deprecated::
    Import from :mod:`app.services.asr.engine` instead.
"""
from app.services.asr.engine import BaseASR, Qwen3ASR, MockASR

__all__ = ["BaseASR", "Qwen3ASR", "MockASR"]
