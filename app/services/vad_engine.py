"""VAD (Voice Activity Detection) engine module.

.. deprecated::
    Import from :mod:`app.services.asr.vad_engine` instead.
"""
from app.services.asr.vad_engine import BaseVAD, EnergyVAD

__all__ = ["BaseVAD", "EnergyVAD"]
