"""Audio processor modules."""

from .stable_audio import StableAudioJobProcessor
from .foley import FoleyJobProcessor
from .vibevoice import VibeVoiceJobProcessor, VibeVoiceMultiCloneJobProcessor
from .diffrhythm_cli import DiffRhythmCLIJobProcessor
from .chatterbox import ChatterboxTTSJobProcessor, ChatterboxVoiceCloneJobProcessor

__all__ = [
    "StableAudioJobProcessor",
    "FoleyJobProcessor",
    "VibeVoiceJobProcessor",
    "VibeVoiceMultiCloneJobProcessor",
    "DiffRhythmCLIJobProcessor",
    "ChatterboxTTSJobProcessor",
    "ChatterboxVoiceCloneJobProcessor",
]

