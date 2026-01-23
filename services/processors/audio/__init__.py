"""Audio processor modules."""

from .stable_audio import StableAudioJobProcessor
from .foley import FoleyJobProcessor
from .vibevoice import VibeVoiceJobProcessor, VibeVoiceMultiCloneJobProcessor
from .diffrhythm_cli import DiffRhythmCLIJobProcessor
from .chatterbox import ChatterboxTTSJobProcessor, ChatterboxVoiceCloneJobProcessor
from .qwen3_tts import Qwen3TTSJobProcessor, Qwen3VoiceDesignJobProcessor, Qwen3VoiceCloneJobProcessor

__all__ = [
    "StableAudioJobProcessor",
    "FoleyJobProcessor",
    "VibeVoiceJobProcessor",
    "VibeVoiceMultiCloneJobProcessor",
    "DiffRhythmCLIJobProcessor",
    "ChatterboxTTSJobProcessor",
    "ChatterboxVoiceCloneJobProcessor",
    "Qwen3TTSJobProcessor",
    "Qwen3VoiceDesignJobProcessor",
    "Qwen3VoiceCloneJobProcessor",
]

