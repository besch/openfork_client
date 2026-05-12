"""Audio processor modules."""

from .stable_audio import StableAudioJobProcessor
from .audiox import AudioXJobProcessor

from .vibevoice import VibeVoiceJobProcessor, VibeVoiceMultiCloneJobProcessor
from .diffrhythm_cli import DiffRhythmCLIJobProcessor
from .chatterbox import ChatterboxTTSJobProcessor, ChatterboxVoiceCloneJobProcessor
from .qwen3_tts import Qwen3TTSJobProcessor, Qwen3VoiceDesignJobProcessor, Qwen3VoiceCloneJobProcessor
from .mmaudio import MMAudioJobProcessor
from .lavasr import LavaSRJobProcessor
from .prismaudio import PrismAudioJobProcessor

__all__ = [
    "StableAudioJobProcessor",
    "AudioXJobProcessor",
    "VibeVoiceJobProcessor",
    "VibeVoiceMultiCloneJobProcessor",
    "DiffRhythmCLIJobProcessor",
    "ChatterboxTTSJobProcessor",
    "ChatterboxVoiceCloneJobProcessor",
    "Qwen3TTSJobProcessor",
    "Qwen3VoiceDesignJobProcessor",
    "Qwen3VoiceCloneJobProcessor",
    "MMAudioJobProcessor",
    "LavaSRJobProcessor",
    "PrismAudioJobProcessor",
]

