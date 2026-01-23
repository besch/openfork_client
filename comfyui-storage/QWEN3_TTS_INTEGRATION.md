# Qwen3-TTS Integration Summary

## Overview

Qwen3-TTS (Alibaba's multilingual TTS model) has been integrated into the OpenFork platform with support for:
- **CustomVoice TTS**: 22 built-in speakers across 10 languages
- **Voice Design**: Create custom voices using natural language descriptions
- **Voice Cloning**: Clone voices from reference audio with optional transcript

## Files Created

### Backend (Python)

| File | Description |
|------|-------------|
| `client/comfyui-storage/Dockerfile.qwen3-tts` | Docker image with Python 3.12, CUDA 12.4, qwen-tts package, FlashAttention 2 |
| `client/comfyui-storage/qwen3_tts_api.py` | FastAPI wrapper with async job processing for all 3 modes |
| `client/comfyui-storage/docker-compose.qwen3-tts.yaml` | Docker compose config with GPU support and persistent model storage |
| `client/services/processors/audio/qwen3_tts.py` | Job processors: `Qwen3TTSJobProcessor`, `Qwen3VoiceDesignJobProcessor`, `Qwen3VoiceCloneJobProcessor` |
| `client/workflows/qwen3-tts.api.json` | REST API specification with endpoints and supported languages/speakers |

### Frontend (TypeScript/React)

| File | Description |
|------|-------------|
| `website/src/components/forms/audio/qwen3-tts-form.tsx` | TTS form with language/speaker selection, emotion presets, example prompts |

### Modified Files

| File | Changes |
|------|---------|
| `client/services/processors/audio/__init__.py` | Added exports for Qwen3 processors |
| `website/services.json` | Added `qwen3-tts` and `qwen3-tts-16gb` services + workflow entries |
| `website/src/lib/types.ts` | Added `QWEN3_TTS`, `QWEN3_VOICE_DESIGN`, `QWEN3_VOICE_CLONE` workflow types |
| `website/src/components/forms/audio/tts-form.tsx` | Added Qwen3-TTS form integration with language/speaker/instruct props |
| `website/src/components/forms/audio/tts-model-selector.tsx` | Added Qwen3-TTS option with type updates |
| `website/src/components/forms/audio/voice-clone-form.tsx` | Added model selector (Chatterbox vs Qwen3), language selection, reference text |
| `website/src/components/forms/audio/audio-form.tsx` | Added Qwen3 state variables and prop passing |
| `website/src/app/actions/audio-actions.ts` | Added Qwen3 params to schema and workflow type selection logic |

## Features

### Supported Languages
- Auto (auto-detect)
- Chinese, English, Japanese, Korean
- German, French, Russian
- Portuguese, Spanish, Italian

### Built-in Speakers (22 total)
Organized by language with male and female options:
- **Chinese**: Vivian, Chelsie, Austin, Ethan
- **English**: Ryan, Olivia, Lily, Emma
- **Japanese**: Haruto, Yuki
- **Korean**: Minjun, Sora
- **German**: Leon, Mia
- **French**: Louis, Chloe
- **Russian**: Ivan, Anna
- **Portuguese**: Miguel, Sofia
- **Spanish**: Mateo, Lucia
- **Italian**: Leonardo, Giulia

### Emotion/Style Instructions
Users can describe speaking style in natural language:
- "Speak with a happy and cheerful tone"
- "Speak like a professional narrator"
- "Speak in a soft whisper"

### Voice Cloning
- Upload 10-60s reference audio (WAV/MP3)
- Optional reference text for better quality
- Supports all 10+ languages

## API Endpoints

```
POST /generate              - Submit TTS job (custom_voice or voice_design mode)
POST /generate/voice-clone  - Submit voice clone with audio upload
POST /generate/voice-clone-path - Submit voice clone with existing path
GET  /status/{job_id}       - Check job status
GET  /download/{job_id}     - Download generated audio
DELETE /job/{job_id}        - Clean up job files
GET  /health                - Health check
GET  /info                  - Get supported languages/speakers
```

## Model Variants

| Service | Model Size | VRAM | Use Case |
|---------|------------|------|----------|
| `qwen3-tts` | 0.6B | 8GB | CustomVoice, Voice Cloning |
| `qwen3-tts-16gb` | 1.7B | 16GB | VoiceDesign |

## Next Steps

1. Build Docker images: `docker compose -f docker-compose.qwen3-tts.yaml build`
2. Push to registry: Update `build_and_push.py` with qwen3-tts entry
3. Run migrations if any database changes needed
4. Test all 3 modes: TTS, Voice Design, Voice Cloning
