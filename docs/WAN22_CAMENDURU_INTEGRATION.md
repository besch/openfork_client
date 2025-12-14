# WAN22 Camenduru Image Integration Notes

## Current Setup

### Image
- **Docker Image**: `camenduru/wan-2-1-i2v-comfyui:fp8`
- **Optimization**: FP8 quantization for RTX 4060 8GB VRAM
- **Type**: ComfyUI-based (works with existing workflows)

### Form Integration
The WAN22 image form (`website/src/components/forms/video-model-forms/wan-2-2-image-form.tsx`) is **compatible** with the camenduru image:

**Form Parameters:**
- ✅ Prompt (text input)
- ✅ Camera Movement (dropdown)
- ✅ Start Image (file upload or asset selection)
- ✅ Upscale options (checkbox + parameters)

These map directly to the workflow JSON parameters.

### Workflow Compatibility

**Current Workflow** (`client/workflows/wan22-image-to-video.api.json`):
- Uses GGUF quantized models: `wan2.2_i2v_high_noise_14B_Q4_K_S.gguf`
- Uses LoRA: `Wan21_lightx2v_I2V_14B_480p_cfg_step_distill_rank32_bf16.safetensors`
- Uses SageAttention nodes for optimization

**Camenduru Image Models:**
- WAN 2.1 models in FP8 format (not GGUF)
- Pre-configured ComfyUI environment
- May have different model file names

## Potential Issues & Solutions

### Issue 1: Model File Names
The workflow references specific model files that may not match the camenduru image's files.

**Solution:**
1. After the image finishes downloading, run a test job
2. Check container logs for "file not found" errors
3. Update workflow JSON with correct model paths if needed

To inspect the container:
```bash
docker run -it --rm --gpus all camenduru/wan-2-1-i2v-comfyui:fp8 /bin/bash
ls /workspace/ComfyUI/models/checkpoints/
ls /workspace/ComfyUI/models/loras/
```

### Issue 2: ComfyUI Custom Nodes
The workflow uses custom nodes:
- `PathchSageAttentionKJ` (SageAttention)
- `ImageResizeKJv2` (KJ nodes)
- `WanImageToVideo` (WAN nodes)

**Solution:**
The camenduru image likely includes these nodes. If not, they'll need to be installed.

### Issue 3: Camera Movement
The camera movement parameter from the form isn't directly used in the workflow. It's currently just a UI element.

**Solution Options:**
1. **Keep as-is**: Use camera movement as part of the prompt (append to user's prompt)
2. **Enhance**: Create a prompt template that incorporates camera movement
3. **Advanced**: Find if WAN 2.2 has specific camera control parameters

## Testing Checklist

After image download completes:

- [ ] Test text-to-video workflow (`wan22-text-to-video.api.json`)
- [ ] Test image-to-video workflow (`wan22-image-to-video.api.json`)
- [ ] Verify model files exist in container
- [ ] Check if custom nodes are installed
- [ ] Test with actual scene generation from UI
- [ ] Monitor VRAM usage (should stay under 8GB with FP8)
- [ ] Verify output video quality

## If Models Don't Match

If the camenduru image has different model names, we can either:

1. **Update the workflow JSON** to match camenduru's model names
2. **Use a different workflow** optimized for WAN 2.1 FP8
3. **Keep current setup** if you prefer your beschiak image models

## Camera Movement Integration

Currently in the form but not workflow. To use it effectively:

```typescript
// In the scene generation action
const enhancedPrompt = cameraMovement !== 'none' 
  ? `${prompt}, ${cameraMovement} camera movement`
  : prompt;
```

This would append camera instructions to the user's prompt.

## Next Steps

1. Wait for docker pull to complete
2. Test with a simple scene
3. Check logs for any errors
4. Update workflow if models don't match
5. Consider camera movement integration

## Rollback Plan

If issues arise, revert to previous image:
```json
{
  "prod_image": "beschiak/openfork-wan22-8gb:latest",
  "docker_image_name": "openfork-wan22-8gb"
}
```
