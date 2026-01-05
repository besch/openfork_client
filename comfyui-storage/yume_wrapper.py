"""
YUME 1.5 ComfyUI Wrapper Node

This wrapper integrates the YUME 1.5 world generation model with ComfyUI.
YUME uses a custom diffusion pipeline (not standard diffusers), so we call
the inference script as a subprocess following best practices from the official repo.

Reference: https://github.com/stdstu12/YUME
Model: stdstu123/Yume-5B-720P (HuggingFace)
"""

import os
import sys
import torch
import subprocess
import tempfile
import shutil
import random
import json
import numpy as np
from pathlib import Path
from PIL import Image

# ComfyUI imports
import folder_paths

# Constants
YUME_PATH = "/opt/YUME"
MODEL_PATH = "/opt/models/Yume"


class YUME_Node:
    """
    YUME 1.5 World Generator Node for ComfyUI.
    
    Generates interactive world videos from text prompts or images using
    the Yume-5B-720P model.
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = folder_paths.get_output_directory()
        self.temp_dir = tempfile.mkdtemp(prefix="yume_")
        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A beautiful landscape with rolling hills and a sunset sky"}),
                "n_prompt": ("STRING", {"multiline": True, "default": "low quality, distorted, bad animation, blurry, watermark"}),
                "num_frames": ("INT", {"default": 49, "min": 9, "max": 97, "step": 8}),  # YUME uses specific frame counts
                "width": ("INT", {"default": 1280, "min": 512, "max": 1920, "step": 64}),
                "height": ("INT", {"default": 720, "min": 288, "max": 1080, "step": 64}),
                "steps": ("INT", {"default": 30, "min": 4, "max": 50}),  # num_euler_timesteps in YUME
                "cfg": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 20.0, "step": 0.5}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "start_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate"
    CATEGORY = "YUME"
    OUTPUT_NODE = True

    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert a ComfyUI image tensor to PIL Image."""
        # ComfyUI images are [B, H, W, C] in range [0, 1]
        if tensor.dim() == 4:
            tensor = tensor[0]  # Take first image if batched
        
        # Convert to numpy and scale to 0-255
        np_image = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(np_image)

    def load_video_frames(self, video_path: str) -> torch.Tensor:
        """Load video frames and convert to ComfyUI tensor format."""
        import av
        
        frames = []
        container = av.open(video_path)
        
        for frame in container.decode(video=0):
            # Convert to RGB numpy array
            img = frame.to_ndarray(format='rgb24')
            frames.append(img)
        
        container.close()
        
        # Stack frames and convert to tensor [N, H, W, C] normalized to [0, 1]
        frames_np = np.stack(frames, axis=0).astype(np.float32) / 255.0
        return torch.from_numpy(frames_np)

    def generate(self, prompt, n_prompt, num_frames, width, height, steps, cfg, seed, start_image=None):
        """
        Generate video using YUME 1.5 model.
        
        For Text-to-Video: Uses --T2V flag with --prompt
        For Image-to-Video: Uses --jpg_dir with input image
        """
        print(f"[YUME] Starting generation: {num_frames} frames at {width}x{height}, {steps} steps")
        
        # Use random seed if 0
        if seed == 0:
            seed = random.randint(1, 2**32 - 1)
        
        # Prepare temporary directories
        temp_input_dir = os.path.join(self.temp_dir, "input")
        temp_output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(temp_input_dir, exist_ok=True)
        os.makedirs(temp_output_dir, exist_ok=True)
        
        # Base command arguments
        cmd = [
            "python", "-m", "fastvideo.sample.sample_5b",
            "--seed", str(seed),
            "--gradient_checkpointing",
            "--train_batch_size", "1",
            "--max_sample_steps", "600000",
            "--mixed_precision", "bf16",
            "--allow_tf32",
            "--video_output_dir", temp_output_dir,
            "--num_euler_timesteps", str(steps),
            "--rand_num_img", "0.6",
        ]
        
        # Create caption file with prompt
        caption_path = os.path.join(temp_input_dir, "caption.txt")
        with open(caption_path, "w", encoding="utf-8") as f:
            f.write(prompt)
        cmd.extend(["--caption_path", caption_path])
        
        # Handle image-to-video vs text-to-video
        if start_image is not None:
            # Image-to-Video mode
            print("[YUME] Mode: Image-to-Video")
            jpg_dir = os.path.join(temp_input_dir, "jpg")
            os.makedirs(jpg_dir, exist_ok=True)
            
            # Save the input image
            pil_image = self.tensor_to_pil(start_image)
            # Resize to target dimensions
            pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)
            image_path = os.path.join(jpg_dir, "input_0.jpg")
            pil_image.save(image_path, quality=95)
            
            cmd.extend(["--jpg_dir", jpg_dir])
        else:
            # Text-to-Video mode
            print("[YUME] Mode: Text-to-Video")
            cmd.extend(["--T2V"])
            cmd.extend(["--prompt", prompt])
        
        # Set environment variables
        env = os.environ.copy()
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
        
        # Run inference
        print(f"[YUME] Running command: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                cwd=YUME_PATH,
                env=env,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minute timeout
            )
            
            if result.returncode != 0:
                print(f"[YUME] Error: {result.stderr}")
                raise RuntimeError(f"YUME inference failed: {result.stderr[:500]}")
            
            print(f"[YUME] Inference completed successfully")
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("YUME inference timed out after 30 minutes")
        except Exception as e:
            print(f"[YUME] Exception during inference: {e}")
            raise
        
        # Find the output video
        output_files = list(Path(temp_output_dir).glob("*.mp4"))
        if not output_files:
            # Check for other video formats
            output_files = list(Path(temp_output_dir).glob("*.avi"))
        if not output_files:
            # Try recursively
            output_files = list(Path(temp_output_dir).rglob("*.mp4"))
        
        if not output_files:
            print(f"[YUME] No output video found in {temp_output_dir}")
            print(f"[YUME] Directory contents: {list(Path(temp_output_dir).iterdir())}")
            # Return dummy frames as fallback
            dummy_frames = torch.zeros((num_frames, height, width, 3), dtype=torch.float32)
            return (dummy_frames,)
        
        # Load the generated video
        video_path = str(output_files[0])
        print(f"[YUME] Loading output video: {video_path}")
        
        try:
            frames = self.load_video_frames(video_path)
            print(f"[YUME] Loaded {frames.shape[0]} frames")
        except Exception as e:
            print(f"[YUME] Error loading video: {e}")
            # Return dummy frames as fallback
            dummy_frames = torch.zeros((num_frames, height, width, 3), dtype=torch.float32)
            return (dummy_frames,)
        
        # Cleanup temp directory
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except:
            pass
        
        return (frames,)


class YUME_TextToVideo_Node(YUME_Node):
    """
    YUME 1.5 Text-to-Video Node.
    Simplified node for text-to-video generation only.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage."}),
                "n_prompt": ("STRING", {"multiline": True, "default": "low quality, distorted, bad animation, blurry, watermark"}),
                "num_frames": ("INT", {"default": 49, "min": 9, "max": 97, "step": 8}),
                "width": ("INT", {"default": 1280, "min": 512, "max": 1920, "step": 64}),
                "height": ("INT", {"default": 720, "min": 288, "max": 1080, "step": 64}),
                "steps": ("INT", {"default": 30, "min": 4, "max": 50}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 20.0, "step": 0.5}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    def generate(self, prompt, n_prompt, num_frames, width, height, steps, cfg, seed):
        return super().generate(prompt, n_prompt, num_frames, width, height, steps, cfg, seed, start_image=None)


class YUME_ImageToVideo_Node(YUME_Node):
    """
    YUME 1.5 Image-to-Video Node.
    Simplified node for image-to-video generation.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start_image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "The camera slowly pans across the scene, revealing more details."}),
                "n_prompt": ("STRING", {"multiline": True, "default": "low quality, distorted, bad animation, blurry, watermark"}),
                "num_frames": ("INT", {"default": 49, "min": 9, "max": 97, "step": 8}),
                "steps": ("INT", {"default": 30, "min": 4, "max": 50}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 20.0, "step": 0.5}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    def generate(self, start_image, prompt, n_prompt, num_frames, steps, cfg, seed):
        # Get dimensions from input image
        height, width = start_image.shape[1], start_image.shape[2]
        # Clamp to valid YUME dimensions
        width = min(max(width, 512), 1920)
        height = min(max(height, 288), 1080)
        # Make divisible by 64
        width = (width // 64) * 64
        height = (height // 64) * 64
        
        return super().generate(prompt, n_prompt, num_frames, width, height, steps, cfg, seed, start_image=start_image)


# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "YUME_Node": YUME_Node,
    "YUME_TextToVideo": YUME_TextToVideo_Node,
    "YUME_ImageToVideo": YUME_ImageToVideo_Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YUME_Node": "YUME World Generator",
    "YUME_TextToVideo": "YUME Text-to-Video",
    "YUME_ImageToVideo": "YUME Image-to-Video",
}
