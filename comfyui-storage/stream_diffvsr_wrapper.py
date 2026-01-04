import os
import torch
import folder_paths
import subprocess
import shutil
import glob
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor

class StreamDiffVSR_Node:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "steps": ("INT", {"default": 4, "min": 1, "max": 50}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "upscale"
    CATEGORY = "Upscaling"

    def upscale(self, images, steps):
        # Setup temp directories
        input_dir = "/tmp/stream_diffvsr_in"
        output_dir = "/tmp/stream_diffvsr_out"
        
        if os.path.exists(input_dir):
            shutil.rmtree(input_dir)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            
        os.makedirs(input_dir)
        os.makedirs(output_dir)

        # Save input images
        print(f"Saving {len(images)} frames to {input_dir}")
        for i, img_tensor in enumerate(images):
            # Convert tensor (Batch, H, W, C) -> PIL
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            img = Image.fromarray(img_np)
            img.save(os.path.join(input_dir, f"{i:05d}.png"))

        # Run inference script
        # Assuming inference.py structure based on standard diffusers/research code
        # python inference.py --model_id ... --in_path ... --out_path ... --num_inference_steps ...
        
        model_id = "/opt/models/Stream-DiffVSR"
        script_path = "/opt/Stream-DiffVSR/inference.py"
        
        cmd = [
            "python", script_path,
            "--model_id", model_id,
            "--in_path", input_dir,
            "--out_path", output_dir,
            "--num_inference_steps", str(steps),
            "--height", "0", # 0 might mean auto/keep aspect, or we assume script handles it. 
            # If the script requires specific height, we might need to adjust. 
            # Based on usage: just path args usually work.
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running Stream-DiffVSR: {e.stdout} {e.stderr}")
            raise RuntimeError(f"Stream-DiffVSR failed: {e.stderr}")

        # Load results
        output_files = sorted(glob.glob(os.path.join(output_dir, "*.png")))
        if not output_files:
            raise RuntimeError("No output images generated from Stream-DiffVSR")

        output_images = []
        for file in output_files:
            img = Image.open(file)
            # Convert PIL -> Tensor (H,W,C)
            # ToTensor gives (C,H,W), we need (H,W,C) for ComfyUI? 
            # ComfyUI Image is (Batch, H, W, Channel) in range 0-1
            img_np = np.array(img).astype(np.float32) / 255.0
            output_images.append(torch.from_numpy(img_np))

        # Stack to (Batch, H, W, C)
        output_tensor = torch.stack(output_images)
        
        # Cleanup
        # shutil.rmtree(input_dir) # Keep for debug
        # shutil.rmtree(output_dir) 

        return (output_tensor,)

NODE_CLASS_MAPPINGS = {
    "StreamDiffVSR_Node": StreamDiffVSR_Node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StreamDiffVSR_Node": "Stream-DiffVSR Upscaler"
}
