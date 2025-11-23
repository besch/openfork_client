
import subprocess
import logging
from typing import Union
import os

def get_audio_duration(file_path: str) -> float:
    """Gets the duration of an audio file using ffprobe."""
    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        logging.error(f"Error getting audio duration for {file_path}: {e}")
        return 0.0

def get_video_duration(file_path: str) -> float:
    """Gets the duration of a video file using ffprobe."""
    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        logging.error(f"Error getting video duration for {file_path}: {e}")
        return 0.0

def generate_thumbnail(video_path: str, thumbnail_path: str) -> bool:
    """Generates a thumbnail for a video file."""
    command = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-ss", "00:00:01.000",
        "-vframes", "1",
        thumbnail_path
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=60)
        logging.info(f"Thumbnail generated at {thumbnail_path}")
        return True
    except subprocess.TimeoutExpired:
        logging.error(f"Timeout generating thumbnail for {video_path}")
        return False
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logging.error(f"Error generating thumbnail for {video_path}: {e}")
        if isinstance(e, subprocess.CalledProcessError):
            logging.error(f"ffmpeg stderr: {e.stderr.decode()}")
        return False

def find_video_in_output(outputs: dict) -> Union[tuple[str, str], None]:
    """Finds the output video details from the ComfyUI workflow output."""
    # Look for 'gifs' which is what VideoCombine nodes output
    for node_id, node_output in outputs.items():
        if 'gifs' in node_output:
            for item in node_output['gifs']:
                filename = item.get('filename')
                subfolder = item.get('subfolder', '')
                if filename and (item.get('format') == 'video/h264-mp4' or filename.endswith('.mp4')):
                     return filename, subfolder
    # Fallback for older formats or other nodes
    for node_id, node_output in outputs.items():
        if 'videos' in node_output:
            for video in node_output['videos']:
                filename = video.get('filename')
                subfolder = video.get('subfolder')
                if filename:
                    return filename, subfolder
    return None

def find_audio_in_output(outputs: dict) -> Union[tuple[str, str], None]:
    """Finds the output audio details from the ComfyUI workflow output."""
    for node_id, node_output in outputs.items():
        if 'audio' in node_output:
            for audio in node_output['audio']:
                filename = audio.get('filename')
                subfolder = audio.get('subfolder', '')
                if filename:
                    return filename, subfolder
    return None

def find_image_in_output(outputs: dict) -> Union[tuple[str, str], None]:
    """Finds the output image details from the ComfyUI workflow output."""
    for node_id, node_output in outputs.items():
        if 'images' in node_output:
            for image in node_output['images']:
                filename = image.get('filename')
                subfolder = image.get('subfolder')
                if filename:
                    return filename, subfolder
    return None

def generate_placeholder_video(output_path: str, duration: int = 5) -> bool:
    """Generates a silent, black placeholder video."""
    command = [
        "ffmpeg",
        "-f", "lavfi",
        "-i", f"color=c=black:s=1024x576:d={duration}",
        "-f", "lavfi",
        "-i", "anullsrc=cl=mono:r=44100",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-shortest",
        "-y", # Overwrite output file if it exists
        output_path
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
        logging.info(f"Placeholder video generated at {output_path}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logging.error(f"Error generating placeholder video: {e}")
        if isinstance(e, subprocess.CalledProcessError):
            logging.error(f"ffmpeg stderr: {e.stderr.decode()}")
        return False


def get_video_dimensions(video_path: str) -> tuple:
    """
    Get the width and height of a video file.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Tuple of (width, height)
        
    Raises:
        RuntimeError if dimensions cannot be determined
    """
    import subprocess
    import json
    
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'json',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        if 'streams' in data and len(data['streams']) > 0:
            stream = data['streams'][0]
            width = stream.get('width')
            height = stream.get('height')
            
            if width and height:
                return (width, height)
        
        raise RuntimeError("Could not extract dimensions from ffprobe output")
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffprobe failed: {e.stderr}")
    except Exception as e:
        raise RuntimeError(f"Failed to get video dimensions: {e}")