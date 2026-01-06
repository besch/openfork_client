
import subprocess
import logging
from typing import Union
import os
import json
from services.docker_utils import get_subprocess_hidden_kwargs

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
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=60, **get_subprocess_hidden_kwargs())
        return float(result.stdout.strip())
    except subprocess.TimeoutExpired:
        logging.error(f"Timeout getting audio duration for {file_path}")
        return 0.0
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
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=60, **get_subprocess_hidden_kwargs())
        return float(result.stdout.strip())
    except subprocess.TimeoutExpired:
        logging.error(f"Timeout getting video duration for {file_path}")
        return 0.0
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        logging.error(f"Error getting video duration for {file_path}: {e}")
        return 0.0

def generate_thumbnail(video_path: str, thumbnail_path: str, width: int = 100) -> bool:
    """Generates a thumbnail for a video file with specified width (default 100px)."""
    command = [
        "ffmpeg",
        "-y",
        "-ss", "00:00:01.000",
        "-i", video_path,
        "-vframes", "1",
        "-vf", f"scale={width}:-1",  # Scale to specified width, maintain aspect ratio
        "-nostdin",
        "-v", "error",
        thumbnail_path
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=120, **get_subprocess_hidden_kwargs())
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
    # Standard ComfyUI audio output format (from SaveAudio nodes)
    for node_id, node_output in outputs.items():
        if 'audio' in node_output:
            for audio in node_output['audio']:
                filename = audio.get('filename')
                subfolder = audio.get('subfolder', '')
                if filename:
                    return filename, subfolder
    
    # Check for StableAudio_ node's custom output format
    # This node returns a file reference dict directly in the outputs
    for node_id, node_output in outputs.items():
        # StableAudio_ returns a tuple with a single dict: ({...},)
        # When accessed through outputs, it appears as the first element
        if isinstance(node_output, dict):
            # Check if this looks like a StableAudio_ output
            if 'filename' in node_output and 'type' in node_output:
                filename = node_output.get('filename')
                subfolder = node_output.get('subfolder', '')
                if filename and node_output.get('type') == 'output':
                    logging.info(f"Found StableAudio_ output: {filename} in subfolder '{subfolder}'")
                    return filename, subfolder
    
    return None

def find_audio_file_in_directory(output_dir: str, job_id: str = None) -> Union[tuple[str, str], None]:
    """
    Finds audio files in the output directory for workflows that save directly to disk.
    Used by workflows like StableAudio_ that don't output through ComfyUI's standard format.
    
    Args:
        output_dir: Path to the output directory to search
        job_id: Optional job ID to filter files
        
    Returns:
        Tuple of (filename, subfolder) if found, None otherwise
    """
    audio_extensions = ['.wav', '.flac', '.mp3', '.ogg', '.m4a']
    try:
        if os.path.exists(output_dir):
            # Get all files sorted by modification time (most recent first)
            files = []
            for root, dirs, filenames in os.walk(output_dir):
                for filename in filenames:
                    if any(filename.lower().endswith(ext) for ext in audio_extensions):
                        filepath = os.path.join(root, filename)
                        files.append((filepath, os.path.getmtime(filepath)))
            
            if files:
                # Sort by modification time, most recent first
                files.sort(key=lambda x: x[1], reverse=True)
                most_recent = files[0][0]
                
                # Get relative path components
                rel_path = os.path.relpath(most_recent, output_dir)
                subfolder = os.path.dirname(rel_path)
                filename = os.path.basename(most_recent)
                
                logging.info(f"Found audio file in directory: {filename} (subfolder: {subfolder})")
                return filename, subfolder
    except Exception as e:
        logging.warning(f"Error searching for audio files in output directory: {e}")
    
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
        subprocess.run(command, check=True, capture_output=True, timeout=60, **get_subprocess_hidden_kwargs())
        logging.info(f"Placeholder video generated at {output_path}")
        return True
    except subprocess.TimeoutExpired:
        logging.error(f"Timeout generating placeholder video for {output_path}")
        return False
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logging.error(f"Error generating placeholder video for {output_path}: {e}")
        return False

def get_video_framerate(video_path: str) -> float:
    """Gets the framerate of a video file using ffprobe."""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate',
            '-of', 'json',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60, **get_subprocess_hidden_kwargs())
        data = json.loads(result.stdout)
        
        if 'streams' in data and len(data['streams']) > 0:
            stream = data['streams'][0]
            r_frame_rate = stream.get('r_frame_rate')
            if r_frame_rate:
                if '/' in r_frame_rate:
                    num, den = map(int, r_frame_rate.split('/'))
                    if den > 0:
                        return num / den
                else:
                    return float(r_frame_rate)
        
        return 30.0 # Default fallback
        
    except subprocess.TimeoutExpired:
        logging.error(f"Timeout getting video framerate for {video_path}")
        return 30.0
    except Exception as e:
        logging.error(f"Failed to get video framerate for {video_path}: {e}")
        return 30.0

def get_video_dimensions(video_path: str) -> tuple:
    """
    Get the width and height of a video file, accounting for rotation metadata and display aspect ratio.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Tuple of (width, height)
        
    Raises:
        RuntimeError if dimensions cannot be determined
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,display_aspect_ratio,tags:stream_tags=rotate',
            '-of', 'json',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60, **get_subprocess_hidden_kwargs())
        data = json.loads(result.stdout)
        
        if 'streams' in data and len(data['streams']) > 0:
            stream = data['streams'][0]
            width = stream.get('width')
            height = stream.get('height')
            
            # Check for rotation
            rotation = 0
            tags = stream.get('tags', {})
            if 'rotate' in tags:
                try:
                    rotation = int(tags.get('rotate'))
                except ValueError:
                    pass
            
            # Also check side_data_list for rotation if tags fail
            if rotation == 0 and 'side_data_list' in stream:
                 for side_data in stream['side_data_list']:
                     if 'rotation' in side_data:
                         try:
                             rotation = int(side_data['rotation'])
                         except ValueError:
                             pass

            if width and height:
                # Handle Display Aspect Ratio (DAR) for anamorphic content
                dar_str = stream.get('display_aspect_ratio')
                if dar_str and ':' in dar_str:
                    try:
                        num, den = map(int, dar_str.split(':'))
                        if den > 0:
                            dar = num / den
                            sar = width / height
                            # If DAR differs significantly from SAR, adjust width to match DAR (assuming square pixels for output)
                            if abs(dar - sar) > 0.01:
                                logging.info(f"Non-square pixels detected. DAR: {dar}, SAR: {sar}. Adjusting width.")
                                width = int(height * dar)
                    except ValueError:
                        pass

                # Swap dimensions if rotated 90 or 270 degrees
                if abs(rotation) in [90, 270]:
                    logging.info(f"Video is rotated {rotation} degrees. Swapping dimensions.")
                    return (height, width)
                return (width, height)
        
        raise RuntimeError("Could not extract dimensions from ffprobe output")
        
    except subprocess.TimeoutExpired:
        logging.error(f"Timeout getting video dimensions for {video_path}")
        raise RuntimeError(f"Timeout getting video dimensions")
    except subprocess.CalledProcessError as e:
        logging.error(f"ffprobe failed for {video_path}: {e.stderr}")
        raise RuntimeError(f"ffprobe failed: {e.stderr}")
    except Exception as e:
        logging.error(f"Failed to get video dimensions for {video_path}: {e}")
        raise RuntimeError(f"Failed to get video dimensions: {e}")

def extract_last_frame(video_path: str, output_image_path: str, timeout: int = 60) -> bool:
    """
    Extracts the last frame from a video file using ffmpeg.

    This function first tries to get the video duration using ffprobe.
    If successful, it seeks to a point just before the end to extract the last frame.
    This method is generally faster than seeking from the end (-sseof).

    Args:
        video_path (str): Path to the input video file.
        output_image_path (str): Path to save the output image file.
        timeout (int): Timeout in seconds for the ffmpeg command.

    Returns:
        bool: True if the frame was extracted successfully, False otherwise.
    """
    try:
        # Get video duration using ffprobe
        duration = get_video_duration(video_path)

        if duration > 0:
            # Seek to 0.5 seconds before the end
            seek_position = max(0, duration - 0.5)
            
            logging.info(f"Extracting last frame from {video_path} using seek to {seek_position:.2f}s (duration: {duration:.2f}s)")

            ffmpeg_cmd = [
                'ffmpeg',
                '-y',
                '-nostdin',
                '-ss', str(seek_position),
                '-i', video_path,
                '-frames:v', '1',
                '-q:v', '2', # High quality
                output_image_path
            ]
        else:
            raise ValueError("Could not determine video duration.")

    except (ValueError, RuntimeError) as e:
        logging.warning(f"Could not get video duration ({e}), falling back to -sseof method.")
        # Fallback to original method if ffprobe fails
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',
            '-nostdin',
            '-sseof', '-0.1',
            '-i', video_path,
            '-frames:v', '1',
            '-q:v', '2',
            output_image_path
        ]

    try:
        logging.info(f"Running ffmpeg command: {' '.join(ffmpeg_cmd)}")
        process = subprocess.run(
            ffmpeg_cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            **get_subprocess_hidden_kwargs()
        )
        logging.info(f"Successfully extracted last frame to {output_image_path}")
        logging.debug(f"ffmpeg stdout: {process.stdout}")
        logging.debug(f"ffmpeg stderr: {process.stderr}")
        return True
    except subprocess.TimeoutExpired as e:
        logging.error(f"Timeout extracting last frame from {video_path}")
        if e.stdout:
            logging.error(f"ffmpeg stdout on timeout: {e.stdout}")
        if e.stderr:
            logging.error(f"ffmpeg stderr on timeout: {e.stderr}")
        return False
    except subprocess.CalledProcessError as e:
        logging.error(f"ffmpeg error extracting last frame from {video_path}: {e.stderr}")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred during frame extraction: {e}")
        return False