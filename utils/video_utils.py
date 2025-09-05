import ffmpeg
import logging
import os

def generate_thumbnail(video_path: str, thumbnail_path: str, timestamp: str = "00:00:01.000"):
    """
    Generates a thumbnail from a video file using ffmpeg-python by piping the
    output directly to a file, avoiding filesystem race conditions.
    """
    try:
        process = (
            ffmpeg
            .input(video_path, ss=timestamp)
            .output('pipe:', format='image2', vcodec='mjpeg', vframes=1)
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            logging.error(f"ffmpeg failed to generate thumbnail for {video_path}: {stderr.decode()}")
            return False

        with open(thumbnail_path, 'wb') as f:
            f.write(stdout)

        logging.info(f"Thumbnail generated for {video_path} at {thumbnail_path}")
        return True
    except ffmpeg.Error as e:
        logging.error(f"ffmpeg error during thumbnail generation setup for {video_path}: {e.stderr.decode()}")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred during thumbnail generation for {video_path}: {e}")
        return False

def get_video_duration(video_path: str) -> float:
    """Gets the duration of a video file in seconds using ffmpeg.probe."""
    try:
        probe = ffmpeg.probe(video_path)
        duration = float(probe['format']['duration'])
        return duration
    except ffmpeg.Error as e:
        logging.error(f"ffmpeg error during duration probe for {video_path}: {e.stderr.decode() if e.stderr else 'Unknown ffmpeg error'}")
        return 0.0
    except (KeyError, ValueError) as e:
        logging.error(f"Could not parse duration for {video_path}: {e}")
        return 0.0

def find_video_in_output(outputs: dict):
    """Parses ComfyUI workflow output to find the first video file."""
    if not isinstance(outputs, dict):
        return None
    for node_id, node_output in outputs.items():
        if isinstance(node_output, dict):
            # Check for 'files' key (standard ComfyUI output)
            if 'files' in node_output:
                file_list = node_output['files']
            # Check for 'gifs' key (observed in some video workflows)
            elif 'gifs' in node_output:
                file_list = node_output['gifs']
            else:
                continue # No relevant file list found in this node output

            for file_info in file_list:
                if isinstance(file_info, dict) and file_info.get('type') == 'output' and \
                   isinstance(file_info.get('filename'), str) and \
                   file_info.get('filename', '').lower().endswith(('.mp4', '.webm', '.gif')):
                    return file_info['filename'], file_info.get('subfolder', '')
    return None

def generate_placeholder_video(output_dir: str, job_id: str) -> str:
    """Generates a placeholder video using ffmpeg."""
    placeholder_filename = f"placeholder_{job_id}.mp4"
    placeholder_path = os.path.join(output_dir, placeholder_filename)
    
    # Create a simple 1-second black video with a timestamp
    try:
        (
            ffmpeg
            .input('color=c=black:s=1280x720:r=30', f='lavfi', t=1)
            .output(placeholder_path, vcodec='libx264', pix_fmt='yuv420p')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        logging.info(f"Generated placeholder video at {placeholder_path}")
        return placeholder_filename
    except ffmpeg.Error as e:
        logging.error(f"ffmpeg failed to generate placeholder video: {e.stderr.decode()}")
        return None