# run

python3 dgn_client.py

The script will check for Docker and NVIDIA drivers.
It will profile the hardware and print the profile.
It will build the Docker image locally (since no registry is set up for Part 1).
It will run the ComfyUI workflow in the container, producing an output video in the output/ directory.
Output:
The output video will be saved in the output/ directory with a filename like 2025-07-25/wanvid_XXXX.mp4 (based on the VHS_VideoCombine node's configuration).
