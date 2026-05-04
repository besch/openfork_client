# OpenFork DGN Client

The OpenFork DGN Client is a command-line tool that works with the [openfork.video](https://openfork.video) website. It allows users to contribute their computer's processing power to the OpenFork network to collaboratively create videos.

This directory contains the source code for the OpenFork Distributed GPU Network (DGN) Client. This is a Python application that allows users to contribute their computer's processing power to the OpenFork network to collaboratively create videos.

## How it Works

The DGN Client connects to the central OpenFork orchestrator, authenticates using a user's account, and waits for video processing jobs. When a job is received, the client:

1.  Downloads the necessary AI model and dependencies into a Docker container.
2.  Executes the AI workflow (e.g., text-to-video, video-foley) with the parameters specified in the job.
3.  Uploads the resulting video or audio back to the OpenFork platform.
4.  Waits for the next job.

The client is designed to be run in the background, managed primarily by the [OpenFork Desktop application](../desktop/README.md), which provides a user-friendly interface for starting, stopping, and configuring the client.

## Core Components

-   **`dgn_client.py`**: The main class that manages the connection to the orchestrator and job lifecycle.
-   **`cli.py`**: The command-line entry point for starting the client and passing configuration arguments.
-   **`services/`**: Contains modules for interacting with external services:
    -   `orchestrator_service.py`: Handles all API communication with the main OpenFork server.
    -   `comfyui_service.py`: Interacts with ComfyUI workflows.
    -   `docker_manager.py`: Manages starting and stopping the Docker containers required for different AI models.
    -   `job_processors.py`: Contains the logic for handling specific types of generation jobs.
-   **`workflows/`**: Contains the ComfyUI workflow definitions in JSON format.

## Installation

1.  Install Python 3.9 or higher.
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Install Docker Desktop and ensure it is running.

### Environment Variables

The client requires the following Supabase configuration:

-   `SUPABASE_URL`: Your Supabase project URL (default: OpenFork production).
-   `SUPABASE_PUBLISHABLE_KEY`: Your Supabase publishable key (`sb_publishable_...` format). This is used for Realtime WebSocket connections. You can also use `NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY`.
-   `SUPABASE_ANON_KEY`: Legacy anon key (JWT format). Fallback for older Supabase projects.

If these are not set, Realtime job notifications will be disabled and the client will fall back to polling.

## Usage (Standalone)

While the client is intended to be run by the desktop application, it can be run directly from the command line for development or debugging purposes.

You will need to provide valid Supabase access and refresh tokens.

```bash
python cli.py --access-token <YOUR_ACCESS_TOKEN> --refresh-token <YOUR_REFRESH_TOKEN>
```

### Command-Line Arguments

-   `--access-token`: (Required) Your Supabase auth access token.
-   `--refresh-token`: (Required) Your Supabase auth refresh token.
-   `--service`: The specific service to run (e.g., `wan22`, `foley`). Defaults to `auto`, which allows the client to accept any job type.
-   `--accept-policy`: Defines which jobs to accept.
    -   `mine` (default): Only accept jobs from your own projects.
    -   `all`: Accept jobs from any public project.
    -   `project`: Accept jobs only from specific projects (requires `--allowed-targets`).
    -   `users`: Accept jobs only from specific users (requires `--allowed-targets`).
-   `--allowed-targets`: A comma-separated list of project slugs or user IDs for the `project` or `users` policies.