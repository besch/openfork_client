# DGN Client

This is a DGN client that uses Docker to run a ComfyUI workflow in a containerized environment.

## Setup

1.  Install Docker on your system.
2.  Install Python 3 and pip.
3.  Install the required Python packages:

    ```
    pip install -r requirements.txt
    ```

## Usage

To run the DGN client, execute the following command:

```
python dgn_client.py
```

This will:

1.  Build the Docker image with ComfyUI and the necessary dependencies.
2.  Run the Docker container.
3.  Trigger the ComfyUI workflow.
4.  Stop the Docker container when the workflow is complete.