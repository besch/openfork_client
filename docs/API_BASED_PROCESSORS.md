# API-Based Job Processor - Setup Guide

## Overview

The OpenFork DGN client now supports Docker images that expose REST APIs instead of ComfyUI workflows. This guide explains how to configure and use API-based processors.

## How It Works

### Traditional ComfyUI Workflow
1. Start Docker container running ComfyUI
2. Connect to WebSocket at `ws://localhost:8188`
3. Submit workflow JSON
4. Wait for completion via WebSocket
5. Copy output files from container

### API-Based Workflow  
1. Start Docker container exposing HTTP API
2. Wait for API health check at `http://localhost:PORT/health`
3. Submit POST request to API endpoint with JSON payload
4. Receive response with output file path
5. Copy output files from container

## Supported Patterns

The framework supports common API patterns:

### 1. RunPod Serverless Style
```python
# Request
POST http://localhost:8000/run
{
    "input": {
        "prompt": "A cinematic video...",
        "negative_prompt": "blurry...",
        "width": 832,
        "height": 480
    }
}

# Response
{
    "output": "/output/generated_video.mp4"
}
```

### 2. Direct API Style
```python
# Request
POST http://localhost:8000/generate
{
    "prompt": "A cinematic video...",
    "params": {
        "width": 832,
        "height": 480
    }
}

# Response
{
    "video_path": "/workspace/output/video.mp4",
    "duration": 5.0
}
```

## Finding API Documentation

To use an API-based Docker image, you need to know:

### 1. **API Port**
The port the container exposes for HTTP requests.

**How to find:**
- Check `docker run` command for `-p` flags
- Inspect container: `docker inspect <container_id> | grep ExposedPorts`
- Common ports: `8000`, `8080`, `5000`, `7860`

### 2. **API Endpoints**
The URL paths for submitting jobs.

**How to find:**
- Look for documentation in Docker Hub description
- Check GitHub repository README
- Common endpoints:
  - `/run` or `/runsync` (RunPod style)
  - `/generate` or `/infer`
  - `/text-to-video` or `/image-to-video`

### 3. **Request Payload Format**
The JSON structure expected by the API.

**How to find:**
- Check API documentation
- Look at example curl commands
- Inspect handler.py or main.py in the image
- Common formats:
  ```json
  {"input": {...}}  // RunPod style
  {"prompt": "...", "params": {...}}  // Direct style
  ```

### 4. **Response Format**
How the API returns the output file path.

**How to find:**
- Check API documentation
- Common patterns:
  ```json
  {"output": "path"}
  {"output": {"video": "path"}}
  {"result": "path"}
  {"video_path": "path"}
  ```

## Example: Testing an Unknown Docker Image

```bash
# 1. Pull and run the image
docker pull antilopax/wan22:latest
docker run -d -p 8000:8000 --name test-wan22 antilopax/wan22:latest

# 2. Check if it's running
docker ps

# 3. Try to find API endpoints
docker exec test-wan22 ls -la /
docker exec test-wan22 cat /app/handler.py  # If it exists
docker exec test-wan22 cat /workspace/README.md  # If it exists

# 4. Test health endpoint
curl http://localhost:8000/health

# 5. Try common endpoints
curl -X POST http://localhost:8000/run -H "Content-Type: application/json" -d '{"input": {"prompt": "test"}}'
curl -X POST http://localhost:8000/generate -H "Content-Type: application/json" -d '{"prompt": "test"}'

# 6. Check container logs for hints
docker logs test-wan22

# 7. Cleanup
docker stop test-wan22
docker rm test-wan22
```

## Creating a Custom Processor

Once you have the API information, create a processor class:

```python
from services.processors.api_based import APIBasedJobProcessor

class YourCustomProcessor(APIBasedJobProcessor):
    @property
    def api_port(self) -> int:
        return 8000  # Your API port
    
    @property
    def api_endpoint(self) -> str:
        return "/run"  # Your API endpoint
    
    def prepare_api_payload(self) -> Dict:
        return {
            "input": {
                "prompt": self.positive_prompt,
                # Add other parameters...
            }
        }
    
    def extract_output_from_response(self, response: Dict) -> Union[str, None]:
        # Extract the output path from response
        return response.get("output")
```

## Registering the Processor

Add your processor to `services.json`:

```json
{
  "workflows": {
    "your-workflow-name": {
      "service_name": "your-service",
      "workflow_file": "dummy.json",  // Not used for API-based
      "processor": "YourCustomProcessor"
    }
  }
}
```

## Troubleshooting

### Container starts but API times out
- Check if the container is actually running the API: `docker logs <container>`
- Verify the port mapping: `docker port <container>`
- Try different health check endpoints: `/health`, `/ping`, `/status`

### API returns error
- Check the request payload format
- Review container logs: `docker logs <container>`
- Verify required parameters are provided

### Output file not found
- Check the response format
- Verify the file path is absolute
- Check if output directory exists in container: `docker exec <container> ls /output`

## Next Steps for antilopax/wan22

To use `antilopax/wan22`, you need to:

1. **Find or contact the image creator** to get:
   - API documentation
   - Example usage
   - Expected payload format

2. **Or reverse-engineer it yourself:**
   - Run the container and inspect it
   - Check for a GitHub repository
   - Look for handler.py or similar files

3. **Alternative:** Use a well-documented image like:
   - `camenduru/wan-2-1-i2v-comfyui:fp8` (ComfyUI-based)
   - Other community images with clear documentation

## Contact & Support

If you find documentation for `antilopax/wan22`, update the `WAN22APITextToVideoJobProcessor` class in `services/processors/api_based.py` with the correct:
- `api_port`
- `api_endpoint`
- `prepare_api_payload()` format
- `extract_output_from_response()` logic
