# OpenFork DGN Client (CLI)

**The Engine Under the Hood.**

This is the Python core that powers the OpenFork Desktop app. It connects to [openfork.video](https://openfork.video) to execute various AI video/image/audio workflows.

It handles the heavy lifting:
1.  **Workflow Automation**: Automatically pulls and runs specialized Docker containers for AI models (ComfyUI, Wan2.1, etc).
2.  **Job Processing**: Listens for generation requests from the web platform and executes them.
3.  **Compute Sharing**: Manages the logic for local generation vs. network distributed processing.

**Note**: Most users should use the **[Desktop App](../desktop/README.md)** for a visual experience. This CLI is strictly for headless servers or developers.

---

### Developer Usage

**Prerequisites**: Python 3.10+, Docker Desktop.

**Installation**:
```bash
pip install -r requirements.txt
```

**Running**:
```bash
python cli.py --access-token <TOKEN> --refresh-token <REFRESH_TOKEN>
```