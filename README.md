# OpenFork DGN Client

The OpenFork DGN Client is the Python worker that turns a GPU into an OpenFork
provider. It connects to the website orchestrator, advertises compatible AI
services, downloads Docker images when needed, runs generation jobs, uploads
results, and reports heartbeats, cache state, credits, and monetize earnings.

Most users run this client through OpenFork Desktop. Developers and operators can
also run it directly for debugging, headless providers, or cloud images.

## Responsibilities

- Register a provider with the OpenFork orchestrator.
- Detect compatible services from local GPU/VRAM, RAM, CUDA, and storage budget.
- Pull and manage Docker images for model backends.
- Execute workflow processors for video, image, audio, speech, music, and utility
  jobs.
- Track cached and downloading images so the server can route jobs intelligently.
- Listen for jobs via Supabase Realtime when authenticated with OAuth tokens.
- Poll quickly in headless/API-key mode for cloud or dedicated provider images.
- Reset interrupted jobs on shutdown so work can return to the queue.

## Key Files

```text
client/
  cli.py                         Command-line entry point
  dgn_client.py                  Provider lifecycle and routing config
  config.py                      Supabase/orchestrator, timeout, cache settings
  services/                      Docker, orchestrator, listener, heartbeat code
  services/processors/           Workflow processor implementations
  utils/                         Logs, media helpers, shutdown handling
  workflows/                     ComfyUI/API workflow JSON files
  comfyui-storage/               Dockerfiles, compose files, backend API wrappers
  tests/                         Pytest coverage for routing/cache behavior
```

## Requirements

- Python 3.10+ recommended.
- Docker Engine with NVIDIA GPU support for local providers.
- NVIDIA driver and CUDA-compatible GPU.
- Enough disk space for selected OpenFork Docker images. Current images commonly
  require 70-220 GB each.

Windows users normally run the desktop app, which installs an OpenFork Ubuntu WSL2
engine and Docker environment automatically.

## Install For Development

```bash
cd client
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\python -m pip install -r requirements.txt
.\venv\Scripts\python -m pip install -r requirements-dev.txt
```

Linux/macOS shell:

```bash
./venv/bin/python -m pip install -r requirements.txt
./venv/bin/python -m pip install -r requirements-dev.txt
```

## Configuration

The production defaults point at `https://www.openfork.video`. Override these for
local development or staging:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_PUBLISHABLE_KEY=sb_publishable_...
SUPABASE_ANON_KEY=legacy_anon_jwt_if_needed
ORCHESTRATOR_URL_PROD=https://www.openfork.video
ORCHESTRATOR_URL_DEV=http://localhost:3000
DOCKER_IMAGE_CACHE_LIMIT_GB=250
DISK_PRESSURE_HEALTHY_GB=50
DISK_PRESSURE_CRITICAL_GB=20
```

Useful headless/cloud variables:

```env
HEADLESS_MODE=true
SELECTED_WORKFLOWS=wan22-text-to-video-8gb,qwen-image-edit-8gb
RUNPOD_POD_ID=...
VAST_CONTAINERLABEL=...
```

`config.py` also supports `OPENFORK_CONFIG_OVERRIDES_PATH`, a JSON file used by
desktop to push storage/cache settings into a running client.

## Authentication Modes

OAuth token mode is used by desktop:

```bash
python cli.py --access-token <SUPABASE_ACCESS_TOKEN> --refresh-token <SUPABASE_REFRESH_TOKEN>
```

API-key mode is used for headless providers and cloud instances:

```bash
python cli.py --dgn-api-key <DGN_API_KEY> --service wan22 --community-mode all
```

If OAuth tokens are not passed as CLI arguments, `cli.py` waits for an initial
`UPDATE_TOKENS` JSON message on stdin. Desktop uses that path so access tokens do
not show up in process lists.

## Routing Options

```text
--service auto|<service-name>       Run all compatible services or a single service.
--process-own-jobs                  Poll the user's own mine-policy jobs first.
--community-mode none               Private mode. Own jobs only when process-own is set.
--community-mode trusted_users      Accept jobs from trusted users.
--community-mode trusted_projects   Accept jobs from trusted projects.
--community-mode all                Accept public community jobs.
--allowed-targets id1,id2           Trusted user/project ids for trusted modes.
--monetize-mode                     Poll paid monetize jobs only.
--dgn-api-key <key>                 Headless authentication.
--root-dir <path>                   Client source/runtime root.
--data-dir <path>                   User data directory.
```

Desktop maps these flags to three primary modes:

- Private: `community_mode=none`, usually with `process_own_jobs=true`.
- Public: `community_mode=all`, optionally prioritizing own jobs.
- Monetize: `monetize_mode=true`, paid queue only.

Trusted Group is a Private sub-setting that selects `trusted_users` or
`trusted_projects` and sends `--allowed-targets`.

## Docker Image Cache

The client maintains a user-facing Docker image budget with LRU eviction:

- Healthy: more than 50 GB free, enforce the configured image budget.
- Pressure: 20-50 GB free, enforce budget and shorten idle cleanup.
- Critical: 20 GB or less free, evict aggressively and block new pulls until
  space recovers.

The server also tracks `cached_images` and `downloading_images` so providers do
not all download the same image for the same queue demand.

## Tests

```bash
python -m pytest
```

Focused tests live under `tests/` for model selection, Docker cache policy,
shutdown reset behavior, and workflow utilities.

## Relationship To Other Projects

- `../website` is the orchestrator, web app, Supabase schema, and admin surface.
- `../desktop` is the Electron UI that embeds this client for normal users.
