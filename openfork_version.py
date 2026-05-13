import os


CLIENT_VERSION = os.environ.get("OPENFORK_CLIENT_VERSION", "0.0.1")
DESKTOP_VERSION = os.environ.get("OPENFORK_DESKTOP_VERSION") or None
CLIENT_BUILD_SHA = os.environ.get("OPENFORK_BUILD_SHA") or None
try:
    DGN_PROTOCOL_VERSION = int(os.environ.get("OPENFORK_DGN_PROTOCOL_VERSION", "1"))
except ValueError:
    DGN_PROTOCOL_VERSION = 1

if os.environ.get("OPENFORK_CLIENT_KIND"):
    CLIENT_KIND = os.environ["OPENFORK_CLIENT_KIND"]
elif os.environ.get("RUNPOD_POD_ID") or os.environ.get("CONTAINER_ID"):
    CLIENT_KIND = "cloud"
elif DESKTOP_VERSION:
    CLIENT_KIND = "desktop"
else:
    CLIENT_KIND = "python"
