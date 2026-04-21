"""
Post-download helper: symlink all files from the HF hub cache snapshot into
/opt/wan2gp/ckpts/ so that both paths work:

  1. hf_hub_download("DeepBeepMeep/LTX-2", "file.gguf")  → HF cache ✓
  2. open("/opt/wan2gp/ckpts/file.gguf")                  → symlink → blob ✓

Each ckpts/ entry resolves through the snapshot symlink to the actual blob,
so no extra disk space is used beyond the HF cache itself.
"""

import os
import sys
from pathlib import Path


HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
CACHE_ROOT = HF_HOME / "hub"
MODEL_DIR = CACHE_ROOT / "models--DeepBeepMeep--LTX-2"
SNAPSHOTS_DIR = MODEL_DIR / "snapshots"
CKPTS = Path("/opt/wan2gp/ckpts")


def symlink_tree(src: Path, dst: Path) -> None:
    """Recursively create symlinks in dst mirroring src."""
    dst.mkdir(parents=True, exist_ok=True)
    for item in sorted(src.iterdir()):
        target = dst / item.name
        if target.exists() or target.is_symlink():
            continue
        if item.is_dir():
            symlink_tree(item, target)
        else:
            # Resolve through any existing symlinks to the actual blob file
            actual = item.resolve()
            target.symlink_to(actual)
            print(f"  linked {target.name} -> {actual}")


def main() -> None:
    if not SNAPSHOTS_DIR.exists():
        print(f"ERROR: No HF snapshots found at {SNAPSHOTS_DIR}", file=sys.stderr)
        sys.exit(1)

    snaps = sorted(SNAPSHOTS_DIR.iterdir())
    if not snaps:
        print(f"ERROR: snapshots dir is empty: {SNAPSHOTS_DIR}", file=sys.stderr)
        sys.exit(1)

    snap = snaps[-1]
    print(f"Using snapshot: {snap.name}")
    symlink_tree(snap, CKPTS)
    print(f"ckpts/ populated from HF cache snapshot.")


if __name__ == "__main__":
    main()
