"""
Wan2GP + LTX-2.3 post-install verification script.
Run by the Dockerfile RUN step — exits non-zero on any failure.
"""
import os
import sys
import pkgutil
import json

sys.path.insert(0, "/opt/wan2gp")

CKPTS = "/opt/wan2gp/ckpts"

DEFAULT_TRANSFORMER = "ltx-2.3-22b-distilled-Q8_0_light.gguf"


def env_flag(name):
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def required_transformers():
    configured = os.environ.get("LTX23_REQUIRED_TRANSFORMERS") or os.environ.get(
        "LTX23_REQUIRED_TRANSFORMER"
    )
    if not configured:
        return [DEFAULT_TRANSFORMER]
    return [name.strip() for name in configured.split(",") if name.strip()]


COMMON_REQUIRED_FILES = [
    f"{CKPTS}/ltx-2.3-22b_vae.safetensors",
    f"{CKPTS}/ltx-2.3-22b_audio_vae.safetensors",
    f"{CKPTS}/ltx-2.3-22b_vocoder.safetensors",
    f"{CKPTS}/ltx-2.3-22b_text_embedding_projection.safetensors",
    f"{CKPTS}/ltx-2.3-22b_embeddings_connector.safetensors",
    f"{CKPTS}/ltx-2.3-22b-distilled-lora-384.safetensors",
    f"{CKPTS}/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    f"{CKPTS}/ltx-2.3-temporal-upscaler-x2-1.0.safetensors",
    f"{CKPTS}/gemma-3-12b-it-qat-q4_0-unquantized/gemma-3-12b-it-qat-q4_0-unquantized.safetensors",
    f"{CKPTS}/gemma-3-12b-it-qat-q4_0-unquantized/gemma-3-12b-it-qat-q4_0-unquantized_quanto_bf16_int8.safetensors",
    f"{CKPTS}/gemma-3-12b-it-qat-q4_0-unquantized/config_light.json",
    f"{CKPTS}/gemma-3-12b-it-qat-q4_0-unquantized/tokenizer.json",
    f"{CKPTS}/gemma-3-12b-it-qat-q4_0-unquantized/tokenizer.model",
]

SHARED_RUNTIME_FILES = [
    f"{CKPTS}/pose/dw-ll_ucoco_384.onnx",
    f"{CKPTS}/pose/yolox_l.onnx",
    f"{CKPTS}/scribble/netG_A_latest.pth",
    f"{CKPTS}/flow/raft-things.pth",
    f"{CKPTS}/depth/depth_anything_v2_vitl.pth",
    f"{CKPTS}/depth/depth_anything_v2_vitb.pth",
    f"{CKPTS}/wav2vec/config.json",
    f"{CKPTS}/wav2vec/feature_extractor_config.json",
    f"{CKPTS}/wav2vec/model.safetensors",
    f"{CKPTS}/wav2vec/preprocessor_config.json",
    f"{CKPTS}/wav2vec/special_tokens_map.json",
    f"{CKPTS}/wav2vec/tokenizer_config.json",
    f"{CKPTS}/wav2vec/vocab.json",
    f"{CKPTS}/chinese-wav2vec2-base/config.json",
    f"{CKPTS}/chinese-wav2vec2-base/pytorch_model.bin",
    f"{CKPTS}/chinese-wav2vec2-base/preprocessor_config.json",
    f"{CKPTS}/roformer/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    f"{CKPTS}/roformer/model_bs_roformer_ep_317_sdr_12.9755.yaml",
    f"{CKPTS}/roformer/download_checks.json",
    f"{CKPTS}/pyannote/pyannote_model_wespeaker-voxceleb-resnet34-LM.bin",
    f"{CKPTS}/pyannote/pytorch_model_segmentation-3.0.bin",
    f"{CKPTS}/det_align/detface.pt",
    f"{CKPTS}/mask/sam_vit_h_4b8939_fp16.safetensors",
    f"{CKPTS}/mask/matanyone.safetensors",
    f"{CKPTS}/mask/config.json",
    f"{CKPTS}/rife4.26.pkl",
]

HDR_REQUIRED_FILES = [
    f"{CKPTS}/ltx-2.3-22b-ic-lora-hdr-0.9.safetensors",
    f"{CKPTS}/ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors",
]

REQUIRED_FILES = [
    *(f"{CKPTS}/{name}" for name in required_transformers()),
    *COMMON_REQUIRED_FILES,
]

VERIFY_SHARED_RUNTIME_FILES = env_flag("LTX23_VERIFY_SHARED_RUNTIME_FILES")
if VERIFY_SHARED_RUNTIME_FILES:
    REQUIRED_FILES.extend(SHARED_RUNTIME_FILES)

REQUIRE_HDR_FILES = env_flag("LTX23_REQUIRE_HDR")
if REQUIRE_HDR_FILES:
    REQUIRED_FILES.extend(HDR_REQUIRED_FILES)

# Config file may vary by repo revision — match any *config*.json under ckpts/
def find_config_json(root):
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if "config" in f and f.endswith(".json"):
                return os.path.join(dirpath, f)
    return None


def find_broken_safetensors_indexes(root):
    broken = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if not filename.endswith(".safetensors.index.json"):
                continue
            index_path = os.path.join(dirpath, filename)
            with open(index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            referenced = set(data.get("weight_map", {}).values())
            missing = sorted(
                name
                for name in referenced
                if not os.path.isfile(os.path.join(dirpath, name))
            )
            if missing:
                broken.append((index_path, missing))
    return broken

# ── List ckpts contents for diagnostics ───────────────────────────────────────
print("--- Contents of ckpts/ ---")
for dirpath, dirnames, filenames in os.walk(CKPTS):
    level = dirpath.replace(CKPTS, "").count(os.sep)
    indent = "  " * level
    print(f"{indent}{os.path.basename(dirpath)}/")
    for f in filenames:
        size_mb = os.path.getsize(os.path.join(dirpath, f)) / 1024 / 1024
        print(f"{indent}  {f}  ({size_mb:.1f} MB)")

# ── Check required files ──────────────────────────────────────────────────────
print("\n--- Checking required model files ---")
if VERIFY_SHARED_RUNTIME_FILES:
    print("Shared WanGP runtime file verification: enabled")
else:
    print(
        "Shared WanGP runtime file verification: skipped "
        "(set LTX23_VERIFY_SHARED_RUNTIME_FILES=1 to require it)"
    )
print(
    "HDR LoRA verification: "
    + ("enabled" if REQUIRE_HDR_FILES else "skipped (set LTX23_REQUIRE_HDR=1 to require it)")
)
missing = [f for f in REQUIRED_FILES if not os.path.isfile(f)]
if missing:
    for f in missing:
        print(f"MISSING: {f}")
    sys.exit(1)
print("Required model files: OK")

# ── Check safetensors shard indexes point only at local files ────────────────
print("\n--- Checking safetensors shard indexes ---")
broken_indexes = find_broken_safetensors_indexes(CKPTS)
if broken_indexes:
    for index_path, missing in broken_indexes:
        print(f"BROKEN INDEX: {index_path}")
        print(f"  Missing {len(missing)} referenced shard(s): {missing[:10]}")
    sys.exit(1)
print("Safetensors shard indexes: OK")

# ── Check config JSON (flexible match) ───────────────────────────────────────
print("\n--- Checking config JSON ---")
config_path = find_config_json(CKPTS)
if config_path:
    print(f"Config JSON found: {config_path}")
else:
    print(f"WARNING: no *config*.json found under {CKPTS}")
    print("Wan2GP may still work if it locates config via another path.")

# ── List available shared submodules (aids future debugging) ──────────────────
print("\n--- Checking shared.api import ---")
try:
    import shared
    mods = [m.name for m in pkgutil.iter_modules(shared.__path__)]
    print(f"shared submodules available: {mods}")
except Exception as e:
    print(f"WARNING: could not list shared submodules: {e}")

# ── Verify shared.api.init is importable ──────────────────────────────────────
try:
    from shared.api import init  # noqa: F401
    print("shared.api.init import: OK")
except ImportError as e:
    print(f"ERROR: shared.api import failed: {e}")
    print("Hint: the Wan2GP API surface may have changed. Check the submodule list above")
    print("      and update wan2gp_processor.py to match the available import path.")
    sys.exit(1)

print("\nAll Wan2GP checks passed.")
