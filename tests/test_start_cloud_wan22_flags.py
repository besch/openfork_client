from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "start_cloud.sh"


def test_start_cloud_wan22_comfyui_tiers_keep_vae_on_gpu():
    text = SCRIPT.read_text(encoding="utf-8")
    wan22_start = text.index("wan22|wan22-8gb)")
    wan22_end = text.index("*ltx2*-8gb*", wan22_start)
    wan22_block = text[wan22_start:wan22_end]

    assert "wan22-16gb)" in wan22_block
    assert "wan22-24gb)" in wan22_block
    assert "--cpu-vae" not in wan22_block
    assert "--lowvram --fp16-vae --reserve-vram 0.5" in wan22_block
    assert "--lowvram --fp16-vae --reserve-vram 0.75" in wan22_block
    assert "--normalvram --fp16-vae --reserve-vram 1.0" in wan22_block
