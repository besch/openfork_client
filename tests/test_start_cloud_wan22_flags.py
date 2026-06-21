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
    assert "--lowvram --fp16-vae --reserve-vram 1.0" in wan22_block


def test_start_cloud_ltx23_wan2gp_tiers_have_explicit_runtime_flags():
    text = SCRIPT.read_text(encoding="utf-8")
    ltx_start = text.index('[[ "${SERVICE_TYPE:-}" == *"ltx23"* ]]')
    ltx_end = text.index('[[ "${SERVICE_TYPE:-}" == *"wan22-wan2gp"* ]]', ltx_start)
    ltx_block = text[ltx_start:ltx_end]

    assert '--perc-reserved-mem-max 0.55 --vram-safety-coefficient 0.70' in ltx_block
    assert '--profile 4 --attention sdpa --perc-reserved-mem-max 0.55' in ltx_block
    assert '--profile 4.5 --attention sdpa --perc-reserved-mem-max 0.45' in ltx_block
