#!/bin/bash
# ACE-Step Diagnostic Tool

echo "=== ACE-Step Diagnostic ==="
echo "Time: $(date)"
echo ""

echo "--- System Info ---"
python --version
pip --version
echo ""

echo "--- CUDA Check ---"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
nvidia-smi -L || echo "NVIDIA SMI not found"
echo ""

echo "--- Network check ---"
curl -s -I https://huggingface.co | grep -i server || echo "HuggingFace not reachable"
echo ""

echo "--- ACE-Step Pipeline Check ---"
python -c "from acestep.acestep_v15_pipeline import AceStepV15Pipeline; print('✓ Pipeline import OK')" || echo "✗ Pipeline import FAILED"
echo ""

echo "--- Model Checkpoints ---"
ls -R checkpoints || echo "Checkpoints directory empty or missing"
echo ""

echo "--- API Connectivity ---"
curl -s http://localhost:8000/health || echo "API not responding on port 8000"
echo ""

echo "=== Diagnostic Complete ==="
