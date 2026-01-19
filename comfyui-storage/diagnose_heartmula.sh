#!/bin/bash
# HeartMuLa Diagnostic Script
# Run this script to diagnose why the API is not loading properly

echo "========================================"
echo "HeartMuLa Diagnostic Report"
echo "========================================"
echo ""

# 1. Check if API is running
echo "1. Checking if HeartMuLa API process is running..."
if pgrep -f "heartmula_api.py" > /dev/null; then
    echo "✓ HeartMuLa API process is running"
    echo "  PID(s): $(pgrep -f heartmula_api.py | tr '\n' ' ')"
else
    echo "✗ HeartMuLa API process is NOT running"
fi
echo ""

# 2. Check port 8000
echo "2. Checking if port 8000 is listening..."
if netstat -tln 2>/dev/null | grep -q ":8000 " || ss -tln 2>/dev/null | grep -q ":8000 "; then
    echo "✓ Port 8000 is listening"
else
    echo "✗ Port 8000 is NOT listening"
fi
echo ""

# 3. Check API logs
echo "3. Checking HeartMuLa API logs..."
if [ -f "/tmp/heartmula_api.log" ]; then
    echo "✓ Log file exists: /tmp/heartmula_api.log"
    echo "  Size: $(du -h /tmp/heartmula_api.log | cut -f1)"
    echo ""
    echo "--- LAST 50 LINES OF LOG ---"
    tail -n 50 /tmp/heartmula_api.log
    echo "--- END OF LOG ---"
    echo ""
    
    # Check for common errors
    echo "4. Analyzing logs for common issues..."
    
    if grep -q "CUDA out of memory" /tmp/heartmula_api.log; then
        echo "⚠️  FOUND: CUDA out of memory errors"
        echo "   This means your GPU doesn't have enough VRAM"
        echo "   Solutions:"
        echo "   - Use 4-bit quantization (set HEARTMULA_QUANTIZATION=4bit)"
        echo "   - Use a smaller model"
        echo "   - Use CPU offloading"
    fi
    
    if grep -q "Killed" /tmp/heartmula_api.log; then
        echo "⚠️  FOUND: Process was killed (likely OOM)"
        echo "   The system killed the process due to memory pressure"
        echo "   Solutions:"
        echo "   - Check system memory: free -h"
        echo "   - Increase swap space"
        echo "   - Use smaller model or quantization"
    fi
    
    if grep -q "FileNotFoundError" /tmp/heartmula_api.log; then
        echo "⚠️  FOUND: Missing files"
        echo "   Some model files are missing"
        grep "FileNotFoundError" /tmp/heartmula_api.log | tail -3
    fi
    
    if grep -q "ImportError\|ModuleNotFoundError" /tmp/heartmula_api.log; then
        echo "⚠️  FOUND: Missing Python dependencies"
        echo "   Some required packages are not installed"
        grep -E "ImportError|ModuleNotFoundError" /tmp/heartmula_api.log | tail -3
    fi
    
    if grep -q "Loading checkpoint shards" /tmp/heartmula_api.log; then
        echo "ℹ️  Model is loading checkpoint shards..."
        grep "Loading checkpoint shards" /tmp/heartmula_api.log | tail -1
    fi
    
    if ! grep -q "HeartMuLa pipeline loaded successfully" /tmp/heartmula_api.log; then
        echo "⚠️  Model has NOT completed loading yet"
        echo "   Last status line:"
        grep -E "Loading|Initializing|Starting" /tmp/heartmula_api.log | tail -1
    else
        echo "✓ Model loaded successfully"
    fi
    
else
    echo "✗ Log file NOT found: /tmp/heartmula_api.log"
    echo "  The API may not have started at all"
fi
echo ""

# 5. Check model files
echo "5. Checking model files..."
if [ -d "/app/ckpt" ]; then
    echo "✓ Model directory exists: /app/ckpt"
    echo ""
    echo "Directory structure:"
    ls -lh /app/ckpt/ 2>/dev/null | head -20
    echo ""
    
    # Check for required files
    if [ -f "/app/ckpt/gen_config.json" ]; then
        echo "✓ gen_config.json found"
    else
        echo "✗ gen_config.json NOT found"
    fi
    
    if [ -f "/app/ckpt/tokenizer.json" ]; then
        echo "✓ tokenizer.json found"
    else
        echo "✗ tokenizer.json NOT found"
    fi
    
    if [ -d "/app/ckpt/HeartMuLa-oss-3B" ]; then
        echo "✓ HeartMuLa-oss-3B directory found"
        echo "  Files: $(ls /app/ckpt/HeartMuLa-oss-3B | wc -l)"
    else
        echo "✗ HeartMuLa-oss-3B directory NOT found"
    fi
    
    if [ -d "/app/ckpt/HeartCodec-oss" ]; then
        echo "✓ HeartCodec-oss directory found"
        echo "  Files: $(ls /app/ckpt/HeartCodec-oss | wc -l)"
    else
        echo "✗ HeartCodec-oss directory NOT found"
    fi
else
    echo "✗ Model directory NOT found: /app/ckpt"
fi
echo ""

# 6. Check Python environment
echo "6. Checking Python environment..."
if python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null; then
    python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
    python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
        python3 -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
        python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
        python3 -c "import torch; print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')"
    fi
else
    echo "✗ PyTorch not available"
fi
echo ""

if python3 -c "import heartlib" 2>/dev/null; then
    echo "✓ heartlib is installed"
else
    echo "✗ heartlib is NOT installed"
fi
echo ""

# 7. Check system resources
echo "7. Checking system resources..."
echo "Memory:"
free -h
echo ""
echo "GPU Memory:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv
else
    echo "nvidia-smi not available"
fi
echo ""

# 8. Check for zombie processes
echo "8. Checking for zombie processes..."
if ps aux | grep -E "heartmula|python.*api" | grep -v grep; then
    echo "Found related processes:"
    ps aux | grep -E "heartmula|python.*api" | grep -v grep
else
    echo "No related processes found"
fi
echo ""

# 9. Recommendations
echo "========================================"
echo "RECOMMENDATIONS"
echo "========================================"
echo ""

if [ ! -f "/tmp/heartmula_api.log" ]; then
    echo "1. Start the HeartMuLa API first:"
    echo "   cd /app && python3 heartmula_api.py > /tmp/heartmula_api.log 2>&1 &"
    echo ""
elif ! grep -q "HeartMuLa pipeline loaded successfully" /tmp/heartmula_api.log 2>/dev/null; then
    echo "1. The model is still loading or failed to load"
    echo "   - Wait longer (model can take 10-20 minutes to load)"
    echo "   - Check the log for specific errors: tail -f /tmp/heartmula_api.log"
    echo ""
fi

# Check if heartlib is installed
if ! python3 -c "import heartlib" 2>/dev/null; then
    echo "2. Install heartlib:"
    echo "   cd /app/heartlib_repo && pip install -e ."
    echo ""
fi

# Check if models are downloaded
if [ ! -d "/app/ckpt/HeartMuLa-oss-3B" ]; then
    echo "3. Download models:"
    echo "   huggingface-cli download HeartMuLa/HeartMuLaGen --local-dir /app/ckpt"
    echo "   huggingface-cli download HeartMuLa/HeartMuLa-oss-3B --local-dir /app/ckpt/HeartMuLa-oss-3B"
    echo "   huggingface-cli download HeartMuLa/HeartCodec-oss --local-dir /app/ckpt/HeartCodec-oss"
    echo ""
fi

echo "4. To monitor the API loading in real-time:"
    echo "   tail -f /tmp/heartmula_api.log"
echo ""

echo "5. To test the API manually:"
    echo "   curl http://localhost:8000/health"
echo ""

echo "========================================"
echo "End of Diagnostic Report"
echo "========================================"
