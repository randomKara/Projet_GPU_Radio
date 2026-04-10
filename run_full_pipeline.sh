#!/bin/bash
set -e

CUDA_PATH=/usr/local/cuda-13.0
export CUDA_PATH
export PATH=$CUDA_PATH/bin:$PATH

echo "=========================================="
echo "GPU Radio Signal Processing Full Pipeline"
echo "=========================================="
echo ""

if [ ! -f "signal.bin" ]; then
    echo "[1/4] Generating signal..."
    python3 generate_signal.py
    echo "✓ Signal generated"
else
    echo "[1/4] Using existing signal.bin"
fi

echo ""
echo "[2/4] Building CUDA application..."
if [ ! -d "build" ]; then
    mkdir build
fi

cd build
cmake -DCMAKE_CUDA_COMPILER=$CUDA_PATH/bin/nvcc .. > /dev/null 2>&1
make > /dev/null 2>&1
cd ..

echo "✓ Build completed"

echo ""
echo "[3/4] Processing signal on GPU..."
./build/radio_fft | grep -E "(Processing batches|Performance Metrics|Throughput|Spectrogram saved)"

echo ""
echo "[4/4] Generating visualization..."
python3 plot_spectrogram.py | grep -E "(Loading|Spectrogram saved|PASS|MHz)"

echo ""
echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  • signal.bin              (input signal)"
echo "  • output_spectrogram.bin  (FFT results)"
echo "  • spectrogram.png         (visualization)"
echo ""
echo "View the spectrogram with: xdg-open spectrogram.png"
