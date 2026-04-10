#!/bin/bash


CUDA_PATH=/usr/local/cuda-13.0
export PATH=$CUDA_PATH/bin:$PATH

echo "=========================================="
echo "CUDA Streams Profiling & Analysis"
echo "=========================================="
echo ""

if ! command -v nsys &> /dev/null; then
    echo "WARNING: nsys not found in PATH"
    echo "Install NVIDIA nsys: part of CUDA Toolkit"
    echo "Expected path: $CUDA_PATH/bin/nsys"
    exit 1
fi

echo "[1] Running profiling..."
nsys profile \
    --stats=true \
    --trace=cuda,cudadrv,osrt \
    --output=profiling_report \
    --force-overwrite true \
    ./build/radio_fft > /dev/null 2>&1

if [ $? -ne 0 ]; then
    echo "ERROR: nsys profile failed"
    exit 1
fi

echo "✓ Profiling completed (output: profiling_report.nsys-rep)"
echo ""

echo "[2] Analyzing stream overlap..."

python3 << 'ANALYSIS_PYTHON'
import subprocess
import re

result = subprocess.run(['nsys', 'export', '--type', 'sqlite', 
                        'profiling_report.nsys-rep', 
                        '--output', 'profiling_report.sqlite'],
                       capture_output=True, text=True)

if result.returncode == 0:
    print("✓ Generated profiling database")
    print("")
    print("--- Stream Overlap Analysis ---")
    print("")
    print("To visualize streams overlap:")
    print("  1. nsys-ui profiling_report.nsys-rep  (GUI visualization)")
    print("  2. Check CUDA/Streams view for H2D, Compute, D2H timeline")
    print("")
    print("Key metrics to look for:")
    print("  • H2D transfer (stream 1) should overlap with GPU compute")
    print("  • D2H transfer (stream 3) should overlap with next batch H2D")
    print("  • Look for gaps in GPU timeline (indicates synchronization stalls)")
    print("")
    print("Expected pattern (ideal overlap):")
    print("  ┌─────────┐")
    print("  │ H2D(1)  │")
    print("  ├─────────┼─────────┐")
    print("  │ COMPUTE │ H2D(2)  │")
    print("  ├─────────┼─────────┼─────────┐")
    print("  │ D2H(1)  │ COMPUTE │ H2D(3)  │")
    print("  └─────────┴─────────┴─────────┘")
else:
    print("Profiling database export not available")
    print("Use nsys-ui for visualization: nsys-ui profiling_report.nsys-rep")

ANALYSIS_PYTHON

echo ""
echo "=========================================="
echo "Profiling Summary"
echo "=========================================="
echo "Report files generated:"
echo "  • profiling_report.nsys-rep  (raw profiling data)"
echo "  • profiling_report.sqlite    (database format)"
echo ""
echo "To view results:"
echo "  nsys-ui profiling_report.nsys-rep"
echo "=========================================="
