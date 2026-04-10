# GPU Radio Signal Processing (QB50 Satellites)

## Overview
This project implements a high-performance signal processing pipeline using CUDA for real-time (or accelerated) analysis of radio waves. It was designed to handle large datasets from SDR (Software Defined Radio) recordings with a focus on Doppler shift detection for Low Earth Orbit (LEO) satellites.

**Topic**: GPU FFT processing for radio waves (Implementation Difficulty: 6/10)

## Dataset
To run the full pipeline, download the sample SDR recording:
- **Download link**: [Zenodo Dataset](https://zenodo.org/records/6402965#.YkYSTIrtaCg)
- **Description**: This is an IQ recording of the QB50 satellite constellation, captured via SatNOGS. QB50 was a network of CubeSats used for thermosphere research. The recording is in stereo WAV format (3 Msps), where the left and right channels represent the I and Q components of the signal.

## Features
- **GPU accelerated FFT**: High-speed processing of interleaved IQ data using cuFFT.
- **Asynchronous Pipeline**: Overlapping Host-to-Device transfers, GPU computation, and Device-to-Host results using CUDA streams.
- **Streaming Architecture**: Optimized for stability with massive files (~16 GB+) using an incremental disk-writing mechanism to prevent OOM (Out Of Memory) issues.
- **CA-CFAR Detection**: Cell-Averaging Constant False Alarm Rate adaptive thresholding for signal detection in noisy environments.
- **Doppler Visualization**: Automated generation of high-contrast spectrograms and Doppler-zoomed waterfalls.

## Hardware Specifications (Target Machine)
This pipeline is optimized for the following configuration:
- **CPU**: 16-core system.
- **RAM**: 32 GB (31 GiB available).
- **GPU**: NVIDIA GeForce RTX 3050 Ti Laptop (Ampere architecture, 4GB VRAM).
- **Optimization**: The code specifically targets the available 31GB RAM by maintaining a memory footprint of ~8GB per 300s segment, preventing OOM crashes while maintaining high PCIe throughput.

## Visualization Guide
The pipeline generates three primary visualizations via `plot_spectrogram.py`:

### 1. `spectrogram.png` (Waterfall Spectrogram)
- **What it is**: A standard waterfall plot showing frequency vs. time.
- **Construction**: Aggregates FFT magnitudes from the whole segment. Values are normalized using the 1st and 99th percentiles to maximize contrast. 
- **Colormap**: Uses a custom `DOPPLER_CMAP` (dark blue to red) designed for highlighting faint signals against thermal noise.

### 2. `spectrogram_doppler.png` (Analysis Zoom)
- **What it is**: An automated zoom on the most "interesting" parts of the spectrum.
- **Construction**: A heuristic algorithm calculates the **variance and average power** for each frequency bin. Bins with high variability (indicating a passing satellite with Doppler shift) are automatically detected and cropped.
- **Purpose**: Directly identifies and visualizes satellite passes without manual searching.

### 3. `spectrum_avg.png` (Spectral Power & Persistence)
- **What it is**: A dual-plot showing the average spectral shape and detection statistics.
- **Top Plot**: Average magnitude (dB) across the segment. Shows the effect of the **GPU Bandpass Filter** (Blue: Raw, Red: Filtered).
- **Bottom Plot**: **CA-CFAR Persistence**. Shows the percentage of time a specific frequency exceeded the adaptive noise floor. Signals persistent for >30% of the time are flagged as detections.

## Installation & Usage
1. **Requirements**: CUDA Toolkit, CMake, Python 3 (NumPy, Pillow), ffmpeg.
2. **Build**:
   ```bash
   ./build.sh
   ```
3. **Run**:
   Place the QB50 `.wav` file in the project root and execute the segment processing script:
   ```bash
   ./process_qb50_segments.sh
   ```

---
*GPU Radio Project - 2026*
