

import numpy as np
import subprocess, json, sys, os, argparse
from pathlib import Path

def get_audio_info(audio_path: str):

    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", audio_path],
        capture_output=True, text=True
    )
    info = json.loads(result.stdout)
    stream = info["streams"][0]
    return {
        "sample_rate": int(stream["sample_rate"]),
        "n_channels": int(stream["channels"]),
        "duration": float(stream.get("duration", 0))
    }

def extract_segment_to_file(audio_path: str, out_path: str, start_sec: float, duration: float):

    cmd = ["ffmpeg", "-y", "-v", "warning",
           "-ss", str(start_sec),
           "-i", audio_path,
           "-t", str(duration),
           "-f", "f32le",
           "-acodec", "pcm_f32le",
           out_path]
    subprocess.run(cmd, check=True)
    return os.path.getsize(out_path)

def main():
    parser = argparse.ArgumentParser(description="Convert audio/IQ to GPU-compatible signal.bin")
    parser.add_argument("input", help="Input file (.ogg, .wav, etc.)")
    parser.add_argument("output", nargs="?", default="signal.bin", help="Output .bin file")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--duration", type=float, default=3600, help="Duration in seconds")
    parser.add_argument("--center-freq", type=float, default=None, help="Center frequency in MHz")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Fichier introuvable : {args.input}")
        sys.exit(1)

    print("=" * 60)
    print("  IQ Signal → GPU Converter (streaming)")
    print("=" * 60)

    info = get_audio_info(args.input)
    fs = info["sample_rate"]
    nch = info["n_channels"]
    total_dur = info["duration"]

    actual_dur = min(args.duration, total_dur - args.start)
    if actual_dur <= 0:
        print(f"[ERROR] Start time {args.start}s exceeds file duration {total_dur}s")
        sys.exit(1)

    print(f"  Fichier      : {os.path.basename(args.input)}")
    print(f"  Sample rate  : {fs:,} Hz ({fs/1e6:.1f} Msps)")
    print(f"  Canaux       : {nch} ({'IQ stéréo' if nch == 2 else 'audio mono'})")
    print(f"  Durée totale : {total_dur:.1f} s")
    print(f"  Extraction   : {args.start:.1f}s → {args.start + actual_dur:.1f}s ({actual_dur:.1f}s)")

    if nch != 2:
        print("[ERROR] Only stereo IQ files supported (mono requires too much RAM for Hilbert)")
        sys.exit(1)

    print(f"  Streaming vers {args.output}...")
    size_bytes = extract_segment_to_file(args.input, args.output, args.start, actual_dur)
    n_samples = size_bytes // 8

    print(f"  [OK] {n_samples:,} IQ samples ({size_bytes/1e6:.1f} MB)")

    fft_size_target = max(1024, int(fs / 150))
    fft_size = 2 ** int(np.ceil(np.log2(fft_size_target)))
    freq_res = fs / fft_size

    center_freq = args.center_freq
    if center_freq is None:
        fname = os.path.basename(args.input).lower()
        if "436500" in fname or "qb50" in fname:
            center_freq = 436.5
        elif "437800" in fname or "iss" in fname:
            center_freq = 437.8
        else:
            center_freq = 0.0

    cfg = {
        "sample_rate":    fs,
        "n_samples":      n_samples,
        "duration_s":     actual_dur,
        "channels":       nch,
        "source":         os.path.basename(args.input),
        "iq_mode":        "stereo_direct",
        "band_low_hz":    0.0,
        "band_high_hz":   fs / 2.0,
        "freq_center_mhz": center_freq,
        "fft_size":       fft_size
    }
    config_path = Path(args.output).stem + "_config.json"
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  Config JSON  : {config_path}")

    print()
    doppler_bins = int(10000 / freq_res)
    print(f"  FFT_SIZE        : {fft_size} bins")
    print(f"  Résolution FFT  : {freq_res:.1f} Hz/bin")
    print(f"  Doppler ±10 kHz ≈ ±{doppler_bins} bins")
    if center_freq > 0:
        print(f"  Centre freq     : {center_freq} MHz")
    print("=" * 60)

if __name__ == "__main__":
    main()
