

import json
import numpy as np
import sys
import os
from PIL import Image, ImageDraw, ImageFont

def load_config(path='signal_config.json'):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {
        "sample_rate": 10e6,
        "band_low_hz": 0.8e6,
        "band_high_hz": 4.8e6,
        "source": "Simulated"
    }

CONFIG = load_config()
FFT_SIZE     = CONFIG.get("fft_size", 1024)
SAMPLE_RATE  = CONFIG["sample_rate"]
BAND_LOW_HZ  = CONFIG.get("band_low_hz", 0)
BAND_HIGH_HZ = CONFIG.get("band_high_hz", SAMPLE_RATE/2)
SOURCE_NAME  = CONFIG.get("source", "Unknown")
BIN_ZOOM     = CONFIG.get("bin_zoom", None)
CENTER_FREQ  = CONFIG.get("freq_center_mhz", 0) * 1e6

DOPPLER_CMAP = np.array([
    [  0,   0,  30], [  0,  20,  60], [  0,  40, 100], [  0,  80, 140],
    [  0, 120, 160], [ 20, 160, 180], [ 60, 200, 200], [120, 220, 180],
    [180, 230, 100], [230, 220,  50], [255, 180,  30], [255, 100,  20],
    [255,  40,  40], [200,  20,  60]], dtype=np.uint8)

def apply_colormap(data_norm: np.ndarray, cmap=None) -> np.ndarray:

    if cmap is None:
        cmap = DOPPLER_CMAP
    idx = (data_norm * (len(cmap)-1)).clip(0, len(cmap)-1).astype(int)
    return cmap[idx]

def load_spectrogram(path='output_spectrogram.bin'):
    data = np.fromfile(path, dtype=np.float32)
    n = len(data) // FFT_SIZE
    return data[:n * FFT_SIZE].reshape(n, FFT_SIZE)

def load_detections(path='output_detections.bin', total=None):
    if not os.path.exists(path): return None
    data = np.fromfile(path, dtype=np.int32)
    if total: data = data[:total]
    n = len(data) // FFT_SIZE
    return data[:n * FFT_SIZE].reshape(n, FFT_SIZE)

def normalize(arr, p_low=2, p_high=98):

    if arr.size > 1_000_000:
        sample = arr.reshape(-1)[::100]
    else:
        sample = arr
    lo, hi = np.percentile(sample, p_low), np.percentile(sample, p_high)
    return ((arr - lo) / max(hi - lo, 1e-6)).clip(0, 1)

def draw_label(draw, x, y, text, color=(255,255,255)):
    text = text.replace('\u2014','--').replace('\u2013','-').replace('\u2014','-')
    try:
        draw.text((x+1, y+1), text, fill=(0,0,0))
        draw.text((x, y), text, fill=color)
    except Exception:
        pass

def make_waterfall(spec_data, title_text, bin_range=None, width=1600, height=800,
                   show_real_freq=True, duration_s=None):

    n_ffts, n_bins_full = spec_data.shape

    if bin_range:
        b1, b2 = bin_range
        spec_view = spec_data[:, b1:b2]
    else:
        spec_view = spec_data

    n_ffts_view, n_bins_view = spec_view.shape

    from PIL import Image as PILImage
    norm = normalize(spec_view, p_low=1, p_high=99).astype(np.float32)
    img_arr = np.array(PILImage.fromarray(norm).resize((width, height), PILImage.BILINEAR))
    rgb = apply_colormap(img_arr)
    img = PILImage.fromarray(rgb, 'RGB')
    draw = ImageDraw.Draw(img)

    freq_res = SAMPLE_RATE / FFT_SIZE
    freq_max = SAMPLE_RATE / 2

    if bin_range:
        b1, b2 = bin_range
        f1_offset = b1 * freq_res
        f2_offset = b2 * freq_res
    else:
        f1_offset = 0
        f2_offset = freq_max

    n_ticks = 6
    for i in range(n_ticks):
        x = int(i * width / (n_ticks - 1))
        f_offset = f1_offset + i * (f2_offset - f1_offset) / (n_ticks - 1)

        if CENTER_FREQ > 0 and show_real_freq:
            f_real = (CENTER_FREQ + f_offset) / 1e6
            label = f"{f_real:.3f}"
        else:
            label = f"{f_offset/1e3:.0f}k"

        draw.line([(x, height-18), (x, height)], fill=(200,200,200), width=1)
        draw_label(draw, max(2, x-20), height-16, label, (200,200,200))

    if duration_s:
        for i in range(5):
            y = int(i * height / 4)
            t = i * duration_s / 4
            draw.line([(0, y), (8, y)], fill=(200,200,200), width=1)
            draw_label(draw, 10, y+2, f"{t:.0f}s", (180,180,180))

    draw_label(draw, 10, 5, title_text, (255,255,255))
    if CENTER_FREQ > 0:
        draw_label(draw, width-180, 5, f"Center: {CENTER_FREQ/1e6:.3f} MHz", (200,200,200))

    return img

def make_spectrum_bar(spec_half, width=1400, height=220):

    n_ffts, n_bins = spec_half.shape
    avg = spec_half.mean(axis=0)

    filtered = avg.copy()
    freq_res = SAMPLE_RATE / FFT_SIZE
    for b in range(n_bins):
        f = b * freq_res
        if f < BAND_LOW_HZ or f > BAND_HIGH_HZ:
            filtered[b] = avg.min()

    db_min, db_max = avg.min(), avg.max()
    def db_to_y(db):
        normed = (db - db_min) / max(db_max - db_min, 1e-6)
        return int((1.0 - normed) * (height - 30))

    img = Image.new('RGB', (width, height), (18, 18, 28))
    draw = ImageDraw.Draw(img)

    freq_max_mhz = SAMPLE_RATE / 2 / 1e6

    for db in range(int(db_min)//10*10, int(db_max)+10, 10):
        y = db_to_y(db)
        if 0 <= y < height:
            draw.line([(0,y),(width,y)], fill=(50,50,60), width=1)
            draw_label(draw, 2, y, f"{db}dB", (120,120,120))

    for f_mhz in [1,2,3,4,5]:
        if f_mhz <= freq_max_mhz:
            x = int(f_mhz / freq_max_mhz * width)
            draw.line([(x,0),(x,height)], fill=(50,50,60), width=1)

    x_lo = int((BAND_LOW_HZ/1e6)  / freq_max_mhz * width)
    x_hi = int((BAND_HIGH_HZ/1e6) / freq_max_mhz * width)
    draw.rectangle([x_lo, 0, x_hi, height], fill=(0, 50, 80))

    pts_raw = [(int(b / n_bins * width), db_to_y(avg[b])) for b in range(n_bins)]
    for i in range(len(pts_raw)-1):
        draw.line([pts_raw[i], pts_raw[i+1]], fill=(80, 150, 230), width=1)

    pts_fil = [(int(b / n_bins * width), db_to_y(filtered[b])) for b in range(n_bins)]
    for i in range(len(pts_fil)-1):
        draw.line([pts_fil[i], pts_fil[i+1]], fill=(230, 80, 80), width=2)

    if SOURCE_NAME == "Simulated":
        emitters = [(1.2,'FM 1.2'),(1.8,'Pulse 1.8'),(2.5,'AM 2.5'),(3.5,'CW 3.5')]
        for f_mhz, lbl in emitters:
            x = int(f_mhz / freq_max_mhz * width)
            draw_label(draw, x+2, 20, lbl, (255,230,0))

    draw_label(draw, 10, height-22, "━ Spectre brut", (80,150,230))
    if SOURCE_NAME == "Simulated":
        draw_label(draw, 200, height-22, "━ Apres filtre passe-bande GPU", (230,80,80))
        draw_label(draw, 600, height-22, "█ Bande passante [0.8–4.8 MHz]", (0,130,200))
    return img

def make_persistence_bar(detections_half, width=1400, height=120):

    persistence = detections_half.mean(axis=0) * 100
    n_bins = len(persistence)
    freq_max_mhz = SAMPLE_RATE / 2 / 1e6

    img = Image.new('RGB', (width, height), (18,18,28))
    draw = ImageDraw.Draw(img)

    for b in range(n_bins):
        x = int(b / n_bins * width)
        bar_h = int(persistence[b] / 100.0 * (height - 20))
        col = (255, 60, 60) if persistence[b] > 30 else (80, 80, 120)
        draw.rectangle([x, height-20-bar_h, x+max(1,width//n_bins), height-20], fill=col)

    y30 = int((1 - 0.30) * (height - 20))
    draw.line([(0, y30), (width, y30)], fill=(255, 200, 0), width=1)
    draw_label(draw, 4, y30-14, "Seuil 30%", (255,200,0))
    draw_label(draw, 10, height-18, "Profil de Persistance CA-CFAR (%)", (200,200,200))
    return img

def find_doppler_regions(spec, detections, min_bins=50, max_regions=4):

    n_ffts, n_bins = spec.shape
    half = n_bins // 2

    avg_power = spec[:, :half].mean(axis=0)
    var_power = spec[:, :half].var(axis=0)

    score = (avg_power - avg_power.min()) / (avg_power.max() - avg_power.min() + 1e-6)
    score += (var_power - var_power.min()) / (var_power.max() - var_power.min() + 1e-6)

    regions = []
    threshold = np.percentile(score, 95)
    in_region = False
    start = 0

    for i in range(half):
        if score[i] > threshold and not in_region:
            start = i
            in_region = True
        elif score[i] <= threshold and in_region:
            if i - start >= min_bins:
                regions.append((start, i))
            in_region = False

    regions.sort(key=lambda r: -score[r[0]:r[1]].sum())
    return regions[:max_regions]

def compose_figures(spec, detections):
    n_ffts, n_bins = spec.shape
    half = n_bins // 2
    spec_h = spec[:, :half]
    det_h  = detections[:, :half] if detections is not None else None

    duration_s = n_ffts * FFT_SIZE / SAMPLE_RATE
    freq_res = SAMPLE_RATE / FFT_SIZE

    W, H = 1800, 1000

    if BIN_ZOOM:
        b1, b2 = BIN_ZOOM
        b1 = max(0, min(b1, half-1))
        b2 = max(b1+1, min(b2, half))
        f1 = (CENTER_FREQ + b1 * freq_res) / 1e6 if CENTER_FREQ > 0 else b1 * freq_res / 1e3
        f2 = (CENTER_FREQ + b2 * freq_res) / 1e6 if CENTER_FREQ > 0 else b2 * freq_res / 1e3
        title = f"QB50 Satellites Doppler - {f1:.3f} to {f2:.3f} MHz"
        wf = make_waterfall(spec_h, title, bin_range=(b1, b2),
                           width=W, height=H, duration_s=duration_s)
    else:
        wf = make_waterfall(spec_h, f"Spectrogramme complet - {SOURCE_NAME}",
                           width=W, height=H, duration_s=duration_s)
    wf.save('captures/spectrogram.png')
    print("[OK] captures/spectrogram.png")

    doppler_regions = find_doppler_regions(spec, detections)

    if BIN_ZOOM:

        b1, b2 = BIN_ZOOM
        margin = (b2 - b1) // 2
        b1_wide = max(0, b1 - margin)
        b2_wide = min(half, b2 + margin)
        f1 = (CENTER_FREQ + b1_wide * freq_res) / 1e6 if CENTER_FREQ > 0 else b1_wide * freq_res / 1e3
        f2 = (CENTER_FREQ + b2_wide * freq_res) / 1e6 if CENTER_FREQ > 0 else b2_wide * freq_res / 1e3
        title = f"Vue élargie - {f1:.3f} to {f2:.3f} MHz"
        wf_doppler = make_waterfall(spec_h, title, bin_range=(b1_wide, b2_wide),
                                   width=W, height=H, duration_s=duration_s)
    elif doppler_regions:
        b1, b2 = doppler_regions[0]
        margin = max(200, (b2 - b1) // 2)
        b1 = max(0, b1 - margin)
        b2 = min(half, b2 + margin)
        f1 = (CENTER_FREQ + b1 * freq_res) / 1e6 if CENTER_FREQ > 0 else b1 * freq_res / 1e3
        f2 = (CENTER_FREQ + b2 * freq_res) / 1e6 if CENTER_FREQ > 0 else b2 * freq_res / 1e3
        title = f"Doppler View - {f1:.3f} to {f2:.3f} MHz"
        wf_doppler = make_waterfall(spec_h, title, bin_range=(b1, b2),
                                   width=W, height=H, duration_s=duration_s)
    else:
        quarter = half // 4
        b1, b2 = quarter, 3*quarter
        wf_doppler = make_waterfall(spec_h, "Doppler View - Center Band",
                                   bin_range=(b1, b2), width=W, height=H,
                                   duration_s=duration_s)
    wf_doppler.save('captures/spectrogram_doppler.png')
    print(f"[OK] captures/spectrogram_doppler.png")

    bar_spec = make_spectrum_bar(spec_h, width=W)
    if det_h is not None:
        bar_pers = make_persistence_bar(det_h, width=W)
        combined = Image.new('RGB', (W, bar_spec.height + bar_pers.height + 4), (10,10,20))
        combined.paste(bar_spec, (0, 0))
        combined.paste(bar_pers, (0, bar_spec.height + 4))
        combined.save('captures/spectrum_avg.png')
    else:
        bar_spec.save('captures/spectrum_avg.png')
    print("[OK] captures/spectrum_avg.png")

def print_summary(spec, detections=None):
    n_ffts, bins = spec.shape
    freq_res = SAMPLE_RATE / FFT_SIZE

    print("=" * 60)
    print("  RAPPORT SPECTROGRAMME - ANALYSE DOPPLER")
    print("=" * 60)
    print(f"  Source          : {SOURCE_NAME}")
    print(f"  FFTs total      : {n_ffts}")
    print(f"  FFT size        : {FFT_SIZE} bins")
    print(f"  Résolution freq : {freq_res:.1f} Hz/bin ({freq_res/1e3:.3f} kHz)")
    print(f"  Durée couverte  : {n_ffts * FFT_SIZE / SAMPLE_RATE:.2f} s")

    if CENTER_FREQ > 0:
        f_low = CENTER_FREQ / 1e6
        f_high = (CENTER_FREQ + SAMPLE_RATE/2) / 1e6
        print(f"  Plage spectrale : {f_low:.3f} — {f_high:.3f} MHz")
        print(f"  Centre freq     : {CENTER_FREQ/1e6:.3f} MHz")
    else:
        print(f"  Plage spectrale : 0 — {SAMPLE_RATE/2/1e6:.1f} MHz (offset)")

    print(f"  Magnitude       : {spec.min():.1f} / {spec.max():.1f} dB")

    doppler_10khz = int(10000 / freq_res)
    print(f"\n  Doppler ±10 kHz = ±{doppler_10khz} bins visibles")

    if detections is not None:
        pers = detections[:, :FFT_SIZE//2].mean(axis=0)
        det_bins = np.where(pers > 0.30)[0]
        print(f"\n  CFAR : {len(det_bins)} bins persistants (>30% FFTs)")
        for b in det_bins[:10]:
            f_offset = b * freq_res
            if CENTER_FREQ > 0:
                f_real = (CENTER_FREQ + f_offset) / 1e6
                print(f"    Bin {b:5d} | {f_real:.4f} MHz | {pers[b]*100:.0f}%")
            else:
                print(f"    Bin {b:5d} | +{f_offset/1e3:.1f} kHz | {pers[b]*100:.0f}%")
        if len(det_bins) > 10:
            print(f"    ... et {len(det_bins)-10} autres bins")
    print("=" * 60)

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'output_spectrogram.bin'
    print(f"Chargement {path}...")
    spec = load_spectrogram(path)
    det  = load_detections('output_detections.bin', spec.size)

    if os.path.exists('output_sparse.bin'):

        dtype_sparse = np.dtype([('idx', np.int64), ('mag', np.float32)])
        raw = np.fromfile('output_sparse.bin', dtype=dtype_sparse)
        n_sparse = len(raw)
        print(f"Sparse: {n_sparse:,} bins conservés / {spec.size:,} "
              f"({100*n_sparse/spec.size:.2f}%)")

    print_summary(spec, det)
    compose_figures(spec, det)
    print("\nTerminé — 3 images générées.")
