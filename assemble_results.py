

from PIL import Image, ImageDraw
import glob
import sys

SEGMENT_DURATION = 300
TOTAL_DURATION = 1388

def assemble_spectrograms():

    seg_files = sorted(glob.glob("captures/spectrogram_seg*.png"))
    if not seg_files:
        print("[ERROR] No spectrogram segments found (captures/spectrogram_seg*.png)")
        sys.exit(1)

    print(f"[INFO] Found {len(seg_files)} segment(s)")

    print("[INFO] Loading images...")
    images = []
    for f in seg_files:
        try:
            img = Image.open(f)
            images.append(img)
            print(f"  ✓ {f} ({img.size})")
        except Exception as e:
            print(f"  ✗ {f}: {e}")
            sys.exit(1)

    if not images:
        print("[ERROR] Could not load any images")
        sys.exit(1)

    width, height = images[0].size

    montage_height = height * len(images)
    print(f"[INFO] Creating montage: {width} × {montage_height} px")
    montage = Image.new('RGB', (width, montage_height), color='black')

    print("[INFO] Assembling timeline...")
    draw = ImageDraw.Draw(montage)

    for i, img in enumerate(images):
        y_offset = i * height
        montage.paste(img, (0, y_offset))

        t_start = i * SEGMENT_DURATION
        t_end = min((i + 1) * SEGMENT_DURATION, TOTAL_DURATION)

        text = f"Segment {i+1}: {t_start:04d}s - {t_end:04d}s"

        text_bbox = draw.textbbox((10, y_offset + 5), text)
        draw.rectangle([0, y_offset, text_bbox[2] + 20, text_bbox[3] + 10], fill='black')
        draw.text((10, y_offset + 5), text, fill='white')

    output_file = 'captures/spectrogram_full_timeline.png'
    print(f"[INFO] Saving montage to {output_file}...")
    montage.save(output_file)

    print(f"\n✓ Assembly complete!")
    print(f"  Output: {output_file}")
    print(f"  Size: {montage.size}")
    print(f"  Segments: {len(images)}")
    print(f"  Total duration: {TOTAL_DURATION}s")

if __name__ == '__main__':
    try:
        assemble_spectrograms()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
