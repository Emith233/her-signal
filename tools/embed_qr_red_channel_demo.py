from pathlib import Path
import subprocess

from PIL import Image, ImageChops, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
OUT = ASSETS / "qr-red-channel-demo.mov"
PREVIEW = ASSETS / "qr-red-channel-extract-sheet.png"

WIDTH, HEIGHT = 1280, 720
FPS = 24
DURATION = 24
TOTAL_FRAMES = FPS * DURATION
HOLD_FRAMES = 6
QR_SIZE = 420
RED_BOOST = 18

HIDE_POINTS = [
    (4, "qr-letter-1.png"),
    (9, "qr-letter-2.png"),
    (14, "qr-letter-3.png"),
    (19, "qr-letter-4.png"),
]


def load_qr_mask(name):
    qr = Image.open(ASSETS / name).convert("L")
    qr = qr.resize((QR_SIZE, QR_SIZE), Image.Resampling.NEAREST)
    mask = qr.point(lambda p: 255 if p < 128 else 0)
    return mask


def base_frame(n):
    pulse = int(5 * ((n % FPS) / FPS))
    r = Image.new("L", (WIDTH, HEIGHT), 32 + pulse)
    g = Image.new("L", (WIDTH, HEIGHT), 38)
    b = Image.new("L", (WIDTH, HEIGHT), 45 - pulse)

    grad_x = Image.linear_gradient("L").rotate(90, expand=True).resize((WIDTH, HEIGHT))
    grad_y = Image.linear_gradient("L").resize((WIDTH, HEIGHT))
    drift = ImageChops.add(
        Image.eval(grad_x, lambda p: int((p - 128) * 0.07 + 8)),
        Image.eval(grad_y, lambda p: int((p - 128) * 0.05 + 8)),
    )
    grain = Image.effect_noise((WIDTH, HEIGHT), 7).convert("L")
    grain = Image.eval(grain, lambda p: int((p - 128) * 0.18 + 5))
    scan = Image.new("L", (WIDTH, HEIGHT), 0)
    scan_draw = ImageDraw.Draw(scan)
    for y in range((n % 7), HEIGHT, 7):
        scan_draw.line((0, y, WIDTH, y), fill=5)

    r = ImageChops.add(ImageChops.add(r, drift), ImageChops.add(grain, scan))
    g = ImageChops.add(ImageChops.add(g, drift), grain)
    b = ImageChops.add(ImageChops.add(b, drift), grain)
    img = Image.merge("RGB", (r, g, b))

    draw = ImageDraw.Draw(img, "RGBA")
    cx = int(WIDTH * (0.5 + 0.03 * ((n % (FPS * 8)) / (FPS * 8) - 0.5)))
    cy = int(HEIGHT * 0.47)
    for radius, alpha in [(310, 20), (220, 18), (140, 12)]:
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=(122, 35, 38, alpha), width=2)
    draw.rectangle((70, 58, WIDTH - 70, HEIGHT - 58), outline=(96, 84, 72, 46), width=1)
    return img


def embed_qr(img, mask, frame_offset):
    x0 = (WIDTH - QR_SIZE) // 2
    y0 = (HEIGHT - QR_SIZE) // 2
    fade = max(0.52, 1 - frame_offset * 0.08)
    crop = img.crop((x0, y0, x0 + QR_SIZE, y0 + QR_SIZE)).convert("RGB")
    r, g, b = crop.split()
    red_overlay = Image.new("L", (QR_SIZE, QR_SIZE), int(RED_BOOST * fade))
    boosted = Image.composite(
        Image.eval(r, lambda p: min(255, p + int(RED_BOOST * fade))),
        r,
        mask,
    )
    crop = Image.merge("RGB", (boosted, g, b))
    img.paste(crop, (x0, y0))
    return img


def extract_preview(img):
    r, g, b = img.split()
    preview = Image.eval(r, lambda p: max(0, min(255, (p - 34) * 10)))
    preview = preview.filter(ImageFilter.SHARPEN)
    return preview.convert("RGB")


def main():
    masks = [(sec, load_qr_mask(name)) for sec, name in HIDE_POINTS]
    preview_frames = []

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "prores_ks",
        "-profile:v",
        "3",
        "-pix_fmt",
        "yuv422p10le",
        str(OUT),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for n in range(TOTAL_FRAMES):
            img = base_frame(n)
            for sec, mask in masks:
                start = sec * FPS
                if start <= n < start + HOLD_FRAMES:
                    img = embed_qr(img, mask, n - start)
                    if n == start:
                        preview_frames.append((sec, extract_preview(img)))
            proc.stdin.write(img.tobytes())
    finally:
        proc.stdin.close()
        code = proc.wait()
        if code:
            raise SystemExit(code)

    sheet = Image.new("RGB", (QR_SIZE * 2 + 48, QR_SIZE * 2 + 92), (18, 18, 18))
    draw = ImageDraw.Draw(sheet)
    for i, (sec, frame) in enumerate(preview_frames):
        x = 16 + (i % 2) * (QR_SIZE + 16)
        y = 46 + (i // 2) * (QR_SIZE + 16)
        sheet.paste(frame.crop(((WIDTH - QR_SIZE) // 2, (HEIGHT - QR_SIZE) // 2, (WIDTH + QR_SIZE) // 2, (HEIGHT + QR_SIZE) // 2)), (x, y))
        draw.text((x, y - 22), f"{sec:02d}s red channel x10", fill=(230, 220, 205))
    sheet.save(PREVIEW)
    print(OUT)
    print(PREVIEW)


if __name__ == "__main__":
    main()
