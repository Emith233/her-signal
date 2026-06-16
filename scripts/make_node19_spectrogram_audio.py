#!/usr/bin/env python3
from __future__ import annotations

import math
import wave
from array import array
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs"

TEXT_TOP = "节点19"
TEXT_BOTTOM = "/sujianwei"
SAMPLE_RATE = 44_100
DURATION = 18.0
FREQ_LOW = 900.0
FREQ_HIGH = 5200.0
BITMAP_W = 360
BITMAP_H = 120
COLUMN_SECONDS = DURATION / BITMAP_W
AMPLITUDE = 0.62


def load_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def centered_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, y: int) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0]
    x = (BITMAP_W - width) // 2
    draw.text((x, y), text, font=font, fill=255)


def make_text_bitmap() -> Image.Image:
    image = Image.new("L", (BITMAP_W, BITMAP_H), 0)
    draw = ImageDraw.Draw(image)
    font_top = load_font(48)
    font_bottom = load_font(34)
    centered_text(draw, TEXT_TOP, font_top, 8)
    centered_text(draw, TEXT_BOTTOM, font_bottom, 70)

    # Slight thickening makes the message readable in common spectrogram tools.
    image = image.filter(ImageFilter.MaxFilter(3))
    return image


def synthesize(bitmap: Image.Image) -> np.ndarray:
    pixels = np.asarray(bitmap, dtype=np.float32) / 255.0
    total_samples = round(DURATION * SAMPLE_RATE)
    audio = np.zeros(total_samples, dtype=np.float32)

    freqs = np.linspace(FREQ_HIGH, FREQ_LOW, BITMAP_H)
    phases = np.zeros(BITMAP_H, dtype=np.float64)
    two_pi = 2 * math.pi

    for x in range(BITMAP_W):
        start = round(x * COLUMN_SECONDS * SAMPLE_RATE)
        end = round((x + 1) * COLUMN_SECONDS * SAMPLE_RATE)
        if end <= start:
            continue
        active = np.flatnonzero(pixels[:, x] > 0.18)
        if active.size == 0:
            continue

        n = end - start
        t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
        column = np.zeros(n, dtype=np.float64)
        weights = pixels[active, x].astype(np.float64)

        for row, weight in zip(active, weights):
            freq = freqs[row]
            phase = phases[row]
            tone = np.sin(two_pi * freq * t + phase)
            column += tone * weight
            phases[row] = (phase + two_pi * freq * n / SAMPLE_RATE) % two_pi

        column /= max(1.0, math.sqrt(active.size) * 0.9)
        fade_len = min(80, n // 4)
        if fade_len:
            fade = np.ones(n, dtype=np.float64)
            ramp = np.linspace(0, 1, fade_len)
            fade[:fade_len] *= ramp
            fade[-fade_len:] *= ramp[::-1]
            column *= fade
        audio[start:end] += column.astype(np.float32)

    # Add a very quiet carrier bed so the file feels intentional, but keep text dominant.
    t_all = np.arange(total_samples, dtype=np.float32) / SAMPLE_RATE
    audio += 0.018 * np.sin(2 * np.pi * 220 * t_all)

    peak = float(np.max(np.abs(audio))) or 1.0
    audio = audio / peak * AMPLITUDE
    return audio


def write_wav(path: Path, audio: np.ndarray) -> None:
    max_i = 32767
    pcm = array("h", np.clip(audio * max_i, -32768, 32767).astype(np.int16))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())


def make_preview(bitmap: Image.Image) -> Image.Image:
    preview = Image.new("RGB", (BITMAP_W, BITMAP_H), (8, 12, 18))
    color = Image.new("RGB", (BITMAP_W, BITMAP_H), (80, 230, 180))
    preview.paste(color, mask=bitmap)
    preview = preview.resize((BITMAP_W * 2, BITMAP_H * 2), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(preview)
    draw.rectangle((0, 0, preview.width - 1, preview.height - 1), outline=(80, 230, 180), width=2)
    return preview


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    bitmap = make_text_bitmap()
    audio = synthesize(bitmap)

    bitmap_path = OUTDIR / "node19_spectrogram_text_bitmap.png"
    preview_path = OUTDIR / "node19_spectrogram_preview.png"
    wav_path = OUTDIR / "node19_spectrogram_clue.wav"

    bitmap.save(bitmap_path)
    make_preview(bitmap).save(preview_path)
    write_wav(wav_path, audio)

    print(bitmap_path)
    print(preview_path)
    print(wav_path)
    print(f"text: {TEXT_TOP} {TEXT_BOTTOM}")
    print(f"duration: {DURATION}s, freq: {FREQ_LOW}-{FREQ_HIGH}Hz")


if __name__ == "__main__":
    main()
