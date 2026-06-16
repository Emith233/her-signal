#!/usr/bin/env python3
from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        data = wav.readframes(wav.getnframes())
    audio = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sample_rate


def render(path: Path, output: Path, low: float = 1300.0, high: float = 2500.0) -> None:
    audio, sample_rate = read_wav(path)
    win = 4096
    hop = 256
    window = np.hanning(win).astype(np.float32)
    freqs = np.fft.rfftfreq(win, 1 / sample_rate)
    band = np.where((freqs >= low) & (freqs <= high))[0]

    cols = []
    for start in range(0, len(audio) - win, hop):
        chunk = audio[start : start + win] * window
        mag = np.abs(np.fft.rfft(chunk))[band]
        cols.append(mag)
    spec = np.stack(cols, axis=1)
    spec = 20 * np.log10(spec + 1e-7)
    floor = np.percentile(spec, 35)
    ceil = np.percentile(spec, 99.7)
    spec = np.clip((spec - floor) / max(1e-6, ceil - floor), 0, 1)
    spec = spec[::-1, :]

    h, w = spec.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = (spec * 40).astype(np.uint8)
    rgb[..., 1] = (spec * 245).astype(np.uint8)
    rgb[..., 2] = (spec * 190).astype(np.uint8)

    image = Image.fromarray(rgb, "RGB").resize((1400, 420), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (1400, 470), (6, 8, 12))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 430), f"{path.name} | zoom {low:.0f}-{high:.0f} Hz", fill=(120, 220, 190))
    draw.text((10, 448), "Morse should read from left to right as high/bright segments: /sujianwei", fill=(120, 220, 190))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: render_spectrogram_zoom.py input.wav output.png")
    render(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
