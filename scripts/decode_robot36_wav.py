#!/usr/bin/env python3
from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np
from PIL import Image


WIDTH = 320
HEIGHT = 240
FREQ_BLACK = 1500
FREQ_WHITE = 2300


def read_mono_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sampwidth = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())

    if sampwidth != 2:
        raise ValueError("Only 16-bit PCM WAV is supported")

    audio = np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sample_rate


def analytic_signal(samples: np.ndarray) -> np.ndarray:
    n = samples.size
    spectrum = np.fft.fft(samples)
    filt = np.zeros(n)
    if n % 2 == 0:
        filt[0] = 1
        filt[n // 2] = 1
        filt[1 : n // 2] = 2
    else:
        filt[0] = 1
        filt[1 : (n + 1) // 2] = 2
    return np.fft.ifft(spectrum * filt)


def fft_bandpass(samples: np.ndarray, sample_rate: int, low: float = 1000.0, high: float = 2600.0) -> np.ndarray:
    spectrum = np.fft.rfft(samples)
    freqs = np.fft.rfftfreq(samples.size, 1 / sample_rate)
    keep = (freqs >= low) & (freqs <= high)
    spectrum[~keep] = 0
    return np.fft.irfft(spectrum, n=samples.size)


def instantaneous_frequency(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    samples = fft_bandpass(samples, sample_rate)
    analytic = analytic_signal(samples)
    phase = np.unwrap(np.angle(analytic))
    freq = np.diff(phase) * sample_rate / (2 * np.pi)
    freq = np.clip(freq, 700, 2800)
    kernel = np.ones(9) / 9
    return np.convolve(freq, kernel, mode="same")


def ms_to_index(ms: float, sample_rate: int) -> int:
    return int(round(ms * sample_rate / 1000))


def zero_crossing_freq(samples: np.ndarray, sample_rate: int, center_ms: float, window_ms: float = 5.0) -> float:
    half = ms_to_index(window_ms / 2, sample_rate)
    center = ms_to_index(center_ms, sample_rate)
    start = max(0, center - half)
    end = min(samples.size, center + half)
    segment = samples[start:end]
    if segment.size < 4:
        return FREQ_BLACK

    signs = np.signbit(segment)
    crossings = np.flatnonzero(signs[1:] != signs[:-1])
    if crossings.size < 2:
        return FREQ_BLACK

    first = float(crossings[0])
    last = float(crossings[-1])
    duration = (last - first) / sample_rate
    if duration <= 0:
        return FREQ_BLACK
    return float((crossings.size - 1) / (2 * duration))


def freq_to_byte(value_freq: float) -> int:
    value = (value_freq - FREQ_BLACK) * 255 / (FREQ_WHITE - FREQ_BLACK)
    return int(round(max(0, min(255, value))))


def decode_robot36(path: Path) -> Image.Image:
    samples, sample_rate = read_mono_wav(path)
    samples = fft_bandpass(samples, sample_rate)

    y_channel = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cb_channel = np.full((HEIGHT, WIDTH), 128, dtype=np.uint8)
    cr_channel = np.full((HEIGHT, WIDTH), 128, dtype=np.uint8)
    cb_known = np.zeros(HEIGHT, dtype=bool)
    cr_known = np.zeros(HEIGHT, dtype=bool)

    vis_ms = 300.0 + 10.0 + 300.0 + 30.0 + 7.0 * 30.0 + 30.0 + 30.0
    line_ms = 9.0 + 3.0 + 88.0 + 4.5 + 1.5 + 44.0
    y_start_in_line = 9.0 + 3.0
    chroma_start_in_line = 9.0 + 3.0 + 88.0 + 4.5 + 1.5
    y_px_ms = 88.0 / WIDTH
    chroma_px_ms = 44.0 / WIDTH

    for line in range(HEIGHT):
        line_start = vis_ms + line * line_ms
        for col in range(WIDTH):
            start = line_start + y_start_in_line + col * y_px_ms
            center = start + y_px_ms / 2
            y_channel[line, col] = freq_to_byte(zero_crossing_freq(samples, sample_rate, center))

        channel = 2 - (line % 2)
        target = cr_channel if channel == 2 else cb_channel
        for col in range(WIDTH):
            start = line_start + chroma_start_in_line + col * chroma_px_ms
            center = start + chroma_px_ms / 2
            target[line, col] = freq_to_byte(zero_crossing_freq(samples, sample_rate, center))

        if channel == 2:
            cr_known[line] = True
        else:
            cb_known[line] = True

    for line in range(HEIGHT):
        if not cb_known[line]:
            source = line + 1 if line + 1 < HEIGHT else line - 1
            cb_channel[line] = cb_channel[source]
        if not cr_known[line]:
            source = line - 1 if line > 0 else line + 1
            cr_channel[line] = cr_channel[source]

    ycbcr = np.dstack([y_channel, cb_channel, cr_channel]).astype(np.uint8)
    return Image.fromarray(ycbcr, "YCbCr").convert("RGB")


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: decode_robot36_wav.py input.wav output.png")
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = decode_robot36(input_path)
    image.save(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
