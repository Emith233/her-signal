#!/usr/bin/env python3
from __future__ import annotations

import math
import wave
from array import array
from pathlib import Path

from PIL import Image, ImageDraw
from pysstv.color import Robot36


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets" / "苏苏.jpg"
OUTDIR = ROOT / "outputs"

SAMPLE_RATE = 48_000
BITS = 16
WIDTH = 320
HEIGHT = 240
SQUARE = 240
X0 = (WIDTH - SQUARE) // 2
X1 = X0 + SQUARE

MESSAGE = "/sujianwei"
MORSE_HZ = 600
WPM = 18
MORSE_GAIN = 0.12


MORSE = {
    "a": ".-",
    "b": "-...",
    "c": "-.-.",
    "d": "-..",
    "e": ".",
    "f": "..-.",
    "g": "--.",
    "h": "....",
    "i": "..",
    "j": ".---",
    "k": "-.-",
    "l": ".-..",
    "m": "--",
    "n": "-.",
    "o": "---",
    "p": ".--.",
    "q": "--.-",
    "r": ".-.",
    "s": "...",
    "t": "-",
    "u": "..-",
    "v": "...-",
    "w": ".--",
    "x": "-..-",
    "y": "-.--",
    "z": "--..",
    "0": "-----",
    "1": ".----",
    "2": "..---",
    "3": "...--",
    "4": "....-",
    "5": ".....",
    "6": "-....",
    "7": "--...",
    "8": "---..",
    "9": "----.",
    "/": "-..-.",
}


def center_crop_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def make_frame() -> Image.Image:
    avatar = center_crop_square(Image.open(SOURCE).convert("RGB"))
    avatar = avatar.resize((SQUARE, SQUARE), Image.Resampling.LANCZOS)
    frame = Image.new("RGB", (WIDTH, HEIGHT), (18, 19, 22))
    frame.paste(avatar, (X0, 0))
    return frame


def write_wave(path: Path, samples: list[float]) -> None:
    max_i = 2 ** (BITS - 1) - 1
    min_i = -(2 ** (BITS - 1))
    pcm = array("h")
    for sample in samples:
        value = int(max(-1.0, min(1.0, sample)) * max_i)
        pcm.append(min_i if value < min_i else value)

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(BITS // 8)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())


def morse_units(message: str) -> list[tuple[bool, int]]:
    units: list[tuple[bool, int]] = []
    words = message.lower().split(" ")
    for wi, word in enumerate(words):
        letters = [MORSE[ch] for ch in word if ch in MORSE]
        for li, code in enumerate(letters):
            for si, symbol in enumerate(code):
                units.append((True, 1 if symbol == "." else 3))
                if si != len(code) - 1:
                    units.append((False, 1))
            if li != len(letters) - 1:
                units.append((False, 3))
        if wi != len(words) - 1:
            units.append((False, 7))
    units.append((False, 12))
    return units


def make_morse_signal(total_samples: int, wpm: int = WPM) -> list[float]:
    unit_seconds = 1.2 / wpm
    unit_samples = max(1, round(unit_seconds * SAMPLE_RATE))
    pattern = morse_units(MESSAGE)
    signal: list[float] = []
    phase = 0.0
    step = 2 * math.pi * MORSE_HZ / SAMPLE_RATE

    while len(signal) < total_samples:
        for tone, units in pattern:
            for _ in range(units * unit_samples):
                if len(signal) >= total_samples:
                    break
                if tone:
                    signal.append(math.sin(phase))
                else:
                    signal.append(0.0)
                phase = (phase + step) % (2 * math.pi)
            if len(signal) >= total_samples:
                break
    return signal


def robot36_square_mask(total_samples: int) -> list[float]:
    mask = [0.0] * total_samples

    vis_ms = 300 + 10 + 300 + 30 + 7 * 30 + 30 + 30
    line_ms = 3 + 88 + 4.5 + 1.5 + 44
    y_scan_start = 3
    px_ms = 88 / WIDTH

    start_x_ms = y_scan_start + X0 * px_ms
    end_x_ms = y_scan_start + X1 * px_ms

    for line in range(HEIGHT):
        line_start_ms = vis_ms + line * line_ms
        start = round((line_start_ms + start_x_ms) * SAMPLE_RATE / 1000)
        end = round((line_start_ms + end_x_ms) * SAMPLE_RATE / 1000)
        start = max(0, min(total_samples, start))
        end = max(0, min(total_samples, end))
        for i in range(start, end):
            mask[i] = 1.0
    return mask


def make_preview(frame: Image.Image) -> Image.Image:
    preview = frame.copy()
    overlay = Image.new("RGBA", preview.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((X0, 0, X1 - 1, HEIGHT - 1), fill=(255, 80, 60, 42), outline=(255, 60, 40, 255), width=2)
    return Image.alpha_composite(preview.convert("RGBA"), overlay)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    frame = make_frame()
    frame_path = OUTDIR / "susu_robot36_square_frame.png"
    preview_path = OUTDIR / "susu_robot36_morse_square_region_preview.png"
    base_wav = OUTDIR / "susu_robot36_base.wav"
    mixed_wav = OUTDIR / "susu_robot36_morse_square.wav"
    audible_square_wav = OUTDIR / "susu_robot36_morse_square_audible.wav"
    prefix_wav = OUTDIR / "susu_morse_prefix_then_robot36_square.wav"

    frame.save(frame_path)
    make_preview(frame).save(preview_path)

    sstv = Robot36(frame, SAMPLE_RATE, BITS)
    base_samples = [sample / (2 ** (BITS - 1)) for sample in sstv.gen_samples()]
    write_wave(base_wav, base_samples)

    morse = make_morse_signal(len(base_samples), WPM)
    mask = robot36_square_mask(len(base_samples))
    mixed = []
    for base, tone, gate in zip(base_samples, morse, mask):
        mixed.append(base * 0.86 + tone * MORSE_GAIN * gate)
    write_wave(mixed_wav, mixed)

    slow_morse = make_morse_signal(len(base_samples), 10)
    audible_mixed = []
    for base, tone, gate in zip(base_samples, slow_morse, mask):
        audible_mixed.append(base * 0.76 + tone * 0.28 * gate)
    write_wave(audible_square_wav, audible_mixed)

    prefix_len = round(8 * SAMPLE_RATE)
    prefix_tone = make_morse_signal(prefix_len, 10)
    prefix = [tone * 0.45 for tone in prefix_tone]
    pause = [0.0] * round(1.2 * SAMPLE_RATE)
    write_wave(prefix_wav, prefix + pause + mixed)

    print(f"message: {MESSAGE}")
    print(f"morse_hz: {MORSE_HZ}")
    print(f"wpm: {WPM}")
    print(f"square_region: x={X0}..{X1 - 1}, y=0..{HEIGHT - 1}")
    print(frame_path)
    print(preview_path)
    print(base_wav)
    print(mixed_wav)
    print(audible_square_wav)
    print(prefix_wav)


if __name__ == "__main__":
    main()
