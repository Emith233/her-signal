#!/usr/bin/env python3
from __future__ import annotations

import wave
from pathlib import Path

from PIL import Image
from pysstv.color import Robot36


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs"
SOURCE = OUTDIR / "decoded_susu_robot36_morse_square_avatar_crop.png"

MESSAGE = "/sujianwei"
WIDTH = 240
HEIGHT = 240
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
FRAME_X = 40
SAMPLE_RATE = 48_000
BITS = 16


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
    units.append((False, 8))
    return units


def expanded_pattern(total_pixels: int) -> list[bool]:
    units = morse_units(MESSAGE)
    total_units = sum(length for _, length in units)
    pattern: list[bool] = []
    consumed = 0
    for index, (tone, length) in enumerate(units):
        if index == len(units) - 1:
            count = total_pixels - consumed
        else:
            count = round(total_pixels * length / total_units)
        pattern.extend([tone] * max(0, count))
        consumed += max(0, count)
    if len(pattern) < total_pixels:
        pattern.extend([False] * (total_pixels - len(pattern)))
    return pattern[:total_pixels]


def apply_brightness(base: Image.Image, mode: str) -> Image.Image:
    image = base.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS).convert("YCbCr")
    pixels = image.load()
    pattern = expanded_pattern(WIDTH * HEIGHT)

    for y in range(HEIGHT):
        for x in range(WIDTH):
            tone = pattern[y * WIDTH + x]
            yy, cb, cr = pixels[x, y]
            if mode == "strong":
                new_y = 232 if tone else 38
            else:
                delta = 58 if tone else -32
                new_y = int(max(16, min(235, yy + delta)))
            pixels[x, y] = (new_y, cb, cr)
    return image.convert("RGB")


def make_robot36_wav(image: Image.Image, path: Path) -> None:
    frame = Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT), (0, 0, 0))
    frame.paste(image, (FRAME_X, 0))
    Robot36(frame, SAMPLE_RATE, BITS).write_wav(str(path))


def wav_info(path: Path) -> str:
    with wave.open(str(path), "rb") as wav:
        seconds = wav.getnframes() / wav.getframerate()
        return f"{wav.getnchannels()}ch {wav.getframerate()}Hz {wav.getsampwidth() * 8}bit {seconds:.2f}s"


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    base = Image.open(SOURCE).convert("RGB")

    for mode in ("strong", "subtle"):
        image = apply_brightness(base, mode)
        image_path = OUTDIR / f"susu_morse_brightness_{mode}.png"
        wav_path = OUTDIR / f"susu_morse_brightness_{mode}_robot36.wav"
        image.save(image_path)
        make_robot36_wav(image, wav_path)
        print(image_path)
        print(wav_path, wav_info(wav_path))


if __name__ == "__main__":
    main()
