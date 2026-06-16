#!/usr/bin/env python3
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pysstv.color import Robot36
from pysstv.grayscale import Robot24BW


ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "assets" / "苏苏.jpg"
OUTDIR = ROOT / "outputs"

MESSAGE = "/sujianwei"
WIDTH = 240
HEIGHT = 240
FRAME_W = 320
FRAME_H = 240
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
                if si < len(code) - 1:
                    units.append((False, 1))
            if li < len(letters) - 1:
                units.append((False, 3))
        if wi < len(words) - 1:
            units.append((False, 7))
    return units


def center_crop_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def pattern_by_row() -> list[bool]:
    units = morse_units(MESSAGE)
    total_units = sum(length for _, length in units)
    rows_per_unit = HEIGHT / total_units
    rows: list[bool] = []
    acc = 0.0

    for tone, length in units:
        acc += length * rows_per_unit
        target = round(acc)
        rows.extend([tone] * max(0, target - len(rows)))

    if len(rows) < HEIGHT:
        rows.extend([False] * (HEIGHT - len(rows)))
    return rows[:HEIGHT]


def make_avatar(mode: str) -> Image.Image:
    base = center_crop_square(Image.open(ASSET).convert("RGB")).resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    image = base.convert("YCbCr")
    pixels = image.load()
    rows = pattern_by_row()

    for y, tone in enumerate(rows):
        for x in range(WIDTH):
            yy, cb, cr = pixels[x, y]
            if mode == "strong":
                new_y = 236 if tone else 28
            elif mode == "highlight":
                new_y = 242 if tone else yy
            else:
                new_y = int(max(18, min(238, yy + (72 if tone else -38))))
            pixels[x, y] = (new_y, cb, cr)

    return image.convert("RGB")


def make_frame(avatar: Image.Image) -> Image.Image:
    frame = Image.new("RGB", (FRAME_W, FRAME_H), (0, 0, 0))
    frame.paste(avatar, (FRAME_X, 0))
    return frame


def write_robot36(frame: Image.Image, path: Path) -> None:
    Robot36(frame, SAMPLE_RATE, BITS).write_wav(str(path))


def write_robot24bw(frame: Image.Image, path: Path) -> None:
    Robot24BW(frame.convert("L"), SAMPLE_RATE, BITS).write_wav(str(path))


def wav_info(path: Path) -> str:
    with wave.open(str(path), "rb") as wav:
        return f"{wav.getnchannels()}ch {wav.getframerate()}Hz {wav.getsampwidth() * 8}bit {wav.getnframes() / wav.getframerate():.2f}s"


def make_row_legend(path: Path) -> None:
    rows = pattern_by_row()
    legend = Image.new("RGB", (WIDTH, HEIGHT), (18, 18, 20))
    draw = ImageDraw.Draw(legend)
    for y, tone in enumerate(rows):
        color = (245, 245, 235) if tone else (12, 12, 14)
        draw.line((0, y, WIDTH, y), fill=color)
    legend.save(path)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    make_row_legend(OUTDIR / "node19_sstv_morse_row_pattern.png")

    for mode in ("strong", "subtle", "highlight"):
        avatar = make_avatar(mode)
        frame = make_frame(avatar)
        gray_avatar = avatar.convert("L").convert("RGB")
        gray_frame = make_frame(gray_avatar)
        avatar_path = OUTDIR / f"node19_sstv_morse_rows_{mode}_avatar.png"
        frame_path = OUTDIR / f"node19_sstv_morse_rows_{mode}_frame.png"
        wav_path = OUTDIR / f"node19_sstv_morse_rows_{mode}_robot36.wav"
        gray_avatar_path = OUTDIR / f"node19_sstv_morse_rows_{mode}_gray_avatar.png"
        gray_frame_path = OUTDIR / f"node19_sstv_morse_rows_{mode}_gray_frame.png"
        gray_wav_path = OUTDIR / f"node19_sstv_morse_rows_{mode}_robot24bw.wav"
        avatar.save(avatar_path)
        frame.save(frame_path)
        write_robot36(frame, wav_path)
        gray_avatar.save(gray_avatar_path)
        gray_frame.save(gray_frame_path)
        write_robot24bw(gray_frame, gray_wav_path)
        print(avatar_path)
        print(frame_path)
        print(wav_path, wav_info(wav_path))
        print(gray_avatar_path)
        print(gray_frame_path)
        print(gray_wav_path, wav_info(gray_wav_path))

    print(f"message: {MESSAGE}")
    print("morse:", " ".join(MORSE[ch] for ch in MESSAGE.lower() if ch in MORSE))


if __name__ == "__main__":
    main()
