#!/usr/bin/env python3
"""Build Instagram Reels from product photos with Ken Burns + music."""

from __future__ import annotations

import math
import struct
import subprocess
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ASSETS = Path("/home/ubuntu/.cursor/projects/workspace/assets")
OUT = Path("/workspace/instagram-reels")
ARTIFACTS = Path("/opt/cursor/artifacts")
FONT = "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Bold.ttf"
FONT_REG = "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf"

W, H = 1080, 1920
FPS = 30

IMAGES = {
    "twins": ASSETS / "01a1cb20-7fc1-7e01-b644-2243aaf44ca9.jpg",
    "red": ASSETS / "01a1cb20-7fec-732d-b766-35a1fa459815.jpg",
    "color": ASSETS / "01a1cb20-80d4-7714-aa6a-a60d9f5916d2.jpg",
    "purple": ASSETS / "01a1cb20-8207-7c18-b1ca-49f8763a0387.jpg",
}


def fa(text: str) -> str:
    # Noto + Pillow already shape Arabic/Persian correctly; avoid double-bidi.
    return text


def write_wav(path: Path, samples: np.ndarray, sr: int = 44100) -> None:
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def note(freq: float, dur: float, sr: int, vol: float = 0.2, kind: str = "pad") -> np.ndarray:
    n = int(sr * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    if kind == "pad":
        wave_ = (
            0.55 * np.sin(2 * math.pi * freq * t)
            + 0.28 * np.sin(2 * math.pi * freq * 2 * t)
            + 0.12 * np.sin(2 * math.pi * freq * 3 * t)
            + 0.05 * np.sin(2 * math.pi * freq * 0.5 * t)
        )
        env = np.minimum(t / 0.35, 1.0) * np.exp(-t * 0.55)
    elif kind == "pluck":
        wave_ = (
            0.7 * np.sin(2 * math.pi * freq * t)
            + 0.25 * np.sin(2 * math.pi * freq * 2.01 * t)
            + 0.1 * np.sin(2 * math.pi * freq * 3.02 * t)
        )
        env = np.exp(-t * 4.2) * (1 - np.exp(-t * 90))
    else:  # soft kick/swell
        wave_ = np.sin(2 * math.pi * (freq * np.exp(-t * 8)) * t)
        env = np.exp(-t * 6)
    return (wave_ * env * vol).astype(np.float64)


def mix_at(buf: np.ndarray, clip: np.ndarray, start: int) -> None:
    end = min(len(buf), start + len(clip))
    if end <= start:
        return
    buf[start:end] += clip[: end - start]


def make_music(path: Path, seconds: float, mood: str = "soft") -> None:
    sr = 44100
    total = int(sr * seconds) + sr
    buf = np.zeros(total, dtype=np.float64)

    # Soft aesthetic chord beds + gentle plucks (original, not sampled)
    if mood == "soft":
        progression = [
            (0.0, [261.63, 329.63, 392.00, 523.25]),  # C
            (2.0, [220.00, 261.63, 329.63, 440.00]),  # Am
            (4.0, [174.61, 220.00, 261.63, 349.23]),  # F
            (6.0, [196.00, 246.94, 293.66, 392.00]),  # G
        ]
        loop = 8.0
        pluck_pattern = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        pluck_notes = [523.25, 587.33, 659.25, 783.99, 659.25, 587.33, 523.25, 440.00]
        pad_vol, pluck_vol = 0.11, 0.055
    elif mood == "warm":
        progression = [
            (0.0, [233.08, 293.66, 349.23, 466.16]),  # Bb
            (2.0, [174.61, 220.00, 261.63, 349.23]),  # F
            (4.0, [196.00, 246.94, 293.66, 392.00]),  # G
            (6.0, [207.65, 261.63, 311.13, 415.30]),  # Ab
        ]
        loop = 8.0
        pluck_pattern = [0.0, 0.75, 1.5, 2.5, 3.25, 4.0, 5.0, 6.0, 7.0]
        pluck_notes = [466.16, 523.25, 587.33, 698.46, 587.33, 523.25, 466.16, 392.00, 349.23]
        pad_vol, pluck_vol = 0.12, 0.06
    else:  # dreamy
        progression = [
            (0.0, [246.94, 311.13, 369.99, 493.88]),  # Bm-ish
            (2.5, [220.00, 277.18, 329.63, 440.00]),
            (5.0, [196.00, 246.94, 293.66, 392.00]),
            (7.5, [174.61, 220.00, 261.63, 349.23]),
        ]
        loop = 10.0
        pluck_pattern = [0.0, 1.25, 2.5, 3.75, 5.0, 6.25, 7.5, 8.75]
        pluck_notes = [493.88, 554.37, 659.25, 739.99, 659.25, 554.37, 493.88, 369.99]
        pad_vol, pluck_vol = 0.10, 0.05

    t = 0.0
    while t < seconds + 1:
        for offset, freqs in progression:
            start = int((t + offset) * sr)
            for f in freqs:
                mix_at(buf, note(f, 2.4, sr, pad_vol, "pad"), start)
        for po, pn in zip(pluck_pattern, pluck_notes):
            mix_at(buf, note(pn, 1.4, sr, pluck_vol, "pluck"), int((t + po) * sr))
        # soft low pulse
        mix_at(buf, note(55.0, 0.35, sr, 0.08, "thud"), int(t * sr))
        mix_at(buf, note(55.0, 0.35, sr, 0.06, "thud"), int((t + 4) * sr))
        t += loop

    # gentle high shimmer
    shimmer_t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    shimmer = 0.008 * np.sin(2 * math.pi * 0.15 * shimmer_t) * np.sin(2 * math.pi * 880 * shimmer_t)
    buf[: len(shimmer)] += shimmer

    # fade in/out
    fade = int(0.8 * sr)
    buf[:fade] *= np.linspace(0, 1, fade)
    out_samples = buf[: int(seconds * sr)]
    out_samples[-fade:] *= np.linspace(1, 0, fade)
    # soft normalize
    peak = np.max(np.abs(out_samples)) or 1.0
    out_samples = out_samples / peak * 0.75
    write_wav(path, out_samples, sr)


def cover_crop(img: Image.Image, tw: int, th: int) -> Image.Image:
    iw, ih = img.size
    scale = max(tw / iw, th / ih)
    nw, nh = int(iw * scale + 0.5), int(ih * scale + 0.5)
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    left = (nw - tw) // 2
    top = (nh - th) // 2
    return img.crop((left, top, left + tw, top + th))


def prepare_base(src: Path, out: Path, overscan: float = 1.25) -> None:
    """Create an oversized vertical frame for Ken Burns zoom/pan."""
    img = Image.open(src).convert("RGB")
    tw, th = int(W * overscan), int(H * overscan)
    cover_crop(img, tw, th).save(out, quality=95)


def make_title_png(path: Path, title: str, subtitle: str = "") -> None:
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    title_font = ImageFont.truetype(FONT, 86)
    sub_font = ImageFont.truetype(FONT_REG, 48)

    # soft bottom gradient bar for readability
    for y in range(H - 560, H):
        a = int(190 * ((y - (H - 560)) / 560))
        draw.rectangle([0, y, W, y + 1], fill=(12, 8, 16, a))

    t = fa(title)
    bbox = draw.textbbox((0, 0), t, font=title_font, direction="rtl", language="fa")
    tw = bbox[2] - bbox[0]
    x = (W - tw) // 2
    y = H - 380
    draw.text((x + 3, y + 4), t, font=title_font, fill=(0, 0, 0, 180), direction="rtl", language="fa")
    draw.text((x, y), t, font=title_font, fill=(255, 250, 253, 245), direction="rtl", language="fa")

    if subtitle:
        s = fa(subtitle)
        bbox = draw.textbbox((0, 0), s, font=sub_font, direction="rtl", language="fa")
        sw = bbox[2] - bbox[0]
        sx = (W - sw) // 2
        draw.text((sx + 2, y + 108), s, font=sub_font, fill=(0, 0, 0, 150), direction="rtl", language="fa")
        draw.text((sx, y + 105), s, font=sub_font, fill=(255, 232, 242, 220), direction="rtl", language="fa")

    img.save(path)


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd[:8]), "...")
    subprocess.run(cmd, check=True)


def ken_burns_clip(
    image: Path,
    out: Path,
    seconds: float,
    mode: str = "in",
    title_png: Path | None = None,
) -> None:
    """Create a vertical Ken Burns clip from a still."""
    frames = int(seconds * FPS)
    # zoompan expressions
    if mode == "in":
        z = "min(zoom+0.0012,1.18)"
        x = "iw/2-(iw/zoom/2)"
        y = "ih/2-(ih/zoom/2)"
    elif mode == "out":
        z = "if(eq(on,1),1.18,max(zoom-0.0012,1.0))"
        x = "iw/2-(iw/zoom/2)"
        y = "ih/2-(ih/zoom/2)"
    elif mode == "left":
        z = "1.12"
        x = "if(eq(on,1),0,min(x+0.55,iw-iw/zoom))"
        y = "ih/2-(ih/zoom/2)"
    elif mode == "right":
        z = "1.12"
        x = "if(eq(on,1),(iw-iw/zoom),max(x-0.55,0))"
        y = "ih/2-(ih/zoom/2)"
    else:  # up
        z = "1.14"
        x = "iw/2-(iw/zoom/2)"
        y = "if(eq(on,1),(ih-ih/zoom),max(y-0.45,0))"

    base = OUT / f"_base_{out.stem}.jpg"
    prepare_base(image, base, 1.28)

    vf = (
        f"zoompan=z='{z}':x='{x}':y='{y}':d={frames}:s={W}x{H}:fps={FPS},"
        f"eq=contrast=1.05:saturation=1.12:brightness=0.02,"
        f"vignette=PI/5,"
        f"format=yuv420p"
    )

    if title_png and title_png.exists():
        # overlay title for last portion via fade (loop BOTH stills)
        fade_st = max(0.2, seconds - 2.4)
        filter_complex = (
            f"[0:v]zoompan=z='{z}':x='{x}':y='{y}':d={frames}:s={W}x{H}:fps={FPS},"
            f"eq=contrast=1.05:saturation=1.12:brightness=0.02,vignette=PI/5[v];"
            f"[1:v]format=rgba,fade=t=in:st={fade_st:.2f}:d=0.6:alpha=1[ov];"
            f"[v][ov]overlay=0:0:format=auto,format=yuv420p"
        )
        run(
            [
                "ffmpeg", "-y",
                "-loop", "1", "-i", str(base),
                "-loop", "1", "-i", str(title_png),
                "-filter_complex", filter_complex,
                "-t", f"{seconds:.2f}",
                "-r", str(FPS),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-preset", "veryfast", "-crf", "18",
                str(out),
            ]
        )
    else:
        run(
            [
                "ffmpeg", "-y",
                "-loop", "1", "-i", str(base),
                "-vf", vf,
                "-t", f"{seconds:.2f}",
                "-r", str(FPS),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-preset", "veryfast", "-crf", "18",
                str(out),
            ]
        )


def xfade_concat(clips: list[Path], out: Path, overlap: float = 0.55) -> float:
    """Crossfade concatenate clips; returns total duration."""
    if len(clips) == 1:
        run(["ffmpeg", "-y", "-i", str(clips[0]), "-c", "copy", str(out)])
        # probe duration
        dur = float(
            subprocess.check_output(
                [
                    "ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1", str(clips[0]),
                ]
            ).decode().strip()
        )
        return dur

    inputs: list[str] = []
    for c in clips:
        inputs += ["-i", str(c)]

    durs = []
    for c in clips:
        d = float(
            subprocess.check_output(
                [
                    "ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1", str(c),
                ]
            ).decode().strip()
        )
        durs.append(d)

    # Build xfade chain
    fc_parts = []
    # offset accumulates
    prev = "[0:v]"
    offset = durs[0] - overlap
    for i in range(1, len(clips)):
        out_label = f"[v{i}]" if i < len(clips) - 1 else "[vout]"
        fc_parts.append(
            f"{prev}[{i}:v]xfade=transition=fade:duration={overlap}:offset={offset:.3f}{out_label}"
        )
        prev = out_label
        if i < len(clips) - 1:
            offset += durs[i] - overlap

    total = sum(durs) - overlap * (len(clips) - 1)
    filter_complex = ";".join(fc_parts)
    run(
        [
            "ffmpeg", "-y", *inputs,
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "veryfast", "-crf", "18",
            "-t", f"{total:.3f}",
            str(out),
        ]
    )
    return total


def mux_audio(video: Path, audio: Path, out: Path, duration: float) -> None:
    run(
        [
            "ffmpeg", "-y",
            "-i", str(video),
            "-i", str(audio),
            "-filter_complex",
            f"[1:a]atrim=0:{duration:.3f},afade=t=in:st=0:d=0.8,afade=t=out:st={max(0, duration-1.2):.3f}:d=1.2[a]",
            "-map", "0:v", "-map", "[a]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            str(out),
        ]
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    # Titles
    titles = {
        "collection": OUT / "title_collection.png",
        "red": OUT / "title_red.png",
        "color": OUT / "title_color.png",
        "purple": OUT / "title_purple.png",
    }
    make_title_png(titles["collection"], "دسته گل دست‌ساز", "هديه خاص و ماندگار")
    make_title_png(titles["red"], "قرمز عاشقانه", "آماده سفارش")
    make_title_png(titles["color"], "رنگ و حال خوب", "کار دست")
    make_title_png(titles["purple"], "بنفش خاص", "جزئیات دست‌ساز")

    # Music
    music_soft = OUT / "music_soft.wav"
    music_warm = OUT / "music_warm.wav"
    music_dream = OUT / "music_dream.wav"
    print("Generating music...")
    make_music(music_soft, 20, "soft")
    make_music(music_warm, 16, "warm")
    make_music(music_dream, 14, "dreamy")

    # --- Reel 1: full collection ---
    print("Building collection clips...")
    c1 = OUT / "clip_twins.mp4"
    c2 = OUT / "clip_red.mp4"
    c3 = OUT / "clip_color.mp4"
    c4 = OUT / "clip_purple.mp4"
    ken_burns_clip(IMAGES["twins"], c1, 3.4, "left")
    ken_burns_clip(IMAGES["red"], c2, 3.4, "in")
    ken_burns_clip(IMAGES["color"], c3, 3.4, "right")
    ken_burns_clip(IMAGES["purple"], c4, 3.8, "in", titles["collection"])

    collection_v = OUT / "collection_v.mp4"
    dur1 = xfade_concat([c1, c2, c3, c4], collection_v, 0.5)
    collection_out = ARTIFACTS / "reel-collection.mp4"
    mux_audio(collection_v, music_soft, collection_out, dur1)

    # --- Reel 2: red romance ---
    print("Building red reel...")
    r1 = OUT / "red_a.mp4"
    r2 = OUT / "red_b.mp4"
    ken_burns_clip(IMAGES["red"], r1, 4.0, "in")
    ken_burns_clip(IMAGES["red"], r2, 4.2, "up", titles["red"])
    red_v = OUT / "red_v.mp4"
    dur2 = xfade_concat([r1, r2], red_v, 0.6)
    red_out = ARTIFACTS / "reel-red-romance.mp4"
    mux_audio(red_v, music_warm, red_out, dur2)

    # --- Reel 3: color pop (twins + colorful) ---
    print("Building color reel...")
    k1 = OUT / "color_a.mp4"
    k2 = OUT / "color_b.mp4"
    ken_burns_clip(IMAGES["twins"], k1, 3.8, "in")
    ken_burns_clip(IMAGES["color"], k2, 4.2, "out", titles["color"])
    color_v = OUT / "color_v.mp4"
    dur3 = xfade_concat([k1, k2], color_v, 0.55)
    color_out = ARTIFACTS / "reel-color-pop.mp4"
    mux_audio(color_v, music_soft, color_out, dur3)

    # --- Reel 4: purple close-up ---
    print("Building purple reel...")
    p1 = OUT / "purple_a.mp4"
    p2 = OUT / "purple_b.mp4"
    ken_burns_clip(IMAGES["purple"], p1, 3.6, "in")
    ken_burns_clip(IMAGES["purple"], p2, 3.8, "left", titles["purple"])
    purple_v = OUT / "purple_v.mp4"
    dur4 = xfade_concat([p1, p2], purple_v, 0.5)
    purple_out = ARTIFACTS / "reel-purple-closeup.mp4"
    mux_audio(purple_v, music_dream, purple_out, dur4)

    # Also copy into workspace for easy access
    for src in [collection_out, red_out, color_out, purple_out]:
        dest = OUT / src.name
        run(["cp", "-f", str(src), str(dest)])

    print("\nDone:")
    for p in [collection_out, red_out, color_out, purple_out]:
        size = p.stat().st_size / (1024 * 1024)
        d = subprocess.check_output(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(p),
            ]
        ).decode().strip()
        print(f"  {p.name}: {float(d):.1f}s, {size:.1f} MB")


if __name__ == "__main__":
    main()
