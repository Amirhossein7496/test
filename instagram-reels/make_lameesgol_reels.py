#!/usr/bin/env python3
"""Free high-quality Lameesgol Instagram Reels: motion graphics + music + HQ encode."""

from __future__ import annotations

import math
import subprocess
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ASSETS = Path("/home/ubuntu/.cursor/projects/workspace/assets")
OUT = Path("/workspace/instagram-reels/v2")
ART = Path("/opt/cursor/artifacts/lameesgol")

W, H = 1080, 1920
FPS = 30
BRAND = "Lameesgol"

IMAGES = {
    "twins": ASSETS / "01a1cb20-7fc1-7e01-b644-2243aaf44ca9.jpg",
    "red": ASSETS / "01a1cb20-7fec-732d-b766-35a1fa459815.jpg",
    "color": ASSETS / "01a1cb20-80d4-7714-aa6a-a60d9f5916d2.jpg",
    "purple": ASSETS / "01a1cb20-8207-7c18-b1ca-49f8763a0387.jpg",
}

FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_FA = "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Bold.ttf"
FONT_FA_REG = "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd[:10]), "...")
    subprocess.run(cmd, check=True)


def write_wav(path: Path, samples: np.ndarray, sr: int = 44100) -> None:
    samples = np.clip(samples, -1.0, 1.0)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        interleaved = np.empty(samples.shape[0] * 2, dtype=np.int16)
        pcm = (samples * 32767).astype(np.int16)
        interleaved[0::2] = pcm[:, 0]
        interleaved[1::2] = pcm[:, 1]
        wf.writeframes(interleaved.tobytes())


def env_adsr(n: int, sr: int, a=0.02, d=0.12, s=0.65, r=0.35) -> np.ndarray:
    attack = max(1, int(a * sr))
    decay = max(1, int(d * sr))
    release = max(1, int(r * sr))
    sustain_n = max(0, n - attack - decay - release)
    parts = [
        np.linspace(0, 1, attack, endpoint=False),
        np.linspace(1, s, decay, endpoint=False),
        np.full(sustain_n, s),
        np.linspace(s, 0, release, endpoint=True),
    ]
    e = np.concatenate(parts)
    if len(e) < n:
        e = np.pad(e, (0, n - len(e)))
    return e[:n]


def tone(freq: float, dur: float, sr: int, kind="pad", vol=0.2) -> np.ndarray:
    n = int(sr * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    if kind == "pad":
        sig = (
            0.55 * np.sin(2 * math.pi * freq * t)
            + 0.25 * np.sin(2 * math.pi * freq * 2.002 * t)
            + 0.12 * np.sin(2 * math.pi * freq * 3.004 * t)
            + 0.08 * np.sin(2 * math.pi * freq * 0.5 * t)
        )
        e = env_adsr(n, sr, 0.25, 0.35, 0.7, 0.8)
    elif kind == "pluck":
        sig = (
            0.7 * np.sin(2 * math.pi * freq * t)
            + 0.22 * np.sin(2 * math.pi * freq * 2.01 * t)
            + 0.08 * np.sin(2 * math.pi * freq * 3.03 * t)
        )
        e = np.exp(-t * 3.8) * (1 - np.exp(-t * 120))
    elif kind == "bell":
        sig = (
            0.5 * np.sin(2 * math.pi * freq * t)
            + 0.3 * np.sin(2 * math.pi * freq * 2.76 * t)
            + 0.15 * np.sin(2 * math.pi * freq * 5.4 * t)
        )
        e = np.exp(-t * 2.2)
    elif kind == "bass":
        sig = 0.8 * np.sin(2 * math.pi * freq * t) + 0.2 * np.sin(2 * math.pi * freq * 2 * t)
        e = env_adsr(n, sr, 0.01, 0.08, 0.55, 0.2)
    else:  # kick
        f = freq * np.exp(-t * 12)
        sig = np.sin(2 * math.pi * f * t)
        e = np.exp(-t * 8)
    return (sig * e * vol).astype(np.float64)


def soft_noise(n: int, vol=0.02) -> np.ndarray:
    return (np.random.randn(n) * vol).astype(np.float64)


def make_music(path: Path, seconds: float, mood: str = "soft") -> None:
    sr = 44100
    total = int(seconds * sr)
    L = np.zeros(total, dtype=np.float64)
    R = np.zeros(total, dtype=np.float64)

    if mood == "soft":
        chords = [
            [261.63, 329.63, 392.00, 523.25],  # C
            [220.00, 261.63, 329.63, 440.00],  # Am
            [174.61, 220.00, 261.63, 349.23],  # F
            [196.00, 246.94, 293.66, 392.00],  # G
        ]
        melody = [523.25, 587.33, 659.25, 523.25, 440.00, 493.88, 523.25, 392.00]
        melody_steps = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        loop = 8.0
        bass_root = [130.81, 110.00, 87.31, 98.00]
    else:
        chords = [
            [246.94, 311.13, 369.99, 493.88],
            [196.00, 246.94, 293.66, 392.00],
            [220.00, 277.18, 329.63, 440.00],
            [174.61, 220.00, 261.63, 349.23],
        ]
        melody = [493.88, 554.37, 659.25, 739.99, 659.25, 554.37, 493.88, 369.99]
        melody_steps = [0, 0.5, 1.0, 1.5, 2.0, 2.75, 3.5, 4.25]
        loop = 8.0
        bass_root = [123.47, 98.00, 110.00, 87.31]

    def place(buf: np.ndarray, clip: np.ndarray, start: int) -> None:
        end = min(len(buf), start + len(clip))
        if end > start:
            buf[start:end] += clip[: end - start]

    t0 = 0.0
    while t0 < seconds + 1:
        for i, freqs in enumerate(chords):
            start = int((t0 + i * 2.0) * sr)
            for j, f in enumerate(freqs):
                clip = tone(f, 2.5, sr, "pad", 0.045 + 0.01 * (j % 2))
                # stereo spread
                place(L, clip * (0.85 if j % 2 == 0 else 1.0), start)
                place(R, clip * (1.0 if j % 2 == 0 else 0.85), start)
            b = tone(bass_root[i], 1.8, sr, "bass", 0.09)
            place(L, b * 0.95, start)
            place(R, b * 0.95, start)

        for step, note in zip(melody_steps, melody):
            start = int((t0 + step) * sr)
            pl = tone(note, 1.1, sr, "pluck", 0.07)
            place(L, pl * 1.05, start)
            place(R, pl * 0.9, start + int(0.008 * sr))  # tiny delay for width
            if mood != "soft" and int(step * 2) % 2 == 0:
                bell = tone(note * 2, 0.8, sr, "bell", 0.025)
                place(L, bell, start)
                place(R, bell, start)

        # soft kick every half bar
        for k in range(0, 8, 2):
            start = int((t0 + k) * sr)
            kick = tone(58, 0.35, sr, "kick", 0.11)
            place(L, kick, start)
            place(R, kick, start)
            # soft hat
            hat = soft_noise(int(0.04 * sr), 0.035) * env_adsr(int(0.04 * sr), sr, 0.001, 0.01, 0.2, 0.03)
            place(L, hat, start + int(0.5 * sr))
            place(R, hat * 0.8, start + int(0.5 * sr))

        t0 += loop

    # simple slap reverb
    delay = int(0.18 * sr)
    wet_L = np.zeros_like(L)
    wet_R = np.zeros_like(R)
    wet_L[delay:] += L[:-delay] * 0.22
    wet_R[delay:] += R[:-delay] * 0.22
    delay2 = int(0.31 * sr)
    wet_L[delay2:] += R[:-delay2] * 0.12
    wet_R[delay2:] += L[:-delay2] * 0.12
    L = L + wet_L
    R = R + wet_R

    # soft air shimmer
    t = np.linspace(0, seconds, total, endpoint=False)
    shimmer = 0.006 * np.sin(2 * math.pi * 0.12 * t) * np.sin(2 * math.pi * 880 * t)
    L += shimmer
    R += shimmer * 0.9

    stereo = np.stack([L[:total], R[:total]], axis=1)
    fade = int(0.9 * sr)
    stereo[:fade] *= np.linspace(0, 1, fade)[:, None]
    stereo[-fade:] *= np.linspace(1, 0, fade)[:, None]
    peak = np.max(np.abs(stereo)) or 1.0
    stereo = stereo / peak * 0.82
    write_wav(path, stereo, sr)


def cover_frame(img: Image.Image, tw: int, th: int) -> Image.Image:
    iw, ih = img.size
    scale = max(tw / iw, th / ih)
    nw, nh = int(iw * scale + 0.5), int(ih * scale + 0.5)
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    left = (nw - tw) // 2
    top = (nh - th) // 2
    return img.crop((left, top, left + tw, top + th))


def make_brand_intro(path: Path, frames: int = 60) -> None:
    """Animated Lameesgol intro card as image sequence then encode."""
    seq = OUT / "intro_frames"
    seq.mkdir(parents=True, exist_ok=True)
    font_big = ImageFont.truetype(FONT_BOLD, 96)
    font_sub = ImageFont.truetype(FONT_REG, 36)
    for i in range(frames):
        p = i / max(1, frames - 1)
        img = Image.new("RGB", (W, H), (18, 12, 18))
        draw = ImageDraw.Draw(img, "RGBA")
        # soft radial glow
        glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        r = int(180 + 420 * min(1, p * 1.4))
        cx, cy = W // 2, H // 2 - 40
        for rr in range(r, 0, -8):
            a = int(28 * (1 - rr / r) * min(1, p * 2))
            gd.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=(255, 120, 160, a))
        img = Image.alpha_composite(img.convert("RGBA"), glow).convert("RGB")
        draw = ImageDraw.Draw(img, "RGBA")

        # brand text fade + slight scale feel via alpha
        alpha = int(255 * min(1, max(0, (p - 0.12) / 0.35)))
        y_off = int((1 - min(1, max(0, (p - 0.1) / 0.4))) * 40)
        bbox = draw.textbbox((0, 0), BRAND, font=font_big)
        tw = bbox[2] - bbox[0]
        x = (W - tw) // 2
        y = H // 2 - 80 + y_off
        draw.text((x + 3, y + 4), BRAND, font=font_big, fill=(0, 0, 0, alpha))
        draw.text((x, y), BRAND, font=font_big, fill=(255, 245, 250, alpha))

        sub = "handmade flower boutique"
        bbox = draw.textbbox((0, 0), sub, font=font_sub)
        sw = bbox[2] - bbox[0]
        sa = int(255 * min(1, max(0, (p - 0.4) / 0.35)))
        draw.text(((W - sw) // 2, y + 120), sub, font=font_sub, fill=(255, 200, 220, sa))

        # thin accent line
        line_w = int(min(1, max(0, (p - 0.25) / 0.4)) * 280)
        if line_w > 0:
            draw.rectangle(
                [W // 2 - line_w, y + 105, W // 2 + line_w, y + 108],
                fill=(255, 160, 190, alpha),
            )
        img.save(seq / f"f{i:04d}.png", optimize=True)

    run(
        [
            "ffmpeg", "-y", "-framerate", str(FPS),
            "-i", str(seq / "f%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "slow", "-crf", "14",
            str(path),
        ]
    )


def make_overlay_png(
    path: Path,
    title: str,
    subtitle: str = "",
    show_brand: bool = True,
) -> None:
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # bottom gradient
    for y in range(H - 620, H):
        a = int(200 * ((y - (H - 620)) / 620))
        draw.rectangle([0, y, W, y + 1], fill=(10, 6, 12, a))
    # top brand chip
    if show_brand:
        brand_font = ImageFont.truetype(FONT_BOLD, 42)
        bbox = draw.textbbox((0, 0), BRAND, font=brand_font)
        bw = bbox[2] - bbox[0]
        bx, by = 56, 64
        # rounded-ish pill background
        pad_x, pad_y = 28, 16
        draw.rounded_rectangle(
            [bx - pad_x, by - pad_y, bx + bw + pad_x, by + 42 + pad_y],
            radius=28,
            fill=(255, 255, 255, 48),
        )
        draw.text((bx + 1, by + 2), BRAND, font=brand_font, fill=(0, 0, 0, 120))
        draw.text((bx, by), BRAND, font=brand_font, fill=(255, 250, 253, 235))

    title_font = ImageFont.truetype(FONT_FA if any(ord(c) > 127 for c in title) else FONT_BOLD, 78)
    sub_font = ImageFont.truetype(FONT_FA_REG if any(ord(c) > 127 for c in subtitle) else FONT_REG, 42)

    kwargs = {}
    if any(ord(c) > 127 for c in title):
        kwargs = {"direction": "rtl", "language": "fa"}
    bbox = draw.textbbox((0, 0), title, font=title_font, **kwargs)
    tw = bbox[2] - bbox[0]
    x = (W - tw) // 2
    y = H - 420
    draw.text((x + 2, y + 3), title, font=title_font, fill=(0, 0, 0, 170), **kwargs)
    draw.text((x, y), title, font=title_font, fill=(255, 248, 252, 245), **kwargs)

    if subtitle:
        skw = {"direction": "rtl", "language": "fa"} if any(ord(c) > 127 for c in subtitle) else {}
        bbox = draw.textbbox((0, 0), subtitle, font=sub_font, **skw)
        sw = bbox[2] - bbox[0]
        sx = (W - sw) // 2
        draw.text((sx + 1, y + 100), subtitle, font=sub_font, fill=(0, 0, 0, 140), **skw)
        draw.text((sx, y + 98), subtitle, font=sub_font, fill=(255, 220, 235, 220), **skw)

    # accent line under title
    draw.rectangle([W // 2 - 90, y + 88, W // 2 + 90, y + 91], fill=(255, 150, 180, 200))
    img.save(path)


def make_sparkle_overlay(seq_dir: Path, frames: int, seed: int = 7) -> Path:
    """Transparent particle motion graphics as PNG sequence; returns pattern path."""
    seq_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n = 28
    xs = rng.uniform(40, W - 40, n)
    ys = rng.uniform(80, H - 200, n)
    phases = rng.uniform(0, 2 * math.pi, n)
    speeds = rng.uniform(0.4, 1.3, n)
    sizes = rng.integers(2, 6, n)
    for i in range(frames):
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        t = i / FPS
        for j in range(n):
            a = int(90 + 110 * (0.5 + 0.5 * math.sin(t * speeds[j] * 2 + phases[j])))
            x = xs[j] + 18 * math.sin(t * 0.7 * speeds[j] + phases[j])
            y = ys[j] - (t * 12 * speeds[j]) % (H * 0.35)
            s = int(sizes[j])
            draw.ellipse([x - s, y - s, x + s, y + s], fill=(255, 230, 245, a))
            if s >= 4:
                draw.ellipse([x - 1, y - s * 2, x + 1, y + s * 2], fill=(255, 255, 255, a // 2))
                draw.ellipse([x - s * 2, y - 1, x + s * 2, y + 1], fill=(255, 255, 255, a // 2))
        img.save(seq_dir / f"f{i:04d}.png")
    return seq_dir / "f%04d.png"


def ken_burns_hq(
    image: Path,
    out: Path,
    seconds: float,
    mode: str = "in",
) -> None:
    """High-quality Ken Burns from full-res source, no early downscale."""
    frames = int(seconds * FPS)
    # Build oversized canvas (~1.35x) from source with LANCZOS
    over = 1.35
    base = OUT / f"_hq_{out.stem}.jpg"
    img = Image.open(image).convert("RGB")
    # slight denoise/sharpen for polish without AI
    img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=80, threshold=2))
    cover_frame(img, int(W * over), int(H * over)).save(base, quality=97, subsampling=0)

    if mode == "in":
        z, x, y = "min(1.0+0.00115*on,1.22)", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    elif mode == "out":
        z, x, y = "if(eq(on,1),1.22,max(1.22-0.00115*on,1.0))", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    elif mode == "left":
        z, x, y = "1.16", "min(0.55*on,iw-iw/zoom)", "ih/2-(ih/zoom/2)"
    elif mode == "right":
        z, x, y = "1.16", "max((iw-iw/zoom)-0.55*on,0)", "ih/2-(ih/zoom/2)"
    else:
        z, x, y = "1.18", "iw/2-(iw/zoom/2)", "max((ih-ih/zoom)-0.4*on,0)"

    vf = (
        f"zoompan=z='{z}':x='{x}':y='{y}':d={frames}:s={W}x{H}:fps={FPS},"
        f"eq=contrast=1.04:saturation=1.08:brightness=0.015,"
        f"unsharp=5:5:0.6:5:5:0.0,"
        f"vignette=PI/6,"
        f"format=yuv420p"
    )
    run(
        [
            "ffmpeg", "-y", "-loop", "1", "-i", str(base),
            "-vf", vf,
            "-t", f"{seconds:.2f}",
            "-r", str(FPS),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "medium", "-crf", "15",
            "-movflags", "+faststart",
            str(out),
        ]
    )


def composite_clip(
    video: Path,
    overlay_png: Path,
    sparkle_pattern: Path,
    out: Path,
    seconds: float,
    fade_text_at: float,
) -> None:
    # text fades in late; sparkles whole time at low opacity
    fc = (
        f"[0:v]format=rgba[base];"
        f"[1:v]format=rgba,fade=t=in:st={fade_text_at:.2f}:d=0.55:alpha=1[txt];"
        f"[2:v]format=rgba,colorchannelmixer=aa=0.55[sp];"
        f"[base][sp]overlay=0:0[v1];"
        f"[v1][txt]overlay=0:0,format=yuv420p"
    )
    run(
        [
            "ffmpeg", "-y",
            "-i", str(video),
            "-loop", "1", "-i", str(overlay_png),
            "-framerate", str(FPS), "-i", str(sparkle_pattern),
            "-filter_complex", fc,
            "-t", f"{seconds:.2f}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "medium", "-crf", "15",
            str(out),
        ]
    )


def xfade_concat(clips: list[Path], out: Path, overlap: float = 0.55) -> float:
    if len(clips) == 1:
        run(["ffmpeg", "-y", "-i", str(clips[0]), "-c", "copy", str(out)])
        return float(
            subprocess.check_output(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(clips[0])]
            ).decode().strip()
        )
    inputs: list[str] = []
    durs = []
    for c in clips:
        inputs += ["-i", str(c)]
        durs.append(
            float(
                subprocess.check_output(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", str(c)]
                ).decode().strip()
            )
        )
    parts = []
    prev = "[0:v]"
    offset = durs[0] - overlap
    for i in range(1, len(clips)):
        label = f"[v{i}]" if i < len(clips) - 1 else "[vout]"
        # nicer transitions alternating
        tr = ["fade", "smoothleft", "smoothright", "fadeblack"][i % 4]
        parts.append(f"{prev}[{i}:v]xfade=transition={tr}:duration={overlap}:offset={offset:.3f}{label}")
        prev = label
        if i < len(clips) - 1:
            offset += durs[i] - overlap
    total = sum(durs) - overlap * (len(clips) - 1)
    run(
        [
            "ffmpeg", "-y", *inputs,
            "-filter_complex", ";".join(parts),
            "-map", "[vout]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "slow", "-crf", "14",
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
            f"[1:a]atrim=0:{duration:.3f},afade=t=in:st=0:d=0.7,afade=t=out:st={max(0, duration-1.1):.3f}:d=1.0,volume=0.95[a]",
            "-map", "0:v", "-map", "[a]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
            "-shortest", "-movflags", "+faststart",
            str(out),
        ]
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)

    print("Music...")
    music_soft = OUT / "music_soft.wav"
    music_up = OUT / "music_upbeat.wav"
    make_music(music_soft, 22, "soft")
    make_music(music_up, 18, "upbeat")

    print("Brand intro + overlays...")
    intro = OUT / "intro.mp4"
    make_brand_intro(intro, frames=48)  # 1.6s

    overlays = {
        "collection": OUT / "ov_collection.png",
        "red": OUT / "ov_red.png",
        "color": OUT / "ov_color.png",
        "purple": OUT / "ov_purple.png",
    }
    make_overlay_png(overlays["collection"], "دسته گل دست‌ساز", "هدیه خاص و ماندگار")
    make_overlay_png(overlays["red"], "قرمز عاشقانه", "آماده سفارش")
    make_overlay_png(overlays["color"], "رنگ و حال خوب", "کار دست")
    make_overlay_png(overlays["purple"], "بنفش خاص", "جزئیات دست‌ساز")

    # sparkle overlays (reuse two lengths)
    sparkle_short = make_sparkle_overlay(OUT / "sparkle_short", frames=int(4.5 * FPS), seed=3)
    sparkle_long = make_sparkle_overlay(OUT / "sparkle_long", frames=int(5.0 * FPS), seed=11)

    print("Ken Burns HQ clips...")
    raw = {
        "twins": OUT / "raw_twins.mp4",
        "red": OUT / "raw_red.mp4",
        "color": OUT / "raw_color.mp4",
        "purple": OUT / "raw_purple.mp4",
        "red2": OUT / "raw_red2.mp4",
        "twins2": OUT / "raw_twins2.mp4",
        "color2": OUT / "raw_color2.mp4",
        "purple2": OUT / "raw_purple2.mp4",
    }
    ken_burns_hq(IMAGES["twins"], raw["twins"], 3.6, "left")
    ken_burns_hq(IMAGES["red"], raw["red"], 3.6, "in")
    ken_burns_hq(IMAGES["color"], raw["color"], 3.6, "right")
    ken_burns_hq(IMAGES["purple"], raw["purple"], 4.0, "in")
    ken_burns_hq(IMAGES["red"], raw["red2"], 4.0, "up")
    ken_burns_hq(IMAGES["twins"], raw["twins2"], 3.8, "in")
    ken_burns_hq(IMAGES["color"], raw["color2"], 4.0, "out")
    ken_burns_hq(IMAGES["purple"], raw["purple2"], 3.8, "left")

    print("Composite motion graphics...")
    comp = {
        "twins": OUT / "c_twins.mp4",
        "red": OUT / "c_red.mp4",
        "color": OUT / "c_color.mp4",
        "purple": OUT / "c_purple.mp4",
        "red2": OUT / "c_red2.mp4",
        "twins2": OUT / "c_twins2.mp4",
        "color2": OUT / "c_color2.mp4",
        "purple2": OUT / "c_purple2.mp4",
    }
    # brand-only overlays early; titles late on last clips
    brand_only = OUT / "ov_brand_only.png"
    make_overlay_png(brand_only, "", "", show_brand=True)
    # clear bottom text for brand-only: remake with empty and just brand chip
    # (make_overlay_png with empty title still draws accent — rebuild minimal)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    brand_font = ImageFont.truetype(FONT_BOLD, 42)
    bbox = draw.textbbox((0, 0), BRAND, font=brand_font)
    bw = bbox[2] - bbox[0]
    bx, by = 56, 64
    draw.rounded_rectangle([bx - 28, by - 16, bx + bw + 28, by + 42 + 16], radius=28, fill=(255, 255, 255, 48))
    draw.text((bx, by), BRAND, font=brand_font, fill=(255, 250, 253, 235))
    img.save(brand_only)

    composite_clip(raw["twins"], brand_only, sparkle_short, comp["twins"], 3.6, 0.2)
    composite_clip(raw["red"], brand_only, sparkle_short, comp["red"], 3.6, 0.2)
    composite_clip(raw["color"], brand_only, sparkle_short, comp["color"], 3.6, 0.2)
    composite_clip(raw["purple"], overlays["collection"], sparkle_long, comp["purple"], 4.0, 1.4)
    composite_clip(raw["red2"], overlays["red"], sparkle_long, comp["red2"], 4.0, 1.5)
    composite_clip(raw["twins2"], brand_only, sparkle_short, comp["twins2"], 3.8, 0.2)
    composite_clip(raw["color2"], overlays["color"], sparkle_long, comp["color2"], 4.0, 1.5)
    composite_clip(raw["purple2"], overlays["purple"], sparkle_long, comp["purple2"], 3.8, 1.4)

    print("Assemble reels...")
    # Collection
    col_v = OUT / "collection_v.mp4"
    d1 = xfade_concat([intro, comp["twins"], comp["red"], comp["color"], comp["purple"]], col_v, 0.5)
    out1 = ART / "Lameesgol-collection.mp4"
    mux_audio(col_v, music_soft, out1, d1)

    # Red
    red_v = OUT / "red_v.mp4"
    d2 = xfade_concat([intro, comp["red"], comp["red2"]], red_v, 0.55)
    out2 = ART / "Lameesgol-red-romance.mp4"
    mux_audio(red_v, music_soft, out2, d2)

    # Color
    color_v = OUT / "color_v.mp4"
    d3 = xfade_concat([intro, comp["twins2"], comp["color2"]], color_v, 0.55)
    out3 = ART / "Lameesgol-color-pop.mp4"
    mux_audio(color_v, music_up, out3, d3)

    # Purple
    purple_v = OUT / "purple_v.mp4"
    d4 = xfade_concat([intro, comp["purple"], comp["purple2"]], purple_v, 0.5)
    out4 = ART / "Lameesgol-purple-closeup.mp4"
    mux_audio(purple_v, music_soft, out4, d4)

    # copy into workspace
    for p in [out1, out2, out3, out4]:
        run(["cp", "-f", str(p), str(OUT / p.name)])
        # also top-level artifacts for easy discovery
        run(["cp", "-f", str(p), str(Path("/opt/cursor/artifacts") / p.name)])

    print("\nDone:")
    for p in [out1, out2, out3, out4]:
        d = float(
            subprocess.check_output(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(p)]
            ).decode().strip()
        )
        print(f"  {p.name}: {d:.1f}s, {p.stat().st_size/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
