"""Benchmark Mojo kernels against Pillow on identical uint8 images."""

from __future__ import annotations

import math
import os
import platform
import sys
import time
from collections.abc import Callable

import numpy as np
from PIL import Image as PILImage
from PIL import ImageFilter as PILImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from mojopillow import Image, ImageFilter  # noqa: E402


def timeit(function: Callable[[], object], repeats: int = 5) -> float:
    best = math.inf
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main() -> None:
    rng = np.random.default_rng(123)
    rgb = rng.integers(0, 256, (2160, 3840, 3), dtype=np.uint8)
    rgba = rng.integers(0, 256, (2160, 3840, 4), dtype=np.uint8)
    rgb_second = np.roll(rgb, 97, axis=1)
    rgba_second = np.roll(rgba, 71, axis=0)

    ours_rgb = Image.fromarray(rgb)
    pillow_rgb = PILImage.fromarray(rgb)
    ours_rgba = Image.fromarray(rgba)
    pillow_rgba = PILImage.fromarray(rgba)
    ours_rgb_second = Image.fromarray(rgb_second)
    pillow_rgb_second = PILImage.fromarray(rgb_second)
    ours_rgba_second = Image.fromarray(rgba_second)
    pillow_rgba_second = PILImage.fromarray(rgba_second)

    cases: list[
        tuple[str, Callable[[], object], Callable[[], object]]
    ] = [
        (
            "RGB bilinear resize 4K -> 1080p",
            lambda: ours_rgb.resize((1920, 1080), Image.Resampling.BILINEAR),
            lambda: pillow_rgb.resize(
                (1920, 1080), PILImage.Resampling.BILINEAR
            ),
        ),
        (
            "RGBA Lanczos resize 4K -> 1080p",
            lambda: ours_rgba.resize((1920, 1080), Image.Resampling.LANCZOS),
            lambda: pillow_rgba.resize(
                (1920, 1080), PILImage.Resampling.LANCZOS
            ),
        ),
        (
            "RGB SHARPEN convolution 4K",
            lambda: ours_rgb.filter(ImageFilter.SHARPEN),
            lambda: pillow_rgb.filter(PILImageFilter.SHARPEN),
        ),
        (
            "RGB -> L conversion 4K",
            lambda: ours_rgb.convert("L"),
            lambda: pillow_rgb.convert("L"),
        ),
        (
            "RGB constant-alpha blend 4K",
            lambda: Image.blend(ours_rgb, ours_rgb_second, 0.35),
            lambda: PILImage.blend(pillow_rgb, pillow_rgb_second, 0.35),
        ),
        (
            "RGBA alpha composite 4K",
            lambda: Image.alpha_composite(ours_rgba, ours_rgba_second),
            lambda: PILImage.alpha_composite(
                pillow_rgba, pillow_rgba_second
            ),
        ),
    ]

    rows: list[tuple[str, float, float, float]] = []
    for name, mojo_function, pillow_function in cases:
        mojo_output = np.asarray(mojo_function())
        pillow_output = np.asarray(pillow_function())
        tolerance = 1 if "blend" in name else 0
        error = np.max(
            np.abs(mojo_output.astype(np.int16) - pillow_output.astype(np.int16))
        )
        if error > tolerance:
            raise RuntimeError(f"{name} parity check failed: max error {error}")
        mojo_seconds = timeit(mojo_function)
        pillow_seconds = timeit(pillow_function)
        rows.append(
            (name, mojo_seconds, pillow_seconds, pillow_seconds / mojo_seconds)
        )

    print(f"Machine: {cpu_name()} ({platform.machine()}, {os.cpu_count()} logical CPUs)")
    print()
    print("| kernel | mojo-pillow | Pillow | speedup |")
    print("| --- | ---: | ---: | ---: |")
    for name, mojo_seconds, pillow_seconds, speedup in rows:
        print(
            f"| {name} | {mojo_seconds * 1000:.2f} ms | "
            f"{pillow_seconds * 1000:.2f} ms | {speedup:.2f}x |"
        )


if __name__ == "__main__":
    main()
