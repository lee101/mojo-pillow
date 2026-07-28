"""ctypes loader for the Mojo image kernel library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.path.join(ROOT, "dist", "libmojo-pillow.so")

I = ctypes.c_int64
F = ctypes.c_float
D = ctypes.c_double

_SIGNATURES = {
    "mp_resize_nearest": ([I] * 7 + [D] * 4, None),
    "mp_resample_horizontal": ([I] * 9, None),
    "mp_resample_vertical": ([I] * 9, None),
    "mp_convolve": ([I] * 7 + [F], None),
    "mp_convert": ([I, I, I, I], None),
    "mp_convert_matrix": ([I, I, I, I, I], None),
    "mp_premultiply_rgba": ([I, I, I, I, I], None),
    "mp_blend": ([I, I, I, I, F], None),
    "mp_chop": ([I, I, I, I, I, F, I], None),
    "mp_composite": ([I, I, I, I, I, I, I], None),
    "mp_alpha_composite_rgba": ([I, I, I, I, I], None),
}


class BuildError(RuntimeError):
    pass


def mojo_command() -> list[str]:
    override = os.environ.get("MOJOPILLOW_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi):
        return [
            pixi,
            "run",
            "--manifest-path",
            os.path.join(ROOT, "pixi.toml"),
            "mojo",
        ]
    raise BuildError("mojo not found; set MOJOPILLOW_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    source = os.path.join(ROOT, "src", "kernels.mojo")
    if (
        not force
        and os.path.exists(LIB)
        and os.path.getmtime(LIB) >= os.path.getmtime(source)
    ):
        return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    command = mojo_command() + [
        "build",
        "--emit",
        "shared-lib",
        source,
        "-o",
        LIB,
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((result.stderr or result.stdout).strip()[:8000])
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def addr(array: np.ndarray) -> int:
    if not isinstance(array, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if array.dtype not in (np.dtype(np.uint8), np.dtype(np.int32), np.dtype(np.float32)):
        raise TypeError(f"unsupported FFI buffer dtype {array.dtype}")
    if not array.flags.c_contiguous:
        raise ValueError("FFI buffers must be C-contiguous")
    if array.size == 0 or array.ctypes.data == 0:
        raise ValueError("FFI buffers must be non-empty")
    return int(array.ctypes.data)


def u8(array: object) -> np.ndarray:
    value = np.asarray(array)
    if value.dtype != np.uint8:
        raise TypeError("only uint8 arrays are supported")
    return np.ascontiguousarray(value)


def main() -> int:
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
