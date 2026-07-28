"""Pillow-compatible channel arithmetic backed by one Mojo kernel."""

from __future__ import annotations

import math

import numpy as np

from . import Image
from ._lib import addr, lib

_OPERATIONS = {
    "lighter": 0,
    "darker": 1,
    "difference": 2,
    "multiply": 3,
    "screen": 4,
    "add": 5,
    "subtract": 6,
    "add_modulo": 7,
    "subtract_modulo": 8,
    "soft_light": 9,
    "hard_light": 10,
    "overlay": 11,
}


def _chop(
    image1: Image.Image,
    image2: Image.Image,
    operation: str,
    scale: float = 1.0,
    offset: float = 0,
) -> Image.Image:
    if image1.mode != image2.mode:
        raise ValueError("images do not match")
    if not math.isfinite(scale) or scale == 0:
        raise ValueError("scale must be finite and non-zero")
    if abs(scale) > np.finfo(np.float32).max:
        raise OverflowError("scale is outside float32 range")
    width = min(image1.width, image2.width)
    height = min(image1.height, image2.height)
    first = np.ascontiguousarray(np.asarray(image1)[:height, :width])
    second = np.ascontiguousarray(np.asarray(image2)[:height, :width])
    result = np.empty_like(first)
    lib().mp_chop(
        addr(first),
        addr(second),
        addr(result),
        result.size,
        _OPERATIONS[operation],
        np.float32(scale),
        int(offset),
    )
    return Image.Image(result, image1.mode)


def duplicate(image: Image.Image) -> Image.Image:
    return image.copy()


def invert(image: Image.Image) -> Image.Image:
    white = Image.new(image.mode, image.size, (255,) * len(image.getbands()))
    return difference(image, white)


def lighter(image1: Image.Image, image2: Image.Image) -> Image.Image:
    return _chop(image1, image2, "lighter")


def darker(image1: Image.Image, image2: Image.Image) -> Image.Image:
    return _chop(image1, image2, "darker")


def difference(image1: Image.Image, image2: Image.Image) -> Image.Image:
    return _chop(image1, image2, "difference")


def multiply(image1: Image.Image, image2: Image.Image) -> Image.Image:
    return _chop(image1, image2, "multiply")


def screen(image1: Image.Image, image2: Image.Image) -> Image.Image:
    return _chop(image1, image2, "screen")


def add(
    image1: Image.Image,
    image2: Image.Image,
    scale: float = 1.0,
    offset: float = 0,
) -> Image.Image:
    return _chop(image1, image2, "add", scale, offset)


def subtract(
    image1: Image.Image,
    image2: Image.Image,
    scale: float = 1.0,
    offset: float = 0,
) -> Image.Image:
    return _chop(image1, image2, "subtract", scale, offset)


def add_modulo(image1: Image.Image, image2: Image.Image) -> Image.Image:
    return _chop(image1, image2, "add_modulo")


def subtract_modulo(image1: Image.Image, image2: Image.Image) -> Image.Image:
    return _chop(image1, image2, "subtract_modulo")


def soft_light(image1: Image.Image, image2: Image.Image) -> Image.Image:
    return _chop(image1, image2, "soft_light")


def hard_light(image1: Image.Image, image2: Image.Image) -> Image.Image:
    return _chop(image1, image2, "hard_light")


def overlay(image1: Image.Image, image2: Image.Image) -> Image.Image:
    return _chop(image1, image2, "overlay")


def blend(
    image1: Image.Image, image2: Image.Image, alpha: float
) -> Image.Image:
    return Image.blend(image1, image2, alpha)


def composite(
    image1: Image.Image, image2: Image.Image, mask: Image.Image
) -> Image.Image:
    return Image.composite(image1, image2, mask)
