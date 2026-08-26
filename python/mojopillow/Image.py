"""A compact Pillow-shaped image API backed by Mojo kernels."""

from __future__ import annotations

import math
from enum import IntEnum
from os import PathLike
from typing import BinaryIO, Sequence

import numpy as np

from ._lib import addr, lib, u8


class Resampling(IntEnum):
    NEAREST = 0
    LANCZOS = 1
    BILINEAR = 2
    BICUBIC = 3
    BOX = 4
    HAMMING = 5


NONE = NEAREST = Resampling.NEAREST
LANCZOS = ANTIALIAS = Resampling.LANCZOS
BILINEAR = Resampling.BILINEAR
BICUBIC = CUBIC = Resampling.BICUBIC
BOX = Resampling.BOX
HAMMING = Resampling.HAMMING

_CHANNELS = {"L": 1, "LA": 2, "RGB": 3, "RGBA": 4, "CMYK": 4}
_CONVERSIONS = {
    ("RGB", "L"): 0,
    ("RGBA", "L"): 1,
    ("L", "RGB"): 2,
    ("L", "RGBA"): 3,
    ("RGB", "RGBA"): 4,
    ("RGBA", "RGB"): 5,
    ("RGB", "CMYK"): 6,
    ("CMYK", "RGB"): 7,
    ("RGB", "LA"): 8,
    ("RGBA", "LA"): 9,
    ("LA", "RGBA"): 10,
    ("LA", "RGB"): 11,
    ("LA", "L"): 12,
    ("L", "CMYK"): 13,
    ("LA", "CMYK"): 14,
}
_SUPPORT = {
    Resampling.BOX: 0.5,
    Resampling.BILINEAR: 1.0,
    Resampling.HAMMING: 1.0,
    Resampling.BICUBIC: 2.0,
    Resampling.LANCZOS: 3.0,
}


def _shape(mode: str, height: int, width: int) -> tuple[int, ...]:
    channels = _CHANNELS[mode]
    return (height, width) if channels == 1 else (height, width, channels)


def _filter_value(resample: Resampling, value: float) -> float:
    if resample == Resampling.BOX:
        return 1.0 if value > -0.5 and value <= 0.5 else 0.0
    value = abs(value)
    if resample == Resampling.BILINEAR:
        return 1.0 - value if value < 1.0 else 0.0
    if resample == Resampling.HAMMING:
        if value == 0.0:
            return 1.0
        if value >= 1.0:
            return 0.0
        angle = value * math.pi
        return math.sin(angle) / angle * (0.54 + 0.46 * math.cos(angle))
    if resample == Resampling.BICUBIC:
        if value < 1.0:
            return ((1.5 * value - 2.5) * value * value) + 1.0
        if value < 2.0:
            return (((value - 5.0) * value + 8.0) * value - 4.0) * -0.5
        return 0.0
    if value >= 3.0:
        return 0.0
    if value == 0.0:
        return 1.0
    angle = value * math.pi
    return (math.sin(angle) / angle) * (
        math.sin(angle / 3.0) / (angle / 3.0)
    )


def _coefficients(
    input_size: int,
    start: float,
    end: float,
    output_size: int,
    resample: Resampling,
) -> tuple[np.ndarray, np.ndarray, int]:
    scale = (end - start) / output_size
    filter_scale = max(scale, 1.0)
    support = _SUPPORT[resample] * filter_scale
    kernel_size = math.ceil(support) * 2 + 1
    bounds = np.zeros((output_size, 2), dtype=np.int32)
    coefficients = np.zeros((output_size, kernel_size), dtype=np.int32)
    inverse = 1.0 / filter_scale
    precision = 1 << 22

    for output_index in range(output_size):
        center = start + (output_index + 0.5) * scale
        first = max(0, int(center - support + 0.5))
        last = min(input_size, int(center + support + 0.5))
        count = last - first
        weights = [
            _filter_value(
                resample, (source + first - center + 0.5) * inverse
            )
            for source in range(count)
        ]
        total = sum(weights)
        if total:
            weights = [weight / total for weight in weights]
        for index, weight in enumerate(weights):
            scaled = weight * precision
            coefficients[output_index, index] = int(
                scaled + 0.5 if scaled >= 0 else scaled - 0.5
            )
        bounds[output_index] = first, count
    return bounds, coefficients, kernel_size


class Image:
    format: str | None = None

    def __init__(self, array: np.ndarray, mode: str):
        if mode not in _CHANNELS:
            raise ValueError(f"unsupported image mode {mode!r}")
        array = np.asarray(array)
        if array.dtype != np.uint8:
            raise TypeError("only uint8 arrays are supported")
        if array.ndim < 2 or array.shape[0] <= 0 or array.shape[1] <= 0:
            raise ValueError("image width and height must be greater than zero")
        expected = _shape(mode, array.shape[0], array.shape[1])
        if tuple(array.shape) != expected:
            raise ValueError(f"array shape {array.shape} does not match mode {mode}")
        self._array = u8(array)
        self.mode = mode
        self.info: dict[str, object] = {}

    @property
    def size(self) -> tuple[int, int]:
        return self._array.shape[1], self._array.shape[0]

    @property
    def width(self) -> int:
        return self._array.shape[1]

    @property
    def height(self) -> int:
        return self._array.shape[0]

    def __array__(
        self, dtype: np.dtype | None = None, copy: bool | None = None
    ) -> np.ndarray:
        array = self._array if dtype is None else self._array.astype(dtype, copy=False)
        if copy is True:
            return array.copy()
        return array

    def __repr__(self) -> str:
        return (
            f"<mojopillow.Image.Image image mode={self.mode} "
            f"size={self.width}x{self.height}>"
        )

    def copy(self) -> Image:
        copied = Image(self._array.copy(), self.mode)
        copied.info = self.info.copy()
        copied.format = self.format
        return copied

    def load(self) -> Image:
        return self

    def getbands(self) -> tuple[str, ...]:
        return {
            "L": ("L",),
            "LA": ("L", "A"),
            "RGB": ("R", "G", "B"),
            "RGBA": ("R", "G", "B", "A"),
            "CMYK": ("C", "M", "Y", "K"),
        }[self.mode]

    def tobytes(self, encoder_name: str = "raw", *args: object) -> bytes:
        if encoder_name != "raw":
            raise ValueError("only raw byte output is supported")
        return self._array.tobytes()

    def getpixel(self, xy: tuple[int, int]) -> int | tuple[int, ...]:
        value = self._array[xy[1], xy[0]]
        if self.mode == "L":
            return int(value)
        return tuple(int(channel) for channel in value)

    def putpixel(self, xy: tuple[int, int], value: int | Sequence[int]) -> None:
        self._array[xy[1], xy[0]] = value

    def save(
        self,
        fp: str | PathLike[str] | BinaryIO,
        format: str | None = None,
        **params: object,
    ) -> None:
        from PIL import Image as PILImage

        PILImage.fromarray(self._array, self.mode).save(fp, format=format, **params)

    def resize(
        self,
        size: tuple[int, int] | list[int] | np.ndarray,
        resample: int | None = None,
        box: tuple[float, float, float, float] | None = None,
        reducing_gap: float | None = None,
    ) -> Image:
        width, height = (int(size[0]), int(size[1]))
        if width <= 0 or height <= 0:
            raise ValueError("height and width must be > 0")
        selected = Resampling.BICUBIC if resample is None else Resampling(resample)
        if reducing_gap is not None and reducing_gap < 1.0:
            raise ValueError("reducing_gap must be 1.0 or greater")
        if box is None:
            box = (0.0, 0.0, float(self.width), float(self.height))
        if (
            len(box) != 4
            or box[0] < 0
            or box[1] < 0
            or box[2] > self.width
            or box[3] > self.height
            or box[2] <= box[0]
            or box[3] <= box[1]
        ):
            raise ValueError("box must be inside the source image")
        box = tuple(float(value) for value in box)
        if not all(math.isfinite(value) for value in box):
            raise ValueError("box coordinates must be finite")
        if self.size == (width, height) and box == (
            0.0,
            0.0,
            float(self.width),
            float(self.height),
        ):
            return self.copy()

        channels = _CHANNELS[self.mode]
        if selected == Resampling.NEAREST:
            result = np.empty(_shape(self.mode, height, width), dtype=np.uint8)
            lib().mp_resize_nearest(
                addr(self._array),
                addr(result),
                self.width,
                self.height,
                width,
                height,
                channels,
                *box,
            )
            return Image(result, self.mode)

        source = self._array
        premultiplied = self.mode in ("LA", "RGBA")
        if premultiplied:
            source = np.empty_like(source)
            lib().mp_premultiply_rgba(
                addr(self._array),
                addr(source),
                self.width * self.height,
                channels,
                0,
            )

        current_width, current_height = self.width, self.height
        need_horizontal = (
            width != self.width or box[0] != 0.0 or box[2] != float(width)
        )
        need_vertical = (
            height != self.height or box[1] != 0.0 or box[3] != float(height)
        )
        if need_horizontal:
            bounds, coefficients, kernel_size = _coefficients(
                self.width, box[0], box[2], width, selected
            )
            horizontal = np.empty(
                _shape(self.mode, current_height, width), dtype=np.uint8
            )
            lib().mp_resample_horizontal(
                addr(source),
                addr(horizontal),
                addr(bounds),
                addr(coefficients),
                current_width,
                current_height,
                width,
                channels,
                kernel_size,
            )
            source = horizontal
            current_width = width
        if need_vertical:
            bounds, coefficients, kernel_size = _coefficients(
                self.height, box[1], box[3], height, selected
            )
            vertical = np.empty(
                _shape(self.mode, height, current_width), dtype=np.uint8
            )
            lib().mp_resample_vertical(
                addr(source),
                addr(vertical),
                addr(bounds),
                addr(coefficients),
                current_width,
                current_height,
                height,
                channels,
                kernel_size,
            )
            source = vertical
        if premultiplied:
            lib().mp_premultiply_rgba(
                addr(source), addr(source), width * height, channels, 1
            )
        return Image(source, self.mode)

    def convert(
        self,
        mode: str | None = None,
        matrix: tuple[float, ...] | None = None,
        dither: object | None = None,
        palette: object = 0,
        colors: int = 256,
    ) -> Image:
        del dither, palette, colors
        if mode is None:
            mode = "RGB"
        if mode == self.mode and matrix is None:
            return self.copy()
        if mode not in _CHANNELS:
            raise ValueError(f"conversion to mode {mode!r} is not covered")
        pixels = self.width * self.height
        if matrix is not None:
            if self.mode != "RGB" or mode not in ("L", "RGB"):
                raise ValueError("illegal conversion")
            needed = 4 if mode == "L" else 12
            if len(matrix) != needed:
                raise TypeError("argument 2 must be sequence of length 4 or 12")
            matrix_values = np.asarray(matrix)
            if not np.issubdtype(matrix_values.dtype, np.number):
                raise TypeError("matrix coefficients must be numeric")
            if not np.all(np.isfinite(matrix_values)):
                raise ValueError("matrix coefficients must be finite")
            float32_limit = np.finfo(np.float32).max
            if np.any(np.abs(matrix_values) > float32_limit):
                raise OverflowError("matrix coefficient is outside float32 range")
            coefficients = np.ascontiguousarray(matrix_values, dtype=np.float32)
            result = np.empty(_shape(mode, self.height, self.width), dtype=np.uint8)
            lib().mp_convert_matrix(
                addr(self._array),
                addr(result),
                addr(coefficients),
                pixels,
                _CHANNELS[mode],
            )
            return Image(result, mode)

        operation = _CONVERSIONS.get((self.mode, mode))
        if operation is None:
            if self.mode != "RGB":
                return self.convert("RGB").convert(mode)
            raise ValueError(f"conversion from {self.mode} to {mode} is not covered")
        result = np.empty(_shape(mode, self.height, self.width), dtype=np.uint8)
        lib().mp_convert(addr(self._array), addr(result), pixels, operation)
        return Image(result, mode)

    def filter(self, filter: object) -> Image:
        from . import ImageFilter

        selected = filter() if isinstance(filter, type) else filter
        if not isinstance(selected, ImageFilter.Kernel):
            if not hasattr(selected, "filterargs"):
                raise TypeError("filter must be an ImageFilter.Kernel")
            size, scale, offset, kernel = selected.filterargs
        else:
            size, scale, offset, kernel = selected.filterargs
        if size not in ((3, 3), (5, 5)):
            raise ValueError("bad kernel size")
        if scale == 0:
            raise ValueError("filter scale cannot be zero")
        kernel_values = np.asarray(kernel)
        if not np.issubdtype(kernel_values.dtype, np.number):
            raise TypeError("kernel coefficients must be numeric")
        if not math.isfinite(float(scale)) or not math.isfinite(float(offset)):
            raise ValueError("filter scale and offset must be finite")
        if not np.all(np.isfinite(kernel_values)):
            raise ValueError("kernel coefficients must be finite")
        float32_limit = np.finfo(np.float32).max
        if abs(float(scale)) > float32_limit or abs(float(offset)) > float32_limit:
            raise OverflowError("filter scale or offset is outside float32 range")
        if kernel_values.size != size[0] * size[1]:
            raise ValueError("kernel coefficient count does not match its size")
        if np.any(np.abs(kernel_values) > float32_limit):
            raise OverflowError("kernel coefficient is outside float32 range")
        coefficients = np.ascontiguousarray(kernel_values, dtype=np.float32)
        coefficients /= np.float32(scale)
        result = np.empty_like(self._array)
        lib().mp_convolve(
            addr(self._array),
            addr(result),
            addr(coefficients),
            self.width,
            self.height,
            _CHANNELS[self.mode],
            size[0],
            np.float32(offset),
        )
        return Image(result, self.mode)


def fromarray(obj: object, mode: str | None = None) -> Image:
    array = np.asarray(obj)
    if array.dtype != np.uint8:
        raise TypeError("only uint8 arrays are supported")
    if mode is None:
        if array.ndim == 2:
            mode = "L"
        elif array.ndim == 3 and array.shape[2] in (2, 3, 4):
            mode = {2: "LA", 3: "RGB", 4: "RGBA"}[array.shape[2]]
        else:
            raise TypeError("cannot infer image mode from array shape")
    return Image(array, mode)


def new(
    mode: str, size: tuple[int, int] | list[int], color: object = 0
) -> Image:
    if mode not in _CHANNELS:
        raise ValueError(f"unsupported image mode {mode!r}")
    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        raise ValueError("height and width must be > 0")
    array = np.empty(_shape(mode, height, width), dtype=np.uint8)
    array[...] = color
    return Image(array, mode)


def open(
    fp: str | PathLike[str] | BinaryIO,
    mode: str = "r",
    formats: list[str] | tuple[str, ...] | None = None,
) -> Image:
    if mode != "r":
        raise ValueError("bad mode")
    from PIL import Image as PILImage

    source = PILImage.open(fp, formats=formats)
    source.load()
    if source.mode not in _CHANNELS:
        source = source.convert("RGBA" if "A" in source.getbands() else "RGB")
    result = fromarray(np.asarray(source), source.mode)
    result.format = source.format
    result.info = source.info.copy()
    return result


def blend(im1: Image, im2: Image, alpha: float) -> Image:
    if im1.mode != im2.mode or im1.size != im2.size:
        raise ValueError("images do not match")
    if not math.isfinite(alpha):
        raise ValueError("alpha must be finite")
    if abs(alpha) > np.finfo(np.float32).max:
        raise OverflowError("alpha is outside float32 range")
    result = np.empty_like(im1._array)
    lib().mp_blend(
        addr(im1._array),
        addr(im2._array),
        addr(result),
        im1._array.size,
        np.float32(alpha),
    )
    return Image(result, im1.mode)


def composite(image1: Image, image2: Image, mask: Image) -> Image:
    if image1.mode != image2.mode or image1.size != image2.size:
        raise ValueError("images do not match")
    if mask.mode not in ("L", "LA", "RGBA") or mask.size != image1.size:
        raise ValueError("bad transparency mask")
    result = np.empty_like(image1._array)
    lib().mp_composite(
        addr(image1._array),
        addr(image2._array),
        addr(mask._array),
        addr(result),
        image1.width * image1.height,
        _CHANNELS[image1.mode],
        _CHANNELS[mask.mode],
    )
    return Image(result, image1.mode)


def alpha_composite(im1: Image, im2: Image) -> Image:
    if (
        im1.mode not in ("LA", "RGBA")
        or im2.mode != im1.mode
        or im1.size != im2.size
    ):
        raise ValueError("images do not match")
    result = np.empty_like(im1._array)
    lib().mp_alpha_composite_rgba(
        addr(im1._array),
        addr(im2._array),
        addr(result),
        im1.width * im1.height,
        _CHANNELS[im1.mode],
    )
    return Image(result, im1.mode)
