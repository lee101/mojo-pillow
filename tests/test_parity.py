"""Numerical and behavioral parity with Pillow on the covered API."""

from __future__ import annotations

import io

import numpy as np
import pytest

PILImage = pytest.importorskip("PIL.Image")
PILImageChops = pytest.importorskip("PIL.ImageChops")
PILImageFilter = pytest.importorskip("PIL.ImageFilter")

from mojopillow import Image, ImageChops, ImageFilter
from mojopillow._lib import addr


@pytest.fixture(scope="module")
def arrays() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260728)
    return {
        "L": rng.integers(0, 256, (37, 43), dtype=np.uint8),
        "LA": rng.integers(0, 256, (37, 43, 2), dtype=np.uint8),
        "RGB": rng.integers(0, 256, (37, 43, 3), dtype=np.uint8),
        "RGBA": rng.integers(0, 256, (37, 43, 4), dtype=np.uint8),
        "CMYK": rng.integers(0, 256, (37, 43, 4), dtype=np.uint8),
    }


def assert_same(ours: Image.Image, upstream: PILImage.Image) -> None:
    assert ours.mode == upstream.mode
    assert ours.size == upstream.size
    assert np.array_equal(np.asarray(ours), np.asarray(upstream))


def assert_u8_close(
    ours: Image.Image, upstream: PILImage.Image, tolerance: int = 1
) -> None:
    assert ours.mode == upstream.mode
    assert ours.size == upstream.size
    difference = np.abs(
        np.asarray(ours).astype(np.int16)
        - np.asarray(upstream).astype(np.int16)
    )
    assert difference.max() <= tolerance


@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA", "CMYK"])
def test_fromarray_and_image_protocol(arrays, mode):
    ours = Image.fromarray(arrays[mode], mode)
    assert ours.mode == mode
    assert ours.size == (43, 37)
    assert ours.width == 43 and ours.height == 37
    assert ours.tobytes() == arrays[mode].tobytes()
    assert np.array_equal(np.asarray(ours), arrays[mode])
    assert ours.copy().tobytes() == ours.tobytes()


@pytest.mark.parametrize(
    "resample",
    [
        PILImage.Resampling.NEAREST,
        PILImage.Resampling.BOX,
        PILImage.Resampling.BILINEAR,
        PILImage.Resampling.HAMMING,
        PILImage.Resampling.BICUBIC,
        PILImage.Resampling.LANCZOS,
    ],
)
@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA", "CMYK"])
def test_resize_all_filters(arrays, mode, resample):
    source = arrays[mode]
    upstream = PILImage.fromarray(source, mode).resize((29, 51), resample)
    ours = Image.fromarray(source, mode).resize((29, 51), int(resample))
    assert_same(ours, upstream)


@pytest.mark.parametrize(
    "resample",
    [
        PILImage.Resampling.NEAREST,
        PILImage.Resampling.BILINEAR,
        PILImage.Resampling.BICUBIC,
        PILImage.Resampling.LANCZOS,
    ],
)
def test_resize_box_parity(arrays, resample):
    box = (2.25, 3.5, 39.75, 35.0)
    upstream = PILImage.fromarray(arrays["RGB"]).resize(
        (31, 19), resample, box=box
    )
    ours = Image.fromarray(arrays["RGB"]).resize(
        (31, 19), int(resample), box=box
    )
    assert_same(ours, upstream)


def test_resize_validation(arrays):
    image = Image.fromarray(arrays["RGB"])
    with pytest.raises(ValueError):
        image.resize((0, 10))
    with pytest.raises(ValueError):
        image.resize((10, 10), box=(-1, 0, 10, 10))
    with pytest.raises(ValueError):
        image.resize((10, 10), reducing_gap=0.5)
    with pytest.raises(ValueError):
        image.resize((10, 10), box=(0, 0, float("nan"), 10))


@pytest.mark.parametrize(
    ("upstream_filter", "our_filter"),
    [
        (PILImageFilter.BLUR, ImageFilter.BLUR),
        (PILImageFilter.CONTOUR, ImageFilter.CONTOUR),
        (PILImageFilter.DETAIL, ImageFilter.DETAIL),
        (PILImageFilter.EDGE_ENHANCE, ImageFilter.EDGE_ENHANCE),
        (PILImageFilter.EDGE_ENHANCE_MORE, ImageFilter.EDGE_ENHANCE_MORE),
        (PILImageFilter.EMBOSS, ImageFilter.EMBOSS),
        (PILImageFilter.FIND_EDGES, ImageFilter.FIND_EDGES),
        (PILImageFilter.SHARPEN, ImageFilter.SHARPEN),
        (PILImageFilter.SMOOTH, ImageFilter.SMOOTH),
        (PILImageFilter.SMOOTH_MORE, ImageFilter.SMOOTH_MORE),
    ],
)
@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA", "CMYK"])
def test_builtin_convolution_filters(
    arrays, mode, upstream_filter, our_filter
):
    upstream = PILImage.fromarray(arrays[mode], mode).filter(upstream_filter)
    ours = Image.fromarray(arrays[mode], mode).filter(our_filter)
    assert_u8_close(ours, upstream)


def test_custom_asymmetric_kernel_is_flipped_like_pillow(arrays):
    kernel = (1.25, -0.5, 0, 2, 3, -1, 0.25, 0, 4)
    upstream_filter = PILImageFilter.Kernel(
        (3, 3), kernel, scale=3.25, offset=7
    )
    our_filter = ImageFilter.Kernel((3, 3), kernel, scale=3.25, offset=7)
    upstream = PILImage.fromarray(arrays["RGB"]).filter(upstream_filter)
    ours = Image.fromarray(arrays["RGB"]).filter(our_filter)
    assert_same(ours, upstream)


@pytest.mark.parametrize("source_mode", ["L", "LA", "RGB", "RGBA", "CMYK"])
@pytest.mark.parametrize("target_mode", ["L", "LA", "RGB", "RGBA", "CMYK"])
def test_mode_conversion_parity(arrays, source_mode, target_mode):
    upstream = PILImage.fromarray(
        arrays[source_mode], source_mode
    ).convert(target_mode)
    ours = Image.fromarray(arrays[source_mode], source_mode).convert(target_mode)
    assert_same(ours, upstream)


@pytest.mark.parametrize(
    ("mode", "matrix"),
    [
        ("L", (0.31, 0.52, 0.17, 3.25)),
        (
            "RGB",
            (
                1.1, -0.1, 0.0, 2.0,
                0.2, 0.7, 0.1, -3.0,
                -0.2, 0.1, 1.1, 4.5,
            ),
        ),
    ],
)
def test_color_matrix_parity(arrays, mode, matrix):
    upstream = np.asarray(PILImage.fromarray(arrays["RGB"]).convert(mode, matrix))
    ours = np.asarray(Image.fromarray(arrays["RGB"]).convert(mode, matrix))
    assert np.max(np.abs(ours.astype(int) - upstream.astype(int))) <= 1


@pytest.mark.parametrize(
    "operation",
    [
        "lighter",
        "darker",
        "difference",
        "multiply",
        "screen",
        "add",
        "subtract",
        "add_modulo",
        "subtract_modulo",
        "soft_light",
        "hard_light",
        "overlay",
    ],
)
def test_image_chops_parity(arrays, operation):
    rng = np.random.default_rng(9)
    second = rng.integers(0, 256, arrays["RGB"].shape, dtype=np.uint8)
    upstream_first = PILImage.fromarray(arrays["RGB"])
    upstream_second = PILImage.fromarray(second)
    our_first = Image.fromarray(arrays["RGB"])
    our_second = Image.fromarray(second)
    upstream = getattr(PILImageChops, operation)(upstream_first, upstream_second)
    ours = getattr(ImageChops, operation)(our_first, our_second)
    assert_same(ours, upstream)


def test_scaled_add_and_subtract_parity(arrays):
    second = np.flip(arrays["RGB"], axis=1).copy()
    for operation in ("add", "subtract"):
        upstream = getattr(PILImageChops, operation)(
            PILImage.fromarray(arrays["RGB"]),
            PILImage.fromarray(second),
            scale=2.5,
            offset=13,
        )
        ours = getattr(ImageChops, operation)(
            Image.fromarray(arrays["RGB"]),
            Image.fromarray(second),
            scale=2.5,
            offset=13,
        )
        assert_u8_close(ours, upstream)


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.7, 1.0, 1.4, -0.4])
def test_blend_parity(arrays, alpha):
    second = np.roll(arrays["RGBA"], 5, axis=1)
    upstream = PILImage.blend(
        PILImage.fromarray(arrays["RGBA"]),
        PILImage.fromarray(second),
        alpha,
    )
    ours = Image.blend(
        Image.fromarray(arrays["RGBA"]),
        Image.fromarray(second),
        alpha,
    )
    difference = np.abs(
        np.asarray(ours).astype(int) - np.asarray(upstream).astype(int)
    )
    assert difference.max() <= 1


@pytest.mark.parametrize("mask_mode", ["L", "LA", "RGBA"])
def test_composite_parity(arrays, mask_mode):
    first = arrays["RGBA"]
    second = np.flip(first, axis=0).copy()
    mask = arrays[mask_mode]
    upstream = PILImage.composite(
        PILImage.fromarray(first),
        PILImage.fromarray(second),
        PILImage.fromarray(mask, mask_mode),
    )
    ours = Image.composite(
        Image.fromarray(first),
        Image.fromarray(second),
        Image.fromarray(mask, mask_mode),
    )
    assert_same(ours, upstream)


@pytest.mark.parametrize("mode", ["LA", "RGBA"])
def test_alpha_composite_parity(arrays, mode):
    second = np.roll(arrays[mode], 7, axis=0)
    upstream = PILImage.alpha_composite(
        PILImage.fromarray(arrays[mode], mode), PILImage.fromarray(second, mode)
    )
    ours = Image.alpha_composite(
        Image.fromarray(arrays[mode], mode), Image.fromarray(second, mode)
    )
    assert_same(ours, upstream)


def test_simd_tail_parity():
    rng = np.random.default_rng(44)
    rgb = rng.integers(0, 256, (3, 17, 3), dtype=np.uint8)
    rgb_second = rng.integers(0, 256, rgb.shape, dtype=np.uint8)
    rgba = rng.integers(0, 256, (3, 17, 4), dtype=np.uint8)
    rgba_second = rng.integers(0, 256, rgba.shape, dtype=np.uint8)

    assert_same(
        Image.fromarray(rgb).convert("L"),
        PILImage.fromarray(rgb).convert("L"),
    )
    assert_same(
        Image.fromarray(rgb).resize((11, 5), Image.Resampling.LANCZOS),
        PILImage.fromarray(rgb).resize(
            (11, 5), PILImage.Resampling.LANCZOS
        ),
    )
    assert_u8_close(
        Image.blend(
            Image.fromarray(rgb), Image.fromarray(rgb_second), 0.35
        ),
        PILImage.blend(
            PILImage.fromarray(rgb), PILImage.fromarray(rgb_second), 0.35
        ),
    )
    assert_same(
        Image.alpha_composite(
            Image.fromarray(rgba), Image.fromarray(rgba_second)
        ),
        PILImage.alpha_composite(
            PILImage.fromarray(rgba), PILImage.fromarray(rgba_second)
        ),
    )


@pytest.mark.parametrize("shape", [(511, 513), (512, 513)])
def test_parallel_threshold_parity(shape):
    rng = np.random.default_rng(shape[0])
    first_l = rng.integers(0, 256, shape, dtype=np.uint8)
    second_l = rng.integers(0, 256, shape, dtype=np.uint8)
    first_rgba = rng.integers(0, 256, (*shape, 4), dtype=np.uint8)
    second_rgba = rng.integers(0, 256, (*shape, 4), dtype=np.uint8)

    assert_u8_close(
        Image.blend(
            Image.fromarray(first_l), Image.fromarray(second_l), 0.35
        ),
        PILImage.blend(
            PILImage.fromarray(first_l), PILImage.fromarray(second_l), 0.35
        ),
    )
    assert_same(
        Image.alpha_composite(
            Image.fromarray(first_rgba), Image.fromarray(second_rgba)
        ),
        PILImage.alpha_composite(
            PILImage.fromarray(first_rgba), PILImage.fromarray(second_rgba)
        ),
    )
    assert_same(
        Image.fromarray(first_rgba).resize((257, 255), Image.Resampling.BILINEAR),
        PILImage.fromarray(first_rgba).resize(
            (257, 255), PILImage.Resampling.BILINEAR
        ),
    )


def test_new_getpixel_and_putpixel():
    ours = Image.new("RGB", (7, 5), (1, 2, 3))
    assert ours.getpixel((2, 1)) == (1, 2, 3)
    ours.putpixel((2, 1), (40, 50, 60))
    assert ours.getpixel((2, 1)) == (40, 50, 60)


def test_ffi_input_validation():
    with pytest.raises(TypeError):
        Image.Image(np.ones((2, 2), dtype=np.uint16), "L")
    with pytest.raises(ValueError):
        Image.fromarray(np.empty((0, 2), dtype=np.uint8), "L")
    with pytest.raises(ValueError):
        Image.new("RGB", (0, 2))
    with pytest.raises(TypeError):
        addr(np.ones((2, 2), dtype=np.float64))
    with pytest.raises(ValueError):
        addr(np.ones((2, 2), dtype=np.uint8)[:, ::2])
    with pytest.raises(ValueError):
        addr(np.empty((0,), dtype=np.uint8))


def test_float_parameter_validation(arrays):
    image = Image.fromarray(arrays["RGB"])
    with pytest.raises(ValueError):
        Image.blend(image, image, float("nan"))
    with pytest.raises(ValueError):
        ImageChops.add(image, image, scale=0)
    with pytest.raises(ValueError):
        image.convert("L", (float("nan"), 0, 0, 0))
    with pytest.raises(ValueError):
        image.filter(ImageFilter.Kernel((3, 3), [float("nan")] * 9, scale=1))


def test_png_open_and_save_roundtrip(arrays):
    buffer = io.BytesIO()
    source = Image.fromarray(arrays["RGBA"])
    source.save(buffer, "PNG")
    buffer.seek(0)
    decoded = Image.open(buffer)
    assert decoded.format == "PNG"
    assert np.array_equal(np.asarray(decoded), arrays["RGBA"])
