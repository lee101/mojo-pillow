# mojo-pillow

`mojo-pillow` ports the compute-heavy core of common Pillow image operations to
compiled [Mojo](https://www.modular.com/mojo) kernels. Its Python surface follows
Pillow's module layout:

```python
from mojopillow import Image, ImageChops, ImageFilter
```

This is a focused kernel port, not a second image-format ecosystem. NumPy owns the
pixels, Mojo performs the resizing, convolution, color conversion, and blending,
and Pillow remains an optional codec adapter for `Image.open()` and `Image.save()`.

## Covered subset

- `Image.fromarray`, `Image.new`, `Image.open`, `Image.save`, NumPy conversion,
  `copy`, `getpixel`, `putpixel`, and raw `tobytes`
- `Image.Image.resize(size, resample=None, box=None, reducing_gap=None)` with
  `NEAREST`, `BOX`, `BILINEAR`, `HAMMING`, `BICUBIC`, and `LANCZOS`
- uint8 modes `L`, `LA`, `RGB`, `RGBA`, and `CMYK`, including custom RGB-to-L
  and RGB-to-RGB conversion matrices
- `ImageFilter.Kernel` for 3x3 and 5x5 convolution plus Pillow's `BLUR`,
  `CONTOUR`, `DETAIL`, `EDGE_ENHANCE`, `EDGE_ENHANCE_MORE`, `EMBOSS`,
  `FIND_EDGES`, `SHARPEN`, `SMOOTH`, and `SMOOTH_MORE`
- `Image.blend`, `Image.composite`, `Image.alpha_composite`, and the
  `ImageChops` lighter/darker/difference/multiply/screen/add/subtract/modulo,
  soft-light, hard-light, and overlay operations

The parity suite compares every covered operation with Pillow 12.3.0. Resize,
fixed mode conversions, integer chops, masks, and alpha compositing are
byte-exact. Floating convolution, scaled chops, and custom color matrices have a
tested maximum error of one uint8 level because Mojo and Pillow can contract
floating-point expressions differently.

Not covered are palette and bilevel images (`P`, `PA`, `1`), integer/float image
modes, quantization or dithering, EXIF transforms, drawing, geometric transforms,
rank/Gaussian/unsharp filters, and Pillow's full plugin API. `reducing_gap` is
validated and accepted, but the current implementation always performs the fair
resample rather than Pillow's optional integer-reduction shortcut. Image decoding
and encoding are delegated to Pillow; they are not Mojo kernels.

## Install

The pinned Mojo nightly, Python, NumPy, Pillow, and test tooling are managed by
Pixi:

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` creates `dist/libmojo-pillow.so`. The Python loader also rebuilds
the library when the Mojo source is newer. This release supports source checkouts
on Linux x86-64; it does not publish prebuilt wheels.

## Usage

This example is self-contained:

```python
import numpy as np
from mojopillow import Image, ImageChops, ImageFilter

pixels = np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3)
image = Image.fromarray(pixels, "RGB")

small = image.resize((96, 96), Image.Resampling.LANCZOS)
sharp = small.filter(ImageFilter.SHARPEN)
gray = sharp.convert("L")
mixed = ImageChops.screen(sharp, Image.new("RGB", sharp.size, (32, 64, 96)))

assert gray.size == (96, 96)
assert np.asarray(mixed).shape == (96, 96, 3)
```

With Pillow available, `Image.open("input.png")` and
`image.save("output.webp")` use its mature codecs while retaining Mojo-backed
compute operations.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 machine with 72
logical CPUs, Linux x86-64. Times are the best of five runs on 3840x2160 uint8
images and include output allocation and Python/ctypes overhead.

| kernel | mojo-pillow | Pillow | speedup |
| --- | ---: | ---: | ---: |
| RGB bilinear resize 4K -> 1080p | 37.01 ms | 66.56 ms | 1.80x |
| RGBA Lanczos resize 4K -> 1080p | 197.07 ms | 324.42 ms | 1.65x |
| RGB SHARPEN convolution 4K | 66.72 ms | 276.41 ms | 4.14x |
| RGB -> L conversion 4K | 5.79 ms | 9.42 ms | 1.63x |
| RGB constant-alpha blend 4K | 10.06 ms | 21.73 ms | 2.16x |
| RGBA alpha composite 4K | 20.58 ms | 90.89 ms | 4.42x |

All measured kernels are faster than Pillow on this dual-socket machine. The
targeted RGB-to-L, blend, and alpha-composite kernels use SIMD with scalar
remainder handling. Alpha compositing and large blends also split independent
work above a size threshold; RGB-to-L remains serial because its memory-bound
4K kernel was faster without thread-launch overhead.

There is no GPU path.

## How it works

All kernels live in one Mojo compilation unit to avoid repeated fixed compiler
startup cost. Python passes C-contiguous NumPy buffer addresses and scalar
metadata through `ctypes`; exported Mojo functions reconstruct mutable
`UnsafePointer` values and write into Python-owned output arrays. No buffer is
allocated or retained across the FFI boundary.

Pixels use interleaved, row-major uint8 memory: `(height, width)` for `L` and
`(height, width, channels)` for multiband modes. Resize mirrors Pillow's
separable algorithm: Python computes the same filter supports and normalized
weights, quantizes them to 22-bit fixed-point coefficients, and Mojo applies
horizontal and vertical passes with uint8 rounding between passes. `LA` and
`RGBA` resize through premultiplied alpha, matching Pillow's transparent-edge
behavior. Large, write-disjoint kernels split work across physical CPU cores
when measurement shows that the launch cost is worthwhile.

MIT licensed.
