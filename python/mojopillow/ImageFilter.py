"""Pillow-compatible fixed convolution filters."""

from __future__ import annotations

from collections.abc import Sequence


class Kernel:
    name = "Kernel"

    def __init__(
        self,
        size: tuple[int, int],
        kernel: Sequence[float],
        scale: float | None = None,
        offset: float = 0,
    ) -> None:
        if scale is None:
            scale = sum(kernel)
        if size[0] * size[1] != len(kernel):
            raise ValueError("not enough coefficients in kernel")
        self.filterargs = size, scale, offset, tuple(kernel)


class _Builtin(Kernel):
    filterargs: tuple[object, ...]

    def __init__(self) -> None:
        pass


class BLUR(_Builtin):
    name = "Blur"
    filterargs = (5, 5), 16, 0, (
        1, 1, 1, 1, 1,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        1, 1, 1, 1, 1,
    )


class CONTOUR(_Builtin):
    name = "Contour"
    filterargs = (3, 3), 1, 255, (
        -1, -1, -1, -1, 8, -1, -1, -1, -1,
    )


class DETAIL(_Builtin):
    name = "Detail"
    filterargs = (3, 3), 6, 0, (0, -1, 0, -1, 10, -1, 0, -1, 0)


class EDGE_ENHANCE(_Builtin):
    name = "Edge-enhance"
    filterargs = (3, 3), 2, 0, (
        -1, -1, -1, -1, 10, -1, -1, -1, -1,
    )


class EDGE_ENHANCE_MORE(_Builtin):
    name = "Edge-enhance More"
    filterargs = (3, 3), 1, 0, (
        -1, -1, -1, -1, 9, -1, -1, -1, -1,
    )


class EMBOSS(_Builtin):
    name = "Emboss"
    filterargs = (3, 3), 1, 128, (-1, 0, 0, 0, 1, 0, 0, 0, 0)


class FIND_EDGES(_Builtin):
    name = "Find Edges"
    filterargs = (3, 3), 1, 0, (
        -1, -1, -1, -1, 8, -1, -1, -1, -1,
    )


class SHARPEN(_Builtin):
    name = "Sharpen"
    filterargs = (3, 3), 16, 0, (
        -2, -2, -2, -2, 32, -2, -2, -2, -2,
    )


class SMOOTH(_Builtin):
    name = "Smooth"
    filterargs = (3, 3), 13, 0, (1, 1, 1, 1, 5, 1, 1, 1, 1)


class SMOOTH_MORE(_Builtin):
    name = "Smooth More"
    filterargs = (5, 5), 100, 0, (
        1, 1, 1, 1, 1,
        1, 5, 5, 5, 1,
        1, 5, 44, 5, 1,
        1, 5, 5, 5, 1,
        1, 1, 1, 1, 1,
    )
