"""Pillow-compatible uint8 image kernels exposed through a C ABI."""

from max.algorithm import parallelize
from std.runtime import initialize_runtime
from std.sys.info import num_physical_cores, simd_width_of

comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int32, AnyOrigin[mut=True]]
comptime FPtr = UnsafePointer[Float32, AnyOrigin[mut=True]]
comptime PARALLEL_PIXELS = 262144


@always_inline
def prepare_runtime():
    initialize_runtime()


def bp(addr: Int) -> BPtr:
    return BPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


@always_inline
def clip_u8(value: Int) -> UInt8:
    if value <= 0:
        return UInt8(0)
    if value >= 255:
        return UInt8(255)
    return UInt8(value)


@always_inline
def shift_div255(value: Int) -> Int:
    return ((value >> 8) + value) >> 8


@always_inline
def div255(value: Int) -> Int:
    return shift_div255(value + 128)


@export("mp_resize_nearest")
def mp_resize_nearest(
    src_addr: Int,
    dst_addr: Int,
    src_w: Int,
    src_h: Int,
    dst_w: Int,
    dst_h: Int,
    channels: Int,
    box_left: Float64,
    box_top: Float64,
    box_right: Float64,
    box_bottom: Float64,
) abi("C"):
    var src = bp(src_addr)
    var dst = bp(dst_addr)
    var scale_x = (box_right - box_left) / Float64(dst_w)
    var scale_y = (box_bottom - box_top) / Float64(dst_h)
    var workers = num_physical_cores() if dst_w * dst_h >= PARALLEL_PIXELS else 1

    @parameter
    def process(worker: Int):
        var y0 = worker * dst_h // workers
        var y1 = (worker + 1) * dst_h // workers
        for y in range(y0, y1):
            var sy = Int(box_top + (Float64(y) + 0.5) * scale_y)
            sy = min(max(sy, 0), src_h - 1)
            for x in range(dst_w):
                var sx = Int(box_left + (Float64(x) + 0.5) * scale_x)
                sx = min(max(sx, 0), src_w - 1)
                var source = (sy * src_w + sx) * channels
                var target = (y * dst_w + x) * channels
                for c in range(channels):
                    dst[target + c] = src[source + c]

    if workers > 1:
        prepare_runtime()
        parallelize[process](workers, workers)
    else:
        process(0)


@export("mp_resample_horizontal")
def mp_resample_horizontal(
    src_addr: Int,
    dst_addr: Int,
    bounds_addr: Int,
    coeffs_addr: Int,
    src_w: Int,
    height: Int,
    dst_w: Int,
    channels: Int,
    kernel_size: Int,
) abi("C"):
    var src = bp(src_addr)
    var dst = bp(dst_addr)
    var bounds = ip(bounds_addr)
    var coeffs = ip(coeffs_addr)
    var workers = num_physical_cores() if dst_w * height >= PARALLEL_PIXELS else 1

    @parameter
    def process(worker: Int):
        var y0 = worker * height // workers
        var y1 = (worker + 1) * height // workers
        for y in range(y0, y1):
            for x in range(dst_w):
                var first = Int(bounds[x * 2])
                var count = Int(bounds[x * 2 + 1])
                for c in range(channels):
                    var acc = Int64(1 << 21)
                    for k in range(count):
                        acc += (
                            Int64(src[(y * src_w + first + k) * channels + c])
                            * Int64(coeffs[x * kernel_size + k])
                        )
                    dst[(y * dst_w + x) * channels + c] = clip_u8(Int(acc >> 22))

    if workers > 1:
        prepare_runtime()
        parallelize[process](workers, workers)
    else:
        process(0)


@export("mp_resample_vertical")
def mp_resample_vertical(
    src_addr: Int,
    dst_addr: Int,
    bounds_addr: Int,
    coeffs_addr: Int,
    width: Int,
    src_h: Int,
    dst_h: Int,
    channels: Int,
    kernel_size: Int,
) abi("C"):
    var src = bp(src_addr)
    var dst = bp(dst_addr)
    var bounds = ip(bounds_addr)
    var coeffs = ip(coeffs_addr)
    var workers = num_physical_cores() if width * dst_h >= PARALLEL_PIXELS else 1

    @parameter
    def process(worker: Int):
        var y0 = worker * dst_h // workers
        var y1 = (worker + 1) * dst_h // workers
        for y in range(y0, y1):
            var first = Int(bounds[y * 2])
            var count = Int(bounds[y * 2 + 1])
            for x in range(width):
                for c in range(channels):
                    var acc = Int64(1 << 21)
                    for k in range(count):
                        acc += (
                            Int64(src[((first + k) * width + x) * channels + c])
                            * Int64(coeffs[y * kernel_size + k])
                        )
                    dst[(y * width + x) * channels + c] = clip_u8(Int(acc >> 22))

    if workers > 1:
        prepare_runtime()
        parallelize[process](workers, workers)
    else:
        process(0)


@export("mp_convolve")
def mp_convolve(
    src_addr: Int,
    dst_addr: Int,
    kernel_addr: Int,
    width: Int,
    height: Int,
    channels: Int,
    kernel_size: Int,
    offset: Float32,
) abi("C"):
    var src = bp(src_addr)
    var dst = bp(dst_addr)
    var kernel = fp(kernel_addr)
    var radius = kernel_size // 2
    var workers = num_physical_cores() if width * height >= PARALLEL_PIXELS else 1

    @parameter
    def process(worker: Int):
        var y0 = worker * height // workers
        var y1 = (worker + 1) * height // workers
        for y in range(y0, y1):
            for x in range(width):
                var pixel = (y * width + x) * channels
                if x < radius or x >= width - radius or y < radius or y >= height - radius:
                    for c in range(channels):
                        dst[pixel + c] = src[pixel + c]
                    continue
                for c in range(channels):
                    var acc = offset + 0.5
                    for ky in range(kernel_size):
                        var sy = y + radius - ky
                        for kx in range(kernel_size):
                            var sx = x - radius + kx
                            acc += (
                                Float32(src[(sy * width + sx) * channels + c])
                                * kernel[ky * kernel_size + kx]
                            )
                    dst[pixel + c] = clip_u8(Int(acc))

    if workers > 1:
        prepare_runtime()
        parallelize[process](workers, workers)
    else:
        process(0)


@export("mp_convert")
def mp_convert(
    src_addr: Int, dst_addr: Int, pixels: Int, operation: Int
) abi("C"):
    var src = bp(src_addr)
    var dst = bp(dst_addr)
    var workers = num_physical_cores() if pixels >= PARALLEL_PIXELS else 1

    if operation == 0:
        comptime W = 16

        @parameter
        def process_rgb_l(worker: Int):
            var start = 0
            var end = pixels
            var vector_end = end - (end - start) % W
            var i = start
            while i < vector_end:
                var source = i * 3
                var first = src.load[width=W](source)
                var second = src.load[width=W](source + W)
                var third = src.load[width=W](source + W * 2)
                var red = SIMD[DType.uint8, W](
                    first[0],
                    first[3],
                    first[6],
                    first[9],
                    first[12],
                    first[15],
                    second[2],
                    second[5],
                    second[8],
                    second[11],
                    second[14],
                    third[1],
                    third[4],
                    third[7],
                    third[10],
                    third[13],
                ).cast[DType.int32]()
                var green = SIMD[DType.uint8, W](
                    first[1],
                    first[4],
                    first[7],
                    first[10],
                    first[13],
                    second[0],
                    second[3],
                    second[6],
                    second[9],
                    second[12],
                    second[15],
                    third[2],
                    third[5],
                    third[8],
                    third[11],
                    third[14],
                ).cast[DType.int32]()
                var blue = SIMD[DType.uint8, W](
                    first[2],
                    first[5],
                    first[8],
                    first[11],
                    first[14],
                    second[1],
                    second[4],
                    second[7],
                    second[10],
                    second[13],
                    third[0],
                    third[3],
                    third[6],
                    third[9],
                    third[12],
                    third[15],
                ).cast[DType.int32]()
                var luminance = (
                    red * SIMD[DType.int32, W](19595)
                    + green * SIMD[DType.int32, W](38470)
                    + blue * SIMD[DType.int32, W](7471)
                    + SIMD[DType.int32, W](32768)
                ) >> 16
                dst.store(i, luminance.cast[DType.uint8]())
                i += W
            while i < end:
                var source = i * 3
                var luminance = (
                    Int(src[source]) * 19595
                    + Int(src[source + 1]) * 38470
                    + Int(src[source + 2]) * 7471
                    + 32768
                ) >> 16
                dst[i] = UInt8(luminance)
                i += 1

        process_rgb_l(0)
        return

    @parameter
    def process(worker: Int):
        var start = worker * pixels // workers
        var end = (worker + 1) * pixels // workers
        for i in range(start, end):
            if operation == 0 or operation == 1:
                var step = 3 if operation == 0 else 4
                var source = i * step
                var lum = (
                    Int(src[source]) * 19595
                    + Int(src[source + 1]) * 38470
                    + Int(src[source + 2]) * 7471
                    + 32768
                ) >> 16
                dst[i] = UInt8(lum)
            elif operation == 2 or operation == 3:
                var target_step = 3 if operation == 2 else 4
                var target = i * target_step
                var value = src[i]
                dst[target] = value
                dst[target + 1] = value
                dst[target + 2] = value
                if target_step == 4:
                    dst[target + 3] = UInt8(255)
            elif operation == 4:
                var source = i * 3
                var target = i * 4
                dst[target] = src[source]
                dst[target + 1] = src[source + 1]
                dst[target + 2] = src[source + 2]
                dst[target + 3] = UInt8(255)
            elif operation == 5:
                var source = i * 4
                var target = i * 3
                dst[target] = src[source]
                dst[target + 1] = src[source + 1]
                dst[target + 2] = src[source + 2]
            elif operation == 6:
                var source = i * 3
                var target = i * 4
                dst[target] = UInt8(255 - Int(src[source]))
                dst[target + 1] = UInt8(255 - Int(src[source + 1]))
                dst[target + 2] = UInt8(255 - Int(src[source + 2]))
                dst[target + 3] = UInt8(0)
            elif operation == 7:
                var source = i * 4
                var target = i * 3
                var nk = 255 - Int(src[source + 3])
                dst[target] = clip_u8(
                    nk - div255(Int(src[source]) * nk)
                )
                dst[target + 1] = clip_u8(
                    nk - div255(Int(src[source + 1]) * nk)
                )
                dst[target + 2] = clip_u8(
                    nk - div255(Int(src[source + 2]) * nk)
                )
            elif operation == 8:
                var source = i * 3
                var target = i * 2
                var lum = (
                    Int(src[source]) * 19595
                    + Int(src[source + 1]) * 38470
                    + Int(src[source + 2]) * 7471
                    + 32768
                ) >> 16
                dst[target] = UInt8(lum)
                dst[target + 1] = UInt8(255)
            elif operation == 9:
                var source = i * 4
                var target = i * 2
                var lum = (
                    Int(src[source]) * 19595
                    + Int(src[source + 1]) * 38470
                    + Int(src[source + 2]) * 7471
                    + 32768
                ) >> 16
                dst[target] = UInt8(lum)
                dst[target + 1] = src[source + 3]
            elif operation == 10:
                var source = i * 2
                var target = i * 4
                var value = src[source]
                dst[target] = value
                dst[target + 1] = value
                dst[target + 2] = value
                dst[target + 3] = src[source + 1]
            elif operation == 11:
                var source = i * 2
                var target = i * 3
                var value = src[source]
                dst[target] = value
                dst[target + 1] = value
                dst[target + 2] = value
            elif operation == 12:
                var source = i * 2
                dst[i] = src[source]
            elif operation == 13 or operation == 14:
                var source_step = 1 if operation == 13 else 2
                var target = i * 4
                dst[target] = UInt8(0)
                dst[target + 1] = UInt8(0)
                dst[target + 2] = UInt8(0)
                dst[target + 3] = UInt8(255 - Int(src[i * source_step]))

    if workers > 1:
        prepare_runtime()
        parallelize[process](workers, workers)
    else:
        process(0)


@export("mp_convert_matrix")
def mp_convert_matrix(
    src_addr: Int,
    dst_addr: Int,
    matrix_addr: Int,
    pixels: Int,
    dst_channels: Int,
) abi("C"):
    var src = bp(src_addr)
    var dst = bp(dst_addr)
    var matrix = fp(matrix_addr)
    var workers = num_physical_cores() if pixels >= PARALLEL_PIXELS else 1

    @parameter
    def process(worker: Int):
        var start = worker * pixels // workers
        var end = (worker + 1) * pixels // workers
        for i in range(start, end):
            var source = i * 3
            for c in range(dst_channels):
                var m = c * 4
                var value = (
                    matrix[m] * Float32(src[source])
                    + matrix[m + 1] * Float32(src[source + 1])
                    + matrix[m + 2] * Float32(src[source + 2])
                    + matrix[m + 3]
                    + 0.5
                )
                dst[i * dst_channels + c] = clip_u8(Int(value))

    if workers > 1:
        prepare_runtime()
        parallelize[process](workers, workers)
    else:
        process(0)


@export("mp_premultiply_rgba")
def mp_premultiply_rgba(
    src_addr: Int, dst_addr: Int, pixels: Int, channels: Int, inverse: Int
) abi("C"):
    var src = bp(src_addr)
    var dst = bp(dst_addr)
    for i in range(pixels):
        var source = i * channels
        var alpha = Int(src[source + channels - 1])
        for c in range(channels - 1):
            if inverse == 0:
                dst[source + c] = UInt8(div255(Int(src[source + c]) * alpha))
            elif alpha == 0 or alpha == 255:
                dst[source + c] = src[source + c]
            else:
                dst[source + c] = clip_u8(255 * Int(src[source + c]) // alpha)
        dst[source + channels - 1] = src[source + channels - 1]


@export("mp_blend")
def mp_blend(
    first_addr: Int,
    second_addr: Int,
    dst_addr: Int,
    count: Int,
    alpha: Float32,
) abi("C"):
    var first = bp(first_addr)
    var second = bp(second_addr)
    var dst = bp(dst_addr)
    var workers = num_physical_cores() if count >= PARALLEL_PIXELS else 1
    comptime W = simd_width_of[DType.float64]()

    @parameter
    def process(worker: Int):
        var start = worker * count // workers
        var end = (worker + 1) * count // workers
        var vector_end = end - (end - start) % W
        var i = start
        while i < vector_end:
            var first_values = first.load[width=W](i).cast[DType.float32]()
            var second_values = second.load[width=W](i).cast[DType.float32]()
            var values = (
                first_values
                + SIMD[DType.float32, W](alpha)
                * (second_values - first_values)
            ).cast[DType.int32]()
            var clipped = min(
                max(values, SIMD[DType.int32, W](0)),
                SIMD[DType.int32, W](255),
            )
            dst.store(i, clipped.cast[DType.uint8]())
            i += W
        while i < end:
            var value = (
                Float32(first[i])
                + alpha * Float32(Int(second[i]) - Int(first[i]))
            )
            dst[i] = clip_u8(Int(value))
            i += 1

    if workers > 1:
        prepare_runtime()
        parallelize[process](workers, workers)
    else:
        process(0)


@export("mp_chop")
def mp_chop(
    first_addr: Int,
    second_addr: Int,
    dst_addr: Int,
    count: Int,
    operation: Int,
    scale: Float32,
    offset: Int,
) abi("C"):
    var first = bp(first_addr)
    var second = bp(second_addr)
    var dst = bp(dst_addr)
    var workers = num_physical_cores() if count >= PARALLEL_PIXELS else 1

    @parameter
    def process(worker: Int):
        var start = worker * count // workers
        var end = (worker + 1) * count // workers
        for i in range(start, end):
            var a = Int(first[i])
            var b = Int(second[i])
            var value = 0
            if operation == 0:
                value = max(a, b)
            elif operation == 1:
                value = min(a, b)
            elif operation == 2:
                value = abs(a - b)
            elif operation == 3:
                value = a * b // 255
            elif operation == 4:
                value = 255 - (255 - a) * (255 - b) // 255
            elif operation == 5:
                value = Int(Float32(a + b) / scale) + offset
            elif operation == 6:
                value = Int(Float32(a - b) / scale) + offset
            elif operation == 7:
                value = (a + b) & 255
            elif operation == 8:
                value = (a - b) & 255
            elif operation == 9:
                value = (
                    ((255 - a) * (a * b)) // 65536
                    + (a * (255 - ((255 - a) * (255 - b) // 255))) // 255
                )
            elif operation == 10:
                value = (
                    a * b // 127
                    if b < 128
                    else 255 - ((255 - b) * (255 - a) // 127)
                )
            elif operation == 11:
                value = (
                    a * b // 127
                    if a < 128
                    else 255 - ((255 - a) * (255 - b) // 127)
                )
            dst[i] = clip_u8(value)

    if workers > 1:
        prepare_runtime()
        parallelize[process](workers, workers)
    else:
        process(0)


@export("mp_composite")
def mp_composite(
    first_addr: Int,
    second_addr: Int,
    mask_addr: Int,
    dst_addr: Int,
    pixels: Int,
    channels: Int,
    mask_channels: Int,
) abi("C"):
    var first = bp(first_addr)
    var second = bp(second_addr)
    var mask = bp(mask_addr)
    var dst = bp(dst_addr)
    for i in range(pixels):
        var m = Int(mask[i * mask_channels + mask_channels - 1])
        for c in range(channels):
            var a = Int(second[i * channels + c])
            var b = Int(first[i * channels + c])
            dst[i * channels + c] = UInt8(div255(a * (255 - m) + b * m))


@export("mp_alpha_composite_rgba")
def mp_alpha_composite_rgba(
    first_addr: Int,
    second_addr: Int,
    dst_addr: Int,
    pixels: Int,
    channels: Int,
) abi("C"):
    var first = bp(first_addr)
    var second = bp(second_addr)
    var dst = bp(dst_addr)
    var workers = num_physical_cores() if pixels >= PARALLEL_PIXELS else 1
    comptime W = simd_width_of[DType.float64]()

    @parameter
    def process(worker: Int):
        var start = worker * pixels // workers
        var end = (worker + 1) * pixels // workers
        var vector_end = end - (end - start) % W
        var i = start
        while i < vector_end:
            var base = i * channels
            var src_alpha = (
                second + base + channels - 1
            ).strided_load[width=W](channels).cast[DType.int32]()
            var dst_alpha = (
                first + base + channels - 1
            ).strided_load[width=W](channels).cast[DType.int32]()
            var transparent = src_alpha.eq(SIMD[DType.int32, W](0))
            var blend = (
                dst_alpha
                * (SIMD[DType.int32, W](255) - src_alpha)
            )
            var alpha255 = src_alpha * SIMD[DType.int32, W](255) + blend
            var denominator = max(alpha255, SIMD[DType.int32, W](1))
            var coef1 = (
                src_alpha
                * SIMD[DType.int32, W](255 * 255 * 128)
                // denominator
            )
            var coef2 = SIMD[DType.int32, W](255 * 128) - coef1
            for c in range(channels - 1):
                var foreground = (
                    second + base + c
                ).strided_load[width=W](channels).cast[DType.int32]()
                var background = (
                    first + base + c
                ).strided_load[width=W](channels).cast[DType.int32]()
                var mixed = foreground * coef1 + background * coef2
                var rounded = mixed + SIMD[DType.int32, W](128 << 7)
                var value = (((rounded >> 8) + rounded) >> 8) >> 7
                var selected = transparent.select(background, value)
                (dst + base + c).strided_store[width=W](
                    selected.cast[DType.uint8](), channels
                )
            var rounded_alpha = alpha255 + SIMD[DType.int32, W](128)
            var output_alpha = (
                (rounded_alpha >> 8) + rounded_alpha
            ) >> 8
            (dst + base + channels - 1).strided_store[width=W](
                output_alpha.cast[DType.uint8](), channels
            )
            i += W
        while i < end:
            var base = i * channels
            var src_alpha = Int(second[base + channels - 1])
            if src_alpha == 0:
                for c in range(channels):
                    dst[base + c] = first[base + c]
                i += 1
                continue
            var blend = Int(first[base + channels - 1]) * (255 - src_alpha)
            var alpha255 = src_alpha * 255 + blend
            var coef1 = src_alpha * 255 * 255 * 128 // alpha255
            var coef2 = 255 * 128 - coef1
            for c in range(channels - 1):
                var mixed = (
                    Int(second[base + c]) * coef1
                    + Int(first[base + c]) * coef2
                )
                dst[base + c] = UInt8(
                    shift_div255(mixed + (128 << 7)) >> 7
                )
            dst[base + channels - 1] = UInt8(
                shift_div255(alpha255 + 128)
            )
            i += 1

    if workers > 1:
        prepare_runtime()
        parallelize[process](workers, workers)
    else:
        process(0)
