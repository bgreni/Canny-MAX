import compiler
from max.tensor import OutputTensor, InputTensor, foreach
from utils.index import IndexList
from runtime.asyncrt import DeviceContextPtr
from math import pi
from utils._select import _select_register_value as select

@compiler.register("grayscale")
struct Grayscale:

    @staticmethod
    fn execute[target: StaticString](
        output: OutputTensor,
        image: InputTensor,
        ctx: DeviceContextPtr
    ) raises:

        @parameter
        @always_inline
        fn do[width: Int](idx: IndexList[output.rank]) -> SIMD[output.dtype, width]:
            var b = image[idx[0], idx[1], 0]
            var g = image[idx[0], idx[1], 1]
            var r = image[idx[0], idx[1], 2]

            return SIMD[output.dtype, width](0.299 * r + 0.587 * g + 0.114 * b)

        foreach[do, target=target, simd_width=1](output, ctx)

@compiler.register("atan2")
struct Atan2:

    @staticmethod
    fn execute[target: StaticString](
        output: OutputTensor,
        x: InputTensor[dtype=output.dtype, rank=output.rank],
        y: InputTensor[dtype=output.dtype, rank=output.rank],
        ctx: DeviceContextPtr
    ) raises:

        @always_inline
        fn atan2[width: Int](y: SIMD[output.dtype, width], x: SIMD[output.dtype, width]) -> SIMD[output.dtype, width]:
            var t3 = abs(x)
            var t1 = abs(y)
            var t0 = max(t3, t1)
            t1 = min(t3, t1)
            t3 = 1.0 / t0
            t3 = t1 * t3

            t4 = t3 * t3
            t0 = -0.013480470
            t0 = t0 * t4 + 0.057477314
            t0 = t0 * t4 - 0.121239071
            t0 = t0 * t4 + 0.195635925
            t0 = t0 * t4 - 0.332994597
            t0 = t0 * t4 + 0.999995630
            t3 = t0 * t3

            t3 = select(all(abs(y) > abs(x)), 1.570796327 - t3, t3)
            t3 = select(all(x < 0), 3.141592654 - t3, t3)
            t3 = select(all(y < 0), -t3, t3)

            return t3

        @parameter
        @always_inline
        fn do[width: Int](idx: IndexList[output.rank]) -> SIMD[output.dtype, width]:
            var x_i = x.load[width](idx)
            var y_i = y.load[width](idx)
            return atan2[width](x_i, y_i)

        foreach[do, target=target, simd_width=1](output, ctx)


@compiler.register("non_max_supression")
struct NonMaxSupression:

    @staticmethod
    fn execute[target: StaticString](
        output: OutputTensor,
        mag: InputTensor[dtype=output.dtype, rank=output.rank],
        direction: InputTensor[dtype=output.dtype, rank=output.rank],
        ctx: DeviceContextPtr
    ) raises:

        @parameter
        @always_inline
        fn do[width: Int](idx: IndexList[output.rank]) -> SIMD[output.dtype, width]:

            var x = idx[1]
            var y = idx[0]

            # skip edges
            if x == 0 or y == 0 or x == mag.dim_size[1]() - 1 or y == mag.dim_size[0]() - 1:
                return 0.0

            var m = mag[idx]
            var d = direction[idx]
            var q: Scalar[output.dtype] = 0.0
            var r: Scalar[output.dtype] = 0.0

            var angle = d * 180.0 / pi
            angle += select(angle < 0, __type_of(q)(180.0), 0.0)

            if (angle >= 0 and angle < 22.5) or (angle >= 157.5 and angle <= 180.0):
                q = mag[y+1, x, 1]
                r = mag[y-1, x, 1]
            elif angle >= 22.5 and angle < 67.5:
                q = mag[y+1, x - 1, 1]
                r = mag[y-1, x + 1, 1]
            elif angle >= 67.5 and angle < 112.5:
                q = mag[y, x - 1, 1]
                r = mag[y, x + 1, 1]
            elif angle >= 112.5 and angle < 157.5:
                q = mag[y-1, x + 1, 1]
                r = mag[y+1, x - 1, 1]

            return select(m >= q and m >= r, m, 0.0)



        foreach[do, target=target, simd_width=1](output, ctx)

@compiler.register("init_thresholding")
struct InitThresholding:

    @staticmethod
    fn execute[target: StaticString](
        output: OutputTensor,
        image: InputTensor[dtype=output.dtype, rank=output.rank],
        low: Scalar[output.dtype],
        high: Scalar[output.dtype],
        ctx: DeviceContextPtr
    ) raises:
        alias T = Scalar[output.dtype]
        @parameter
        @always_inline
        fn do[width: Int](idx: IndexList[output.rank]) -> SIMD[output.dtype, width]:
            var pixel = image[idx]
            var values = InlineArray[T, 3](0, 255, 175)
            var i: T = 0
            i += T(pixel >= high)
            i += T(pixel >= low)
            return values[i]

        foreach[do, target=target, simd_width=1](output, ctx)

@compiler.register("thresholding")
struct Thresholding:

    @staticmethod
    fn execute[target: StaticString](
        output: OutputTensor,
        image: InputTensor[dtype=output.dtype, rank=output.rank],
        ctx: DeviceContextPtr
    ) raises:

        @parameter
        @always_inline
        fn do[width: Int](idx: IndexList[output.rank]) -> SIMD[output.dtype, width]:
            var pixel = image[idx]
            var x = idx[1]
            var y = idx[0]
            # skip edges
            if x == 0 or y == 0 or x == image.dim_size[1]() - 1 or y == image.dim_size[0]() - 1:
                return 0.0

            if pixel == 175:
                var arr = SIMD[DType.bool, 8](False)
                arr[0] = offset_load[
                    height_offset= -1, width_offset= -1
                ](image, idx) == 255
                arr[1] = offset_load[
                    height_offset= -1, width_offset=0
                ](image, idx) == 255
                arr[2] = offset_load[
                    height_offset= -1, width_offset=1
                ](image, idx) == 255
                arr[3] = offset_load[
                    height_offset=0, width_offset= -1
                ](image, idx) == 255
                arr[4] = offset_load[
                    height_offset=0, width_offset=1
                ](image, idx) == 255
                arr[5] = offset_load[
                    height_offset=1, width_offset= -1
                ](image, idx) == 255
                arr[6] = offset_load[
                    height_offset=1, width_offset=0
                ](image, idx) == 255
                arr[7] = offset_load[
                    height_offset=1, width_offset=1
                ](image, idx) == 255

                return select(any(arr), 255, 0)
            return pixel
        foreach[do, target=target, simd_width=1](output, ctx)


fn offset_load[
    _rank: Int, type: DType, //, height_offset: Int, width_offset: Int
](tensor: InputTensor[dtype=type, rank=_rank], index: IndexList[_rank]) -> Scalar[
    type
]:
    var clamped_index = index
    clamped_index[0] = clamped_index[0] + height_offset
    clamped_index[1] = clamped_index[1] + width_offset
    return tensor[clamped_index]
