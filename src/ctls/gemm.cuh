#include <ctls/base/buffer.cuh>
#include <ctls/pipe/prefetch.cuh>
#include <ctls/pipe/mma.cuh>
#include <ctls/pipe/buffers.cuh>

namespace cuda
{

    template <unsigned Rows, unsigned Cols>
    struct Shape
    {
        static constexpr unsigned rows = Rows;
        static constexpr unsigned cols = Cols;
    };

    template <unsigned RowStride, unsigned ColStride>
    struct Stride
    {
        static constexpr unsigned row = RowStride;
        static constexpr unsigned col = ColStride;
    };

    template <typename ShapeT, typename StrideT>
    struct Layout
    {
        using shape_type = ShapeT;
        using stride_type = StrideT;

        __host__ __device__ static constexpr unsigned offset(
            unsigned row,
            unsigned col)
        {
            return row * StrideT::row + col * StrideT::col;
        }
    };

    template <typename data_t, data_t alpha, data_t beta>
    class Epilogue
    {
        static constexpr data_t alpha_value = alpha;
        static constexpr data_t beta_value = beta;

    public:
        static constexpr __global__ void apply(data_t *output, data_t *accumulated, std::size_t size)
        {
            const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size)
            {
                output[idx] = alpha_value * accumulated[idx] + beta_value * output[idx];
            }
        }
    };

    template <typename data_t, typename LayoutT>
    class Matrix
    {
        data_t *data;

    public:
        __host__ __device__ Matrix(data_t *pointer) : data(pointer) {}

        using Layout = LayoutT;
    };

    template <class LayoutA, class LayoutB, class LayoutC, typename data_ta, typename data_tb, typename data_tc>
    class GemmParams
    {
        Matrix<data_ta, LayoutA> A;
        Matrix<data_tb, LayoutB> B;
        Matrix<data_tc, LayoutC> C;
    };

}