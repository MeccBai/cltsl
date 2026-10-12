#pragma once

#include <cstddef>
#include <ctls/base/buffer.cuh>

namespace cuda
{
    template <typename data_t>
    class ReduceParams
    {
    public:
        using value_type = data_t;

        __host__ __device__ ReduceParams(DeviceSlice<data_t> input_slice,
                                         DeviceSlice<data_t> output_slice,
                                         unsigned int elements_per_thread)
            : input(input_slice), output(output_slice),
              per_element_thread(elements_per_thread) {}

        __device__ DeviceSlice<data_t> get_input() const { return input; }
        __device__ DeviceSlice<data_t> get_output() const { return output; }
        __device__ unsigned int get_per_element_thread() const
        {
            return per_element_thread;
        }

    private:
        DeviceSlice<data_t> input;
        DeviceSlice<data_t> output;
        unsigned int per_element_thread;
    };

    template <typename data_t>
    struct IdentityMap
    {
        __host__ __device__ data_t operator()(data_t value) const
        {
            return value;
        }
    };

    template <typename data_t, class map_t, class reduce_t>
    class WarpReduce
    {
    public:
        __host__ __device__ WarpReduce(map_t map_op, reduce_t reduce_op)
            : f_map(map_op), f_reduce(reduce_op) {}

        __device__ data_t map(data_t value) const
        {
            return f_map(value);
        }

        // Every lane named in the active mask calls each shuffle. The butterfly
        // order preserves lane order, so associative non-commutative operators
        // (such as concatenation) are also handled correctly.
        __device__ data_t operator()(data_t value, bool active) const
        {
            const unsigned int participating = __activemask();
            const unsigned int active_lanes =
                __ballot_sync(participating, active);
            const unsigned int lane = threadIdx.x & 31u;

            for (unsigned int offset = 1; offset < 32; offset <<= 1)
            {
                const data_t other =
                    __shfl_xor_sync(participating, value, offset);
                const unsigned int partner = lane ^ offset;
                if (active && (active_lanes & (1u << partner)) != 0)
                {
                    value = f_reduce(value, other);
                }
            }
            return value;
        }

    private:
        map_t f_map;
        reduce_t f_reduce;
    };

    template <typename data_t, class reduce_t>
    class FinalReduce
    {
    public:
        __host__ __device__ explicit FinalReduce(reduce_t reduce_op)
            : warp_reduce(IdentityMap<data_t>{}, reduce_op) {}

        __device__ void operator()(data_t *shared,
                                   unsigned int partial_count,
                                   data_t *output,
                                   std::size_t output_index) const
        {
            const unsigned int lane = threadIdx.x & 31u;
            const bool active = lane < partial_count;
            data_t value = active ? shared[lane] : data_t{};
            value = warp_reduce(value, active);
            if (lane == 0)
            {
                output[output_index] = value;
            }
        }

    private:
        WarpReduce<data_t, IdentityMap<data_t>, reduce_t> warp_reduce;
    };

    template <typename data_t, class map_t, class reduce_t>
    class ReduceOp
    {
    public:
        __host__ __device__ ReduceOp(map_t map_op, reduce_t reduce_op)
            : warp_reduce(map_op, reduce_op), final_reduce(reduce_op) {}

        // Launch one block per output. Dynamic shared memory must be at least
        // ceil(blockDim.x / 32) * sizeof(data_t) bytes.
        __device__ void operator()(ReduceParams<data_t> params,
                                   data_t *shared) const
        {
            const unsigned int elements_per_output =
                params.get_per_element_thread();
            const DeviceSlice<data_t> input = params.get_input();
            const DeviceSlice<data_t> output = params.get_output();
            const data_t *input_data =
                static_cast<const data_t *>(input.get_data());
            data_t *output_data =
                static_cast<data_t *>(output.get_data());
            const std::size_t output_index = blockIdx.x;

            // These conditions are block-uniform, so every thread either
            // returns or reaches the block-wide barrier below.
            if (elements_per_output == 0 ||
                output_index >= output.get_size())
            {
                return;
            }

            const std::size_t input_offset =
                output_index * static_cast<std::size_t>(blockDim.x) *
                elements_per_output;
            if (input_offset >= input.get_size())
            {
                return;
            }

            const std::size_t remaining = input.get_size() - input_offset;
            const std::size_t block_input_count =
                static_cast<std::size_t>(blockDim.x) * elements_per_output;
            const std::size_t count =
                remaining < block_input_count ? remaining : block_input_count;
            const std::size_t thread_offset =
                static_cast<std::size_t>(threadIdx.x) * elements_per_output;

            data_t local{};
            bool active = false;
            for (unsigned int index = 0;
                 index < elements_per_output && thread_offset + index < count;
                 ++index)
            {
                const data_t mapped =
                    warp_reduce.map(input_data[input_offset + thread_offset + index]);
                local = active ? f_reduce(local, mapped) : mapped;
                active = true;
            }

            local = warp_reduce(local, active);
            const unsigned int warp = threadIdx.x >> 5;
            const unsigned int warp_count = (blockDim.x + 31u) >> 5;
            if ((threadIdx.x & 31u) == 0)
            {
                shared[warp] = local;
            }
            __syncthreads();

            if (warp == 0)
            {
                final_reduce(shared, warp_count, output_data, output_index);
            }
        }

    private:
        WarpReduce<data_t, map_t, reduce_t> warp_reduce;
        FinalReduce<data_t, reduce_t> final_reduce;
        reduce_t f_reduce;
    };

    template <class Operator, class Params>
    __global__ void ReduceKernel(Operator op, Params params)
    {
        using data_t = typename Params::value_type;
        extern __shared__ __align__(16) unsigned char storage[];
        op(params, reinterpret_cast<data_t *>(storage));
    }
}
