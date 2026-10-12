#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

#include <ctls/base/buffer.cuh>
#include <ctls/reduce.cuh>
#include <ctls/gemm.cuh>

struct Identity
{
  __host__ __device__ float operator()(float value) const { return value; }
};

struct Add
{
  __host__ __device__ float operator()(float lhs, float rhs) const
  {
    return lhs + rhs;
  }
};

int main()
{
  constexpr unsigned int input_size = 1025;
  constexpr unsigned int block_size = 100;
  constexpr unsigned int elements_per_thread = 8;
  constexpr unsigned int elements_per_output = block_size * elements_per_thread;
  constexpr unsigned int output_size =
      (input_size + elements_per_output - 1) / elements_per_output;

  std::vector<float> host_input(input_size);
  for (unsigned int i = 0; i < input_size; ++i)
  {
    host_input[i] = static_cast<float>(i + 1);
  }
  std::vector<float> host_output(output_size, 0.0f);
  Buffer<float> input(input_size);
  Buffer<float> output(output_size);
  input.copy_from_host(host_input.data(), host_input.size());

  auto input_slice = input.slice(0, input_size);
  auto output_slice = output.slice(0, output_size);
  cuda::ReduceParams<float> params(input_slice, output_slice,
                                   elements_per_thread);
  using Reduce = cuda::ReduceOp<float, Identity, Add>;
  Reduce op(Identity{}, Add{});

  cuda::ReduceKernel<Reduce, cuda::ReduceParams<float>>
      <<<output_size, block_size, ((block_size + 31) / 32) * sizeof(float)>>>(
          op, params);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess)
  {
    std::fprintf(stderr, "ReduceKernel launch failed: %s\n",
                 cudaGetErrorString(status));
    return 1;
  }
  status = cudaDeviceSynchronize();
  if (status != cudaSuccess)
  {
    std::fprintf(stderr, "ReduceKernel execution failed: %s\n",
                 cudaGetErrorString(status));
    return 1;
  }
  output.copy_to_host(host_output.data(), host_output.size());

  const float expected[] = {320400.0f, 205425.0f};
  for (unsigned int i = 0; i < output_size; ++i)
  {
    std::printf("output[%u]=%.1f (expected %.1f)\n", i, host_output[i],
                expected[i]);
    if (host_output[i] != expected[i])
    {
      return 2;
    }
  }
  return 0;
}
