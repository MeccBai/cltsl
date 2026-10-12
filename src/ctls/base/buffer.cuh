#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

template <typename data_t>
class DeviceSlice
{
public:
  __host__ __device__ DeviceSlice(data_t *pointer, std::size_t count)
      : data(pointer), size(count) {}

  __host__ __device__ data_t *get_data() const { return data; }
  __host__ __device__ std::size_t get_size() const { return size; }

  __host__ __device__ DeviceSlice create(std::size_t offset,
                                         std::size_t length) const
  {
    if (offset > size || length > size - offset)
    {
      return DeviceSlice<data_t>(nullptr, 0);
    }
    return DeviceSlice<data_t>(data ? data + offset : nullptr, length);
  }

private:
  data_t *data;
  std::size_t size;
};

template <typename data_t>
class Buffer
{
public:
  explicit Buffer(std::size_t count) : data(nullptr), size(count)
  {
    if (count == 0)
    {
      return;
    }
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(data_t))
    {
      throw std::length_error("Buffer size overflows size_t");
    }
    const cudaError_t status =
        cudaMalloc(reinterpret_cast<void **>(&data), count * sizeof(data_t));
    if (status != cudaSuccess)
    {
      data = nullptr;
      size = 0;
      throw std::runtime_error(std::string("cudaMalloc failed: ") +
                               cudaGetErrorString(status));
    }
  }

  ~Buffer()
  {
    if (data)
    {
      cudaFree(data);
    }
  }

  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;

  Buffer(Buffer &&other) noexcept : data(other.data), size(other.size)
  {
    other.data = nullptr;
    other.size = 0;
  }

  void copy_from_host(const data_t *source, std::size_t count,
                      std::size_t offset = 0)
  {
    check_range(offset, count);
    if (count == 0)
    {
      return;
    }
    if (!source)
    {
      throw std::invalid_argument("copy_from_host received a null source");
    }
    check_cuda(cudaMemcpy(data + offset, source, count * sizeof(data_t),
                          cudaMemcpyHostToDevice), "cudaMemcpy host to device");
  }

  void copy_to_host(data_t *destination, std::size_t count,
                    std::size_t offset = 0) const
  {
    check_range(offset, count);
    if (count == 0)
    {
      return;
    }
    if (!destination)
    {
      throw std::invalid_argument("copy_to_host received a null destination");
    }
    check_cuda(cudaMemcpy(destination, data + offset, count * sizeof(data_t),
                          cudaMemcpyDeviceToHost), "cudaMemcpy device to host");
  }

  DeviceSlice<data_t> slice(std::size_t offset, std::size_t length) const
  {
    if (offset > size || length > size - offset)
    {
      return DeviceSlice<data_t>(nullptr, 0);
    }
    return DeviceSlice<data_t>(data ? data + offset : nullptr, length);
  }

private:
  static void check_cuda(cudaError_t status, const char *operation)
  {
    if (status != cudaSuccess)
    {
      throw std::runtime_error(std::string(operation) + " failed: " +
                               cudaGetErrorString(status));
    }
  }

  void check_range(std::size_t offset, std::size_t count) const
  {
    if (offset > size || count > size - offset)
    {
      throw std::out_of_range("Buffer copy range is out of bounds");
    }
  }

  data_t *data;
  std::size_t size;
};
