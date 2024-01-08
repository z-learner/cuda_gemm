#pragma once

#include "cuda_runtime.h"

#include <cstddef>
#include <cstdint>

namespace tools {

// Z - Y - X ...
template<typename T, uint32_t... Dims> class NdarrayWarp {
public:
  __host__ __device__ NdarrayWarp(T* memory)
    : data_(memory) {}

  template<typename... Indices> __host__ __device__ T& operator()(Indices... indices) {
    static_assert(sizeof...(Indices) == sizeof...(Dims), "Incorrect number of indices");
    return data_[index(indices...)];
  }

  template<typename... Indices> __host__ __device__ const T& operator()(Indices... indices) const {
    static_assert(sizeof...(Indices) == sizeof...(Dims), "Incorrect number of indices");
    return data_[index(indices...)];
  }

  template<typename... Indices> __host__ __device__ uint32_t index(Indices... indices) const { return index_helper<sizeof...(Dims) - 1, Dims...>(indices...); }

private:
  template<size_t N, uint32_t First, uint32_t... Rest, typename... Indices> __host__ __device__ uint32_t index_helper(uint32_t first, Indices... rest) const {
    if constexpr (N > 0) {
      return first * product<Rest...>() + index_helper<N - 1, Rest...>(rest...);
    } else {
      return first;
    }
  }

  template<uint32_t First, uint32_t... Rest> __host__ __device__ static constexpr uint32_t product() {
    if constexpr (sizeof...(Rest) > 0) {
      return First * product<Rest...>();
    } else {
      return First;
    }
  }

  T* data_;
};

}   // namespace tools
