#include "kernel/gmm.hpp"


namespace cuda_kernel {


template<typename T> __global__ void noraml_gmm(T* input_a, T* input_b, T* output, int N, int M, int P) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= N || col >= P) { return; }

  T sum = 0.0f;
  for (int index = 0; index < M; ++index) { sum += input_a[index + row * M] * input_b[index * P + col]; }

  output[row * P + col] = sum;
}


template<typename T, int BlockSize> __global__ void shared_gmm(T* input_a, T* input_b, T* output, int N, int M, int P) {

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= N || col >= P) { return; }

  __shared__ float shared_sub_block_a[BlockSize][BlockSize];
  __shared__ float shared_sub_block_b[BlockSize][BlockSize];

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  T sum = 0;

  for (int index = 0; index < (M + BlockSize - 1) / BlockSize; ++index) {
    // load A
    if (index * BlockSize + tx < M) {
      shared_sub_block_a[ty][tx] = input_a[row * M + index * BlockSize + tx];
    } else {
      shared_sub_block_a[ty][tx] = 0;
    }

    // load B
    if (index * BlockSize + ty < M) {
      shared_sub_block_b[ty][tx] = input_b[(index * BlockSize + ty) * P + col];
    } else {
      shared_sub_block_b[ty][tx] = 0;
    }

    __syncthreads();

    // calculdata sum
    for (int k = 0; k < BlockSize; ++k) { sum += (shared_sub_block_a[ty][k] * shared_sub_block_b[k][tx]); }
  }
  output[row * P + col] = sum;
}


template<typename T, int BlockSize> __global__ void shared_gmm_bank_opt(T* input_a, T* input_b, T* output, int N, int M, int P) {}

}   // namespace cuda_kernel

void luanch_gmm(float* input_a, float* input_b, float* output, int N, int M, int P, GmmAlog gmm_algo) {

  static constexpr size_t BlockSize = 16;

  dim3 grid_dim((N + BlockSize - 1) / BlockSize, (P + BlockSize - 1) / BlockSize);
  dim3 block_dim(BlockSize, BlockSize);

  switch (gmm_algo) {
  case GmmAlog::kNormal: cuda_kernel::noraml_gmm<float><<<grid_dim, block_dim>>>(input_a, input_b, output, N, M, P); break;
  case GmmAlog::kShared: cuda_kernel::shared_gmm<float, BlockSize><<<grid_dim, block_dim>>>(input_a, input_b, output, N, M, P); break;
  case GmmAlog::kSharedOpt: cuda_kernel::shared_gmm_bank_opt<float, BlockSize><<<grid_dim, block_dim>>>(input_a, input_b, output, N, M, P); break;
  default: assert(false); break;
  }
  return;
}