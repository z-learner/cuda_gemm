#include "kernel/gmm.hpp"

#define TILE_K 4
#define TILE_SIZE 16

namespace cuda_kernel {

__device__ void load_gmem_tile_to_smem(const float* A, int tile_idx, int row, int col, float smemA[TILE_SIZE][TILE_SIZE]) {
  // load data to shared memory from global memory
  int tx           = threadIdx.x;
  int ty           = threadIdx.y;
  int gmem_idx_row = row + ty;
  int gmem_idx_col = tile_idx + col + tx;
  smemA[ty][tx]    = A[gmem_idx_row * K + gmem_idx_col];
}

__device__ void load_smem_tile_to_reg(const float smem[TILE_SIZE][TILE_SIZE], int row, int col, float reg[4]) {
  // load data to reg from shared memory
  reg[0] = smem[row][col];
  reg[1] = smem[row + TILE_SIZE][col];
  reg[2] = smem[row + 2 * TILE_SIZE][col];
  reg[3] = smem[row + 3 * TILE_SIZE][col];
}


__device__ void mma4x4(const float a[4], const float b[4], float c[4][4]) {
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) { c[i][j] += a[i] * b[j]; }
  }
}



__global__ void gemm(float* A, float* B, float* C, int M, int N, int K) {
  __shared__ float smemA[TILE_SIZE][TILE_SIZE];
  __shared__ float smemB[TILE_SIZE][TILE_SIZE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int row = by * TILE_SIZE + ty;
  int col = bx * TILE_SIZE + tx;

  float c_local[4][4] = {{0.0f}};

  for (int i = 0; i < K; i += TILE_K) {
    __syncthreads();
    load_gmem_tile_to_smem(A, i, row, col, smemA);
    load_gmem_tile_to_smem(B, i, row, col, smemB);
    __syncthreads();

#pragma unroll
    for (int j = 0; j < TILE_K; ++j) {
      float a_reg[4], b_reg[4];
      load_smem_tile_to_reg(smemA, ty, j, a_reg);
      load_smem_tile_to_reg(smemB, j, tx, b_reg);

      mma4x4(a_reg, b_reg, c_local);
    }
  }

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      if (row + i < M && col + j < N) { atomicAdd(&C[(row + i) * N + col + j], c_local[i][j]); }
    }
  }
}


}   // namespace cuda_kernel

void luanch_gmm(float* input_a, float* input_b, float* output, int N, int M, int P) {


  dim3 grid_dim((N + TILE_SIZE - 1) / TILE_SIZE, (P + TILE_SIZE - 1) / TILE_SIZE);
  dim3 block_dim(TILE_SIZE, TILE_SIZE);

  cuda_kernel::gemm<<<grid_dim, block_dim>>>(input_a, input_b, output, N, M, P);

  return;
}