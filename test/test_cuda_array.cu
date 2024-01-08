
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "tools/ndarra_wrap.hpp"

#include <iostream>
__global__ void check_array_kernal() {
  int a[32 * 32];
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  tools::NdarrayWarp<int, 32, 32> array_warp(a);

  array_warp(ty, tx) = tx + ty;

  printf("ty:%d,tx:%d,index:%d,value:%d\n", ty, tx, array_warp.index(ty, tx), array_warp(ty, tx));
}

int main() {
  dim3 grid_dim(1, 1);
  dim3 block_dim(32, 32);

  check_array_kernal<<<grid_dim, block_dim>>>();
  cudaDeviceSynchronize();

  return 0;
}