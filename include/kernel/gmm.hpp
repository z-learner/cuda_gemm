#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

#include <assert.h>
#include <cstddef>
#include <cstdint>



void luanch_gmm(float* input_a, float* input_b, float* output, int N, int M, int P);