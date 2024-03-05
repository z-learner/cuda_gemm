#include "cuda/cuda_helper.hpp"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "cxxopts/cxxopts.hpp"
#include "kernel/gmm.hpp"

#include <iostream>
#include <random>
#include <vector>

int main(int argc, char** argv) {
  cxxopts::Options options("MyProgram", "One line description of MyProgram");
  // clang-format off
  options.add_options()
  ("n,ND", "Int param A Row", cxxopts::value<int>()->default_value("1024"))
  ("m,MD", "Int param A Col", cxxopts::value<int>()->default_value("1024"))
  ("p,PD", "Int param B Col", cxxopts::value<int>()->default_value("1024"))
  ("i,iter", "run iter", cxxopts::value<int>()->default_value("100"))
  ("t,type", "run iter", cxxopts::value<std::string>()->default_value("normal"))
  ("c,check", "is check output")
  ("h,help", "Print usage")
  ;
  // clang-format on
  auto result = options.parse(argc, argv);

  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    exit(0);
  }

  int M = result["MD"].as<int>();
  int N = result["ND"].as<int>();
  int P = result["PD"].as<int>();


  int iter = result["iter"].as<int>();

  printf("N:%d,M:%d,P:%d,iter:%d\n", N, M, P, iter);

  std::vector<std::vector<float>> A(N, std::vector<float>(M));
  std::vector<std::vector<float>> B(M, std::vector<float>(P));

  std::random_device         rd;          // 随机数种子
  std::mt19937               gen(rd());   // 标准的mersenne_twister_engine
  std::normal_distribution<> dis(0, 1);

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) { A[i][j] = dis(gen); }
  }

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < P; ++j) { B[i][j] = dis(gen); }
  }

  float* dev_input_a = nullptr;
  float* dev_input_b = nullptr;
  float* dev_output  = nullptr;

  CUDA_CHECK(cudaMalloc((void**)&dev_input_a, N * M * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dev_input_b, M * P * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dev_output, N * P * sizeof(float)));

  for (int index = 0; index < N; ++index) { CUDA_CHECK(cudaMemcpy(dev_input_a + index * M, A[index].data(), M * sizeof(float), cudaMemcpyHostToDevice)); }

  for (int index = 0; index < M; ++index) { CUDA_CHECK(cudaMemcpy(dev_input_b + index * P, B[index].data(), P * sizeof(float), cudaMemcpyHostToDevice)); }


  float total_run_time_ms = 0;

  for (int index = 0; index < iter; ++index) {
    auto start = std::chrono::system_clock::now();
    luanch_gmm(dev_input_a, dev_input_b, dev_output, N, M, P);
    CUDA_CHECK_LAST_ERROR;
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::system_clock::now();
    total_run_time_ms += (end - start).count() / 1e6;
  }

  printf("avg run time : %fms in %d\n", total_run_time_ms / iter, iter);



  if (result.count("check")) {

    std::vector<std::vector<float>> C(N, std::vector<float>(P));

    for (int index = 0; index < N; ++index) { CUDA_CHECK(cudaMemcpy(C[index].data(), dev_output + index * P, P * sizeof(float), cudaMemcpyDeviceToHost)); }

    std::cout << "check 10 points" << std::endl;
    for (int index = 0; index < 10; ++index) {
      std::uniform_real_distribution<> index_dis(0, 1);   // 定义一个范围
      int                              row = index_dis(gen) * N;
      int                              col = index_dis(gen) * P;
      float                            sum = 0;
      for (int k = 0; k < M; ++k) { sum += A[row][k] * B[k][col]; }
      if (std::abs(sum - C[row][col]) < 1e-3) {
        printf("check (%d, %d) success, cpu value %f, gpu value %f\n", row, col, sum, C[row][col]);
      } else {
        printf("check (%d, %d)  failed, cpu value %f, gpu value %f\n", row, col, sum, C[row][col]);
      }
    }
  }

  return 0;
}