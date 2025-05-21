#pragma once 
// this should be the used filed for any funxtion
#include <cuda_runtime.h>

#define bli blockIdx
#define bld blockDim
#define thi threadIdx

// Vector
__global__ void _vak(const float* A,const float* B,float* C, int N); // vector add kernel
void vectorAdd(const float* d_A, const float* d_B, float* d_C, int N);