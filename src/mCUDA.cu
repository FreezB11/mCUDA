#include <cuda_runtime.h>
#include "mCUDA.h"

__global__ void addKernel(float* a, float* b, float* result, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        result[idx] = a[idx] + b[idx];
    }
}

void launchAddKernel(float* a, float* b, float* result, int n) {
    float *d_a, *d_b, *d_result;

    size_t size = n * sizeof(float);
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_result, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    addKernel<<<blocks, threads>>>(d_a, d_b, d_result, n);

    cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}
