#include "mCUDA.cuh"

/**
    @note VectorAdd for the starter
    first with kernel definition
*/

__global__ void _vak(const float* A,const float* B,float* C, int N){
    int i = bli.x * bld.x + thi.x;
    if(i < N){
        C[i] = A[i] + B[i];
    }
}

void vectorAdd(const float* d_A, const float* d_B,float* d_C, int N){
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1)/threadsPerBlock;
    _vak<<<blocksPerGrid, threadsPerBlock>>>(d_A,d_B,d_C, N);
    cudaDeviceSynchronize();
}