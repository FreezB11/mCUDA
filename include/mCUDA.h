/**
    @file: mCUDA.h
    @author: HSAY
*/

// #include <cuda_runtime.h>

#ifdef __cplusplus
    extern "C" {
#endif

void strassenMultiply(const float* A, const float* B, float* C, int n);

#ifdef __cplusplus
}
#endif