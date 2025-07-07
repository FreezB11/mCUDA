#include <cstring>
#include <cstdlib>
#include "mCUDA.h"

extern void gpuMatrixMul(const float* A, const float* B, float* C, int n);

void addMatrix(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n * n; ++i)
        C[i] = A[i] + B[i];
}

void subMatrix(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n * n; ++i)
        C[i] = A[i] - B[i];
}

void strassenMultiply(const float* A, const float* B, float* C, int n) {
    if (n <= 64) {  // base case uses CUDA
        gpuMatrixMul(A, B, C, n);
        return;
    }

    int newSize = n / 2;
    int sz = newSize * newSize;

    const float *A11 = A, *A12 = A + newSize;
    const float *A21 = A + newSize * n, *A22 = A + newSize * n + newSize;
    const float *B11 = B, *B12 = B + newSize;
    const float *B21 = B + newSize * n, *B22 = B + newSize * n + newSize;

    float *M1 = new float[sz], *M2 = new float[sz], *M3 = new float[sz];
    float *M4 = new float[sz], *M5 = new float[sz], *M6 = new float[sz], *M7 = new float[sz];
    float *Atemp = new float[sz], *Btemp = new float[sz];

    addMatrix(A11, A22, Atemp, newSize);
    addMatrix(B11, B22, Btemp, newSize);
    strassenMultiply(Atemp, Btemp, M1, newSize);

    addMatrix(A21, A22, Atemp, newSize);
    strassenMultiply(Atemp, B11, M2, newSize);

    subMatrix(B12, B22, Btemp, newSize);
    strassenMultiply(A11, Btemp, M3, newSize);

    subMatrix(B21, B11, Btemp, newSize);
    strassenMultiply(A22, Btemp, M4, newSize);

    addMatrix(A11, A12, Atemp, newSize);
    strassenMultiply(Atemp, B22, M5, newSize);

    subMatrix(A21, A11, Atemp, newSize);
    addMatrix(B11, B12, Btemp, newSize);
    strassenMultiply(Atemp, Btemp, M6, newSize);

    subMatrix(A12, A22, Atemp, newSize);
    addMatrix(B21, B22, Btemp, newSize);
    strassenMultiply(Atemp, Btemp, M7, newSize);

    float *C11 = new float[sz], *C12 = new float[sz], *C21 = new float[sz], *C22 = new float[sz];

    for (int i = 0; i < sz; ++i) {
        C11[i] = M1[i] + M4[i] - M5[i] + M7[i];
        C12[i] = M3[i] + M5[i];
        C21[i] = M2[i] + M4[i];
        C22[i] = M1[i] - M2[i] + M3[i] + M6[i];
    }

    for (int i = 0; i < newSize; ++i) {
        std::memcpy(&C[i * n], &C11[i * newSize], sizeof(float) * newSize);
        std::memcpy(&C[i * n + newSize], &C12[i * newSize], sizeof(float) * newSize);
        std::memcpy(&C[(i + newSize) * n], &C21[i * newSize], sizeof(float) * newSize);
        std::memcpy(&C[(i + newSize) * n + newSize], &C22[i * newSize], sizeof(float) * newSize);
    }

    delete[] M1; delete[] M2; delete[] M3; delete[] M4;
    delete[] M5; delete[] M6; delete[] M7;
    delete[] Atemp; delete[] Btemp;
    delete[] C11; delete[] C12; delete[] C21; delete[] C22;
}
