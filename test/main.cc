#include <iostream>
#include <cstdlib>   // for rand()
#include <ctime>     // for time()
#include <chrono>
#include "mCUDA.h"

void printMatrix(const float* M, int n) {
    for (int i = 0; i < n * n; ++i) {
        std::cout << M[i] << " ";
        if ((i + 1) % n == 0) std::cout << "\n";
    }
    std::cout << "\n";
}

int main() {
    const int N = 1024; // Must be power of 2 for Strassen
    float* A = new float[N * N];
    float* B = new float[N * N];
    float* C = new float[N * N];

    std::srand(static_cast<unsigned int>(std::time(0))); // seed RNG

    for (int i = 0; i < N * N; ++i) {
        A[i] = static_cast<float>(std::rand() % 100); // random values [0, 99]
        B[i] = static_cast<float>(std::rand() % 100);
    }

    auto start = std::chrono::high_resolution_clock::now();
    strassenMultiply(A, B, C, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "Strassen multiplication for " << N << "x" << N << " took ";
    std::cout << duration.count() << " ms\n";

    // Optional for small N
    // printMatrix(C, N);

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
