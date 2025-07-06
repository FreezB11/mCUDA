#include <iostream>
#include "mCUDA.h"

int main() {
    const int n = 10;
    float a[n], b[n], result[n];

    for (int i = 0; i < n; ++i) {
        a[i] = i;
        b[i] = 2 * i;
    }

    launchAddKernel(a, b, result, n);

    for (int i = 0; i < n; ++i)
        std::cout << result[i] << " ";
    std::cout << std::endl;

    return 0;
}
