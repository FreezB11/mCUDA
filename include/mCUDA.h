#ifndef MCUDA_H
#define MCUDA_H

#ifdef __cplusplus
extern "C" {
#endif

void launchAddKernel(float* a, float* b, float* result, int n);

#ifdef __cplusplus
}
#endif

#endif // MCUDA_H
