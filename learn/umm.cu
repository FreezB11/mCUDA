// kernel definition
__global__ void VecAdd(float* A, float* B, float* C){
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main(){
    /// kernel  invocation with N threads
    int N = 512;
    float *A, *B, *C;
    VecAdd<<<1,N>>>(A,B,C);
}