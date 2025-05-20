
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N]){
    int i = threadIdx.x;
    int j = threadIdx.y;
    C[i][j] = A[i][j] + B[i][j];
}

int main(){
    //kernel invoked with N*N*1 threads
    int numBlocks = 1;
    dim3 threadsPerBlock(N,N);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A,B,C);
}
//////////////////////////////////////////////////////
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N]){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
    C[i][j] = A[i][j] + B[i][j];
}

int main(){
    int N = 512;
    float *A, *B, *C;
    ∕∕ Kernel invocation
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N ∕ threadsPerBlock.x, N ∕ threadsPerBlock.y);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
}