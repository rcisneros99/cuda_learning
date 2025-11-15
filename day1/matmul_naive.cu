extern "C"
#include <cuda_runtime.h>
//kernel 
__global__
void matmul_naive(const float* A, const float* B, float* C,
                  int M, int N, int K) {

    // we'll compute the threads 
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Guard against threads outside valid region
    if (row < M && col < K) {

        float sum = 0.0f;
        //loop
        for (int p = 0; p < N; ++p) {
            sum += A[row * N + p] * B[p * K + col];
        }
        //write it to output C
        C[row * K + col] = sum;
    }
}

void solve(const float* A, const float* B, float* C,
            int M, int N, int K) {

    // threads
    dim3 threadsPerBlock(16, 16);

    // compute
    dim3 numBlocks((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // launch the kernel
    matmul_naive<<<numBlocks, threadsPerBlock>>>(A, B, C, M, N, K);

    // Wait for completion before returning to CPU
    cudaDeviceSynchronize();
}