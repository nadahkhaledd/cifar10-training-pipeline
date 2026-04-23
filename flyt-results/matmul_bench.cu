#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

// Simple matrix multiply kernel
__global__ void matmul(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char **argv) {
    int N = 1024;  // Matrix size
    int iterations = 5;
    
    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) iterations = atoi(argv[2]);
    
    size_t bytes = N * N * sizeof(float);
    
    // Get GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s, SMs: %d\n", prop.name, prop.multiProcessorCount);
    printf("Matrix size: %dx%d, Iterations: %d\n", N, N, iterations);
    
    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    
    // Initialize matrices
    srand(42);
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)(rand() % 100) / 100.0f;
        h_B[i] = (float)(rand() % 100) / 100.0f;
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    
    // Copy to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);
    
    // Warmup
    matmul<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    
    // Verify correctness (check one element)
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    float expected = 0.0f;
    for (int k = 0; k < N; k++) expected += h_A[k] * h_B[k * N];
    float diff = h_C[0] - expected;
    if (diff < -0.01f || diff > 0.01f) {
        printf("VERIFICATION FAILED: got %f expected %f\n", h_C[0], expected);
    } else {
        printf("Verification passed (C[0,0] = %f)\n", h_C[0]);
    }
    
    // Benchmark
    double total_time = 0;
    for (int iter = 0; iter < iterations; iter++) {
        double start = get_time();
        matmul<<<grid, block>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();
        double elapsed = get_time() - start;
        total_time += elapsed;
        double gflops = (2.0 * N * N * N) / (elapsed * 1e9);
        printf("Iteration %d: %.4f sec (%.2f GFLOPS)\n", iter + 1, elapsed, gflops);
    }
    
    double avg_time = total_time / iterations;
    double avg_gflops = (2.0 * N * N * N) / (avg_time * 1e9);
    printf("\n=== SUMMARY ===\n");
    printf("Avg time: %.4f sec\n", avg_time);
    printf("Avg GFLOPS: %.2f\n", avg_gflops);
    printf("Total time: %.4f sec\n", total_time);
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
