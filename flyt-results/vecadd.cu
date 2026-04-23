#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vecAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    int n = 1024;
    size_t bytes = n * sizeof(float);
    
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    
    for (int i = 0; i < n; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }
    
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    printf("d_a=%p d_b=%p d_c=%p\n", d_a, d_b, d_c);
    
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    
    vecAdd<<<(n+255)/256, 256>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    
    int errors = 0;
    for (int i = 0; i < n; i++) {
        float expected = (float)i + (float)(i * 2);
        if (h_c[i] != expected) {
            if (errors < 5) printf("MISMATCH [%d]: got %f expected %f\n", i, h_c[i], expected);
            errors++;
        }
    }
    
    if (errors == 0) printf("SUCCESS: All %d elements correct!\n", n);
    else printf("FAILED: %d/%d mismatches\n", errors, n);
    
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return errors > 0 ? 1 : 0;
}
