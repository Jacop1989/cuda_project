#include <stdio.h>
#include <cuda_runtime.h>

#define N 1000  // ขนาดของเมทริกซ์

// CUDA kernel สำหรับการคูณเมทริกซ์
__global__ void matrix_mul(float *a, float *b, float *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0;
        for (int i = 0; i < n; i++) {
            sum += a[row * n + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

int main() {
    size_t size = N * N * sizeof(float);

    // จองหน่วยความจำบน Host
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);

    // กำหนดค่าเริ่มต้นในเมทริกซ์
    for (int i = 0; i < N * N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // จองหน่วยความจำบน Device
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // คัดลอกข้อมูลจาก Host ไปยัง Device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // กำหนดจำนวนบล็อกและเทรด (จาก 16x16 เป็น 32x32)
    dim3 threads_per_block(32, 32);
    dim3 number_of_blocks((N + threads_per_block.x - 1) / threads_per_block.x, 
                          (N + threads_per_block.y - 1) / threads_per_block.y);

    // จับเวลาบน GPU
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrix_mul<<<number_of_blocks, threads_per_block>>>(d_a, d_b, d_c, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("GPU Time for Matrix Multiplication with 32x32 blocks: %f ms\n", milliseconds);

    // คัดลอกผลลัพธ์กลับมายัง Host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // ลบหน่วยความจำ
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
