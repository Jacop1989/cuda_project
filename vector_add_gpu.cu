#include <stdio.h>
#include <cuda_runtime.h>

#define N 100000000  // เพิ่มขนาดเวกเตอร์

// CUDA kernel สำหรับการบวกเวกเตอร์
__global__ void vector_add(float *a, float *b, float *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    float *h_a = (float *)malloc(N * sizeof(float));
    float *h_b = (float *)malloc(N * sizeof(float));
    float *h_c = (float *)malloc(N * sizeof(float));

    // กำหนดค่าเริ่มต้น
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // จองหน่วยความจำบน Device
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    // คัดลอกข้อมูลจาก Host ไปยัง Device
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    // จับเวลาบน GPU
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threads_per_block = 256;
    int number_of_blocks = (N + threads_per_block - 1) / threads_per_block;

    cudaEventRecord(start);
    vector_add<<<number_of_blocks, threads_per_block>>>(d_a, d_b, d_c, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("GPU Time: %f ms\n", milliseconds);

    // คัดลอกผลลัพธ์กลับมายัง Host
    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    // ลบหน่วยความจำ
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
