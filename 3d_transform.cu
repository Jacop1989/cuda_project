#include <stdio.h>
#include <cuda_runtime.h>

#define N 100000  // จำนวนจุดใน 3D space

// CUDA kernel สำหรับ 3D transformations (Rotation และ Translation)
__global__ void transform_points(float *x, float *y, float *z, int n, float theta, float tx, float ty, float tz) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        // การหมุนรอบแกน z-axis (2D rotation)
        float new_x = x[idx] * cos(theta) - y[idx] * sin(theta);
        float new_y = x[idx] * sin(theta) + y[idx] * cos(theta);

        // การย้ายตำแหน่ง (Translation)
        x[idx] = new_x + tx;
        y[idx] = new_y + ty;
        z[idx] = z[idx] + tz;
    }
}

int main() {
    size_t size = N * sizeof(float);

    // จองหน่วยความจำบน Host สำหรับพิกัด (x, y, z)
    float *h_x = (float *)malloc(size);
    float *h_y = (float *)malloc(size);
    float *h_z = (float *)malloc(size);

    // กำหนดค่าเริ่มต้นในพิกัด
    for (int i = 0; i < N; i++) {
        h_x[i] = i * 0.1f;
        h_y[i] = i * 0.1f;
        h_z[i] = i * 0.1f;
    }

    // จองหน่วยความจำบน Device สำหรับพิกัด (x, y, z)
    float *d_x, *d_y, *d_z;
    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);
    cudaMalloc(&d_z, size);

    // คัดลอกข้อมูลจาก Host ไปยัง Device
    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z, size, cudaMemcpyHostToDevice);

    // กำหนดพารามิเตอร์การหมุนและการย้ายตำแหน่ง
    float theta = 3.14159 / 4;  // หมุน 45 องศา
    float tx = 1.0f, ty = 2.0f, tz = 3.0f;  // การย้ายตำแหน่ง

    // กำหนดจำนวนบล็อกและเทรด
    int threads_per_block = 256;
    int number_of_blocks = (N + threads_per_block - 1) / threads_per_block;

    // จับเวลาบน GPU
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    transform_points<<<number_of_blocks, threads_per_block>>>(d_x, d_y, d_z, N, theta, tx, ty, tz);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("GPU Time for 3D Transformations: %f ms\n", milliseconds);

    // คัดลอกผลลัพธ์กลับมายัง Host
    cudaMemcpy(h_x, d_x, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_z, d_z, size, cudaMemcpyDeviceToHost);

    // แสดงพิกัดบางส่วน
    for (int i = 0; i < 5; i++) {
        printf("Point %d -> (x: %f, y: %f, z: %f)\n", i, h_x[i], h_y[i], h_z[i]);
    }

    // ลบหน่วยความจำ
    free(h_x);
    free(h_y);
    free(h_z);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    return 0;
}
