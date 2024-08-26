#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel สำหรับการบวกเวกเตอร์
__global__ void vector_add(float *a, float *b, float *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n = 1000;  // ขนาดของเวกเตอร์
    size_t size = n * sizeof(float);

    // จองหน่วยความจำสำหรับเวกเตอร์บน Host (CPU)
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);

    // จองหน่วยความจำบน Device (GPU)
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // กำหนดค่าข้อมูลในเวกเตอร์ a และ b
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // คัดลอกข้อมูลจาก Host ไปยัง Device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // ระบุจำนวนบล็อกและเทรด
    int threads_per_block = 256;
    int number_of_blocks = (n + threads_per_block - 1) / threads_per_block;

    // เรียกใช้ CUDA kernel สำหรับการบวกเวกเตอร์
    vector_add<<<number_of_blocks, threads_per_block>>>(d_a, d_b, d_c, n);

    // รอให้ GPU ทำงานเสร็จ
    cudaDeviceSynchronize();

    // คัดลอกผลลัพธ์จาก Device กลับมายัง Host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // แสดงผลลัพธ์บางส่วน
    for (int i = 0; i < 10; i++) {
        printf("%f + %f = %f\n", h_a[i], h_b[i], h_c[i]);
    }

    // ลบหน่วยความจำบน Host และ Device
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
