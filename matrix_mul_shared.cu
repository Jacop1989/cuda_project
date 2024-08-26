#include <stdio.h>
#include <cuda_runtime.h>

#define N 1000  // ขนาดของเมทริกซ์
#define TILE_SIZE 32  // ขนาดของบล็อก

// CUDA kernel สำหรับการคูณเมทริกซ์โดยใช้ shared memory
__global__ void matrix_mul_shared(float *a, float *b, float *c, int n) {
    __shared__ float shared_a[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_b[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int tile = 0; tile < n / TILE_SIZE; ++tile) {
        shared_a[threadIdx.y][threadIdx.x] = a[row * n + (tile * TILE_SIZE + threadIdx.x)];
        shared_b[threadIdx.y][threadIdx.x] = b[(tile * TILE_SIZE + threadIdx.y) * n + col];

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += shared_a[threadIdx.y][k] * shared_b[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
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

    // กำหนดจำนวนบล็อกและเทรด
    dim3 threads_per_block(TILE_SIZE, TILE_SIZE);
    dim3 number_of_blocks((N + threads_per_block.x - 1) / threads_per_block.x, 
                          (N + threads_per_block.y - 1) / threads_per_block.y);

    // จับเวลาบน GPU
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrix_mul_shared<<<number_of_blocks, threads_per_block>>>(d_a, d_b, d_c, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("GPU Time for Matrix Multiplication with shared memory: %f ms\n", milliseconds);

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
