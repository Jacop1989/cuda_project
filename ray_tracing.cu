#include <stdio.h>
#include <cuda_runtime.h>

#define WIDTH  800  // ความกว้างของภาพ
#define HEIGHT 600  // ความสูงของภาพ

// โครงสร้างสำหรับสี
struct Color {
    float r, g, b;
};

// CUDA kernel สำหรับการทำ Ray Tracing เบื้องต้น
__global__ void ray_tracing(Color *image) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < WIDTH && y < HEIGHT) {
        int idx = y * WIDTH + x;

        // แปลงพิกัดของภาพเป็นสีแบบง่าย (สำหรับทดสอบ)
        float r = float(x) / WIDTH;
        float g = float(y) / HEIGHT;
        float b = 0.2f;
        
        image[idx].r = r;
        image[idx].g = g;
        image[idx].b = b;
    }
}

int main() {
    size_t size = WIDTH * HEIGHT * sizeof(Color);

    // จองหน่วยความจำบน Host สำหรับภาพ
    Color *h_image = (Color *)malloc(size);

    // จองหน่วยความจำบน Device สำหรับภาพ
    Color *d_image;
    cudaMalloc(&d_image, size);

    // กำหนดจำนวนบล็อกและเทรด
    dim3 threads_per_block(16, 16);
    dim3 number_of_blocks((WIDTH + threads_per_block.x - 1) / threads_per_block.x, 
                          (HEIGHT + threads_per_block.y - 1) / threads_per_block.y);

    // จับเวลาบน GPU
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    ray_tracing<<<number_of_blocks, threads_per_block>>>(d_image);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("GPU Time for Ray Tracing: %f ms\n", milliseconds);

    // คัดลอกผลลัพธ์กลับมายัง Host
    cudaMemcpy(h_image, d_image, size, cudaMemcpyDeviceToHost);

    // เขียนผลลัพธ์ภาพลงไฟล์ (PPM format)
    FILE *f = fopen("ray_tracing.ppm", "w");
    fprintf(f, "P3\n%d %d\n255\n", WIDTH, HEIGHT);
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        int r = int(255.99 * h_image[i].r);
        int g = int(255.99 * h_image[i].g);
        int b = int(255.99 * h_image[i].b);
        fprintf(f, "%d %d %d\n", r, g, b);
    }
    fclose(f);

    // ลบหน่วยความจำ
    free(h_image);
    cudaFree(d_image);

    return 0;
}
