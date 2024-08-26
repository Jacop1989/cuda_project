#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define WIDTH  800  // ความกว้างของภาพ
#define HEIGHT 600  // ความสูงของภาพ

// โครงสร้างสำหรับสี
struct Color {
    float r, g, b;
};

// โครงสร้างสำหรับเวกเตอร์ 3 มิติ
struct Vec3 {
    float x, y, z;

    __device__ Vec3 operator+(const Vec3& v) const {
        return Vec3{x + v.x, y + v.y, z + v.z};
    }

    __device__ Vec3 operator-(const Vec3& v) const {
        return Vec3{x - v.x, y - v.y, z - v.z};
    }

    __device__ Vec3 operator*(float t) const {
        return Vec3{x * t, y * t, z * t};
    }

    __device__ float dot(const Vec3& v) const {
        return x * v.x + y * v.y + z * v.z;
    }

    __device__ Vec3 normalize() const {
        float length = sqrtf(x * x + y * y + z * z);
        return Vec3{x / length, y / length, z / length};
    }
};

// Function สำหรับตรวจสอบการชนระหว่าง Ray กับทรงกลม
__device__ bool hit_sphere(const Vec3& sphere_center, float sphere_radius, const Vec3& ray_origin, const Vec3& ray_direction) {
    Vec3 oc = ray_origin - sphere_center;
    float a = ray_direction.dot(ray_direction);
    float b = 2.0f * oc.dot(ray_direction);
    float c = oc.dot(oc) - sphere_radius * sphere_radius;
    float discriminant = b * b - 4 * a * c;
    return (discriminant > 0);
}

// Function สำหรับ trace ray ใน scene ที่มีหลายทรงกลม
__device__ Color trace_ray(const Vec3& ray_origin, const Vec3& ray_direction) {
    Vec3 sphere_center1 = {0.0f, 0.0f, -1.0f};  // ทรงกลมแรก
    Vec3 sphere_center2 = {0.5f, 0.0f, -1.5f};  // ทรงกลมที่สอง
    float sphere_radius1 = 0.5f;
    float sphere_radius2 = 0.3f;

    // ตรวจสอบการชนระหว่าง Ray กับทรงกลมแรก
    if (hit_sphere(sphere_center1, sphere_radius1, ray_origin, ray_direction)) {
        return Color{1.0f, 0.0f, 0.0f};  // สีแดงสำหรับทรงกลมแรก
    }
    
    // ตรวจสอบการชนระหว่าง Ray กับทรงกลมที่สอง
    if (hit_sphere(sphere_center2, sphere_radius2, ray_origin, ray_direction)) {
        return Color{0.0f, 1.0f, 0.0f};  // สีเขียวสำหรับทรงกลมที่สอง
    }

    // พื้นหลังถ้าไม่มีการชน
    return Color{0.5f, 0.7f, 1.0f};  // พื้นหลังสีฟ้า
}

// CUDA kernel สำหรับ Ray Tracing พร้อมวัตถุหลายชิ้น
__global__ void ray_tracing(Color *image) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < WIDTH && y < HEIGHT) {
        int idx = y * WIDTH + x;

        // คำนวณพิกัดของรังสี (Ray) ออกจากกล้อง
        float u = float(x) / WIDTH;
        float v = float(y) / HEIGHT;
        Vec3 ray_origin = {0.0f, 0.0f, 0.0f};  // กล้องอยู่ที่ (0, 0, 0)
        Vec3 ray_direction = {2.0f * u - 1.0f, 2.0f * v - 1.0f, -1.0f};  // รังสีออกจากกล้อง
        ray_direction = ray_direction.normalize();

        // คำนวณสีของพิกเซล
        image[idx] = trace_ray(ray_origin, ray_direction);
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

    printf("GPU Time for Ray Tracing with Multiple Objects: %f ms\n", milliseconds);

    // คัดลอกผลลัพธ์กลับมายัง Host
    cudaMemcpy(h_image, d_image, size, cudaMemcpyDeviceToHost);

    // เขียนผลลัพธ์ภาพลงไฟล์ (PPM format)
    FILE *f = fopen("ray_tracing_multi_object.ppm", "w");
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
