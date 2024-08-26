#include <stdio.h>

__global__ void hello_from_gpu() {
    printf("Hello World from GPU!\n");
}

int main() {
    hello_from_gpu<<<1, 1>>>();
    cudaDeviceSynchronize();  // รอให้ GPU ทำงานเสร็จ
    return 0;
}
