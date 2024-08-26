#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 100000000  // เพิ่มขนาดเวกเตอร์

void vector_add(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    float *a = (float *)malloc(N * sizeof(float));
    float *b = (float *)malloc(N * sizeof(float));
    float *c = (float *)malloc(N * sizeof(float));

    // กำหนดค่าเริ่มต้น
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    // จับเวลา
    clock_t start, end;
    double cpu_time_used;
    start = clock();

    // การบวกเวกเตอร์
    vector_add(a, b, c, N);

    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("CPU Time: %f seconds\n", cpu_time_used);

    // ลบหน่วยความจำ
    free(a);
    free(b);
    free(c);

    return 0;
}
