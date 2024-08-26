#include <stdio.h>
#include <math.h>

typedef struct {
    float m[4][4];
} Matrix4x4;

// ฟังก์ชันสำหรับสร้าง Identity Matrix
Matrix4x4 identity_matrix() {
    Matrix4x4 mat = {{{0}}};
    for (int i = 0; i < 4; i++) {
        mat.m[i][i] = 1.0f;
    }
    return mat;
}

// ฟังก์ชันสำหรับการพิมพ์ Matrix ออกมาให้เห็นค่า
void print_matrix(const Matrix4x4 *mat) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%f ", mat->m[i][j]);
        }
        printf("\n");
    }
}

// ฟังก์ชัน Translation (การเลื่อนวัตถุ)
Matrix4x4 translation_matrix(float tx, float ty, float tz) {
    Matrix4x4 mat = identity_matrix();
    mat.m[3][0] = tx;
    mat.m[3][1] = ty;
    mat.m[3][2] = tz;
    return mat;
}

// ฟังก์ชัน Scaling (การย่อ-ขยายวัตถุ)
Matrix4x4 scaling_matrix(float sx, float sy, float sz) {
    Matrix4x4 mat = identity_matrix();
    mat.m[0][0] = sx;
    mat.m[1][1] = sy;
    mat.m[2][2] = sz;
    return mat;
}

// ฟังก์ชัน Rotation ในแกน X
Matrix4x4 rotation_matrix_x(float angle) {
    Matrix4x4 mat = identity_matrix();
    float rad = angle * M_PI / 180.0f;
    mat.m[1][1] = cosf(rad);
    mat.m[1][2] = -sinf(rad);
    mat.m[2][1] = sinf(rad);
    mat.m[2][2] = cosf(rad);
    return mat;
}

// ฟังก์ชัน Rotation ในแกน Y
Matrix4x4 rotation_matrix_y(float angle) {
    Matrix4x4 mat = identity_matrix();
    float rad = angle * M_PI / 180.0f;
    mat.m[0][0] = cosf(rad);
    mat.m[0][2] = sinf(rad);
    mat.m[2][0] = -sinf(rad);
    mat.m[2][2] = cosf(rad);
    return mat;
}

// ฟังก์ชัน Rotation ในแกน Z
Matrix4x4 rotation_matrix_z(float angle) {
    Matrix4x4 mat = identity_matrix();
    float rad = angle * M_PI / 180.0f;
    mat.m[0][0] = cosf(rad);
    mat.m[0][1] = -sinf(rad);
    mat.m[1][0] = sinf(rad);
    mat.m[1][1] = cosf(rad);
    return mat;
}

int main() {
    Matrix4x4 trans = translation_matrix(1.0f, 2.0f, 3.0f);
    Matrix4x4 scale = scaling_matrix(2.0f, 2.0f, 2.0f);
    Matrix4x4 rotX = rotation_matrix_x(45.0f);
    
    printf("Translation Matrix:\n");
    print_matrix(&trans);
    
    printf("Scaling Matrix:\n");
    print_matrix(&scale);
    
    printf("Rotation Matrix X (45 degrees):\n");
    print_matrix(&rotX);
    
    return 0;
}
