// gcc transformation_3d_points.c -o transformation_3d_points -lm
// ./transformation_3d_points

#include <stdio.h>
#include <math.h>



typedef struct {
    float m[4][4];
} Matrix4x4;

typedef struct {
    float x, y, z, w; // w ใช้สำหรับการแปลงแบบ Homogeneous Coordinates
} Vec3D;


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

Vec3D multiply_matrix_vector(const Matrix4x4 *mat, const Vec3D *vec) {
    Vec3D result;
    result.x = mat->m[0][0] * vec->x + mat->m[1][0] * vec->y + mat->m[2][0] * vec->z + mat->m[3][0] * vec->w;
    result.y = mat->m[0][1] * vec->x + mat->m[1][1] * vec->y + mat->m[2][1] * vec->z + mat->m[3][1] * vec->w;
    result.z = mat->m[0][2] * vec->x + mat->m[1][2] * vec->y + mat->m[2][2] * vec->z + mat->m[3][2] * vec->w;
    result.w = mat->m[0][3] * vec->x + mat->m[1][3] * vec->y + mat->m[2][3] * vec->z + mat->m[3][3] * vec->w;
    return result;
}


int main() {
    Vec3D point = {1.0f, 1.0f, 1.0f, 1.0f}; // จุดที่ต้องการแปลง

    Matrix4x4 trans = translation_matrix(2.0f, 3.0f, 4.0f);
    Matrix4x4 scale = scaling_matrix(1.5f, 1.5f, 1.5f);
    Matrix4x4 rotX = rotation_matrix_x(45.0f);
    
    // แปลงจุดด้วย Translation
    Vec3D translated_point = multiply_matrix_vector(&trans, &point);
    printf("Translated Point: x = %f, y = %f, z = %f\n", translated_point.x, translated_point.y, translated_point.z);

    // แปลงจุดด้วย Scaling
    Vec3D scaled_point = multiply_matrix_vector(&scale, &point);
    printf("Scaled Point: x = %f, y = %f, z = %f\n", scaled_point.x, scaled_point.y, scaled_point.z);

    // แปลงจุดด้วย Rotation ในแกน X
    Vec3D rotated_point = multiply_matrix_vector(&rotX, &point);
    printf("Rotated Point (X-axis): x = %f, y = %f, z = %f\n", rotated_point.x, rotated_point.y, rotated_point.z);

    return 0;
}

