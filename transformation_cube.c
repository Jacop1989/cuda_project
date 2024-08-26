// gcc transformation_cube.c -o transformation_cube -lm
// ./transformation_cube

#include <stdio.h>
#include <math.h>



typedef struct {
    float m[4][4];
} Matrix4x4;

typedef struct {
    float x, y, z, w; // w ใช้สำหรับการแปลงแบบ Homogeneous Coordinates
} Vec3D;

typedef struct {
    Vec3D vertices[8]; // Cube มี 8 vertices (จุด)
} Cube;

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

Cube create_cube(float size) {
    Cube cube;
    float half_size = size / 2.0f;

    // กำหนดตำแหน่งของ 8 vertices ของลูกบาศก์
    cube.vertices[0] = (Vec3D){-half_size, -half_size, -half_size, 1.0f};
    cube.vertices[1] = (Vec3D){ half_size, -half_size, -half_size, 1.0f};
    cube.vertices[2] = (Vec3D){ half_size,  half_size, -half_size, 1.0f};
    cube.vertices[3] = (Vec3D){-half_size,  half_size, -half_size, 1.0f};
    cube.vertices[4] = (Vec3D){-half_size, -half_size,  half_size, 1.0f};
    cube.vertices[5] = (Vec3D){ half_size, -half_size,  half_size, 1.0f};
    cube.vertices[6] = (Vec3D){ half_size,  half_size,  half_size, 1.0f};
    cube.vertices[7] = (Vec3D){-half_size,  half_size,  half_size, 1.0f};

    return cube;
}

void transform_cube(Cube *cube, const Matrix4x4 *transformation) {
    for (int i = 0; i < 8; i++) {
        cube->vertices[i] = multiply_matrix_vector(transformation, &cube->vertices[i]);
    }
}

int main() {
    Cube cube = create_cube(2.0f); // สร้าง Cube ขนาด 2 หน่วย
    Matrix4x4 scale = scaling_matrix(1.5f, 1.5f, 1.5f); // ขยายขนาดลูกบาศก์
    Matrix4x4 rotY = rotation_matrix_y(45.0f); // หมุนลูกบาศก์รอบแกน Y

    // ทำการแปลงลูกบาศก์ด้วย Scaling
    transform_cube(&cube, &scale);
    
    // ทำการแปลงลูกบาศก์ด้วย Rotation
    transform_cube(&cube, &rotY);

    // พิมพ์ค่าของ vertices ของลูกบาศก์ที่ถูกแปลงแล้ว
    for (int i = 0; i < 8; i++) {
        printf("Vertex %d: x = %f, y = %f, z = %f\n", i, cube.vertices[i].x, cube.vertices[i].y, cube.vertices[i].z);
    }

    return 0;
}

