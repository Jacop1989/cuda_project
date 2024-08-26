#include <stdio.h>
#include <math.h>

typedef struct {
    float x, y, z, w;
} Vec3D;

typedef struct {
    Vec3D vertices[8]; // Cube มี 8 vertices
} Cube;

typedef struct {
    int start, end;
} Edge;

Edge cube_edges[12] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 0}, // ฐานล่าง
    {4, 5}, {5, 6}, {6, 7}, {7, 4}, // ฐานบน
    {0, 4}, {1, 5}, {2, 6}, {3, 7}  // เชื่อมระหว่างฐานล่างและฐานบน
};

typedef struct {
    float m[4][4];
} Matrix4x4;

// ฟังก์ชันสำหรับสร้าง Cube
Cube create_cube(float size) {
    Cube cube;
    float half_size = size / 2.0f;
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

// ฟังก์ชันคูณ Matrix กับ Vector
Vec3D multiply_matrix_vector(const Matrix4x4 *mat, const Vec3D *vec) {
    Vec3D result;
    result.x = mat->m[0][0] * vec->x + mat->m[1][0] * vec->y + mat->m[2][0] * vec->z + mat->m[3][0] * vec->w;
    result.y = mat->m[0][1] * vec->x + mat->m[1][1] * vec->y + mat->m[2][1] * vec->z + mat->m[3][1] * vec->w;
    result.z = mat->m[0][2] * vec->x + mat->m[1][2] * vec->y + mat->m[2][2] * vec->z + mat->m[3][2] * vec->w;
    result.w = mat->m[0][3] * vec->x + mat->m[1][3] * vec->y + mat->m[2][3] * vec->z + mat->m[3][3] * vec->w;
    return result;
}

// ฟังก์ชันการแปลง (Transformation)
void transform_cube(Cube *cube, const Matrix4x4 *transformation) {
    for (int i = 0; i < 8; i++) {
        cube->vertices[i] = multiply_matrix_vector(transformation, &cube->vertices[i]);
    }
}

// ฟังก์ชันแสดงเส้นโครงของ Cube
void print_wireframe(Cube *cube) {
    for (int i = 0; i < 12; i++) {
        int start_idx = cube_edges[i].start;
        int end_idx = cube_edges[i].end;
        Vec3D start = cube->vertices[start_idx];
        Vec3D end = cube->vertices[end_idx];
        printf("Edge from (%f, %f, %f) to (%f, %f, %f)\n", 
            start.x, start.y, start.z, 
            end.x, end.y, end.z);
    }
}

// ฟังก์ชันสร้าง Identity Matrix
Matrix4x4 identity_matrix() {
    Matrix4x4 mat = {{{0}}};
    for (int i = 0; i < 4; i++) {
        mat.m[i][i] = 1.0f;
    }
    return mat;
}

// ฟังก์ชันการเลื่อนวัตถุ (Translation)
Matrix4x4 translation_matrix(float tx, float ty, float tz) {
    Matrix4x4 mat = identity_matrix();
    mat.m[3][0] = tx;
    mat.m[3][1] = ty;
    mat.m[3][2] = tz;
    return mat;
}

// ฟังก์ชันการหมุนรอบแกน Y (Rotation)
Matrix4x4 rotation_matrix_y(float angle) {
    Matrix4x4 mat = identity_matrix();
    float rad = angle * M_PI / 180.0f;
    mat.m[0][0] = cosf(rad);
    mat.m[0][2] = sinf(rad);
    mat.m[2][0] = -sinf(rad);
    mat.m[2][2] = cosf(rad);
    return mat;
}

int main() {
    Cube cube = create_cube(2.0f); // สร้าง Cube ขนาด 2 หน่วย
    Matrix4x4 trans = translation_matrix(0.0f, 0.0f, 0.0f); // เลื่อนลูกบาศก์
    Matrix4x4 rotY = rotation_matrix_y(45.0f); // หมุนลูกบาศก์รอบแกน Y

    // ทำการแปลงลูกบาศก์ด้วย Translation และ Rotation
    transform_cube(&cube, &trans);
    transform_cube(&cube, &rotY);

    // แสดงเส้นโครงของลูกบาศก์
    print_wireframe(&cube);

    return 0;
}
