#include "cgrad_tensor.h"
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>

#define EPSILON 1e-5

void test_gemm_simple() {
    // 2x3 * 3x2 = 2x2
    cgrad_tensor_f32 a, b, c;
    uint32_t shapeA[] = {1, 1, 2, 3};
    uint32_t shapeB[] = {1, 1, 3, 2};
    float dataA[6] = {1, 2, 3, 4, 5, 6}; // row-major: [ [1,2,3], [4,5,6] ]
    float dataB[6] = {7, 8, 9, 10, 11, 12}; // row-major: [ [7,8], [9,10], [11,12] ]
    float expected[4] = {58, 64, 139, 154}; // [ [58,64], [139,154] ]

    cgrad_tensor_f32_init(&a, shapeA);
    cgrad_tensor_f32_init(&b, shapeB);

    // Copy data
    for (int i = 0; i < 6; i++) {
        a.data[i] = dataA[i];
        b.data[i] = dataB[i];
    }

    int err = cgrad_tensor_f32_gemm(&a, &b, &c);
    assert(err == 0);

    // Check result
    for (int i = 0; i < 4; i++) {
        if (fabsf(c.data[i] - expected[i]) > EPSILON) {
            printf("GEMM test failed: c.data[%d] = %f, expected %f\n", i, c.data[i], expected[i]);
            assert(0);
        }
    }
    printf("GEMM simple test passed.\n");

    cgrad_tensor_f32_free(&a);
    cgrad_tensor_f32_free(&b);
    cgrad_tensor_f32_free(&c);
}

void test_gemm_batched() {
    // Batch size 2, each batch: 2x2 * 2x2 = 2x2
    cgrad_tensor_f32 a, b, c;
    uint32_t shape[] = {2, 1, 2, 2}; // batch=2, 2x2 matrices

    // Batch 0: A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    // Batch 1: A = [[9,10],[11,12]], B = [[13,14],[15,16]]
    float dataA[8] = {
        1,2,3,4,    // batch 0
        9,10,11,12  // batch 1
    };
    float dataB[8] = {
        5,6,7,8,    // batch 0
        13,14,15,16 // batch 1
    };
    float expected[8] = {
        // batch 0 result: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        19, 22, 43, 50,
        // batch 1 result: [[9*13+10*15, 9*14+10*16], [11*13+12*15, 11*14+12*16]]
        267, 286, 323, 346
    };

    cgrad_tensor_f32_init(&a, shape);
    cgrad_tensor_f32_init(&b, shape);

    // Copy data for both batches
    for (int i = 0; i < 8; i++) {
        a.data[i] = dataA[i];
        b.data[i] = dataB[i];
    }

    int err = cgrad_tensor_f32_gemm(&a, &b, &c);
    assert(err == 0);

    // Check result for both batches
    for (int i = 0; i < 8; i++) {
        if (fabsf(c.data[i] - expected[i]) > EPSILON) {
            printf("Batched GEMM test failed: c.data[%d] = %f, expected %f\n", i, c.data[i], expected[i]);
            assert(0);
        }
    }
    printf("GEMM batched test passed.\n");

    cgrad_tensor_f32_free(&a);
    cgrad_tensor_f32_free(&b);
    cgrad_tensor_f32_free(&c);
}

int main() {
    test_gemm_simple();
    test_gemm_batched();
    return 0;
}
