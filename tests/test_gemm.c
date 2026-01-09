#include "backends/cgrad_tensor_f32_cpu.h"
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

void test_transpose() {
    cgrad_tensor_f32 t;
    uint32_t shape[] = {1, 2, 3, 4};
    cgrad_tensor_f32_init(&t, shape);

    // Save original strides and shape
    uint32_t orig_shape[MAX_TENSOR_DIM], orig_strides[MAX_TENSOR_DIM];
    for (int i = 0; i < MAX_TENSOR_DIM; i++) {
        orig_shape[i] = t.layout.shape[i];
        orig_strides[i] = t.layout.strides[i];
    }

    // Permute: swap last two axes (perm = [0,1,3,2])
    uint32_t perm[MAX_TENSOR_DIM] = {0, 1, 3, 2};
    cgrad_tensor_f32_transpose(&t, perm);

    // Check shape and strides
    for (int i = 0; i < MAX_TENSOR_DIM; i++) {
        assert(t.layout.shape[i] == orig_shape[perm[i]]);
        assert(t.layout.strides[i] == orig_strides[perm[i]]);
    }
    printf("Transpose test passed.\n");

    cgrad_tensor_f32_free(&t);
}

void test_gemm_with_transpose() {
    // A: shape [1,1,2,3], B: shape [1,1,3,2]
    cgrad_tensor_f32 a, b, c;
    uint32_t shapeA[] = {1, 1, 2, 3};
    uint32_t shapeB[] = {1, 1, 3, 2};
    float dataA[6] = {1, 2, 3, 4, 5, 6}; // [ [1,2,3], [4,5,6] ]
    float dataB[6] = {7, 8, 9, 10, 11, 12}; // [ [7,8], [9,10], [11,12] ]
    // Transpose A to shape [1,1,3,2], so A^T = [ [1,4], [2,5], [3,6] ]
    // GEMM: (A^T) [3x2] * B [3x2] is invalid, but B^T [2x3] * (A^T) [3x2] is valid
    // Instead, let's transpose A to [1,1,3,2] and B to [1,1,2,3], then do GEMM

    // For this test, let's just swap the last two axes of A and B, and do GEMM
    cgrad_tensor_f32_init(&a, shapeA);
    cgrad_tensor_f32_init(&b, shapeB);

    for (int i = 0; i < 6; i++) {
        a.data[i] = dataA[i];
        b.data[i] = dataB[i];
    }

    // Transpose A and B: swap last two axes
    uint32_t perm[MAX_TENSOR_DIM] = {0, 1, 3, 2};
    cgrad_tensor_f32_transpose(&a, perm);
    cgrad_tensor_f32_transpose(&b, perm);

    // Now a.shape = [1,1,3,2], b.shape = [1,1,2,3]
    // GEMM: [3x2] * [2x3] = [3x3]
    // Let's set up expected result for the first (and only) batch:
    // A^T = [ [1,4], [2,5], [3,6] ] (3x2)
    // B^T = [ [7,9,11], [8,10,12] ] (2x3)
    // C = A^T * B^T = [ [1*7+4*8, 1*9+4*10, 1*11+4*12],
    //                   [2*7+5*8, 2*9+5*10, 2*11+5*12],
    //                   [3*7+6*8, 3*9+6*10, 3*11+6*12] ]
    float expected[9] = {
        39, 49, 59,
        54, 68, 82,
        69, 87, 105
    };

    // GEMM
    cgrad_tensor_f32_gemm(&a, &b, &c);

    // Check result
    for (int i = 0; i < 9; i++) {
        if (fabsf(c.data[i] - expected[i]) > EPSILON) {
            printf("GEMM with transpose test failed: c.data[%d] = %f, expected %f\n", i, c.data[i], expected[i]);
            assert(0);
        }
    }
    printf("GEMM with transpose test passed.\n");

    cgrad_tensor_f32_free(&a);
    cgrad_tensor_f32_free(&b);
    cgrad_tensor_f32_free(&c);
}

int main() {
    test_gemm_simple();
    test_gemm_batched();
    test_transpose();
    test_gemm_with_transpose();
    return 0;
}
