#include "cgrad_tensor.h"
#include <stdio.h>
#include <stdint.h>
#include <assert.h>

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

int main() {
    test_transpose();
    return 0;
}