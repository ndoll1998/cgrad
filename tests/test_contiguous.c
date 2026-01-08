#include "cgrad_tensor.h"
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>

#define EPSILON 1e-5

void test_make_contiguous() {
    cgrad_tensor_f32 t, t_contig;
    uint32_t shape[] = {2, 3, 4, 5};
    cgrad_tensor_f32_init(&t, shape);

    // Fill with unique values based on indices
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 3; j++)
        for (int k = 0; k < 4; k++)
          for (int l = 0; l < 5; l++) {
            uint32_t idx[4] = {i, j, k, l};
            float val = 1000*i + 100*j + 10*k + l;
            cgrad_tensor_f32_set(&t, idx, val);
          }

    // Transpose: swap axes 1 and 2
    uint32_t perm[4] = {0, 2, 1, 3};
    cgrad_tensor_f32_transpose(&t, perm);

    // Make contiguous
    int err = cgrad_tensor_f32_make_contiguous(&t, &t_contig);
    assert(err == 0);

    // Check that all values match logical order
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 3; j++)
        for (int k = 0; k < 4; k++)
          for (int l = 0; l < 5; l++) {
            // Logical index in original order
            uint32_t orig_idx[4] = {i, j, k, l};
            // Logical index in transposed order
            uint32_t trans_idx[4] = {i, k, j, l};
            float expected = 1000*i + 100*j + 10*k + l;
            float got = *cgrad_tensor_f32_ptr(&t_contig, trans_idx);
            if (fabsf(got - expected) > EPSILON) {
              printf("Contiguous test failed: got %f, expected %f at %d,%d,%d,%d\n", got, expected, i, k, j, l);
              assert(0);
            }
          }
    printf("Contiguous copy test passed.\n");

    cgrad_tensor_f32_free(&t);
    cgrad_tensor_f32_free(&t_contig);
}

int main() {
    test_make_contiguous();
    return 0;
}