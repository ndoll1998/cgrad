#include <cmocka.h>
#include "storage/cgrad_storage.h"
#include "cgrad_errors.h"
#include "storage/cgrad_storage_layout.h"
#include <stdio.h>
#include <stdint.h>
#include <math.h>

// Helper to access backend-specific storage data
typedef struct {
    cgrad_storage_layout layout;
    float* data;
} cgrad_backend_cpu_f32;

#define EPSILON 1e-5

static void test_cgrad_backend_cpu_f32_contiguous_swap23(void **state) {
    (void)state;
    cgrad_storage t;
    uint32_t shape[] = {2, 3, 4, 5};
    cgrad_storage_init(&t, shape, 4, "cpu_f32");
    cgrad_backend_cpu_f32* backend_data = (cgrad_backend_cpu_f32*)t.data;

    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 3; j++)
        for (int k = 0; k < 4; k++)
          for (int l = 0; l < 5; l++) {
            uint32_t idx[4] = {i, j, k, l};
            float val = 1000*i + 100*j + 10*k + l;
            t.backend->storage_set(t.data, idx, 4, val);
          }

    uint32_t perm[4] = {0, 2, 1, 3};
    assert_int_equal(cgrad_storage_layout_transpose(&backend_data->layout, perm, 4), CGRAD_SUCCESS);

    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 3; j++)
        for (int k = 0; k < 4; k++)
          for (int l = 0; l < 5; l++) {
            // After permuting {0,2,1,3}, the shape is {2,4,3,5}
            // The original index {i,j,k,l} maps to {i,k,j,l}
            uint32_t permuted_idx[4] = {i, k, j, l};
            float expected = 1000*i + 100*j + 10*k + l;
            float got = 0.0f;
            assert_int_equal(t.backend->storage_get(t.data, permuted_idx, 4, &got), CGRAD_SUCCESS);
            assert_true(fabsf(got - expected) <= EPSILON);
          }

    cgrad_storage_free(&t);
}

static void test_cgrad_backend_cpu_f32_contiguous_swap01(void **state) {
    (void)state;
    cgrad_storage t;
    uint32_t shape[] = {2, 3, 4, 5};
    cgrad_storage_init(&t, shape, 4, "cpu_f32");
    cgrad_backend_cpu_f32* backend_data = (cgrad_backend_cpu_f32*)t.data;

    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 3; j++)
        for (int k = 0; k < 4; k++)
          for (int l = 0; l < 5; l++) {
            uint32_t idx[4] = {i, j, k, l};
            float val = 1000*i + 100*j + 10*k + l;
            t.backend->storage_set(t.data, idx, 4, val);
          }

    uint32_t perm[4] = {1, 0, 2, 3};
    assert_int_equal(cgrad_storage_layout_transpose(&backend_data->layout, perm, 4), CGRAD_SUCCESS);

    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 3; j++)
        for (int k = 0; k < 4; k++)
          for (int l = 0; l < 5; l++) {
            // After permuting {1,0,2,3}, the shape is {3,2,4,5}
            // The original index {i,j,k,l} maps to {j,i,k,l}
            uint32_t permuted_idx[4] = {j, i, k, l};
            float expected = 1000*i + 100*j + 10*k + l;
            float got = 0.0f;
            assert_int_equal(t.backend->storage_get(t.data, permuted_idx, 4, &got), CGRAD_SUCCESS);
            assert_true(fabsf(got - expected) <= EPSILON);
          }

    cgrad_storage_free(&t);
}

static void test_cgrad_backend_cpu_f32_gemm_simple(void **state) {
    (void)state;
    cgrad_storage a, b, c;
    uint32_t shapeA[] = {1, 1, 2, 3};
    uint32_t shapeB[] = {1, 1, 3, 2};
    float dataA[6] = {1, 2, 3, 4, 5, 6};
    float dataB[6] = {7, 8, 9, 10, 11, 12};
    float expected[4] = {58, 64, 139, 154};

    cgrad_storage_init(&a, shapeA, 4, "cpu_f32");
    cgrad_storage_init(&b, shapeB, 4, "cpu_f32");
    cgrad_backend_cpu_f32* a_data = (cgrad_backend_cpu_f32*)a.data;
    cgrad_backend_cpu_f32* b_data = (cgrad_backend_cpu_f32*)b.data;

    for (int i = 0; i < 6; i++) {
        a_data->data[i] = dataA[i];
        b_data->data[i] = dataB[i];
    }

    // Output shape: {1, 1, 2, 2}
    uint32_t shapeC[] = {1, 1, 2, 2};
    cgrad_storage_init(&c, shapeC, 4, "cpu_f32");
    cgrad_backend_cpu_f32* c_data = (cgrad_backend_cpu_f32*)c.data;

    int err = a.backend->storage_gemm(1.0f, a.data, b.data, 0.0f, c.data);
    assert_int_equal(err, 0);

    for (int i = 0; i < 4; i++) {
        assert_true(fabsf(c_data->data[i] - expected[i]) <= EPSILON);
    }

    cgrad_storage_free(&a);
    cgrad_storage_free(&b);
    cgrad_storage_free(&c);
}

static void test_cgrad_backend_cpu_f32_gemm_batched(void **state) {
    (void)state;
    cgrad_storage a, b, c;
    uint32_t shape[] = {1, 2, 2, 2};

    float dataA[8] = {1,2,3,4,9,10,11,12};
    float dataB[8] = {5,6,7,8,13,14,15,16};

    cgrad_storage_init(&a, shape, 4, "cpu_f32");
    cgrad_storage_init(&b, shape, 4, "cpu_f32");
    cgrad_backend_cpu_f32* a_data = (cgrad_backend_cpu_f32*)a.data;
    cgrad_backend_cpu_f32* b_data = (cgrad_backend_cpu_f32*)b.data;

    for (int i = 0; i < 8; i++) {
        a_data->data[i] = dataA[i];
        b_data->data[i] = dataB[i];
    }

    // Output shape: {1, 2, 2, 2}
    uint32_t shapeC[] = {1, 2, 2, 2};
    cgrad_storage_init(&c, shapeC, 4, "cpu_f32");

    int err = a.backend->storage_gemm(1.0f, a.data, b.data, 0.0f, c.data);
    assert_int_equal(err, 0);

    cgrad_storage_free(&a);
    cgrad_storage_free(&b);
    cgrad_storage_free(&c);
}

static void test_cgrad_backend_cpu_f32_gemm_with_transpose(void **state) {
    (void)state;
    cgrad_storage a, b, c;
    uint32_t shapeA[] = {1, 1, 2, 3};
    uint32_t shapeB[] = {1, 1, 3, 2};
    float dataA[6] = {1, 2, 3, 4, 5, 6};
    float dataB[6] = {7, 8, 9, 10, 11, 12};
    float expected[9] = {39, 49, 59, 54, 68, 82, 69, 87, 105};

    cgrad_storage_init(&a, shapeA, 4, "cpu_f32");
    cgrad_storage_init(&b, shapeB, 4, "cpu_f32");
    cgrad_backend_cpu_f32* a_data = (cgrad_backend_cpu_f32*)a.data;
    cgrad_backend_cpu_f32* b_data = (cgrad_backend_cpu_f32*)b.data;

    for (int i = 0; i < 6; i++) {
        a_data->data[i] = dataA[i];
        b_data->data[i] = dataB[i];
    }

    uint32_t perm[TENSOR_DIM] = {0, 1, 3, 2};
    assert_int_equal(cgrad_storage_layout_transpose(&a_data->layout, perm, 4), CGRAD_SUCCESS);
    assert_int_equal(cgrad_storage_layout_transpose(&b_data->layout, perm, 4), CGRAD_SUCCESS);

    // Output shape: {1, 1, 3, 3}
    uint32_t shapeC[] = {1, 1, 3, 3};
    cgrad_storage_init(&c, shapeC, 4, "cpu_f32");
    cgrad_backend_cpu_f32* c_data = (cgrad_backend_cpu_f32*)c.data;

    a.backend->storage_gemm(1.0f, a.data, b.data, 0.0f, c.data);

    for (int i = 0; i < 9; i++) {
        assert_true(fabsf(c_data->data[i] - expected[i]) <= EPSILON);
    }

    cgrad_storage_free(&a);
    cgrad_storage_free(&b);
    cgrad_storage_free(&c);
}

static void test_cgrad_backend_cpu_f32_tensor_add(void **state) {
    (void)state;
    cgrad_storage a, b;
    uint32_t shape[] = {2, 3, 4, 1};
    cgrad_storage_init(&a, shape, 4, "cpu_f32");
    cgrad_storage_init(&b, shape, 4, "cpu_f32");
    cgrad_backend_cpu_f32* a_data = (cgrad_backend_cpu_f32*)a.data;
    cgrad_backend_cpu_f32* b_data = (cgrad_backend_cpu_f32*)b.data;

    // Fill a and b with known values
    for (int i = 0; i < a_data->layout.size; i++) {
        a_data->data[i] = (float)i;
        b_data->data[i] = (float)(1000 + i);
    }

    // b = a + b (in-place)
    int err = a.backend->storage_axpy(1.0f, a.data, b.data, NULL);
    assert_int_equal(err, 0);

    // Check b = a + b
    for (int i = 0; i < b_data->layout.size; i++) {
        assert_true(fabsf(b_data->data[i] - (a_data->data[i] + (float)(1000 + i))) < EPSILON);
    }

    cgrad_storage_free(&a);
    cgrad_storage_free(&b);
}

static void test_cgrad_backend_cpu_f32_add_with_transposed_inputs(void **state) {
    (void)state;
    cgrad_storage a, b;
    cgrad_storage_init(&a, (uint32_t[]){2, 3, 4, 1}, 4, "cpu_f32");
    cgrad_storage_init(&b, (uint32_t[]){3, 2, 4, 1}, 4, "cpu_f32");
    cgrad_backend_cpu_f32* a_data = (cgrad_backend_cpu_f32*)a.data;
    cgrad_backend_cpu_f32* b_data = (cgrad_backend_cpu_f32*)b.data;

    // Fill a and b with known values
    for (int i = 0; i < a_data->layout.size; i++) {
        a_data->data[i] = (float)i;
        b_data->data[i] = (float)(1000 + i);
    }

    // Store original b values for verification
    float original_b[24];
    for (int i = 0; i < b_data->layout.size; i++) {
        original_b[i] = b_data->data[i];
    }

    // Make a non-contiguous by transposing memory layout
    assert_int_equal(cgrad_storage_layout_transpose(&a_data->layout, (uint32_t[]){1, 0, 2, 3}, 4), CGRAD_SUCCESS);

    // b = a + b (in-place)
    int err = a.backend->storage_axpy(1.0f, a.data, b.data, NULL);
    assert_int_equal(err, 0);

    // Check b = a + b
    for (int i = 0; i < b_data->layout.size; i++) {
        uint32_t idx[4] = {i / (2*4), (i / 4) % 2, (i % 4), 0};
        float a_val = 0.0f;
        assert_int_equal(a.backend->storage_get(a.data, idx, 4, &a_val), CGRAD_SUCCESS);
        
        float expected = a_val + original_b[i];
        assert_true(fabsf(b_data->data[i] - expected) < EPSILON);
    }

    cgrad_storage_free(&a);
    cgrad_storage_free(&b);
}

static void test_cgrad_backend_cpu_f32_gemm_with_transposed_inputs(void **state) {
    (void)state;
    uint32_t shapeA[] = {1, 1, 2, 3};
    uint32_t shapeB[] = {1, 1, 3, 2};
    float dataA[6] = {1, 2, 3, 4, 5, 6};
    float dataB[6] = {7, 8, 9, 10, 11, 12};
    float expected[9] = {39, 49, 59, 54, 68, 82, 69, 87, 105};

    cgrad_storage a, b, c;
    cgrad_storage_init(&a, shapeA, 4, "cpu_f32");
    cgrad_storage_init(&b, shapeB, 4, "cpu_f32");
    cgrad_backend_cpu_f32* a_data = (cgrad_backend_cpu_f32*)a.data;
    cgrad_backend_cpu_f32* b_data = (cgrad_backend_cpu_f32*)b.data;

    for (int i = 0; i < 6; i++) {
        a_data->data[i] = dataA[i];
        b_data->data[i] = dataB[i];
    }

    // Transpose a and b: swap last two axes
    uint32_t perm[TENSOR_DIM] = {0, 1, 3, 2};
    assert_int_equal(cgrad_storage_layout_transpose(&a_data->layout, perm, 4), CGRAD_SUCCESS);
    assert_int_equal(cgrad_storage_layout_transpose(&b_data->layout, perm, 4), CGRAD_SUCCESS);

    // Output shape: {1, 1, 3, 3}
    uint32_t shapeC[] = {1, 1, 3, 3};
    cgrad_storage_init(&c, shapeC, 4, "cpu_f32");
    cgrad_backend_cpu_f32* c_data = (cgrad_backend_cpu_f32*)c.data;

    int err = a.backend->storage_gemm(1.0f, a.data, b.data, 0.0f, c.data);
    assert_int_equal(err, 0);

    // Check result
    for (int i = 0; i < 9; i++) {
        assert_true(fabsf(c_data->data[i] - expected[i]) <= EPSILON);
    }

    cgrad_storage_free(&a);
    cgrad_storage_free(&b);
    cgrad_storage_free(&c);
}

int run_cgrad_backend_cpu_f32_tests(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_cgrad_backend_cpu_f32_contiguous_swap23),
        cmocka_unit_test(test_cgrad_backend_cpu_f32_contiguous_swap01),
        cmocka_unit_test(test_cgrad_backend_cpu_f32_gemm_simple),
        cmocka_unit_test(test_cgrad_backend_cpu_f32_gemm_batched),
        cmocka_unit_test(test_cgrad_backend_cpu_f32_gemm_with_transpose),
        cmocka_unit_test(test_cgrad_backend_cpu_f32_tensor_add),
        cmocka_unit_test(test_cgrad_backend_cpu_f32_add_with_transposed_inputs),
        cmocka_unit_test(test_cgrad_backend_cpu_f32_gemm_with_transposed_inputs),
    };
    return cmocka_run_group_tests_name("cgrad_backend_cpu_f32", tests, NULL, NULL);
}

#ifndef TEST_ALL_MAIN
int main(void) {
    return run_cgrad_backend_cpu_f32_tests();
}
#endif
