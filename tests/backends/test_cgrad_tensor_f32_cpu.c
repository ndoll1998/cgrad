#include <cmocka.h>
#include "backends/cgrad_tensor_f32_cpu.h"
#include "cgrad_errors.h"
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#define EPSILON 1e-5

static void test_cgrad_tensor_f32_contiguous_swap23(void **state) {
    (void)state;
    cgrad_tensor_f32_cpu t, t_contig;
    uint32_t shape[] = {2, 3, 4, 5};
    cgrad_tensor_f32_cpu_init(&t, shape, 4);

    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 3; j++)
        for (int k = 0; k < 4; k++)
          for (int l = 0; l < 5; l++) {
            uint32_t idx[4] = {i, j, k, l};
            float val = 1000*i + 100*j + 10*k + l;
            cgrad_tensor_f32_cpu_set(&t, idx, val);
          }

    uint32_t perm[4] = {0, 2, 1, 3};
    assert_int_equal(cgrad_tensor_f32_cpu_transpose(&t, perm, 4), CGRAD_SUCCESS);

    int err = cgrad_tensor_f32_cpu_contiguous(&t, &t_contig);
    assert_int_equal(err, 0);

    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 3; j++)
        for (int k = 0; k < 4; k++)
          for (int l = 0; l < 5; l++) {
            uint32_t orig_idx[4] = {i, j, k, l};
            uint32_t trans_idx[4] = {i, k, j, l};
            float expected = 1000*i + 100*j + 10*k + l;
            float got = *cgrad_tensor_f32_cpu_ptr(&t_contig, trans_idx);
            assert_true(fabsf(got - expected) <= EPSILON);
          }

    cgrad_tensor_f32_cpu_free(&t);
    cgrad_tensor_f32_cpu_free(&t_contig);
}

static void test_cgrad_tensor_f32_contiguous_swap01(void **state) {
    (void)state;
    cgrad_tensor_f32_cpu t, t_contig;
    uint32_t shape[] = {2, 3, 4, 5};
    cgrad_tensor_f32_cpu_init(&t, shape, 4);

    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 3; j++)
        for (int k = 0; k < 4; k++)
          for (int l = 0; l < 5; l++) {
            uint32_t idx[4] = {i, j, k, l};
            float val = 1000*i + 100*j + 10*k + l;
            cgrad_tensor_f32_cpu_set(&t, idx, val);
          }

    uint32_t perm[4] = {1, 0, 2, 3};
    assert_int_equal(cgrad_tensor_f32_cpu_transpose(&t, perm, 4), CGRAD_SUCCESS);

    int err = cgrad_tensor_f32_cpu_contiguous(&t, &t_contig);
    assert_int_equal(err, 0);

    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 3; j++)
        for (int k = 0; k < 4; k++)
          for (int l = 0; l < 5; l++) {
            uint32_t orig_idx[4] = {i, j, k, l};
            uint32_t trans_idx[4] = {j, i, k, l};
            float expected = 1000*i + 100*j + 10*k + l;
            float got = *cgrad_tensor_f32_cpu_ptr(&t_contig, trans_idx);
            assert_true(fabsf(got - expected) <= EPSILON);
          }

    cgrad_tensor_f32_cpu_free(&t);
    cgrad_tensor_f32_cpu_free(&t_contig);
}

static void test_gemm_simple(void **state) {
    (void)state;
    cgrad_tensor_f32_cpu a, b, c;
    uint32_t shapeA[] = {1, 1, 2, 3};
    uint32_t shapeB[] = {1, 1, 3, 2};
    float dataA[6] = {1, 2, 3, 4, 5, 6};
    float dataB[6] = {7, 8, 9, 10, 11, 12};
    float expected[4] = {58, 64, 139, 154};

    cgrad_tensor_f32_cpu_init(&a, shapeA, 4);
    cgrad_tensor_f32_cpu_init(&b, shapeB, 4);

    for (int i = 0; i < 6; i++) {
        a.data[i] = dataA[i];
        b.data[i] = dataB[i];
    }

    // Output shape: {1, 1, 2, 2}
    uint32_t shapeC[] = {1, 1, 2, 2};
    cgrad_tensor_f32_cpu_init(&c, shapeC, 4);

    int err = cgrad_tensor_f32_cpu_gemm(&a, &b, &c);
    assert_int_equal(err, 0);

    for (int i = 0; i < 4; i++) {
        assert_true(fabsf(c.data[i] - expected[i]) <= EPSILON);
    }

    cgrad_tensor_f32_cpu_free(&a);
    cgrad_tensor_f32_cpu_free(&b);
    cgrad_tensor_f32_cpu_free(&c);
}

static void test_gemm_batched(void **state) {
    (void)state;
    cgrad_tensor_f32_cpu a, b, c;
    uint32_t shape[] = {2, 1, 2, 2};

    float dataA[8] = {1,2,3,4,9,10,11,12};
    float dataB[8] = {5,6,7,8,13,14,15,16};
    float expected[8] = {19, 22, 43, 50, 267, 286, 323, 346};

    cgrad_tensor_f32_cpu_init(&a, shape, 4);
    cgrad_tensor_f32_cpu_init(&b, shape, 4);

    for (int i = 0; i < 8; i++) {
        a.data[i] = dataA[i];
        b.data[i] = dataB[i];
    }

    // Output shape: {2, 1, 2, 2}
    uint32_t shapeC[] = {2, 1, 2, 2};
    cgrad_tensor_f32_cpu_init(&c, shapeC, 4);

    int err = cgrad_tensor_f32_cpu_gemm(&a, &b, &c);
    assert_int_equal(err, 0);

    for (int i = 0; i < 8; i++) {
        assert_true(fabsf(c.data[i] - expected[i]) <= EPSILON);
    }

    cgrad_tensor_f32_cpu_free(&a);
    cgrad_tensor_f32_cpu_free(&b);
    cgrad_tensor_f32_cpu_free(&c);
}

static void test_gemm_with_transpose(void **state) {
    (void)state;
    cgrad_tensor_f32_cpu a, b, c;
    uint32_t shapeA[] = {1, 1, 2, 3};
    uint32_t shapeB[] = {1, 1, 3, 2};
    float dataA[6] = {1, 2, 3, 4, 5, 6};
    float dataB[6] = {7, 8, 9, 10, 11, 12};
    float expected[9] = {39, 49, 59, 54, 68, 82, 69, 87, 105};

    cgrad_tensor_f32_cpu_init(&a, shapeA, 4);
    cgrad_tensor_f32_cpu_init(&b, shapeB, 4);

    for (int i = 0; i < 6; i++) {
        a.data[i] = dataA[i];
        b.data[i] = dataB[i];
    }

    uint32_t perm[TENSOR_DIM] = {0, 1, 3, 2};
    assert_int_equal(cgrad_tensor_f32_cpu_transpose(&a, perm, 4), CGRAD_SUCCESS);
    assert_int_equal(cgrad_tensor_f32_cpu_transpose(&b, perm, 4), CGRAD_SUCCESS);

    // Output shape: {1, 1, 3, 3}
    uint32_t shapeC[] = {1, 1, 3, 3};
    cgrad_tensor_f32_cpu_init(&c, shapeC, 4);

    cgrad_tensor_f32_cpu_gemm(&a, &b, &c);

    for (int i = 0; i < 9; i++) {
        assert_true(fabsf(c.data[i] - expected[i]) <= EPSILON);
    }

    cgrad_tensor_f32_cpu_free(&a);
    cgrad_tensor_f32_cpu_free(&b);
    cgrad_tensor_f32_cpu_free(&c);
}

static void test_tensor_add(void **state) {
    (void)state;
    cgrad_tensor_f32_cpu a, b, c;
    uint32_t shape[] = {2, 3, 4, 1};
    cgrad_tensor_f32_cpu_init(&a, shape, 4);
    cgrad_tensor_f32_cpu_init(&b, shape, 4);

    // Fill a and b with known values
    for (int i = 0; i < a.layout.size; i++) {
        a.data[i] = (float)i;
        b.data[i] = (float)(1000 + i);
    }

    int err = cgrad_tensor_f32_cpu_add(&a, &b, &c);
    assert_int_equal(err, 0);

    // Check c = a + b
    for (int i = 0; i < c.layout.size; i++) {
        assert_true(fabsf(c.data[i] - (a.data[i] + b.data[i])) < EPSILON);
    }

    cgrad_tensor_f32_cpu_free(&a);
    cgrad_tensor_f32_cpu_free(&b);
    cgrad_tensor_f32_cpu_free(&c);
}

static void test_add_with_transposed_inputs(void **state) {
    (void)state;
    cgrad_tensor_f32_cpu a, b, c;
    cgrad_tensor_f32_cpu_init(&a, (uint32_t[]){2,3,4,1}, 4);
    cgrad_tensor_f32_cpu_init(&b, (uint32_t[]){2,3,4,1}, 4);

    // Fill a and b with known values
    for (int i = 0; i < a.layout.size; i++) {
        a.data[i] = (float)i;
        b.data[i] = (float)(1000 + i);
    }

    // Make a non-contiguous by transposing memory layout but keep shape the same
    assert_int_equal(cgrad_tensor_f32_cpu_transpose(&a, (uint32_t[]){1, 0, 2, 3}, 4), CGRAD_SUCCESS);
    assert_int_equal(cgrad_tensor_f32_cpu_transpose(&b, (uint32_t[]){1, 0, 2, 3}, 4), CGRAD_SUCCESS);

    // c = a_t + b
    int err = cgrad_tensor_f32_cpu_add(&a, &b, &c);
    assert_int_equal(err, 0);

    // Check c = a.t + b.t
    for (int i = 0; i < c.layout.size; i++) {

        float x = *cgrad_tensor_f32_cpu_ptr(&a, (uint32_t[]){i / (2*4), (i / 4) % 2, (i % 4), 0});
        float y = *cgrad_tensor_f32_cpu_ptr(&b, (uint32_t[]){i / (2*4), (i / 4) % 2, (i % 4), 0});
        float z = *cgrad_tensor_f32_cpu_ptr(&c, (uint32_t[]){i / (2*4), (i / 4) % 2, (i % 4), 0});

        assert_true(fabsf(z - (x + y)) < EPSILON);
    }

    cgrad_tensor_f32_cpu_free(&a);
    cgrad_tensor_f32_cpu_free(&b);
    cgrad_tensor_f32_cpu_free(&c);
}

static void test_gemm_with_transposed_inputs(void **state) {
    (void)state;
    // a: [1, 1, 2, 3], b: [1, 1, 3, 2]
    uint32_t shapeA[] = {1, 1, 2, 3};
    uint32_t shapeB[] = {1, 1, 3, 2};
    float dataA[6] = {1, 2, 3, 4, 5, 6};
    float dataB[6] = {7, 8, 9, 10, 11, 12};
    float expected[9] = {39, 49, 59, 54, 68, 82, 69, 87, 105};

    cgrad_tensor_f32_cpu a, b, c;
    cgrad_tensor_f32_cpu_init(&a, shapeA, 4);
    cgrad_tensor_f32_cpu_init(&b, shapeB, 4);

    for (int i = 0; i < 6; i++) {
        a.data[i] = dataA[i];
        b.data[i] = dataB[i];
    }

    // Transpose a and b: swap last two axes
    uint32_t perm[TENSOR_DIM] = {0, 1, 3, 2};
    assert_int_equal(cgrad_tensor_f32_cpu_transpose(&a, perm, 4), CGRAD_SUCCESS);
    assert_int_equal(cgrad_tensor_f32_cpu_transpose(&b, perm, 4), CGRAD_SUCCESS);

    // Output shape: {1, 1, 3, 3}
    uint32_t shapeC[] = {1, 1, 3, 3};
    cgrad_tensor_f32_cpu_init(&c, shapeC, 4);

    int err = cgrad_tensor_f32_cpu_gemm(&a, &b, &c);
    assert_int_equal(err, 0);

    // Check result
    for (int i = 0; i < 9; i++) {
        cgrad_tensor_f32_cpu_ptr(&c, (uint32_t[]){0,0,i/3,i%3});
        assert_true(fabsf(c.data[i] - expected[i]) <= EPSILON);
    }

    cgrad_tensor_f32_cpu_free(&a);
    cgrad_tensor_f32_cpu_free(&b);
    cgrad_tensor_f32_cpu_free(&c);
}

static void test_transpose(void **state) {
    (void)state;
    cgrad_tensor_f32_cpu t;
    uint32_t shape[] = {1, 2, 3, 4};
    cgrad_tensor_f32_cpu_init(&t, shape, 4);

    uint32_t orig_shape[TENSOR_DIM], orig_strides[TENSOR_DIM];
    for (int i = 0; i < TENSOR_DIM; i++) {
        orig_shape[i] = t.layout.shape[i];
        orig_strides[i] = t.layout.strides[i];
    }

    uint32_t perm[TENSOR_DIM] = {0, 1, 3, 2};
    assert_int_equal(cgrad_tensor_f32_cpu_transpose(&t, perm, 4), CGRAD_SUCCESS);

    for (int i = 0; i < TENSOR_DIM; i++) {
        assert_int_equal(t.layout.shape[i], orig_shape[perm[i]]);
        assert_int_equal(t.layout.strides[i], orig_strides[perm[i]]);
    }

    cgrad_tensor_f32_cpu_free(&t);
}


int run_cgrad_tensor_f32_cpu_tests(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_cgrad_tensor_f32_contiguous_swap23),
        cmocka_unit_test(test_cgrad_tensor_f32_contiguous_swap01),
        cmocka_unit_test(test_gemm_simple),
        cmocka_unit_test(test_gemm_batched),
        cmocka_unit_test(test_gemm_with_transpose),
        cmocka_unit_test(test_tensor_add),
        cmocka_unit_test(test_transpose),
        cmocka_unit_test(test_add_with_transposed_inputs),
        cmocka_unit_test(test_gemm_with_transposed_inputs),
    };
    return _cmocka_run_group_tests("cgrad_tensor_f32_cpu", tests, sizeof(tests)/sizeof(tests[0]), NULL, NULL);
}

#ifndef TEST_ALL_MAIN
int main(void) {
    return run_cgrad_tensor_f32_cpu_tests();
}
#endif

