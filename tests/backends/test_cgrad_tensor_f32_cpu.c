#include <cmocka.h>
#include "backends/cgrad_tensor_f32_cpu.h"
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#define EPSILON 1e-5
#define MAX_TENSOR_DIM 4

static void test_cgrad_tensor_f32_contiguous_swap23(void **state) {
    (void)state;
    cgrad_tensor_f32_cpu t, t_contig;
    uint32_t shape[] = {2, 3, 4, 5};
    cgrad_tensor_f32_cpu_init(&t, shape);

    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 3; j++)
        for (int k = 0; k < 4; k++)
          for (int l = 0; l < 5; l++) {
            uint32_t idx[4] = {i, j, k, l};
            float val = 1000*i + 100*j + 10*k + l;
            cgrad_tensor_f32_cpu_set(&t, idx, val);
          }

    uint32_t perm[4] = {0, 2, 1, 3};
    cgrad_tensor_f32_cpu_transpose(&t, perm);

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
    cgrad_tensor_f32_cpu_init(&t, shape);

    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 3; j++)
        for (int k = 0; k < 4; k++)
          for (int l = 0; l < 5; l++) {
            uint32_t idx[4] = {i, j, k, l};
            float val = 1000*i + 100*j + 10*k + l;
            cgrad_tensor_f32_cpu_set(&t, idx, val);
          }

    uint32_t perm[4] = {1, 0, 2, 3};
    cgrad_tensor_f32_cpu_transpose(&t, perm);

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

    cgrad_tensor_f32_cpu_init(&a, shapeA);
    cgrad_tensor_f32_cpu_init(&b, shapeB);

    for (int i = 0; i < 6; i++) {
        a.data[i] = dataA[i];
        b.data[i] = dataB[i];
    }

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

    cgrad_tensor_f32_cpu_init(&a, shape);
    cgrad_tensor_f32_cpu_init(&b, shape);

    for (int i = 0; i < 8; i++) {
        a.data[i] = dataA[i];
        b.data[i] = dataB[i];
    }

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

    cgrad_tensor_f32_cpu_init(&a, shapeA);
    cgrad_tensor_f32_cpu_init(&b, shapeB);

    for (int i = 0; i < 6; i++) {
        a.data[i] = dataA[i];
        b.data[i] = dataB[i];
    }

    uint32_t perm[MAX_TENSOR_DIM] = {0, 1, 3, 2};
    cgrad_tensor_f32_cpu_transpose(&a, perm);
    cgrad_tensor_f32_cpu_transpose(&b, perm);

    cgrad_tensor_f32_cpu_gemm(&a, &b, &c);

    for (int i = 0; i < 9; i++) {
        assert_true(fabsf(c.data[i] - expected[i]) <= EPSILON);
    }

    cgrad_tensor_f32_cpu_free(&a);
    cgrad_tensor_f32_cpu_free(&b);
    cgrad_tensor_f32_cpu_free(&c);
}

static void test_gemm_broadcasting(void **state) {
    (void)state;
    cgrad_tensor_f32_cpu a, b, c;
    uint32_t shapeA[] = {2, 1, 2, 2};
    uint32_t shapeB[] = {1, 3, 2, 2};
    cgrad_tensor_f32_cpu_init(&a, shapeA);
    cgrad_tensor_f32_cpu_init(&b, shapeB);

    for (int i = 0; i < 2; i++) {
        for (int m = 0; m < 2; m++) {
            for (int k = 0; k < 2; k++) {
                int idx = i*4 + m*2 + k;
                a.data[idx] = (float)(i+1);
            }
        }
    }
    for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 2; k++) {
            for (int n = 0; n < 2; n++) {
                int idx = j*4 + k*2 + n;
                b.data[idx] = (float)(j+10);
            }
        }
    }

    int err = cgrad_tensor_f32_cpu_gemm(&a, &b, &c);
    assert_int_equal(err, 0);

    assert_int_equal(c.layout.shape[0], 2);
    assert_int_equal(c.layout.shape[1], 3);
    assert_int_equal(c.layout.shape[2], 2);
    assert_int_equal(c.layout.shape[3], 2);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            for (int m = 0; m < 2; m++) {
                for (int n = 0; n < 2; n++) {
                    float expected = 0.0f;
                    for (int k = 0; k < 2; k++) {
                        expected += (float)(i+1) * (float)(j+10);
                    }
                    int idx = ((i*3 + j)*4) + m*2 + n;
                    assert_true(fabsf(c.data[idx] - expected) <= EPSILON);
                }
            }
        }
    }

    cgrad_tensor_f32_cpu_free(&a);
    cgrad_tensor_f32_cpu_free(&b);
    cgrad_tensor_f32_cpu_free(&c);
}

static void test_tensor_add_broadcast(void **state) {
    (void)state;
    // a: [2, 3, 4, 1], b: [1, 3, 4, 1] -> c: [2, 3, 4, 1]
    uint32_t shape_a[] = {2, 3, 4, 1};
    uint32_t shape_b[] = {1, 3, 4, 1};
    cgrad_tensor_f32_cpu a, b, c;
    cgrad_tensor_f32_cpu_init(&a, shape_a);
    cgrad_tensor_f32_cpu_init(&b, shape_b);

    // Fill a and b with known values
    for (int i = 0; i < a.layout.size; i++) {
        a.data[i] = (float)(i + 1);
    }
    for (int i = 0; i < b.layout.size; i++) {
        b.data[i] = (float)(1000 + i);
    }

    int err = cgrad_tensor_f32_cpu_add(&a, &b, &c);
    assert_int_equal(err, 0);

    // Check c = a + b (with broadcasting)
    for (int i0 = 0; i0 < 2; i0++) {
        for (int i1 = 0; i1 < 3; i1++) {
            for (int i2 = 0; i2 < 4; i2++) {
                for (int i3 = 0; i3 < 1; i3++) {
                    int idx_a = ((i0 * 3 + i1) * 4 + i2) * 1 + i3;
                    int idx_b = ((0 * 3 + i1) * 4 + i2) * 1 + i3;
                    int idx_c = idx_a;
                    float expected = a.data[idx_a] + b.data[idx_b];
                    assert_true(fabsf(c.data[idx_c] - expected) < 1e-5);
                }
            }
        }
    }

    cgrad_tensor_f32_cpu_free(&a);
    cgrad_tensor_f32_cpu_free(&b);
    cgrad_tensor_f32_cpu_free(&c);
}

static void test_tensor_add(void **state) {
    (void)state;
    cgrad_tensor_f32_cpu a, b, c;
    uint32_t shape[] = {2, 3, 4, 1};
    cgrad_tensor_f32_cpu_init(&a, shape);
    cgrad_tensor_f32_cpu_init(&b, shape);

    // Fill a and b with known values
    for (int i = 0; i < a.layout.size; i++) {
        a.data[i] = (float)i;
        b.data[i] = (float)(1000 + i);
    }

    int err = cgrad_tensor_f32_cpu_add(&a, &b, &c);
    assert_int_equal(err, 0);

    // Check c = a + b
    for (int i = 0; i < c.layout.size; i++) {
        assert_true(fabsf(c.data[i] - (a.data[i] + b.data[i])) < 1e-5);
    }

    cgrad_tensor_f32_cpu_free(&a);
    cgrad_tensor_f32_cpu_free(&b);
    cgrad_tensor_f32_cpu_free(&c);
}

static void test_transpose(void **state) {
    (void)state;
    cgrad_tensor_f32_cpu t;
    uint32_t shape[] = {1, 2, 3, 4};
    cgrad_tensor_f32_cpu_init(&t, shape);

    uint32_t orig_shape[MAX_TENSOR_DIM], orig_strides[MAX_TENSOR_DIM];
    for (int i = 0; i < MAX_TENSOR_DIM; i++) {
        orig_shape[i] = t.layout.shape[i];
        orig_strides[i] = t.layout.strides[i];
    }

    uint32_t perm[MAX_TENSOR_DIM] = {0, 1, 3, 2};
    cgrad_tensor_f32_cpu_transpose(&t, perm);

    for (int i = 0; i < MAX_TENSOR_DIM; i++) {
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
        cmocka_unit_test(test_gemm_broadcasting),
        cmocka_unit_test(test_tensor_add),
        cmocka_unit_test(test_tensor_add_broadcast),
        cmocka_unit_test(test_transpose),
    };
    return _cmocka_run_group_tests("cgrad_tensor_f32_cpu", tests, sizeof(tests)/sizeof(tests[0]), NULL, NULL);
}

#ifndef TEST_ALL_MAIN
int main(void) {
    return run_cgrad_tensor_f32_cpu_tests();
}
#endif
