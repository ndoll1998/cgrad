#include <cmocka.h>
#include "cgrad_layout.h"
#include "cgrad_errors.h"
#include <stdint.h>
#include <string.h>

static void test_cgrad_tensor_layout_init_and_copy(void **state) {
    (void)state;
    cgrad_tensor_layout l1, l2;
    uint32_t shape[4] = {2, 3, 4, 5};
    assert_int_equal(cgrad_tensor_layout_init(&l1, shape, 4), 0);
    cgrad_tensor_layout_copy(&l2, &l1);
    assert_memory_equal(&l1, &l2, sizeof(cgrad_tensor_layout));
}

static void test_cgrad_tensor_layout_flat_index(void **state) {
    (void)state;
    cgrad_tensor_layout l;
    uint32_t shape[4] = {2, 3, 4, 5};
    assert_int_equal(cgrad_tensor_layout_init(&l, shape, 4), 0);

    // Valid index
    uint32_t indices_valid[4] = {1, 2, 3, 4};
    size_t idx = 0;
    int err = cgrad_tensor_layout_flat_index(&l, indices_valid, 4, &idx);
    assert_int_equal(err, CGRAD_SUCCESS);
    assert_int_equal(idx, 1*l.strides[0] + 2*l.strides[1] + 3*l.strides[2] + 4*l.strides[3]);

    // Out-of-bounds index
    uint32_t indices_oob[4] = {2, 0, 0, 0}; // 2 >= shape[0]
    err = cgrad_tensor_layout_flat_index(&l, indices_oob, 4, &idx);
    assert_int_equal(err, CGRAD_LAYOUT_ERR_INDEX_OUT_OF_BOUNDS);
}

static void test_cgrad_tensor_layout_transpose(void **state) {
    (void)state;
    cgrad_tensor_layout l;
    uint32_t shape[4] = {2, 3, 4, 5};
    cgrad_tensor_layout_init(&l, shape, 4);
    uint32_t perm[4] = {0, 2, 1, 3};
    cgrad_tensor_layout l_orig;
    cgrad_tensor_layout_copy(&l_orig, &l);
    cgrad_tensor_layout_transpose(&l, perm, 4);
    for (int i = 0; i < 4; i++) {
        assert_int_equal(l.shape[i], l_orig.shape[perm[i]]);
        assert_int_equal(l.strides[i], l_orig.strides[perm[i]]);
    }
}

static void test_cgrad_tensor_layout_is_contiguous(void **state) {
    (void)state;
    cgrad_tensor_layout l;
    uint32_t shape[4] = {2, 3, 4, 5};
    cgrad_tensor_layout_init(&l, shape, 4);
    assert_int_equal(cgrad_tensor_layout_is_contiguous(&l), 1);

    // Make non-contiguous by modifying strides
    l.strides[2] = 100;
    assert_int_equal(cgrad_tensor_layout_is_contiguous(&l), 0);

    // Edge case: shape with 1 in some dims
    uint32_t shape2[4] = {1, 1, 4, 5};
    cgrad_tensor_layout_init(&l, shape2, 4);
    assert_int_equal(cgrad_tensor_layout_is_contiguous(&l), 1);

    // Edge case: NULL pointer
    assert_int_equal(cgrad_tensor_layout_is_contiguous(NULL), 0);
}

static void test_cgrad_tensor_layout_transpose_duplicate_dim(void **state) {
    (void)state;
    cgrad_tensor_layout l;
    uint32_t shape[4] = {2, 3, 4, 5};
    cgrad_tensor_layout_init(&l, shape, 4);
    // Duplicate dimension: 0 appears twice
    uint32_t perm[4] = {0, 1, 2, 0};
    int err = cgrad_tensor_layout_transpose(&l, perm, 4);
    assert_int_equal(err, CGRAD_LAYOUT_ERR_DUPLICATE_DIM);
}

static void test_cgrad_tensor_layout_is_regular(void **state) {
    (void)state;
    cgrad_tensor_layout l;
    uint32_t shape[4] = {2, 3, 4, 5};
    cgrad_tensor_layout_init(&l, shape, 4);
    // Contiguous is regular
    assert_int_equal(cgrad_tensor_layout_is_regular(&l), 1);

    // Make strides a constant multiple of contiguous (e.g., *2)
    for (int i = 0; i < 4; i++) l.strides[i] *= 2;
    assert_int_equal(cgrad_tensor_layout_is_regular(&l), 1);

    // Make strides irregular
    l.strides[2] = 7;
    assert_int_equal(cgrad_tensor_layout_is_regular(&l), 0);

    // Degenerate: shape with 1s
    uint32_t shape2[4] = {1, 1, 4, 5};
    cgrad_tensor_layout_init(&l, shape2, 4);
    assert_int_equal(cgrad_tensor_layout_is_regular(&l), 1);

    // NULL pointer
    assert_int_equal(cgrad_tensor_layout_is_regular(NULL), 0);
}

static void test_partial_shape_and_index(void **state) {
    (void)state;
    cgrad_tensor_layout l;
    uint32_t shape2[2] = {3, 4};
    assert_int_equal(cgrad_tensor_layout_init(&l, shape2, 2), 0);
    // Should fill as {1,1,3,4}
    assert_int_equal(l.shape[0], 1);
    assert_int_equal(l.shape[1], 1);
    assert_int_equal(l.shape[2], 3);
    assert_int_equal(l.shape[3], 4);

    // Valid index for last 2 dims
    uint32_t indices2[2] = {2, 3};
    size_t idx = 0;
    int err = cgrad_tensor_layout_flat_index(&l, indices2, 2, &idx);
    assert_int_equal(err, CGRAD_SUCCESS);
    assert_int_equal(idx, 2*l.strides[2] + 3*l.strides[3]);

    // Out-of-bounds for last 2 dims
    uint32_t indices2_oob[2] = {3, 0};
    err = cgrad_tensor_layout_flat_index(&l, indices2_oob, 2, &idx);
    assert_int_equal(err, CGRAD_LAYOUT_ERR_INDEX_OUT_OF_BOUNDS);

    // Valid index for ndim=1
    uint32_t shape1[1] = {7};
    assert_int_equal(cgrad_tensor_layout_init(&l, shape1, 1), 0);
    assert_int_equal(l.shape[0], 1);
    assert_int_equal(l.shape[1], 1);
    assert_int_equal(l.shape[2], 1);
    assert_int_equal(l.shape[3], 7);
    uint32_t indices1[1] = {6};
    err = cgrad_tensor_layout_flat_index(&l, indices1, 1, &idx);
    assert_int_equal(err, CGRAD_SUCCESS);
    assert_int_equal(idx, 6*l.strides[3]);
}

static void test_partial_transpose(void **state) {
    (void)state;
    cgrad_tensor_layout l;
    uint32_t shape[3] = {2, 3, 4};
    assert_int_equal(cgrad_tensor_layout_init(&l, shape, 3), 0);
    // shape should be {1,2,3,4}
    assert_int_equal(l.shape[0], 1);
    assert_int_equal(l.shape[1], 2);
    assert_int_equal(l.shape[2], 3);
    assert_int_equal(l.shape[3], 4);

    // Transpose last 2 dims (swap 3 and 4)
    uint32_t perm2[2] = {1,0};
    assert_int_equal(cgrad_tensor_layout_transpose(&l, perm2, 2), 0);
    // Now shape should be {1,2,4,3}
    assert_int_equal(l.shape[0], 1);
    assert_int_equal(l.shape[1], 2);
    assert_int_equal(l.shape[2], 4);
    assert_int_equal(l.shape[3], 3);

    // Transpose last 3 dims (permute 2,4,3)
    uint32_t perm3[3] = {2,1,0};
    assert_int_equal(cgrad_tensor_layout_transpose(&l, perm3, 3), 0);
    // Now shape should be {1,3,4,2}
    assert_int_equal(l.shape[0], 1);
    assert_int_equal(l.shape[1], 3);
    assert_int_equal(l.shape[2], 4);
    assert_int_equal(l.shape[3], 2);
}

int run_cgrad_layout_tests(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_cgrad_tensor_layout_init_and_copy),
        cmocka_unit_test(test_cgrad_tensor_layout_flat_index),
        cmocka_unit_test(test_cgrad_tensor_layout_transpose),
        cmocka_unit_test(test_cgrad_tensor_layout_is_contiguous),
        cmocka_unit_test(test_cgrad_tensor_layout_transpose_duplicate_dim),
        cmocka_unit_test(test_cgrad_tensor_layout_is_regular),
        cmocka_unit_test(test_partial_shape_and_index),
        cmocka_unit_test(test_partial_transpose),
    };
    return _cmocka_run_group_tests("cgrad_layout", tests, sizeof(tests)/sizeof(tests[0]), NULL, NULL);
}

#ifndef TEST_ALL_MAIN
int main(void) {
    return run_cgrad_layout_tests();
}
#endif
