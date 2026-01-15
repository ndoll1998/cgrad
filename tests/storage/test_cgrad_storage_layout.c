#include <cmocka.h>
#include "storage/cgrad_storage_layout.h"
#include "cgrad_errors.h"
#include <stdint.h>
#include <string.h>

static void test_cgrad_storage_layout_init_and_copy(void **state) {
    (void)state;
    cgrad_storage_layout l1, l2;
    uint32_t shape[3] = {3, 4, 5};
    assert_int_equal(cgrad_storage_layout_init(&l1, shape, 3), 0);
    // Check right-alignment and leading 1s
    for (int i = 0; i < TENSOR_DIM - 3; ++i) assert_int_equal(l1.shape[i], 1);
    assert_int_equal(l1.shape[TENSOR_DIM - 3], 3);
    assert_int_equal(l1.shape[TENSOR_DIM - 2], 4);
    assert_int_equal(l1.shape[TENSOR_DIM - 1], 5);
    cgrad_storage_layout_copy(&l2, &l1);
    assert_memory_equal(&l1, &l2, sizeof(cgrad_storage_layout));
}

static void test_cgrad_storage_layout_flat_index(void **state) {
    (void)state;
    cgrad_storage_layout l;
    uint32_t shape[2] = {4, 5};
    assert_int_equal(cgrad_storage_layout_init(&l, shape, 2), 0);

    // Valid index (for last 2 dims)
    uint32_t indices_valid[2] = {3, 2};
    size_t idx = 0;
    int err = cgrad_storage_layout_flat_index(&l, indices_valid, 2, &idx);
    assert_int_equal(err, CGRAD_SUCCESS);
    size_t expected_idx = 3 * l.strides[TENSOR_DIM - 2] + 2 * l.strides[TENSOR_DIM - 1];
    // For shape {4,5} right-aligned in TENSOR_DIM, expected_idx should be 3*5 + 2*1 = 17
    assert_int_equal(idx, 17);

    // Out-of-bounds index
    uint32_t indices_oob[2] = {4, 0}; // 4 >= shape[0]
    err = cgrad_storage_layout_flat_index(&l, indices_oob, 2, &idx);
    assert_int_equal(err, CGRAD_STORAGE_LAYOUT_ERR_INDEX_OUT_OF_BOUNDS);
}

static void test_cgrad_storage_layout_transpose(void **state) {
    (void)state;
    cgrad_storage_layout l;
    uint32_t shape[TENSOR_DIM];
    for (int i = 0; i < TENSOR_DIM; ++i) shape[i] = i + 2;
    cgrad_storage_layout_init(&l, shape, TENSOR_DIM);
    uint32_t perm[TENSOR_DIM];
    for (int i = 0; i < TENSOR_DIM; ++i) perm[i] = (TENSOR_DIM - 1) - i; // reverse
    cgrad_storage_layout l_orig;
    cgrad_storage_layout_copy(&l_orig, &l);
    cgrad_storage_layout_transpose(&l, perm, TENSOR_DIM);
    for (int i = 0; i < TENSOR_DIM; i++) {
        assert_int_equal(l.shape[i], l_orig.shape[perm[i]]);
        assert_int_equal(l.strides[i], l_orig.strides[perm[i]]);
    }
}

static void test_cgrad_storage_layout_is_contiguous(void **state) {
    (void)state;
    cgrad_storage_layout l;
    uint32_t shape[TENSOR_DIM];
    for (int i = 0; i < TENSOR_DIM; ++i) shape[i] = i + 2;
    cgrad_storage_layout_init(&l, shape, TENSOR_DIM);
    assert_int_equal(cgrad_storage_layout_is_contiguous(&l), 1);

    // Make non-contiguous by modifying strides
    if (TENSOR_DIM > 2) {
        l.strides[2] = 100;
        assert_int_equal(cgrad_storage_layout_is_contiguous(&l), 0);
    }

    // Edge case: shape with 1 in some dims
    for (int i = 0; i < TENSOR_DIM; ++i) shape[i] = (i < TENSOR_DIM - 2) ? 1 : (i + 2);
    cgrad_storage_layout_init(&l, shape, TENSOR_DIM);
    assert_int_equal(cgrad_storage_layout_is_contiguous(&l), 1);

    // Edge case: NULL pointer
    assert_int_equal(cgrad_storage_layout_is_contiguous(NULL), 0);
}

static void test_cgrad_storage_layout_transpose_duplicate_dim(void **state) {
    (void)state;
    cgrad_storage_layout l;
    uint32_t shape[TENSOR_DIM];
    for (int i = 0; i < TENSOR_DIM; ++i) shape[i] = i + 2;
    cgrad_storage_layout_init(&l, shape, TENSOR_DIM);
    // Duplicate dimension: 0 appears twice
    uint32_t perm[TENSOR_DIM];
    for (int i = 0; i < TENSOR_DIM; ++i) perm[i] = i;
    if (TENSOR_DIM > 1) perm[TENSOR_DIM - 1] = 0; // duplicate 0
    int err = cgrad_storage_layout_transpose(&l, perm, TENSOR_DIM);
    assert_int_equal(err, CGRAD_STORAGE_LAYOUT_ERR_DUPLICATE_DIM);
}

static void test_cgrad_storage_layout_is_regular(void **state) {
    (void)state;
    cgrad_storage_layout l;
    uint32_t shape[TENSOR_DIM];
    for (int i = 0; i < TENSOR_DIM; ++i) shape[i] = i + 2;
    cgrad_storage_layout_init(&l, shape, TENSOR_DIM);
    // Contiguous is regular
    assert_int_equal(cgrad_storage_layout_is_regular(&l), 1);

    // Make strides a constant multiple of contiguous (e.g., *2)
    for (int i = 0; i < TENSOR_DIM; i++) l.strides[i] *= 2;
    assert_int_equal(cgrad_storage_layout_is_regular(&l), 1);

    // Make strides irregular
    if (TENSOR_DIM > 2) {
        l.strides[2] = 7;
        assert_int_equal(cgrad_storage_layout_is_regular(&l), 0);
    }

    // Degenerate: shape with 1s
    for (int i = 0; i < TENSOR_DIM; ++i) shape[i] = (i < TENSOR_DIM - 2) ? 1 : (i + 2);
    cgrad_storage_layout_init(&l, shape, TENSOR_DIM);
    assert_int_equal(cgrad_storage_layout_is_regular(&l), 1);

    // NULL pointer
    assert_int_equal(cgrad_storage_layout_is_regular(NULL), 0);
}

static void test_cgrad_storage_layout_partial_shape_and_index(void **state) {
    (void)state;
    cgrad_storage_layout l;
    // ndim = 2
    uint32_t shape2[2] = {3, 4};
    assert_int_equal(cgrad_storage_layout_init(&l, shape2, 2), 0);
    // Should fill as {1,...,1,3,4}
    for (int i = 0; i < TENSOR_DIM - 2; ++i) assert_int_equal(l.shape[i], 1);
    assert_int_equal(l.shape[TENSOR_DIM - 2], 3);
    assert_int_equal(l.shape[TENSOR_DIM - 1], 4);

    // Valid index for last 2 dims
    uint32_t indices2[2] = {2, 3};
    size_t idx = 0;
    int err = cgrad_storage_layout_flat_index(&l, indices2, 2, &idx);
    assert_int_equal(err, CGRAD_SUCCESS);
    assert_int_equal(idx, 2 * l.strides[TENSOR_DIM - 2] + 3 * l.strides[TENSOR_DIM - 1]);

    // Out-of-bounds for last 2 dims
    uint32_t indices2_oob[2] = {3, 0};
    err = cgrad_storage_layout_flat_index(&l, indices2_oob, 2, &idx);
    assert_int_equal(err, CGRAD_STORAGE_LAYOUT_ERR_INDEX_OUT_OF_BOUNDS);

    // ndim = 1
    uint32_t shape1[1] = {7};
    assert_int_equal(cgrad_storage_layout_init(&l, shape1, 1), 0);
    for (int i = 0; i < TENSOR_DIM - 1; ++i) assert_int_equal(l.shape[i], 1);
    assert_int_equal(l.shape[TENSOR_DIM - 1], 7);
    uint32_t indices1[1] = {6};
    err = cgrad_storage_layout_flat_index(&l, indices1, 1, &idx);
    assert_int_equal(err, CGRAD_SUCCESS);
    assert_int_equal(idx, 6 * l.strides[TENSOR_DIM - 1]);
}

static void test_cgrad_storage_layout_partial_transpose(void **state) {
    (void)state;
    cgrad_storage_layout l;
    // ndim = 3
    uint32_t shape[3] = {2, 3, 4};
    assert_int_equal(cgrad_storage_layout_init(&l, shape, 3), 0);
    // shape should be {1,...,1,2,3,4}
    for (int i = 0; i < TENSOR_DIM - 3; ++i) assert_int_equal(l.shape[i], 1);
    assert_int_equal(l.shape[TENSOR_DIM - 3], 2);
    assert_int_equal(l.shape[TENSOR_DIM - 2], 3);
    assert_int_equal(l.shape[TENSOR_DIM - 1], 4);

    // Transpose last 2 dims (swap last two)
    uint32_t perm2[2] = {1, 0};
    assert_int_equal(cgrad_storage_layout_transpose(&l, perm2, 2), 0);
    // Now shape should be {1,...,1,2,4,3}
    for (int i = 0; i < TENSOR_DIM - 3; ++i) assert_int_equal(l.shape[i], 1);
    assert_int_equal(l.shape[TENSOR_DIM - 3], 2);
    assert_int_equal(l.shape[TENSOR_DIM - 2], 4);
    assert_int_equal(l.shape[TENSOR_DIM - 1], 3);

    // Transpose last 3 dims (permute 2,4,3)
    uint32_t perm3[3] = {2, 1, 0};
    assert_int_equal(cgrad_storage_layout_transpose(&l, perm3, 3), 0);
    // Now shape should be {1,...,1,3,4,2}
    for (int i = 0; i < TENSOR_DIM - 3; ++i) assert_int_equal(l.shape[i], 1);
    assert_int_equal(l.shape[TENSOR_DIM - 3], 3);
    assert_int_equal(l.shape[TENSOR_DIM - 2], 4);
    assert_int_equal(l.shape[TENSOR_DIM - 1], 2);
}

static void test_cgrad_storage_layout_reshape(void **state) {
    (void)state;
    cgrad_storage_layout l;
    // Regular contiguous layout, step = 1, ndim < TENSOR_DIM
    uint32_t shape[3] = {2, 3, 4};
    assert_int_equal(cgrad_storage_layout_init(&l, shape, 3), 0);
    int32_t new_shape1[3] = {4, 3, 2};
    assert_int_equal(cgrad_storage_layout_reshape(&l, new_shape1, 3), CGRAD_SUCCESS);
    for (int i = 0; i < TENSOR_DIM - 3; ++i) assert_int_equal(l.shape[i], 1);
    assert_int_equal(l.shape[TENSOR_DIM - 3], 4);
    assert_int_equal(l.shape[TENSOR_DIM - 2], 3);
    assert_int_equal(l.shape[TENSOR_DIM - 1], 2);

    // Strides should be contiguous (step=1)
    assert_int_equal(l.strides[TENSOR_DIM - 1], 1);
    assert_int_equal(l.strides[TENSOR_DIM - 2], 2);
    assert_int_equal(l.strides[TENSOR_DIM - 3], 6);

    // Reshape with -1 (infer)
    uint32_t shape2[3] = {2, 3, 4};
    assert_int_equal(cgrad_storage_layout_init(&l, shape2, 3), 0);
    int32_t new_shape2[3] = {4, -1, 2}; // -1 should be 3
    assert_int_equal(cgrad_storage_layout_reshape(&l, new_shape2, 3), CGRAD_SUCCESS);
    assert_int_equal(l.shape[TENSOR_DIM - 2], 3);

    // Reshape with -1 (infer) and reduce ndim
    uint32_t shape3[3] = {2, 3, 4};
    assert_int_equal(cgrad_storage_layout_init(&l, shape3, 3), 0);
    int32_t new_shape3[2] = {4, -1}; // -1 should be 6
    assert_int_equal(cgrad_storage_layout_reshape(&l, new_shape3, 2), CGRAD_SUCCESS);
    assert_int_equal(l.shape[TENSOR_DIM - 1], 6);

    // Error: product mismatch
    uint32_t shape4[3] = {2, 3, 4};
    assert_int_equal(cgrad_storage_layout_init(&l, shape4, 3), 0);
    int32_t bad_shape[3] = {2, 2, 2}; // 2*2*2=8 != 24
    assert_int_equal(cgrad_storage_layout_reshape(&l, bad_shape, 3), CGRAD_STORAGE_LAYOUT_ERR_RESHAPE_INVALID_SHAPE);

    // Error: more than one -1
    int32_t bad_shape2[3] = {-1, -1, 2};
    assert_int_equal(cgrad_storage_layout_reshape(&l, bad_shape2, 3), CGRAD_STORAGE_LAYOUT_ERR_RESHAPE_INVALID_SHAPE);

    // Error: non-regular layout
    uint32_t shape5[3] = {2, 3, 4};
    assert_int_equal(cgrad_storage_layout_init(&l, shape5, 3), 0);
    // Make strides irregular
    if (TENSOR_DIM > 2) l.strides[TENSOR_DIM - 2] = 7;
    int32_t new_shape5[3] = {4, 3, 2};
    assert_int_equal(cgrad_storage_layout_reshape(&l, new_shape5, 3), CGRAD_STORAGE_LAYOUT_ERR_NOT_REGULAR);

    // Stride scaling: regular but step != 1
    uint32_t shape6[3] = {2, 3, 4};
    assert_int_equal(cgrad_storage_layout_init(&l, shape6, 3), 0);
    // Make strides a constant multiple of contiguous (e.g., *2)
    for (int i = 0; i < TENSOR_DIM; i++) l.strides[i] *= 2;
    int32_t new_shape6[3] = {4, 3, 2};
    assert_int_equal(cgrad_storage_layout_reshape(&l, new_shape6, 3), CGRAD_SUCCESS);
    // Strides should be contiguous * 2
    assert_int_equal(l.strides[TENSOR_DIM - 1], 2);
    assert_int_equal(l.strides[TENSOR_DIM - 2], 4);
    assert_int_equal(l.strides[TENSOR_DIM - 3], 12);

    // Collapse all dims into a single dimension with -1
    uint32_t shape7[TENSOR_DIM];
    for (int i = 0; i < TENSOR_DIM; ++i) shape7[i] = i + 2;
    assert_int_equal(cgrad_storage_layout_init(&l, shape7, TENSOR_DIM), 0);
    int32_t collapse_all[1] = {-1};
    assert_int_equal(cgrad_storage_layout_reshape(&l, collapse_all, 1), CGRAD_SUCCESS);
    for (int i = 0; i < TENSOR_DIM-1; ++i) {
        assert_int_equal(l.shape[i], 1);
    }
    uint32_t expected_size = 1;
    for (int i = 0; i < TENSOR_DIM; ++i) expected_size *= shape7[i];
    assert_int_equal(l.shape[TENSOR_DIM-1], expected_size);
}

static void test_cgrad_storage_layout_reduce(void **state) {
    (void)state;
    cgrad_storage_layout l;
    
    // Test 1: Reduce single dimension
    uint32_t shape1[2] = {3, 4};
    assert_int_equal(cgrad_storage_layout_init(&l, shape1, 2), 0);
    // Reduce first of last 2 dims: mask={1,0}
    uint8_t mask1[2] = {1, 0};
    assert_int_equal(cgrad_storage_layout_reduce(&l, mask1, 2), CGRAD_SUCCESS);
    // Shape should be (1, 4) in last 2 dims
    assert_int_equal(l.shape[TENSOR_DIM - 2], 1);
    assert_int_equal(l.shape[TENSOR_DIM - 1], 4);
    // Size should be updated
    assert_int_equal(l.size, 4);
    // Strides should be recalculated
    assert_int_equal(l.strides[TENSOR_DIM - 1], 1);
    assert_int_equal(l.strides[TENSOR_DIM - 2], 4);
    
    // Test 2: Reduce second dimension
    uint32_t shape2[2] = {3, 4};
    assert_int_equal(cgrad_storage_layout_init(&l, shape2, 2), 0);
    uint8_t mask2[2] = {0, 1};
    assert_int_equal(cgrad_storage_layout_reduce(&l, mask2, 2), CGRAD_SUCCESS);
    // Shape should be (3, 1)
    assert_int_equal(l.shape[TENSOR_DIM - 2], 3);
    assert_int_equal(l.shape[TENSOR_DIM - 1], 1);
    assert_int_equal(l.size, 3);
    
    // Test 3: Reduce both dimensions
    uint32_t shape3[2] = {3, 4};
    assert_int_equal(cgrad_storage_layout_init(&l, shape3, 2), 0);
    uint8_t mask3[2] = {1, 1};
    assert_int_equal(cgrad_storage_layout_reduce(&l, mask3, 2), CGRAD_SUCCESS);
    assert_int_equal(l.shape[TENSOR_DIM - 2], 1);
    assert_int_equal(l.shape[TENSOR_DIM - 1], 1);
    assert_int_equal(l.size, 1);
    
    // Test 4: Reduce with 3 dimensions
    uint32_t shape4[3] = {2, 3, 4};
    assert_int_equal(cgrad_storage_layout_init(&l, shape4, 3), 0);
    uint8_t mask4[3] = {1, 0, 1};  // Reduce first and last of last 3 dims
    assert_int_equal(cgrad_storage_layout_reduce(&l, mask4, 3), CGRAD_SUCCESS);
    assert_int_equal(l.shape[TENSOR_DIM - 3], 1);
    assert_int_equal(l.shape[TENSOR_DIM - 2], 3);
    assert_int_equal(l.shape[TENSOR_DIM - 1], 1);
    assert_int_equal(l.size, 3);
    
    // Test 5: No reduction (all zeros in mask)
    uint32_t shape5[2] = {3, 4};
    assert_int_equal(cgrad_storage_layout_init(&l, shape5, 2), 0);
    uint8_t mask5[2] = {0, 0};
    assert_int_equal(cgrad_storage_layout_reduce(&l, mask5, 2), CGRAD_SUCCESS);
    assert_int_equal(l.shape[TENSOR_DIM - 2], 3);
    assert_int_equal(l.shape[TENSOR_DIM - 1], 4);
    assert_int_equal(l.size, 12);
    
    // Test 6: Error - NULL pointer
    assert_int_equal(cgrad_storage_layout_reduce(NULL, mask1, 2), CGRAD_STORAGE_LAYOUT_ERR_NULL_POINTER);
    assert_int_equal(cgrad_storage_layout_reduce(&l, NULL, 2), CGRAD_STORAGE_LAYOUT_ERR_NULL_POINTER);
    
    // Test 7: Error - invalid ndim
    assert_int_equal(cgrad_storage_layout_reduce(&l, mask1, -1), CGRAD_STORAGE_LAYOUT_ERR_SHAPE_MISMATCH);
    assert_int_equal(cgrad_storage_layout_reduce(&l, mask1, TENSOR_DIM + 1), CGRAD_STORAGE_LAYOUT_ERR_SHAPE_MISMATCH);
}

int run_cgrad_storage_layout_tests(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_cgrad_storage_layout_init_and_copy),
        cmocka_unit_test(test_cgrad_storage_layout_flat_index),
        cmocka_unit_test(test_cgrad_storage_layout_transpose),
        cmocka_unit_test(test_cgrad_storage_layout_is_contiguous),
        cmocka_unit_test(test_cgrad_storage_layout_transpose_duplicate_dim),
        cmocka_unit_test(test_cgrad_storage_layout_is_regular),
        cmocka_unit_test(test_cgrad_storage_layout_partial_shape_and_index),
        cmocka_unit_test(test_cgrad_storage_layout_partial_transpose),
        cmocka_unit_test(test_cgrad_storage_layout_reshape),
        cmocka_unit_test(test_cgrad_storage_layout_reduce),
    };
    return cmocka_run_group_tests_name("cgrad_storage_layout", tests, NULL, NULL);
}

#ifndef TEST_ALL_MAIN
int main(void) {
    return run_cgrad_storage_layout_tests();
}
#endif
