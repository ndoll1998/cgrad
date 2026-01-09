#include <cmocka.h>
#include "cgrad_layout.h"
#include <stdint.h>
#include <string.h>

#define MAX_TENSOR_DIM 4

static void test_cgrad_tensor_layout_init_and_copy(void **state) {
    (void)state;
    cgrad_tensor_layout l1, l2;
    uint32_t shape[MAX_TENSOR_DIM] = {2, 3, 4, 5};
    assert_int_equal(cgrad_tensor_layout_init(&l1, shape), 0);
    cgrad_tensor_layout_copy(&l2, &l1);
    assert_memory_equal(&l1, &l2, sizeof(cgrad_tensor_layout));
}

static void test_cgrad_tensor_flat_index(void **state) {
    (void)state;
    uint32_t strides[MAX_TENSOR_DIM] = {60, 20, 5, 1};
    uint32_t indices[MAX_TENSOR_DIM] = {1, 2, 3, 4};
    size_t idx = cgrad_tensor_flat_index(indices, strides);
    assert_int_equal(idx, 1*60 + 2*20 + 3*5 + 4*1);
}

static void test_cgrad_tensor_layout_transpose(void **state) {
    (void)state;
    cgrad_tensor_layout l;
    uint32_t shape[MAX_TENSOR_DIM] = {2, 3, 4, 5};
    cgrad_tensor_layout_init(&l, shape);
    uint32_t perm[MAX_TENSOR_DIM] = {0, 2, 1, 3};
    cgrad_tensor_layout l_orig;
    cgrad_tensor_layout_copy(&l_orig, &l);
    cgrad_tensor_layout_transpose(&l, perm);
    for (int i = 0; i < MAX_TENSOR_DIM; i++) {
        assert_int_equal(l.shape[i], l_orig.shape[perm[i]]);
        assert_int_equal(l.strides[i], l_orig.strides[perm[i]]);
    }
}

int run_cgrad_layout_tests(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_cgrad_tensor_layout_init_and_copy),
        cmocka_unit_test(test_cgrad_tensor_flat_index),
        cmocka_unit_test(test_cgrad_tensor_layout_transpose),
    };
    return _cmocka_run_group_tests("cgrad_layout", tests, sizeof(tests)/sizeof(tests[0]), NULL, NULL);
}

#ifndef TEST_ALL_MAIN
int main(void) {
    return run_cgrad_layout_tests();
}
#endif
