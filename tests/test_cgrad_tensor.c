#include <cmocka.h>
#include "cgrad_tensor.h"
#include "backends/cgrad_tensor_f32_cpu.h"
#include "cgrad_errors.h"
#include <stdint.h>
#include <string.h>
#include <stdio.h>

static void test_cgrad_tensor_init_and_free(void **state) {
    cgrad_tensor t;
    uint32_t shape[TENSOR_DIM] = {2, 3, 4, 5};
    assert_int_equal(cgrad_tensor_init(&t, shape, 4, CGRAD_BACKEND_F32_CPU), CGRAD_SUCCESS);
    assert_non_null(t.handle);
    cgrad_tensor_free(&t);
    assert_null(t.handle);
}

static void test_cgrad_tensor_init_errors(void **state) {
    uint32_t shape[TENSOR_DIM] = {2, 3, 4, 5};
    // Null tensor pointer
    assert_int_equal(cgrad_tensor_init(NULL, shape, 4, CGRAD_BACKEND_F32_CPU), CGRAD_TENSOR_ERR_NULL_POINTER);
    // Null shape pointer
    cgrad_tensor t;
    assert_int_equal(cgrad_tensor_init(&t, NULL, 4, CGRAD_BACKEND_F32_CPU), CGRAD_TENSOR_ERR_NULL_POINTER);
}

static void test_cgrad_tensor_fill(void **state) {
    cgrad_tensor t;
    uint32_t shape[TENSOR_DIM] = {2, 3, 4, 5};
    float fill_value = 7.5f;
    assert_int_equal(cgrad_tensor_init(&t, shape, 4, CGRAD_BACKEND_F32_CPU), CGRAD_SUCCESS);
    assert_int_equal(cgrad_tensor_fill(&t, fill_value), CGRAD_SUCCESS);
    cgrad_tensor_f32_cpu* handle = (cgrad_tensor_f32_cpu*)t.handle;
    for (int i = 0; i < handle->layout.size; i++) {
        assert_float_equal(handle->data[i], fill_value, 1e-6);
    }
    cgrad_tensor_free(&t);
}

static void test_cgrad_tensor_reshape(void **state) {
    cgrad_tensor src, dst = {0};
    uint32_t shape[TENSOR_DIM] = {2, 3, 4, 5};
    assert_int_equal(cgrad_tensor_init(&src, shape, 4, CGRAD_BACKEND_F32_CPU), CGRAD_SUCCESS);
    assert_int_equal(cgrad_tensor_fill(&src, 42.0f), CGRAD_SUCCESS);

    // Reshape
    int32_t new_shape[2] = {10, 12};
    assert_int_equal(cgrad_tensor_reshape(&src, &dst, new_shape, 2), CGRAD_SUCCESS);

    // Check dst layout shape
    cgrad_tensor_layout* dst_layout = dst.backend->tensor_get_layout(dst.handle);
    assert_int_equal(dst_layout->shape[TENSOR_DIM-2], 10);
    assert_int_equal(dst_layout->shape[TENSOR_DIM-1], 12);

    cgrad_tensor_free(&src);
    // cgrad_tensor_free(&dst);
}

int run_cgrad_tensor_tests(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_cgrad_tensor_init_and_free),
        cmocka_unit_test(test_cgrad_tensor_init_errors),
        cmocka_unit_test(test_cgrad_tensor_fill),
        cmocka_unit_test(test_cgrad_tensor_reshape),
    };
    return _cmocka_run_group_tests("cgrad_tensor", tests, sizeof(tests)/sizeof(tests[0]), NULL, NULL);
}

#ifndef TEST_ALL_MAIN
int main(void) {
    return run_cgrad_tensor_tests();
}
#endif
