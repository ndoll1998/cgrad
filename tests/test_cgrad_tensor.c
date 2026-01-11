#include <cmocka.h>
#include "cgrad_tensor.h"
#include "backends/cgrad_tensor_f32_cpu.h"
#include "cgrad_errors.h"
#include "cgrad_tensor_registry.h"
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

static void test_cgrad_tensor_init_and_free(void **state) {
    cgrad_tensor t;
    uint32_t shape[TENSOR_DIM] = {2, 3, 4, 5};
    assert_int_equal(cgrad_tensor_init(&t, shape, 4, CGRAD_BACKEND_F32_CPU), CGRAD_SUCCESS);
    assert_non_null(t.data);
    cgrad_tensor_free(&t);
    assert_null(t.data);
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
    cgrad_tensor_f32_cpu* handle = (cgrad_tensor_f32_cpu*)t.data;
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
    cgrad_tensor_layout* dst_layout = dst.backend->tensor_get_layout(dst.data);
    assert_int_equal(dst_layout->shape[TENSOR_DIM-2], 10);
    assert_int_equal(dst_layout->shape[TENSOR_DIM-1], 12);

    cgrad_tensor_free(&src);
    cgrad_tensor_free(&dst);

    // Ensure registry is empty
    assert_int_equal(cgrad_tensor_registry_count(), 0);
}

static int mock_tensor_free_count = 0;
static void* mock_alloc_tensor_handle(void) {
    return malloc(1);
}
static void mock_tensor_free(void* handle) {
    // Do not free here; freeing is handled after test to avoid double free
    mock_tensor_free_count++;
}

static void test_cgrad_tensor_registry_root_freed_only_after_all_children(void **state) {
    (void)state;
    // Setup mock backend
    cgrad_backend mock_backend = {0};
    mock_backend.alloc_tensor_handle = mock_alloc_tensor_handle;
    mock_backend.tensor_free = mock_tensor_free;
    mock_backend.tensor_shallow_copy = NULL; // Not needed for this test
    mock_backend.tensor_init = NULL; // Not needed for this test

    // Create root tensor
    cgrad_tensor root = {0};
    root.backend = &mock_backend;
    root.data = malloc(1);

    // Register root
    cgrad_tensor_registry_register(&root, NULL);

    // Create two children (simulate shallow copies)
    cgrad_tensor child1 = {0}, child2 = {0};
    child1.backend = &mock_backend;
    child1.data = malloc(1);
    cgrad_tensor_registry_register(&child1, &root);

    child2.backend = &mock_backend;
    child2.data = malloc(1);
    cgrad_tensor_registry_register(&child2, &root);

    mock_tensor_free_count = 0;

    // Free one child, root should not be freed
    cgrad_tensor_free(&child1);
    assert_int_equal(mock_tensor_free_count, 0);

    // Free the other child, root should not be freed
    cgrad_tensor_free(&child2);
    assert_int_equal(mock_tensor_free_count, 0);

    // Free the root, now root handle should be freed
    cgrad_tensor_free(&root);
    assert_int_equal(mock_tensor_free_count, 1);

    // Manually free all handles to avoid memory leaks
    free(root.data);
    free(child1.data);
    free(child2.data);
}

int run_cgrad_tensor_tests(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_cgrad_tensor_init_and_free),
        cmocka_unit_test(test_cgrad_tensor_init_errors),
        cmocka_unit_test(test_cgrad_tensor_fill),
        cmocka_unit_test(test_cgrad_tensor_reshape),
        cmocka_unit_test(test_cgrad_tensor_registry_root_freed_only_after_all_children),
    };
    return _cmocka_run_group_tests("cgrad_tensor", tests, sizeof(tests)/sizeof(tests[0]), NULL, NULL);
}

#ifndef TEST_ALL_MAIN
int main(void) {
    return run_cgrad_tensor_tests();
}
#endif
