#include <cmocka.h>
#include "storage/cgrad_storage.h"
#include "storage/backends/cgrad_storage_f32_cpu.h"
#include "cgrad_errors.h"
#include "storage/cgrad_storage_registry.h"
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// Forward declare getter - it's internal to storage module
extern cgrad_storage_registry* get_global_registry(void);

static void test_cgrad_storage_init_and_free(void **state) {
    cgrad_storage t;
    uint32_t shape[TENSOR_DIM] = {2, 3, 4, 5};
    assert_int_equal(cgrad_storage_init(&t, shape, 4, CGRAD_STORAGE_BACKEND_F32_CPU), CGRAD_SUCCESS);
    assert_non_null(t.data);
    cgrad_storage_free(&t);
    assert_null(t.data);
}

static void test_cgrad_storage_init_errors(void **state) {
    uint32_t shape[TENSOR_DIM] = {2, 3, 4, 5};
    // Null tensor pointer
    assert_int_equal(cgrad_storage_init(NULL, shape, 4, CGRAD_STORAGE_BACKEND_F32_CPU), CGRAD_ERR_NULL_POINTER);
    // Null shape pointer
    cgrad_storage t;
    assert_int_equal(cgrad_storage_init(&t, NULL, 4, CGRAD_STORAGE_BACKEND_F32_CPU), CGRAD_ERR_NULL_POINTER);
}

static void test_cgrad_storage_fill(void **state) {
    cgrad_storage t;
    uint32_t shape[TENSOR_DIM] = {2, 3, 4, 5};
    float fill_value = 7.5f;
    assert_int_equal(cgrad_storage_init(&t, shape, 4, CGRAD_STORAGE_BACKEND_F32_CPU), CGRAD_SUCCESS);
    assert_int_equal(cgrad_storage_fill(&t, fill_value), CGRAD_SUCCESS);
    cgrad_storage_f32_cpu* handle = (cgrad_storage_f32_cpu*)t.data;
    for (int i = 0; i < handle->layout.size; i++) {
        assert_float_equal(handle->data[i], fill_value, 1e-6);
    }
    cgrad_storage_free(&t);
}

static void test_cgrad_storage_contiguous(void **state) {
    cgrad_storage src = {0}, dst = {0};
    uint32_t shape[TENSOR_DIM] = {2, 3, 4, 5};
    assert_int_equal(cgrad_storage_init(&src, shape, 4, CGRAD_STORAGE_BACKEND_F32_CPU), CGRAD_SUCCESS);
    assert_int_equal(cgrad_storage_fill(&src, 3.14f), CGRAD_SUCCESS);

    // Make a contiguous copy
    assert_int_equal(cgrad_storage_contiguous(&src, &dst), CGRAD_SUCCESS);

    // Check that the data matches
    cgrad_storage_f32_cpu* src_handle = (cgrad_storage_f32_cpu*)src.data;
    cgrad_storage_f32_cpu* dst_handle = (cgrad_storage_f32_cpu*)dst.data;
    assert_int_equal(src_handle->layout.size, dst_handle->layout.size);
    for (int i = 0; i < src_handle->layout.size; ++i) {
        assert_float_equal(src_handle->data[i], dst_handle->data[i], 1e-6);
    }

    cgrad_storage_free(&src);
    cgrad_storage_free(&dst);
}

static void test_cgrad_storage_reshape(void **state) {
    cgrad_storage src, dst = {0};
    uint32_t shape[TENSOR_DIM] = {2, 3, 4, 5};
    assert_int_equal(cgrad_storage_init(&src, shape, 4, CGRAD_STORAGE_BACKEND_F32_CPU), CGRAD_SUCCESS);
    assert_int_equal(cgrad_storage_fill(&src, 42.0f), CGRAD_SUCCESS);

    // Reshape
    int32_t new_shape[2] = {10, 12};
    assert_int_equal(cgrad_storage_reshape(&src, &dst, new_shape, 2), CGRAD_SUCCESS);

    // Check dst layout shape
    cgrad_storage_layout* dst_layout = dst.backend->storage_get_layout(dst.data);
    assert_int_equal(dst_layout->shape[TENSOR_DIM-2], 10);
    assert_int_equal(dst_layout->shape[TENSOR_DIM-1], 12);

    cgrad_storage_free(&src);
    cgrad_storage_free(&dst);

    // Ensure registry is empty
    cgrad_storage_registry* registry = get_global_registry();
    if (registry) {
        assert_int_equal(cgrad_storage_registry_count(registry), 0);
    }
}

static int mock_storage_free_count = 0;
static void mock_storage_free(void* handle) {
    // Do not free here; freeing is handled after test to avoid double free
    mock_storage_free_count++;
}

static void test_cgrad_storage_registry_root_freed_only_after_all_children(void **state) {
    (void)state;
    cgrad_storage_registry* registry = get_global_registry();
    assert_non_null(registry);
    
    // Setup mock backend
    cgrad_storage_backend mock_backend = {0};
    mock_backend.storage_free = mock_storage_free;
    mock_backend.storage_shallow_copy = NULL; // Not needed for this test
    mock_backend.storage_init = NULL; // Not needed for this test

    // Create root tensor
    cgrad_storage root = {0};

    // Register root
    cgrad_storage_registry_register(registry, &root, NULL);

    // Create two children (simulate shallow copies)
    cgrad_storage child1 = {0}, child2 = {0};
    child1.backend = &mock_backend;
    child1.data = malloc(1);
    cgrad_storage_registry_register(registry, &child1, &root);

    child2.backend = &mock_backend;
    child2.data = malloc(1);
    cgrad_storage_registry_register(registry, &child2, &root);

    mock_storage_free_count = 0;

    // Free one child
    cgrad_storage_free(&child1);

    // Free the other child
    cgrad_storage_free(&child2);

    // Free the root, now root handle should be freed
    cgrad_storage_free(&root);
    assert_int_equal(mock_storage_free_count, 1);

    // Manually free all handles to avoid memory leaks
    free(child1.data);
    free(child2.data);
    
    cgrad_storage_cleanup_global_registry();
}

static void test_cgrad_storage_gemm_write_to_existing_tensor(void **state) {
    (void)state;
    // Create two input tensors for GEMM: a (2x3) and b (3x4)
    cgrad_storage a = {0}, b = {0};
    uint32_t a_shape[TENSOR_DIM] = {1,1,1,1,1,1,2,3};
    uint32_t b_shape[TENSOR_DIM] = {1,1,1,1,1,1,3,4};
    assert_int_equal(cgrad_storage_init(&a, a_shape, 8, CGRAD_STORAGE_BACKEND_F32_CPU), CGRAD_SUCCESS);
    assert_int_equal(cgrad_storage_init(&b, b_shape, 8, CGRAD_STORAGE_BACKEND_F32_CPU), CGRAD_SUCCESS);
    
    // Fill with test values
    assert_int_equal(cgrad_storage_fill(&a, 1.0f), CGRAD_SUCCESS);
    assert_int_equal(cgrad_storage_fill(&b, 2.0f), CGRAD_SUCCESS);
    
    // Test 1: GEMM with uninitialized result tensor (should work)
    cgrad_storage r1 = {0};
    assert_int_equal(cgrad_storage_gemm(1.0f, &a, &b, 0.0f, &r1), CGRAD_SUCCESS);
    cgrad_storage_free(&r1);
    
    // Test 2: GEMM with pre-initialized result tensor with matching shape (should work)
    cgrad_storage r2 = {0};
    uint32_t r_shape[TENSOR_DIM] = {1,1,1,1,1,1,2,4};
    assert_int_equal(cgrad_storage_init(&r2, r_shape, 8, CGRAD_STORAGE_BACKEND_F32_CPU), CGRAD_SUCCESS);
    assert_int_equal(cgrad_storage_gemm(1.0f, &a, &b, 0.0f, &r2), CGRAD_SUCCESS);
    cgrad_storage_free(&r2);
    
    // Test 3: GEMM with pre-initialized result tensor with mismatched shape (should fail with SHAPE_MISMATCH)
    cgrad_storage r3 = {0};
    uint32_t wrong_shape[TENSOR_DIM] = {1,1,1,1,1,1,3,3};  // Wrong shape
    assert_int_equal(cgrad_storage_init(&r3, wrong_shape, 8, CGRAD_STORAGE_BACKEND_F32_CPU), CGRAD_SUCCESS);
    int err = cgrad_storage_gemm(1.0f, &a, &b, 0.0f, &r3);
    assert_int_equal(err, CGRAD_STORAGE_ERR_SHAPE_MISMATCH);
    cgrad_storage_free(&r3);
    
    cgrad_storage_free(&a);
    cgrad_storage_free(&b);
}

static void test_cgrad_storage_sum(void **state) {
    (void)state;
    // Create a 2x3 tensor with values 1,2,3,4,5,6
    cgrad_storage t = {0};
    uint32_t shape[TENSOR_DIM] = {1,1,1,1,1,1,2,3};
    assert_int_equal(cgrad_storage_init(&t, shape, 8, CGRAD_STORAGE_BACKEND_F32_CPU), CGRAD_SUCCESS);

    // Fill with values
    float vals[6] = {1,2,3,4,5,6};
    for (uint32_t i = 0; i < 2; ++i) {
        for (uint32_t j = 0; j < 3; ++j) {
            uint32_t idx[TENSOR_DIM] = {0,0,0,0,0,0,i,j};
            assert_int_equal(t.backend->storage_set(t.data, idx, TENSOR_DIM, vals[i*3+j]), CGRAD_SUCCESS);
        }
    }

    // Sum over axis 1 (columns): mask = [0,1] (right-aligned)
    uint8_t mask1[2] = {0,1};
    cgrad_storage r1 = {0};
    assert_int_equal(cgrad_storage_sum(&t, mask1, 2, &r1), CGRAD_SUCCESS);
    // Should be shape [2,1], values [6,15]
    float expected1[2] = {6,15};
    for (uint32_t i = 0; i < 2; ++i) {
        uint32_t idx[TENSOR_DIM] = {0,0,0,0,0,0,i,0};
        float v = 0;
        assert_int_equal(r1.backend->storage_get(r1.data, idx, TENSOR_DIM, &v), CGRAD_SUCCESS);
        assert_float_equal(v, expected1[i], 1e-6);
    }
    cgrad_storage_free(&r1);

    // Sum over axis 0 (rows): mask = [1,0] (right-aligned)
    uint8_t mask2[2] = {1,0};
    cgrad_storage r2 = {0};
    assert_int_equal(cgrad_storage_sum(&t, mask2, 2, &r2), CGRAD_SUCCESS);
    // Should be shape [1,3], values [5,7,9]
    float expected2[3] = {5,7,9};
    for (uint32_t j = 0; j < 3; ++j) {
        uint32_t idx[TENSOR_DIM] = {0,0,0,0,0,0,0,j};
        float v = 0;
        assert_int_equal(r2.backend->storage_get(r2.data, idx, TENSOR_DIM, &v), CGRAD_SUCCESS);
        assert_float_equal(v, expected2[j], 1e-6);
    }
    cgrad_storage_free(&r2);

    // Sum over all axes: mask = [1,1]
    uint8_t mask3[2] = {1,1};
    cgrad_storage r3 = {0};
    assert_int_equal(cgrad_storage_sum(&t, mask3, 2, &r3), CGRAD_SUCCESS);
    // Should be scalar [21]
    float v3 = 0;
    uint32_t idx3[TENSOR_DIM] = {0,0,0,0,0,0,0,0};
    assert_int_equal(r3.backend->storage_get(r3.data, idx3, TENSOR_DIM, &v3), CGRAD_SUCCESS);
    assert_float_equal(v3, 21.0f, 1e-6);
    cgrad_storage_free(&r3);

    cgrad_storage_free(&t);
}

int run_cgrad_storage_tests(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_cgrad_storage_init_and_free),
        cmocka_unit_test(test_cgrad_storage_init_errors),
        cmocka_unit_test(test_cgrad_storage_fill),
        cmocka_unit_test(test_cgrad_storage_contiguous),
        cmocka_unit_test(test_cgrad_storage_reshape),
        cmocka_unit_test(test_cgrad_storage_registry_root_freed_only_after_all_children),
        cmocka_unit_test(test_cgrad_storage_gemm_write_to_existing_tensor),
        cmocka_unit_test(test_cgrad_storage_sum),
    };
    return cmocka_run_group_tests_name("cgrad_storage", tests, NULL, NULL);
}

#ifndef TEST_ALL_MAIN
int main(void) {
    return run_cgrad_storage_tests();
}
#endif
