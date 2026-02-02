#include <cmocka.h>
#include "cgrad.h"
#include "storage/cgrad_storage.h"
#include "storage/cgrad_storage_layout.h"
#include "cgrad_status.h"
#include "storage/cgrad_storage_registry.h"
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// ============================================================================
// Setup and Teardown
// ============================================================================

static int storage_setup_test(void **state) {
    (void) state;
    cgrad_init();
    return 0;
}

static int storage_teardown_test(void **state) {
    (void) state;
    cgrad_cleanup();
    return 0;
}

static void test_cgrad_storage_init_and_free(void **state) {
    cgrad_storage t;
    uint32_t shape[TENSOR_DIM] = {2, 3, 4, 5};
    assert_int_equal(cgrad_storage_init(&t, shape, 4, "cpu_f32"), CGRAD_SUCCESS);
    assert_non_null(t.data);
    cgrad_storage_free(&t);
    assert_null(t.data);
}

static void test_cgrad_storage_init_errors(void **state) {
    uint32_t shape[TENSOR_DIM] = {2, 3, 4, 5};
    // Null tensor pointer
    assert_int_equal(cgrad_storage_init(NULL, shape, 4, "cpu_f32"), CGRAD_ERR_NULL_POINTER);
    // Null shape pointer
    cgrad_storage t;
    assert_int_equal(cgrad_storage_init(&t, NULL, 4, "cpu_f32"), CGRAD_ERR_NULL_POINTER);
}

static void test_cgrad_storage_fill(void **state) {
    cgrad_storage t;
    uint32_t shape[] = {2, 3, 4, 5};
    float fill_value = 7.5f;
    assert_int_equal(cgrad_storage_init(&t, shape, 4, "cpu_f32"), CGRAD_SUCCESS);
    assert_int_equal(cgrad_storage_fill(&t, fill_value), CGRAD_SUCCESS);
    
    // Use high-level API to check a few values
    uint32_t idx1[TENSOR_DIM] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t idx2[TENSOR_DIM] = {0, 0, 0, 0, 1, 2, 3, 4};
    float value;
    assert_int_equal(cgrad_storage_get(&t, idx1, TENSOR_DIM, &value), CGRAD_SUCCESS);
    assert_float_equal(value, fill_value, 1e-6);
    assert_int_equal(cgrad_storage_get(&t, idx2, TENSOR_DIM, &value), CGRAD_SUCCESS);
    assert_float_equal(value, fill_value, 1e-6);
    
    cgrad_storage_free(&t);
}

static void test_cgrad_storage_contiguous(void **state) {
    cgrad_storage src = {0}, dst = {0};
    uint32_t shape[] = {2, 3, 4, 5};
    assert_int_equal(cgrad_storage_init(&src, shape, 4, "cpu_f32"), CGRAD_SUCCESS);
    assert_int_equal(cgrad_storage_fill(&src, 3.14f), CGRAD_SUCCESS);

    // Make a contiguous copy
    assert_int_equal(cgrad_storage_contiguous(&src, &dst), CGRAD_SUCCESS);

    // Check that the data matches using high-level API
    cgrad_storage_layout* src_layout = src.backend->storage_get_layout(src.data);
    cgrad_storage_layout* dst_layout = dst.backend->storage_get_layout(dst.data);
    assert_int_equal(src_layout->size, dst_layout->size);
    
    // Check a few sample values
    uint32_t idx1[TENSOR_DIM] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t idx2[TENSOR_DIM] = {0, 0, 0, 0, 1, 2, 3, 4};
    float src_value, dst_value;
    
    assert_int_equal(cgrad_storage_get(&src, idx1, TENSOR_DIM, &src_value), CGRAD_SUCCESS);
    assert_int_equal(cgrad_storage_get(&dst, idx1, TENSOR_DIM, &dst_value), CGRAD_SUCCESS);
    assert_float_equal(src_value, dst_value, 1e-6);
    
    assert_int_equal(cgrad_storage_get(&src, idx2, TENSOR_DIM, &src_value), CGRAD_SUCCESS);
    assert_int_equal(cgrad_storage_get(&dst, idx2, TENSOR_DIM, &dst_value), CGRAD_SUCCESS);
    assert_float_equal(src_value, dst_value, 1e-6);

    cgrad_storage_free(&src);
    cgrad_storage_free(&dst);
}

static void test_cgrad_storage_reshape(void **state) {
    cgrad_storage src, dst = {0};
    uint32_t shape[] = {2, 3, 4, 5};
    assert_int_equal(cgrad_storage_init(&src, shape, 4, "cpu_f32"), CGRAD_SUCCESS);
    assert_int_equal(cgrad_storage_fill(&src, 42.0f), CGRAD_SUCCESS);

    // Reshape to same total size: 2*3*4*5 = 120 = 10*12
    int32_t new_shape[2] = {10, 12};
    assert_int_equal(cgrad_storage_reshape(&src, &dst, new_shape, 2), CGRAD_SUCCESS);

    // Check dst layout shape
    cgrad_storage_layout* dst_layout = dst.backend->storage_get_layout(dst.data);
    assert_int_equal(dst_layout->shape[TENSOR_DIM-2], 10);
    assert_int_equal(dst_layout->shape[TENSOR_DIM-1], 12);

    cgrad_storage_free(&src);
    cgrad_storage_free(&dst);

    // try to free the global storage registry
    // this does nothing if the registry is non-empty
    cgrad_storage_registry_free();
}

static void test_cgrad_storage_registry_root_freed_only_after_all_children(void **state) {
    (void)state;
    
    // Create a root storage using the high-level API
    cgrad_storage root = {0};
    uint32_t shape[] = {2, 3};
    assert_int_equal(cgrad_storage_init(&root, shape, 2, "cpu_f32"), CGRAD_SUCCESS);
    assert_int_equal(cgrad_storage_fill(&root, 1.0f), CGRAD_SUCCESS);
    
    // Create two children using view with NULL layout (equivalent to shallow copy)
    cgrad_storage child1 = {0}, child2 = {0};
    assert_int_equal(cgrad_storage_view(&root, &child1, NULL), CGRAD_SUCCESS);
    assert_int_equal(cgrad_storage_view(&root, &child2, NULL), CGRAD_SUCCESS);
    
    // Verify all three have valid data handles
    assert_non_null(root.data);
    assert_non_null(child1.data);
    assert_non_null(child2.data);
    
    // Free one child - root data should still be accessible
    assert_int_equal(cgrad_storage_free(&child1), CGRAD_SUCCESS);
    assert_null(child1.data); // child1 handle should be freed
    
    // Verify root data is still valid by reading a value
    float value;
    uint32_t idx[TENSOR_DIM] = {0,0,0,0,0,0,0,0};
    assert_int_equal(cgrad_storage_get(&root, idx, TENSOR_DIM, &value), CGRAD_SUCCESS);
    assert_float_equal(value, 1.0f, 1e-6);
    
    // Verify child2 data is still valid
    assert_int_equal(cgrad_storage_get(&child2, idx, TENSOR_DIM, &value), CGRAD_SUCCESS);
    assert_float_equal(value, 1.0f, 1e-6);
    
    // Free the other child - root data should still be accessible
    assert_int_equal(cgrad_storage_free(&child2), CGRAD_SUCCESS);
    assert_null(child2.data); // child2 handle should be freed
    
    // Verify root data is still valid
    assert_int_equal(cgrad_storage_get(&root, idx, TENSOR_DIM, &value), CGRAD_SUCCESS);
    assert_float_equal(value, 1.0f, 1e-6);
    
    // Free the root - now the actual data should be freed
    assert_int_equal(cgrad_storage_free(&root), CGRAD_SUCCESS);
    assert_null(root.data); // root handle should be freed
}

static void test_cgrad_storage_gemm_write_to_existing_tensor(void **state) {
    (void)state;
    // Create two input tensors for GEMM: a (2x3) and b (3x4)
    cgrad_storage a = {0}, b = {0};
    uint32_t a_shape[TENSOR_DIM] = {1,1,1,1,1,1,2,3};
    uint32_t b_shape[TENSOR_DIM] = {1,1,1,1,1,1,3,4};
    assert_int_equal(cgrad_storage_init(&a, a_shape, 8, "cpu_f32"), CGRAD_SUCCESS);
    assert_int_equal(cgrad_storage_init(&b, b_shape, 8, "cpu_f32"), CGRAD_SUCCESS);
    
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
    assert_int_equal(cgrad_storage_init(&r2, r_shape, 8, "cpu_f32"), CGRAD_SUCCESS);
    assert_int_equal(cgrad_storage_gemm(1.0f, &a, &b, 0.0f, &r2), CGRAD_SUCCESS);
    cgrad_storage_free(&r2);
    
    // Test 3: GEMM with pre-initialized result tensor with mismatched shape (should fail with SHAPE_MISMATCH)
    cgrad_storage r3 = {0};
    uint32_t wrong_shape[TENSOR_DIM] = {1,1,1,1,1,1,3,3};  // Wrong shape
    assert_int_equal(cgrad_storage_init(&r3, wrong_shape, 8, "cpu_f32"), CGRAD_SUCCESS);
    int err = cgrad_storage_gemm(1.0f, &a, &b, 0.0f, &r3);
    assert_int_equal(err, CGRAD_ERR_STORAGE_SHAPE_MISMATCH);
    cgrad_storage_free(&r3);
    
    cgrad_storage_free(&a);
    cgrad_storage_free(&b);
}

static void test_cgrad_storage_reduce(void **state) {
    (void)state;
    // Create a 2x3 tensor with values 1,2,3,4,5,6
    cgrad_storage t = {0};
    uint32_t shape[TENSOR_DIM] = {1,1,1,1,1,1,2,3};
    assert_int_equal(cgrad_storage_init(&t, shape, 8, "cpu_f32"), CGRAD_SUCCESS);

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
    assert_int_equal(cgrad_storage_reduce(1.0f, &t, mask1, 2, 0.0f, &r1), CGRAD_SUCCESS);
    // Should be shape [2,1], values [6,15]
    float expected1[2] = {6,15};
    for (uint32_t i = 0; i < 2; ++i) {
        uint32_t idx[TENSOR_DIM] = {0,0,0,0,0,0,i,0};
        float v = 0;
        assert_int_equal(cgrad_storage_get(&r1, idx, TENSOR_DIM, &v), CGRAD_SUCCESS);
        assert_float_equal(v, expected1[i], 1e-6);
    }
    cgrad_storage_free(&r1);

    // Sum over axis 0 (rows): mask = [1,0] (right-aligned)
    uint8_t mask2[2] = {1,0};
    cgrad_storage r2 = {0};
    assert_int_equal(cgrad_storage_reduce(1.0f, &t, mask2, 2, 0.0f, &r2), CGRAD_SUCCESS);
    // Should be shape [1,3], values [5,7,9]
    float expected2[3] = {5,7,9};
    for (uint32_t j = 0; j < 3; ++j) {
        uint32_t idx[TENSOR_DIM] = {0,0,0,0,0,0,0,j};
        float v = 0;
        assert_int_equal(cgrad_storage_get(&r2, idx, TENSOR_DIM, &v), CGRAD_SUCCESS);
        assert_float_equal(v, expected2[j], 1e-6);
    }
    cgrad_storage_free(&r2);

    // Sum over all axes: mask = [1,1]
    uint8_t mask3[2] = {1,1};
    cgrad_storage r3 = {0};
    assert_int_equal(cgrad_storage_reduce(1.0f, &t, mask3, 2, 0.0f, &r3), CGRAD_SUCCESS);
    // Should be scalar [21]
    float v3 = 0;
    uint32_t idx3[TENSOR_DIM] = {0,0,0,0,0,0,0,0};
    assert_int_equal(cgrad_storage_get(&r3, idx3, TENSOR_DIM, &v3), CGRAD_SUCCESS);
    assert_float_equal(v3, 21.0f, 1e-6);
    cgrad_storage_free(&r3);

    cgrad_storage_free(&t);
}

static void test_cgrad_storage_view_with_null_layout(void **state) {
    (void)state;
    // View with NULL layout should behave like shallow_copy
    cgrad_storage src, view = {0};
    uint32_t shape[] = {2, 3};
    assert_int_equal(cgrad_storage_init(&src, shape, 2, "cpu_f32"), CGRAD_SUCCESS);
    assert_int_equal(cgrad_storage_fill(&src, 5.0f), CGRAD_SUCCESS);
    
    // Create view with NULL layout
    assert_int_equal(cgrad_storage_view(&src, &view, NULL), CGRAD_SUCCESS);
    
    // Check that view has same shape as source
    cgrad_storage_layout* src_layout = src.backend->storage_get_layout(src.data);
    cgrad_storage_layout* view_layout = view.backend->storage_get_layout(view.data);
    assert_int_equal(src_layout->size, view_layout->size);
    for (int i = 0; i < TENSOR_DIM; i++) {
        assert_int_equal(src_layout->shape[i], view_layout->shape[i]);
    }
    
    // Check that data is shared
    uint32_t idx[TENSOR_DIM] = {0,0,0,0,0,0,0,0};
    float src_val, view_val;
    assert_int_equal(cgrad_storage_get(&src, idx, TENSOR_DIM, &src_val), CGRAD_SUCCESS);
    assert_int_equal(cgrad_storage_get(&view, idx, TENSOR_DIM, &view_val), CGRAD_SUCCESS);
    assert_float_equal(src_val, view_val, 1e-6);
    
    cgrad_storage_free(&src);
    cgrad_storage_free(&view);
}

static void test_cgrad_storage_view_with_contained_layout(void **state) {
    (void)state;
    // Create source storage with shape 4x4
    cgrad_storage src, view = {0};
    uint32_t src_shape[] = {4, 4};
    assert_int_equal(cgrad_storage_init(&src, src_shape, 2, "cpu_f32"), CGRAD_SUCCESS);
    
    // Fill with sequential values
    float count = 0.0f;
    for (uint32_t i = 0; i < 4; ++i) {
        for (uint32_t j = 0; j < 4; ++j) {
            uint32_t idx[TENSOR_DIM] = {0,0,0,0,0,0,i,j};
            assert_int_equal(src.backend->storage_set(src.data, idx, TENSOR_DIM, count++), CGRAD_SUCCESS);
        }
    }
    
    // Create a view with shape 2x2 (contained within 4x4)
    cgrad_storage_layout view_layout;
    uint32_t view_shape[] = {2, 2};
    assert_int_equal(cgrad_storage_layout_init(&view_layout, view_shape, 2), CGRAD_SUCCESS);
    
    // The view should access the first 2x2 block of the source
    assert_int_equal(cgrad_storage_view(&src, &view, &view_layout), CGRAD_SUCCESS);
    
    // Verify view shape
    cgrad_storage_layout* actual_view_layout = view.backend->storage_get_layout(view.data);
    assert_int_equal(actual_view_layout->shape[TENSOR_DIM-2], 2);
    assert_int_equal(actual_view_layout->shape[TENSOR_DIM-1], 2);
    assert_int_equal(actual_view_layout->size, 4);
    
    // Verify data is shared (first 4 values: 0,1,2,3)
    float val;
    uint32_t idx[TENSOR_DIM] = {0,0,0,0,0,0,0,0};
    assert_int_equal(cgrad_storage_get(&view, idx, TENSOR_DIM, &val), CGRAD_SUCCESS);
    assert_float_equal(val, 0.0f, 1e-6);
    
    idx[TENSOR_DIM-1] = 1;
    assert_int_equal(cgrad_storage_get(&view, idx, TENSOR_DIM, &val), CGRAD_SUCCESS);
    assert_float_equal(val, 1.0f, 1e-6);
    
    idx[TENSOR_DIM-2] = 1;
    idx[TENSOR_DIM-1] = 0;
    assert_int_equal(cgrad_storage_get(&view, idx, TENSOR_DIM, &val), CGRAD_SUCCESS);
    assert_float_equal(val, 2.0f, 1e-6);
    
    cgrad_storage_free(&src);
    cgrad_storage_free(&view);
}

static void test_cgrad_storage_view_data_sharing(void **state) {
    (void)state;
    // Verify that modifying the view affects the source (data is shared)
    cgrad_storage src, view = {0};
    uint32_t shape[] = {2, 2};
    assert_int_equal(cgrad_storage_init(&src, shape, 2, "cpu_f32"), CGRAD_SUCCESS);
    assert_int_equal(cgrad_storage_fill(&src, 0.0f), CGRAD_SUCCESS);
    
    // Create view
    assert_int_equal(cgrad_storage_view(&src, &view, NULL), CGRAD_SUCCESS);
    
    // Modify view
    uint32_t idx[TENSOR_DIM] = {0,0,0,0,0,0,0,0};
    assert_int_equal(view.backend->storage_set(view.data, idx, TENSOR_DIM, 42.0f), CGRAD_SUCCESS);
    
    // Check that source is also modified
    float src_val;
    assert_int_equal(cgrad_storage_get(&src, idx, TENSOR_DIM, &src_val), CGRAD_SUCCESS);
    assert_float_equal(src_val, 42.0f, 1e-6);
    
    cgrad_storage_free(&src);
    cgrad_storage_free(&view);
}

static void test_cgrad_storage_view_out_of_bounds_layout(void **state) {
    (void)state;
    // Test that view rejects layouts that are out of bounds
    cgrad_storage src, view = {0};
    uint32_t src_shape[] = {2, 3};
    assert_int_equal(cgrad_storage_init(&src, src_shape, 2, "cpu_f32"), CGRAD_SUCCESS);
    
    // Get source layout to understand its structure
    cgrad_storage_layout* src_layout = src.backend->storage_get_layout(src.data);
    
    // Create a layout with all dimensions specified to avoid leading 1s
    cgrad_storage_layout view_layout;
    uint32_t full_view_shape[TENSOR_DIM] = {1,1,1,1,1,1,2,2};
    assert_int_equal(cgrad_storage_layout_init(&view_layout, full_view_shape, TENSOR_DIM), CGRAD_SUCCESS);
    
    // Modify strides to make the layout out of bounds
    // Source size is 6, last index accessed will be (0,0,0,0,0,0,1,1) 
    // With strides {0,0,0,0,0,0,100,1}, this gives: 0*... + 1*100 + 1*1 = 101
    memset(view_layout.strides, 0, sizeof(view_layout.strides));
    view_layout.strides[TENSOR_DIM-2] = 100;
    view_layout.strides[TENSOR_DIM-1] = 1;
    
    // Should return index out of bounds error
    int err = cgrad_storage_view(&src, &view, &view_layout);
    assert_int_equal(err, CGRAD_ERR_STORAGE_LAYOUT_INDEX_OUT_OF_BOUNDS);
    
    cgrad_storage_free(&src);
}

static void test_cgrad_storage_view_non_contained_layout(void **state) {
    (void)state;
    // Test that view rejects layouts with shapes larger than source
    cgrad_storage src, view = {0};
    uint32_t src_shape[] = {2, 3};
    assert_int_equal(cgrad_storage_init(&src, src_shape, 2, "cpu_f32"), CGRAD_SUCCESS);
    
    // Create a layout with larger shape
    cgrad_storage_layout view_layout;
    uint32_t view_shape[] = {3, 4};
    assert_int_equal(cgrad_storage_layout_init(&view_layout, view_shape, 2), CGRAD_SUCCESS);
    
    // Should return index out of bounds error
    int err = cgrad_storage_view(&src, &view, &view_layout);
    assert_int_equal(err, CGRAD_ERR_STORAGE_LAYOUT_INDEX_OUT_OF_BOUNDS);
    
    cgrad_storage_free(&src);
}

static void test_cgrad_storage_view_registration(void **state) {
    (void)state;
    // Test that view is properly registered with source as parent
    cgrad_storage src, view = {0};
    uint32_t shape[] = {2, 2};
    assert_int_equal(cgrad_storage_init(&src, shape, 2, "cpu_f32"), CGRAD_SUCCESS);
    
    // Create view
    assert_int_equal(cgrad_storage_view(&src, &view, NULL), CGRAD_SUCCESS);
    
    // Get the root of view (should be src)
    cgrad_storage root;
    int err = cgrad_storage_registry_get_root(&view, &root);
    assert_int_equal(err, CGRAD_SUCCESS);
    
    // Check that root is src
    assert_int_equal(uuid_compare(root.uuid, src.uuid), 0);
    
    // Free view - src data should still be valid
    assert_int_equal(cgrad_storage_free(&view), CGRAD_SUCCESS);
    assert_null(view.data);
    
    // Verify src data is still accessible
    uint32_t idx[TENSOR_DIM] = {0,0,0,0,0,0,0,0};
    float val;
    assert_int_equal(cgrad_storage_get(&src, idx, TENSOR_DIM, &val), CGRAD_SUCCESS);
    
    cgrad_storage_free(&src);
}

int run_cgrad_storage_tests(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_cgrad_storage_init_and_free, storage_setup_test, storage_teardown_test),
        cmocka_unit_test_setup_teardown(test_cgrad_storage_init_errors, storage_setup_test, storage_teardown_test),
        cmocka_unit_test_setup_teardown(test_cgrad_storage_fill, storage_setup_test, storage_teardown_test),
        cmocka_unit_test_setup_teardown(test_cgrad_storage_contiguous, storage_setup_test, storage_teardown_test),
        cmocka_unit_test_setup_teardown(test_cgrad_storage_reshape, storage_setup_test, storage_teardown_test),
        cmocka_unit_test_setup_teardown(test_cgrad_storage_registry_root_freed_only_after_all_children, storage_setup_test, storage_teardown_test),
        cmocka_unit_test_setup_teardown(test_cgrad_storage_gemm_write_to_existing_tensor, storage_setup_test, storage_teardown_test),
        cmocka_unit_test_setup_teardown(test_cgrad_storage_reduce, storage_setup_test, storage_teardown_test),
        cmocka_unit_test_setup_teardown(test_cgrad_storage_view_with_null_layout, storage_setup_test, storage_teardown_test),
        cmocka_unit_test_setup_teardown(test_cgrad_storage_view_with_contained_layout, storage_setup_test, storage_teardown_test),
        cmocka_unit_test_setup_teardown(test_cgrad_storage_view_data_sharing, storage_setup_test, storage_teardown_test),
        cmocka_unit_test_setup_teardown(test_cgrad_storage_view_out_of_bounds_layout, storage_setup_test, storage_teardown_test),
        cmocka_unit_test_setup_teardown(test_cgrad_storage_view_non_contained_layout, storage_setup_test, storage_teardown_test),
        cmocka_unit_test_setup_teardown(test_cgrad_storage_view_registration, storage_setup_test, storage_teardown_test),
    };
    return cmocka_run_group_tests_name("cgrad_storage", tests, NULL, NULL);
}

#ifndef TEST_ALL_MAIN
int main(void) {
    return run_cgrad_storage_tests();
}
#endif
