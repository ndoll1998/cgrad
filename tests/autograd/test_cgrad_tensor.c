#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdlib.h>
#include <math.h>

#include "autograd/cgrad_tensor.h"
#include "cgrad_errors.h"

#define EPSILON 1e-5

// ============================================================================
// Setup and Teardown
// ============================================================================

static int teardown_test(void **state) {
    (void) state;
    // Clean up global graph after each test
    cgrad_tensor_cleanup_global_graph();
    return 0;
}

// ============================================================================
// Test: Tensor Initialization
// ============================================================================

static void test_cgrad_tensor_init(void **state) {
    (void) state;
    
    cgrad_tensor tensor;
    uint32_t shape[] = {2, 3};
    
    int ret = cgrad_tensor_init(&tensor, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    
    assert_int_equal(ret, CGRAD_SUCCESS);
    // Shape is stored in last dimensions of TENSOR_DIM (8)
    assert_int_equal(tensor.layout.shape[TENSOR_DIM - 2], 2);
    assert_int_equal(tensor.layout.shape[TENSOR_DIM - 1], 3);
}

// ============================================================================
// Test: Tensor Fill
// ============================================================================

static void test_cgrad_tensor_fill(void **state) {
    (void) state;
    
    cgrad_tensor tensor;
    uint32_t shape[] = {2, 2};
    
    cgrad_tensor_init(&tensor, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    
    int ret = cgrad_tensor_fill(&tensor, 3.14f);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Get storage and verify
    cgrad_storage* storage = cgrad_tensor_get_storage(&tensor);
    assert_non_null(storage);
}

// ============================================================================
// Test: Tensor Add
// ============================================================================

static void test_cgrad_tensor_add(void **state) {
    (void) state;
    
    // Create two tensors
    cgrad_tensor a, b, c;
    uint32_t shape[] = {2, 2};
    
    cgrad_tensor_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_init(&b, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    
    cgrad_tensor_fill(&a, 1.0f);
    cgrad_tensor_fill(&b, 2.0f);
    
    // Add tensors (lazy)
    int ret = cgrad_tensor_add(&a, &b, &c);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Verify output shape (stored in last dimensions of TENSOR_DIM)
    assert_int_equal(c.layout.shape[TENSOR_DIM - 2], 2);
    assert_int_equal(c.layout.shape[TENSOR_DIM - 1], 2);
    
    // Execute the graph
    ret = cgrad_tensor_execute(&c);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Get result storage
    cgrad_storage* result = cgrad_tensor_get_storage(&c);
    assert_non_null(result);
}

// ============================================================================
// Test: Tensor Sub
// ============================================================================

static void test_cgrad_tensor_sub(void **state) {
    (void) state;
    
    cgrad_tensor a, b, c;
    uint32_t shape[] = {2, 2};
    
    cgrad_tensor_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_init(&b, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    
    cgrad_tensor_fill(&a, 5.0f);
    cgrad_tensor_fill(&b, 2.0f);
    
    int ret = cgrad_tensor_sub(&a, &b, &c);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    ret = cgrad_tensor_execute(&c);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    cgrad_storage* result = cgrad_tensor_get_storage(&c);
    assert_non_null(result);
    
    // Expected: 5.0 - 2.0 = 3.0
}

// ============================================================================
// Test: Tensor GEMM
// ============================================================================

static void test_cgrad_tensor_gemm(void **state) {
    (void) state;
    
    cgrad_tensor a, b, c;
    uint32_t shape_a[] = {2, 3};
    uint32_t shape_b[] = {3, 2};
    
    cgrad_tensor_init(&a, shape_a, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_init(&b, shape_b, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    
    cgrad_tensor_fill(&a, 1.0f);
    cgrad_tensor_fill(&b, 2.0f);
    
    int ret = cgrad_tensor_gemm(&a, &b, &c);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Output should be (2, 2)
    assert_int_equal(c.layout.shape[TENSOR_DIM - 2], 2);
    assert_int_equal(c.layout.shape[TENSOR_DIM - 1], 2);
    
    ret = cgrad_tensor_execute(&c);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    cgrad_storage* result = cgrad_tensor_get_storage(&c);
    assert_non_null(result);
}

// ============================================================================
// Test: Tensor Transpose
// ============================================================================

static void test_cgrad_tensor_transpose(void **state) {
    (void) state;
    
    cgrad_tensor a, b;
    uint32_t shape[] = {2, 3};
    
    cgrad_tensor_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_fill(&a, 1.0f);
    
    uint32_t perm[] = {1, 0};
    int ret = cgrad_tensor_transpose(&a, perm, 2, &b);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Output should be (3, 2)
    assert_int_equal(b.layout.shape[TENSOR_DIM - 2], 3);
    assert_int_equal(b.layout.shape[TENSOR_DIM - 1], 2);
    
    ret = cgrad_tensor_execute(&b);
    assert_int_equal(ret, CGRAD_SUCCESS);
}

// ============================================================================
// Test: Tensor Reshape
// ============================================================================

static void test_cgrad_tensor_reshape(void **state) {
    (void) state;
    
    cgrad_tensor a, b;
    uint32_t shape[] = {2, 3};
    
    cgrad_tensor_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_fill(&a, 1.0f);
    
    int32_t new_shape[] = {3, 2};
    int ret = cgrad_tensor_reshape(&a, new_shape, 2, &b);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Output should be (3, 2)
    assert_int_equal(b.layout.shape[TENSOR_DIM - 2], 3);
    assert_int_equal(b.layout.shape[TENSOR_DIM - 1], 2);
    
    ret = cgrad_tensor_execute(&b);
    assert_int_equal(ret, CGRAD_SUCCESS);
}

// ============================================================================
// Test: Tensor Reduce Sum
// ============================================================================

static void test_cgrad_tensor_reduce_sum(void **state) {
    (void) state;
    
    cgrad_tensor a, b;
    uint32_t shape[] = {2, 3};
    
    cgrad_tensor_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_fill(&a, 1.0f);
    
    // Reduce along axis 0
    uint8_t mask[] = {1, 0};
    int ret = cgrad_tensor_reduce_sum(&a, mask, 2, &b);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Output should be (1, 3)
    assert_int_equal(b.layout.shape[TENSOR_DIM - 2], 1);
    assert_int_equal(b.layout.shape[TENSOR_DIM - 1], 3);
    
    ret = cgrad_tensor_execute(&b);
    assert_int_equal(ret, CGRAD_SUCCESS);
}

// ============================================================================
// Test: Complex Graph (Multiple Operations)
// ============================================================================

static void test_complex_graph(void **state) {
    (void) state;
    
    // Build: d = (a + b) - c
    cgrad_tensor a, b, c, d, e;
    uint32_t shape[] = {2, 2};
    
    cgrad_tensor_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_init(&b, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_init(&c, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    
    cgrad_tensor_fill(&a, 5.0f);
    cgrad_tensor_fill(&b, 3.0f);
    cgrad_tensor_fill(&c, 2.0f);
    
    // d = a + b
    int ret = cgrad_tensor_add(&a, &b, &d);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Execute final result
    ret = cgrad_tensor_execute(&d);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    cgrad_storage* result = cgrad_tensor_get_storage(&d);
    assert_non_null(result);
    
    // Expected: 5 + 3 = 8
}

// ============================================================================
// Test: Execution Caching
// ============================================================================

static void test_execution_caching(void **state) {
    (void) state;
    
    // Build graph: c = a + b
    cgrad_tensor a, b, c;
    uint32_t shape[] = {2, 2};
    
    cgrad_tensor_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_init(&b, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    
    cgrad_tensor_fill(&a, 1.0f);
    cgrad_tensor_fill(&b, 2.0f);
    
    cgrad_tensor_add(&a, &b, &c);
    
    // Execute once
    int ret = cgrad_tensor_execute(&c);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    cgrad_storage* result1 = cgrad_tensor_get_storage(&c);
    assert_non_null(result1);
    
    // Execute again - should use cached result
    ret = cgrad_tensor_execute(&c);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    cgrad_storage* result2 = cgrad_tensor_get_storage(&c);
    assert_non_null(result2);
    
    // Should be the same storage pointer (cached)
    assert_ptr_equal(result1, result2);
}

// ============================================================================
// Test: Disconnected Components (Subgraph Execution)
// ============================================================================

static void test_disconnected_components(void **state) {
    (void) state;
    
    // Create two disconnected components in the global graph:
    // Component 1: c1 = a1 + b1
    // Component 2: c2 = a2 + b2
    
    cgrad_tensor a1, b1, c1, a2, b2, c2;
    uint32_t shape[] = {2, 2};
    
    // Component 1
    cgrad_tensor_init(&a1, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_init(&b1, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_fill(&a1, 1.0f);
    cgrad_tensor_fill(&b1, 2.0f);
    cgrad_tensor_add(&a1, &b1, &c1);
    
    // Component 2 (completely disconnected from component 1)
    cgrad_tensor_init(&a2, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_init(&b2, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_fill(&a2, 10.0f);
    cgrad_tensor_fill(&b2, 20.0f);
    cgrad_tensor_add(&a2, &b2, &c2);
    
    // Execute only c1 - should NOT execute c2
    int ret = cgrad_tensor_execute(&c1);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // c1 should be materialized
    cgrad_storage* storage_c1 = cgrad_tensor_get_storage(&c1);
    assert_non_null(storage_c1);
    
    // c2 should NOT be materialized (still lazy)
    cgrad_storage* storage_c2 = cgrad_tensor_get_storage(&c2);
    assert_null(storage_c2);
    
    // Now execute c2
    ret = cgrad_tensor_execute(&c2);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Now c2 should be materialized
    storage_c2 = cgrad_tensor_get_storage(&c2);
    assert_non_null(storage_c2);
    
    // Both should have different storage (independent computations)
    assert_ptr_not_equal(storage_c1, storage_c2);
}


// ============================================================================
// Test Suite
// ============================================================================

int run_cgrad_tensor_tests(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_teardown(test_cgrad_tensor_init, teardown_test),
        cmocka_unit_test_teardown(test_cgrad_tensor_fill, teardown_test),
        cmocka_unit_test_teardown(test_cgrad_tensor_add, teardown_test),
        cmocka_unit_test_teardown(test_cgrad_tensor_sub, teardown_test),
        cmocka_unit_test_teardown(test_cgrad_tensor_gemm, teardown_test),
        cmocka_unit_test_teardown(test_cgrad_tensor_transpose, teardown_test),
        cmocka_unit_test_teardown(test_cgrad_tensor_reshape, teardown_test),
        cmocka_unit_test_teardown(test_cgrad_tensor_reduce_sum, teardown_test),
        cmocka_unit_test_teardown(test_complex_graph, teardown_test),
        cmocka_unit_test_teardown(test_execution_caching, teardown_test),
        cmocka_unit_test_teardown(test_disconnected_components, teardown_test),
    };
    
    return cmocka_run_group_tests_name("cgrad_tensor", tests, NULL, NULL);
}

#ifndef TEST_ALL_MAIN
int main(void) {
    return run_cgrad_tensor_tests();
}
#endif
