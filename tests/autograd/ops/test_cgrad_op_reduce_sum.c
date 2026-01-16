#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdlib.h>
#include <math.h>

#include "autograd/cgrad_ops.h"
#include "cgrad_errors.h"
#include "storage/cgrad_storage.h"
#include "storage/cgrad_storage_layout.h"
#include "storage/backends/cgrad_storage_f32_cpu.h"

#define OP_REDUCE_SUM_EPSILON 1e-4f

// ============================================================================
// Helper Functions
// ============================================================================

static float op_reduce_sum_get_storage_value(cgrad_storage* storage, int idx) {
    cgrad_storage_f32_cpu* cpu_storage = (cgrad_storage_f32_cpu*)storage->data;
    return cpu_storage->data[idx];
}

static int op_reduce_sum_approx_equal(float a, float b, float eps) {
    return fabsf(a - b) < eps;
}

// ============================================================================
// Setup and Teardown
// ============================================================================

static int op_reduce_sum_teardown_test(void **state) {
    (void) state;
    cgrad_storage_cleanup_global_registry();
    return 0;
}

// ============================================================================
// Test: Reduce sum forward - sum all elements
// ============================================================================

static void test_op_reduce_sum_forward_all(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 3};
    cgrad_storage a, b;
    
    cgrad_storage_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_storage_fill(&a, 2.0f);
    
    // Get the REDUCE_SUM operation descriptor
    const cgrad_op_descriptor* op_desc = cgrad_get_op_descriptor(CGRAD_OP_REDUCE_SUM);
    assert_non_null(op_desc);
    
    // Prepare inputs
    cgrad_storage* inputs[1] = {&a};
    
    // Prepare metadata for reduce_sum (mask = {1, 1} - sum all)
    cgrad_op_metadata metadata = {0};
    metadata.reduce_sum.mask[0] = 1;
    metadata.reduce_sum.mask[1] = 1;
    metadata.reduce_sum.ndim = 2;
    
    // Execute forward pass - output will be initialized by reduce_sum
    void* ctx = NULL;
    memset(&b, 0, sizeof(b));
    int ret = op_desc->forward(inputs, 1, &metadata, &b, &ctx, 1);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // 2 * 3 * 2.0 = 12.0
    assert_true(op_reduce_sum_approx_equal(op_reduce_sum_get_storage_value(&b, 0), 12.0f, OP_REDUCE_SUM_EPSILON));
    
    cgrad_storage_free(&a);
    cgrad_storage_free(&b);
}

// ============================================================================
// Test: Reduce sum forward - sum along last axis
// ============================================================================

static void test_op_reduce_sum_forward_last_axis(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 3};
    cgrad_storage a, b;
    
    cgrad_storage_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_storage_fill(&a, 1.0f);
    
    // Get the REDUCE_SUM operation descriptor
    const cgrad_op_descriptor* op_desc = cgrad_get_op_descriptor(CGRAD_OP_REDUCE_SUM);
    assert_non_null(op_desc);
    
    // Prepare inputs
    cgrad_storage* inputs[1] = {&a};
    
    // Prepare metadata for reduce_sum (mask = {0, 1} - sum along last axis)
    cgrad_op_metadata metadata = {0};
    metadata.reduce_sum.mask[0] = 0;
    metadata.reduce_sum.mask[1] = 1;
    metadata.reduce_sum.ndim = 2;
    
    // Execute forward pass
    void* ctx = NULL;
    memset(&b, 0, sizeof(b));
    int ret = op_desc->forward(inputs, 1, &metadata, &b, &ctx, 1);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Each row sums to 3.0 (3 elements * 1.0)
    // Result shape should be (2, 1)
    assert_true(op_reduce_sum_approx_equal(op_reduce_sum_get_storage_value(&b, 0), 3.0f, OP_REDUCE_SUM_EPSILON));
    assert_true(op_reduce_sum_approx_equal(op_reduce_sum_get_storage_value(&b, 1), 3.0f, OP_REDUCE_SUM_EPSILON));
    
    cgrad_storage_free(&a);
    cgrad_storage_free(&b);
}

// ============================================================================
// Test: Reduce sum forward - sum along first axis
// ============================================================================

static void test_op_reduce_sum_forward_first_axis(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 3};
    cgrad_storage a, b;
    
    cgrad_storage_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_storage_fill(&a, 1.0f);
    
    // Get the REDUCE_SUM operation descriptor
    const cgrad_op_descriptor* op_desc = cgrad_get_op_descriptor(CGRAD_OP_REDUCE_SUM);
    assert_non_null(op_desc);
    
    // Prepare inputs
    cgrad_storage* inputs[1] = {&a};
    
    // Prepare metadata for reduce_sum (mask = {1, 0} - sum along first axis)
    cgrad_op_metadata metadata = {0};
    metadata.reduce_sum.mask[0] = 1;
    metadata.reduce_sum.mask[1] = 0;
    metadata.reduce_sum.ndim = 2;
    
    // Execute forward pass
    void* ctx = NULL;
    memset(&b, 0, sizeof(b));
    int ret = op_desc->forward(inputs, 1, &metadata, &b, &ctx, 1);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Each column sums to 2.0 (2 elements * 1.0)
    // Result shape should be (1, 3)
    for (int i = 0; i < 3; i++) {
        assert_true(op_reduce_sum_approx_equal(op_reduce_sum_get_storage_value(&b, i), 2.0f, OP_REDUCE_SUM_EPSILON));
    }
    
    cgrad_storage_free(&a);
    cgrad_storage_free(&b);
}

// ============================================================================
// Test: Reduce sum backward - sum all elements
// ============================================================================

static void test_op_reduce_sum_backward_all(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 3};
    uint32_t shape_reduced[] = {1};
    cgrad_storage a, b;
    cgrad_storage grad_a, grad_b;
    
    cgrad_storage_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_storage_init(&grad_a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_storage_fill(&a, 2.0f);
    cgrad_storage_fill(&grad_a, 0.0f);
    
    // Get the REDUCE_SUM operation descriptor
    const cgrad_op_descriptor* op_desc = cgrad_get_op_descriptor(CGRAD_OP_REDUCE_SUM);
    assert_non_null(op_desc);
    
    // Prepare inputs
    cgrad_storage* inputs[1] = {&a};
    
    // Prepare metadata for reduce_sum (mask = {1, 1} - sum all)
    cgrad_op_metadata metadata = {0};
    metadata.reduce_sum.mask[0] = 1;
    metadata.reduce_sum.mask[1] = 1;
    metadata.reduce_sum.ndim = 2;
    
    // Execute forward pass
    void* ctx = NULL;
    memset(&b, 0, sizeof(b));
    int ret = op_desc->forward(inputs, 1, &metadata, &b, &ctx, 1);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Initialize gradient for output (scalar)
    cgrad_storage_init(&grad_b, shape_reduced, 1, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_storage_fill(&grad_b, 1.0f);
    
    // Execute backward pass
    cgrad_storage* grad_inputs[1] = {&grad_a};
    int input_requires_grad[1] = {1};
    ret = op_desc->backward(inputs, 1, &b, &grad_b, &metadata, ctx, grad_inputs, input_requires_grad);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // For b = sum(a): db/da = 1 for all elements
    for (int i = 0; i < 6; i++) {
        assert_true(op_reduce_sum_approx_equal(op_reduce_sum_get_storage_value(&grad_a, i), 1.0f, OP_REDUCE_SUM_EPSILON));
    }
    
    cgrad_storage_free(&a);
    cgrad_storage_free(&b);
    cgrad_storage_free(&grad_a);
    cgrad_storage_free(&grad_b);
}

// ============================================================================
// Test: Reduce sum backward - sum along last axis
// ============================================================================

static void test_op_reduce_sum_backward_last_axis(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 3};
    uint32_t shape_reduced[] = {2, 1};
    cgrad_storage a, b;
    cgrad_storage grad_a, grad_b;
    
    cgrad_storage_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_storage_init(&grad_a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_storage_fill(&a, 1.0f);
    cgrad_storage_fill(&grad_a, 0.0f);
    
    // Get the REDUCE_SUM operation descriptor
    const cgrad_op_descriptor* op_desc = cgrad_get_op_descriptor(CGRAD_OP_REDUCE_SUM);
    assert_non_null(op_desc);
    
    // Prepare inputs
    cgrad_storage* inputs[1] = {&a};
    
    // Prepare metadata for reduce_sum (mask = {0, 1} - sum along last axis)
    cgrad_op_metadata metadata = {0};
    metadata.reduce_sum.mask[0] = 0;
    metadata.reduce_sum.mask[1] = 1;
    metadata.reduce_sum.ndim = 2;
    
    // Execute forward pass
    void* ctx = NULL;
    memset(&b, 0, sizeof(b));
    int ret = op_desc->forward(inputs, 1, &metadata, &b, &ctx, 1);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Initialize gradient for output (reduced shape)
    cgrad_storage_init(&grad_b, shape_reduced, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_storage_fill(&grad_b, 1.0f);
    
    // Execute backward pass
    cgrad_storage* grad_inputs[1] = {&grad_a};
    int input_requires_grad[1] = {1};
    ret = op_desc->backward(inputs, 1, &b, &grad_b, &metadata, ctx, grad_inputs, input_requires_grad);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Gradient should be broadcast back: all elements get 1.0
    for (int i = 0; i < 6; i++) {
        assert_true(op_reduce_sum_approx_equal(op_reduce_sum_get_storage_value(&grad_a, i), 1.0f, OP_REDUCE_SUM_EPSILON));
    }
    
    cgrad_storage_free(&a);
    cgrad_storage_free(&b);
    cgrad_storage_free(&grad_a);
    cgrad_storage_free(&grad_b);
}

// ============================================================================
// Test: Reduce sum backward - no grad
// ============================================================================

static void test_op_reduce_sum_backward_no_grad(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 3};
    uint32_t shape_reduced[] = {1};
    cgrad_storage a, b;
    cgrad_storage grad_b;
    
    cgrad_storage_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_storage_fill(&a, 1.0f);
    
    // Get the REDUCE_SUM operation descriptor
    const cgrad_op_descriptor* op_desc = cgrad_get_op_descriptor(CGRAD_OP_REDUCE_SUM);
    assert_non_null(op_desc);
    
    // Prepare inputs
    cgrad_storage* inputs[1] = {&a};
    
    // Prepare metadata for reduce_sum (mask = {1, 1} - sum all)
    cgrad_op_metadata metadata = {0};
    metadata.reduce_sum.mask[0] = 1;
    metadata.reduce_sum.mask[1] = 1;
    metadata.reduce_sum.ndim = 2;
    
    // Execute forward pass
    void* ctx = NULL;
    memset(&b, 0, sizeof(b));
    int ret = op_desc->forward(inputs, 1, &metadata, &b, &ctx, 1);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Initialize gradient for output
    cgrad_storage_init(&grad_b, shape_reduced, 1, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_storage_fill(&grad_b, 1.0f);
    
    // Execute backward pass with no grad required
    cgrad_storage* grad_inputs[1] = {NULL};
    int input_requires_grad[1] = {0};
    ret = op_desc->backward(inputs, 1, &b, &grad_b, &metadata, ctx, grad_inputs, input_requires_grad);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    cgrad_storage_free(&a);
    cgrad_storage_free(&b);
    cgrad_storage_free(&grad_b);
}

// ============================================================================
// Test Suite
// ============================================================================

int run_cgrad_op_reduce_sum_tests(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_teardown(test_op_reduce_sum_forward_all, op_reduce_sum_teardown_test),
        cmocka_unit_test_teardown(test_op_reduce_sum_forward_last_axis, op_reduce_sum_teardown_test),
        cmocka_unit_test_teardown(test_op_reduce_sum_forward_first_axis, op_reduce_sum_teardown_test),
        cmocka_unit_test_teardown(test_op_reduce_sum_backward_all, op_reduce_sum_teardown_test),
        cmocka_unit_test_teardown(test_op_reduce_sum_backward_last_axis, op_reduce_sum_teardown_test),
        cmocka_unit_test_teardown(test_op_reduce_sum_backward_no_grad, op_reduce_sum_teardown_test),
    };
    
    return cmocka_run_group_tests_name("cgrad_op_reduce_sum", tests, NULL, NULL);
}

#ifndef TEST_ALL_MAIN
int main(void) {
    return run_cgrad_op_reduce_sum_tests();
}
#endif
