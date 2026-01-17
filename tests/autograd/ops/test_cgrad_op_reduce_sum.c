#include <stdarg.h>
#include "cgrad.h"
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdlib.h>
#include <math.h>

#include "autograd/cgrad_ops.h"
#include "cgrad_status.h"
#include "storage/cgrad_storage.h"
#include "storage/cgrad_storage_layout.h"

#define OP_REDUCE_SUM_EPSILON 1e-4f

// ============================================================================
// Setup and Teardown
// ============================================================================

static int reduce_sum_setup_test(void **state) {
    (void) state;
    cgrad_init();
    return 0;
}

static int reduce_sum_teardown_test(void **state) {
    (void) state;
    cgrad_cleanup();
    return 0;
}

// ============================================================================
// Test: Reduce sum forward - sum all elements
// ============================================================================

static void test_op_reduce_sum_forward_all(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 3};
    cgrad_storage a, b;
    
    cgrad_storage_init(&a, shape, 2, "cpu_f32");
    cgrad_storage_fill(&a, 2.0f);
    
    // Get the REDUCE_SUM operation descriptor
    const cgrad_op_descriptor* op_desc = &cgrad_op_reduce_sum;
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
    
    // 2 * 3 * 2.0 = 12.0 (scalar result)
    float value;
    uint32_t idx[1] = {0};
    cgrad_storage_get(&b, idx, 1, &value);
    assert_true(fabsf(value - 12.0f) < OP_REDUCE_SUM_EPSILON);
    
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
    
    cgrad_storage_init(&a, shape, 2, "cpu_f32");
    cgrad_storage_fill(&a, 1.0f);
    
    // Get the REDUCE_SUM operation descriptor
    const cgrad_op_descriptor* op_desc = &cgrad_op_reduce_sum;
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
    float value;
    uint32_t idx[2];
    for (uint32_t i = 0; i < 2; i++) {
        idx[0] = i; idx[1] = 0;
        cgrad_storage_get(&b, idx, 2, &value);
        assert_true(fabsf(value - 3.0f) < OP_REDUCE_SUM_EPSILON);
    }
    
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
    
    cgrad_storage_init(&a, shape, 2, "cpu_f32");
    cgrad_storage_fill(&a, 1.0f);
    
    // Get the REDUCE_SUM operation descriptor
    const cgrad_op_descriptor* op_desc = &cgrad_op_reduce_sum;
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
    float value;
    uint32_t idx[2];
    for (uint32_t j = 0; j < 3; j++) {
        idx[0] = 0; idx[1] = j;
        cgrad_storage_get(&b, idx, 2, &value);
        assert_true(fabsf(value - 2.0f) < OP_REDUCE_SUM_EPSILON);
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
    
    cgrad_storage_init(&a, shape, 2, "cpu_f32");
    cgrad_storage_init(&grad_a, shape, 2, "cpu_f32");
    cgrad_storage_fill(&a, 2.0f);
    cgrad_storage_fill(&grad_a, 0.0f);
    
    // Get the REDUCE_SUM operation descriptor
    const cgrad_op_descriptor* op_desc = &cgrad_op_reduce_sum;
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
    cgrad_storage_init(&grad_b, shape_reduced, 1, "cpu_f32");
    cgrad_storage_fill(&grad_b, 1.0f);
    
    // Execute backward pass
    cgrad_storage* grad_inputs[1] = {&grad_a};
    int input_requires_grad[1] = {1};
    ret = op_desc->backward(inputs, 1, &b, &grad_b, &metadata, ctx, grad_inputs, input_requires_grad);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // For b = sum(a): db/da = 1 for all elements
    float value;
    uint32_t idx[2];
    for (uint32_t i = 0; i < 2; i++) {
        for (uint32_t j = 0; j < 3; j++) {
            idx[0] = i; idx[1] = j;
            cgrad_storage_get(&grad_a, idx, 2, &value);
            assert_true(fabsf(value - 1.0f) < OP_REDUCE_SUM_EPSILON);
        }
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
    
    cgrad_storage_init(&a, shape, 2, "cpu_f32");
    cgrad_storage_init(&grad_a, shape, 2, "cpu_f32");
    cgrad_storage_fill(&a, 1.0f);
    cgrad_storage_fill(&grad_a, 0.0f);
    
    // Get the REDUCE_SUM operation descriptor
    const cgrad_op_descriptor* op_desc = &cgrad_op_reduce_sum;
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
    cgrad_storage_init(&grad_b, shape_reduced, 2, "cpu_f32");
    cgrad_storage_fill(&grad_b, 1.0f);
    
    // Execute backward pass
    cgrad_storage* grad_inputs[1] = {&grad_a};
    int input_requires_grad[1] = {1};
    ret = op_desc->backward(inputs, 1, &b, &grad_b, &metadata, ctx, grad_inputs, input_requires_grad);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Gradient should be broadcast back: all elements get 1.0
    float value;
    uint32_t idx[2];
    for (uint32_t i = 0; i < 2; i++) {
        for (uint32_t j = 0; j < 3; j++) {
            idx[0] = i; idx[1] = j;
            cgrad_storage_get(&grad_a, idx, 2, &value);
            assert_true(fabsf(value - 1.0f) < OP_REDUCE_SUM_EPSILON);
        }
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
    
    cgrad_storage_init(&a, shape, 2, "cpu_f32");
    cgrad_storage_fill(&a, 1.0f);
    
    // Get the REDUCE_SUM operation descriptor
    const cgrad_op_descriptor* op_desc = &cgrad_op_reduce_sum;
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
    cgrad_storage_init(&grad_b, shape_reduced, 1, "cpu_f32");
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
        cmocka_unit_test_setup_teardown(test_op_reduce_sum_forward_all, reduce_sum_setup_test, reduce_sum_teardown_test),
        cmocka_unit_test_setup_teardown(test_op_reduce_sum_forward_last_axis, reduce_sum_setup_test, reduce_sum_teardown_test),
        cmocka_unit_test_setup_teardown(test_op_reduce_sum_forward_first_axis, reduce_sum_setup_test, reduce_sum_teardown_test),
        cmocka_unit_test_setup_teardown(test_op_reduce_sum_backward_all, reduce_sum_setup_test, reduce_sum_teardown_test),
        cmocka_unit_test_setup_teardown(test_op_reduce_sum_backward_last_axis, reduce_sum_setup_test, reduce_sum_teardown_test),
        cmocka_unit_test_setup_teardown(test_op_reduce_sum_backward_no_grad, reduce_sum_setup_test, reduce_sum_teardown_test),
    };
    
    return cmocka_run_group_tests_name("cgrad_op_reduce_sum", tests, NULL, NULL);
}

#ifndef TEST_ALL_MAIN
int main(void) {
    return run_cgrad_op_reduce_sum_tests();
}
#endif
