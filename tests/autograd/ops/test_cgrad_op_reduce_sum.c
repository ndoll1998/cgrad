#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdlib.h>
#include <math.h>

#include "autograd/cgrad_tensor.h"
#include "cgrad_errors.h"
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
    cgrad_tensor_cleanup_global_graph();
    return 0;
}

// ============================================================================
// Test: Reduce sum forward - sum all elements
// ============================================================================

static void test_op_reduce_sum_forward_all(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 3};
    cgrad_tensor a, b;
    
    cgrad_tensor_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_fill(&a, 2.0f);
    
    // Sum all elements: mask = {1, 1}
    uint8_t mask[] = {1, 1};
    int ret = cgrad_tensor_reduce_sum(&a, mask, 2, &b);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    ret = cgrad_tensor_execute(&b);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    cgrad_storage* result = cgrad_tensor_get_storage(&b);
    assert_non_null(result);
    
    // 2 * 3 * 2.0 = 12.0
    assert_true(op_reduce_sum_approx_equal(op_reduce_sum_get_storage_value(result, 0), 12.0f, OP_REDUCE_SUM_EPSILON));
}

// ============================================================================
// Test: Reduce sum forward - sum along last axis
// ============================================================================

static void test_op_reduce_sum_forward_last_axis(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 3};
    cgrad_tensor a, b;
    
    cgrad_tensor_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_fill(&a, 1.0f);
    
    // Sum along last axis: mask = {0, 1}
    uint8_t mask[] = {0, 1};
    int ret = cgrad_tensor_reduce_sum(&a, mask, 2, &b);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    ret = cgrad_tensor_execute(&b);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    cgrad_storage* result = cgrad_tensor_get_storage(&b);
    assert_non_null(result);
    
    // Each row sums to 3.0 (3 elements * 1.0)
    // Result shape should be (2, 1)
    assert_true(op_reduce_sum_approx_equal(op_reduce_sum_get_storage_value(result, 0), 3.0f, OP_REDUCE_SUM_EPSILON));
    assert_true(op_reduce_sum_approx_equal(op_reduce_sum_get_storage_value(result, 1), 3.0f, OP_REDUCE_SUM_EPSILON));
}

// ============================================================================
// Test: Reduce sum forward - sum along first axis
// ============================================================================

static void test_op_reduce_sum_forward_first_axis(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 3};
    cgrad_tensor a, b;
    
    cgrad_tensor_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_fill(&a, 1.0f);
    
    // Sum along first axis: mask = {1, 0}
    uint8_t mask[] = {1, 0};
    int ret = cgrad_tensor_reduce_sum(&a, mask, 2, &b);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    ret = cgrad_tensor_execute(&b);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    cgrad_storage* result = cgrad_tensor_get_storage(&b);
    assert_non_null(result);
    
    // Each column sums to 2.0 (2 elements * 1.0)
    // Result shape should be (1, 3)
    for (int i = 0; i < 3; i++) {
        assert_true(op_reduce_sum_approx_equal(op_reduce_sum_get_storage_value(result, i), 2.0f, OP_REDUCE_SUM_EPSILON));
    }
}

// ============================================================================
// Test: Reduce sum backward - sum all elements
// ============================================================================

static void test_op_reduce_sum_backward_all(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 3};
    cgrad_tensor a, b;
    
    cgrad_tensor_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_fill(&a, 2.0f);
    
    // Sum all elements
    uint8_t mask[] = {1, 1};
    cgrad_tensor_reduce_sum(&a, mask, 2, &b);
    
    int ret = cgrad_tensor_execute(&b);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    ret = cgrad_tensor_backward(&b);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // For b = sum(a): db/da = 1 for all elements
    cgrad_storage* grad_a = cgrad_tensor_get_grad(&a);
    assert_non_null(grad_a);
    
    for (int i = 0; i < 6; i++) {
        assert_true(op_reduce_sum_approx_equal(op_reduce_sum_get_storage_value(grad_a, i), 1.0f, OP_REDUCE_SUM_EPSILON));
    }
}

// ============================================================================
// Test: Reduce sum backward - sum along last axis
// ============================================================================

static void test_op_reduce_sum_backward_last_axis(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 3};
    cgrad_tensor a, b;
    
    cgrad_tensor_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_fill(&a, 1.0f);
    
    // Sum along last axis
    uint8_t mask[] = {0, 1};
    cgrad_tensor_reduce_sum(&a, mask, 2, &b);
    
    int ret = cgrad_tensor_execute(&b);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    ret = cgrad_tensor_backward(&b);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Gradient should be broadcast back: all elements get 1.0
    cgrad_storage* grad_a = cgrad_tensor_get_grad(&a);
    assert_non_null(grad_a);
    
    for (int i = 0; i < 6; i++) {
        assert_true(op_reduce_sum_approx_equal(op_reduce_sum_get_storage_value(grad_a, i), 1.0f, OP_REDUCE_SUM_EPSILON));
    }
}

// ============================================================================
// Test: Reduce sum backward - no grad
// ============================================================================

static void test_op_reduce_sum_backward_no_grad(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 3};
    cgrad_tensor a, b;
    
    cgrad_tensor_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_set_requires_grad(&a, 0);
    cgrad_tensor_fill(&a, 1.0f);
    
    uint8_t mask[] = {1, 1};
    cgrad_tensor_reduce_sum(&a, mask, 2, &b);
    
    int ret = cgrad_tensor_execute(&b);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // b should not require grad since a doesn't
    int requires_grad;
    cgrad_tensor_get_requires_grad(&b, &requires_grad);
    assert_int_equal(requires_grad, 0);
    
    ret = cgrad_tensor_backward(&b);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // a should not have gradient
    cgrad_storage* grad_a = cgrad_tensor_get_grad(&a);
    assert_null(grad_a);
}

// ============================================================================
// Test: Reduce sum backward - chained with add
// ============================================================================

static void test_op_reduce_sum_backward_chained(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 2};
    cgrad_tensor a, b, c, d;
    
    cgrad_tensor_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_init(&b, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_fill(&a, 1.0f);
    cgrad_tensor_fill(&b, 2.0f);
    
    // c = a + b
    cgrad_tensor_add(&a, &b, &c);
    
    // d = sum(c)
    uint8_t mask[] = {1, 1};
    cgrad_tensor_reduce_sum(&c, mask, 2, &d);
    
    int ret = cgrad_tensor_execute(&d);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // d = sum(a + b) = sum(a) + sum(b) = 4 + 8 = 12
    cgrad_storage* result = cgrad_tensor_get_storage(&d);
    assert_non_null(result);
    assert_true(op_reduce_sum_approx_equal(op_reduce_sum_get_storage_value(result, 0), 12.0f, OP_REDUCE_SUM_EPSILON));
    
    ret = cgrad_tensor_backward(&d);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // dd/da = 1 for all elements, dd/db = 1 for all elements
    cgrad_storage* grad_a = cgrad_tensor_get_grad(&a);
    cgrad_storage* grad_b = cgrad_tensor_get_grad(&b);
    
    assert_non_null(grad_a);
    assert_non_null(grad_b);
    
    for (int i = 0; i < 4; i++) {
        assert_true(op_reduce_sum_approx_equal(op_reduce_sum_get_storage_value(grad_a, i), 1.0f, OP_REDUCE_SUM_EPSILON));
        assert_true(op_reduce_sum_approx_equal(op_reduce_sum_get_storage_value(grad_b, i), 1.0f, OP_REDUCE_SUM_EPSILON));
    }
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
        cmocka_unit_test_teardown(test_op_reduce_sum_backward_chained, op_reduce_sum_teardown_test),
    };
    
    return cmocka_run_group_tests_name("cgrad_op_reduce_sum", tests, NULL, NULL);
}

#ifndef TEST_ALL_MAIN
int main(void) {
    return run_cgrad_op_reduce_sum_tests();
}
#endif
