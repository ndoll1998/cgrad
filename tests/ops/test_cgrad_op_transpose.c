#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdlib.h>
#include <math.h>

#include "cgrad_tensor.h"
#include "cgrad_errors.h"
#include "storage_backends/cgrad_storage_f32_cpu.h"

#define OP_TRANSPOSE_EPSILON 1e-4f

// ============================================================================
// Helper Functions
// ============================================================================

static float op_transpose_get_storage_value(cgrad_storage* storage, int idx) {
    cgrad_storage_f32_cpu* cpu_storage = (cgrad_storage_f32_cpu*)storage->data;
    return cpu_storage->data[idx];
}

static int op_transpose_approx_equal(float a, float b, float eps) {
    return fabsf(a - b) < eps;
}

// ============================================================================
// Setup and Teardown
// ============================================================================

static int op_transpose_teardown_test(void **state) {
    (void) state;
    cgrad_tensor_cleanup_global_graph();
    return 0;
}

// ============================================================================
// Test: Transpose forward
// ============================================================================

static void test_op_transpose_forward(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 3};
    cgrad_tensor a, b;
    
    cgrad_tensor_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_fill(&a, 1.0f);
    
    uint32_t perm[] = {1, 0};
    int ret = cgrad_tensor_transpose(&a, perm, 2, &b);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    ret = cgrad_tensor_execute(&b);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Output should be (3, 2)
    assert_int_equal(b.layout.shape[TENSOR_DIM - 2], 3);
    assert_int_equal(b.layout.shape[TENSOR_DIM - 1], 2);
}

// ============================================================================
// Test: Transpose backward - basic
// ============================================================================

static void test_op_transpose_backward_basic(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 3};
    cgrad_tensor a, b;
    
    cgrad_tensor_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_fill(&a, 1.0f);
    
    uint32_t perm[] = {1, 0};
    cgrad_tensor_transpose(&a, perm, 2, &b);
    
    int ret = cgrad_tensor_execute(&b);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    ret = cgrad_tensor_backward(&b);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Gradient should be transposed back to original shape
    cgrad_storage* grad_a = cgrad_tensor_get_grad(&a);
    assert_non_null(grad_a);
    
    // All gradients should be 1.0
    for (int i = 0; i < 6; i++) {
        assert_true(op_transpose_approx_equal(op_transpose_get_storage_value(grad_a, i), 1.0f, OP_TRANSPOSE_EPSILON));
    }
}

// ============================================================================
// Test: Transpose backward - no grad
// ============================================================================

static void test_op_transpose_backward_no_grad(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 3};
    cgrad_tensor a, b;
    
    cgrad_tensor_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_set_requires_grad(&a, 0);
    cgrad_tensor_fill(&a, 1.0f);
    
    uint32_t perm[] = {1, 0};
    cgrad_tensor_transpose(&a, perm, 2, &b);
    
    int ret = cgrad_tensor_execute(&b);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    ret = cgrad_tensor_backward(&b);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // a should not have gradient
    cgrad_storage* grad_a = cgrad_tensor_get_grad(&a);
    assert_null(grad_a);
}

// ============================================================================
// Test: Transpose backward - double transpose
// ============================================================================

static void test_op_transpose_backward_double(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 3};
    cgrad_tensor a, b, c;
    
    cgrad_tensor_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_fill(&a, 1.0f);
    
    uint32_t perm[] = {1, 0};
    
    // b = transpose(a)
    cgrad_tensor_transpose(&a, perm, 2, &b);
    
    // c = transpose(b) = a (back to original shape)
    cgrad_tensor_transpose(&b, perm, 2, &c);
    
    int ret = cgrad_tensor_execute(&c);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    ret = cgrad_tensor_backward(&c);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    cgrad_storage* grad_a = cgrad_tensor_get_grad(&a);
    assert_non_null(grad_a);
    
    // All gradients should be 1.0
    for (int i = 0; i < 6; i++) {
        assert_true(op_transpose_approx_equal(op_transpose_get_storage_value(grad_a, i), 1.0f, OP_TRANSPOSE_EPSILON));
    }
}

// ============================================================================
// Test Suite
// ============================================================================

#ifndef TEST_ALL_MAIN
int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_teardown(test_op_transpose_forward, op_transpose_teardown_test),
        cmocka_unit_test_teardown(test_op_transpose_backward_basic, op_transpose_teardown_test),
        cmocka_unit_test_teardown(test_op_transpose_backward_no_grad, op_transpose_teardown_test),
        cmocka_unit_test_teardown(test_op_transpose_backward_double, op_transpose_teardown_test),
    };
    
    return cmocka_run_group_tests(tests, NULL, NULL);
}
#else
int test_cgrad_op_transpose_main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_teardown(test_op_transpose_forward, op_transpose_teardown_test),
        cmocka_unit_test_teardown(test_op_transpose_backward_basic, op_transpose_teardown_test),
        cmocka_unit_test_teardown(test_op_transpose_backward_no_grad, op_transpose_teardown_test),
        cmocka_unit_test_teardown(test_op_transpose_backward_double, op_transpose_teardown_test),
    };
    
    return _cmocka_run_group_tests("cgrad_op_transpose", tests, sizeof(tests)/sizeof(tests[0]), NULL, NULL);
}
#endif
