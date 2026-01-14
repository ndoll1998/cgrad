#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdlib.h>
#include <math.h>

#include "cgrad_tensor.h"
#include "cgrad_errors.h"
#include "storage_backends/cgrad_storage_f32_cpu.h"

#define OP_RESHAPE_EPSILON 1e-4f

// ============================================================================
// Helper Functions
// ============================================================================

static float op_reshape_get_storage_value(cgrad_storage* storage, int idx) {
    cgrad_storage_f32_cpu* cpu_storage = (cgrad_storage_f32_cpu*)storage->data;
    return cpu_storage->data[idx];
}

static int op_reshape_approx_equal(float a, float b, float eps) {
    return fabsf(a - b) < eps;
}

// ============================================================================
// Setup and Teardown
// ============================================================================

static int op_reshape_teardown_test(void **state) {
    (void) state;
    cgrad_tensor_cleanup_global_graph();
    return 0;
}

// ============================================================================
// Test: Reshape forward
// ============================================================================

static void test_op_reshape_forward(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 3};
    cgrad_tensor a, b;
    
    cgrad_tensor_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_fill(&a, 1.0f);
    
    int32_t new_shape[] = {3, 2};
    int ret = cgrad_tensor_reshape(&a, new_shape, 2, &b);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    ret = cgrad_tensor_execute(&b);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Output should be (3, 2)
    assert_int_equal(b.layout.shape[TENSOR_DIM - 2], 3);
    assert_int_equal(b.layout.shape[TENSOR_DIM - 1], 2);
}

// ============================================================================
// Test: Reshape backward - basic
// ============================================================================

static void test_op_reshape_backward_basic(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 3};
    cgrad_tensor a, b;
    
    cgrad_tensor_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_fill(&a, 1.0f);
    
    int32_t new_shape[] = {3, 2};
    cgrad_tensor_reshape(&a, new_shape, 2, &b);
    
    int ret = cgrad_tensor_execute(&b);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    ret = cgrad_tensor_backward(&b);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Gradient should be reshaped back to original shape
    cgrad_storage* grad_a = cgrad_tensor_get_grad(&a);
    assert_non_null(grad_a);
    
    // All gradients should be 1.0
    for (int i = 0; i < 6; i++) {
        assert_true(op_reshape_approx_equal(op_reshape_get_storage_value(grad_a, i), 1.0f, OP_RESHAPE_EPSILON));
    }
}

// ============================================================================
// Test: Reshape backward - no grad
// ============================================================================

static void test_op_reshape_backward_no_grad(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 3};
    cgrad_tensor a, b;
    
    cgrad_tensor_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_set_requires_grad(&a, 0);
    cgrad_tensor_fill(&a, 1.0f);
    
    int32_t new_shape[] = {3, 2};
    cgrad_tensor_reshape(&a, new_shape, 2, &b);
    
    int ret = cgrad_tensor_execute(&b);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    ret = cgrad_tensor_backward(&b);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // a should not have gradient
    cgrad_storage* grad_a = cgrad_tensor_get_grad(&a);
    assert_null(grad_a);
}

// ============================================================================
// Test: Reshape backward - flatten
// ============================================================================

static void test_op_reshape_backward_flatten(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 3};
    cgrad_tensor a, b;
    
    cgrad_tensor_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_fill(&a, 1.0f);
    
    // Flatten to 1D
    int32_t new_shape[] = {6};
    cgrad_tensor_reshape(&a, new_shape, 1, &b);
    
    int ret = cgrad_tensor_execute(&b);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    ret = cgrad_tensor_backward(&b);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    cgrad_storage* grad_a = cgrad_tensor_get_grad(&a);
    assert_non_null(grad_a);
    
    // All gradients should be 1.0
    for (int i = 0; i < 6; i++) {
        assert_true(op_reshape_approx_equal(op_reshape_get_storage_value(grad_a, i), 1.0f, OP_RESHAPE_EPSILON));
    }
}

// ============================================================================
// Test: Reshape backward - double reshape
// ============================================================================

static void test_op_reshape_backward_double(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 3};
    cgrad_tensor a, b, c;
    
    cgrad_tensor_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_fill(&a, 1.0f);
    
    // b = reshape(a, [3, 2])
    int32_t shape1[] = {3, 2};
    cgrad_tensor_reshape(&a, shape1, 2, &b);
    
    // c = reshape(b, [6])
    int32_t shape2[] = {6};
    cgrad_tensor_reshape(&b, shape2, 1, &c);
    
    int ret = cgrad_tensor_execute(&c);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    ret = cgrad_tensor_backward(&c);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    cgrad_storage* grad_a = cgrad_tensor_get_grad(&a);
    assert_non_null(grad_a);
    
    // All gradients should be 1.0
    for (int i = 0; i < 6; i++) {
        assert_true(op_reshape_approx_equal(op_reshape_get_storage_value(grad_a, i), 1.0f, OP_RESHAPE_EPSILON));
    }
}

// ============================================================================
// Test Suite
// ============================================================================

#ifndef TEST_ALL_MAIN
int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_teardown(test_op_reshape_forward, op_reshape_teardown_test),
        cmocka_unit_test_teardown(test_op_reshape_backward_basic, op_reshape_teardown_test),
        cmocka_unit_test_teardown(test_op_reshape_backward_no_grad, op_reshape_teardown_test),
        cmocka_unit_test_teardown(test_op_reshape_backward_flatten, op_reshape_teardown_test),
        cmocka_unit_test_teardown(test_op_reshape_backward_double, op_reshape_teardown_test),
    };
    
    return cmocka_run_group_tests(tests, NULL, NULL);
}
#else
int test_cgrad_op_reshape_main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_teardown(test_op_reshape_forward, op_reshape_teardown_test),
        cmocka_unit_test_teardown(test_op_reshape_backward_basic, op_reshape_teardown_test),
        cmocka_unit_test_teardown(test_op_reshape_backward_no_grad, op_reshape_teardown_test),
        cmocka_unit_test_teardown(test_op_reshape_backward_flatten, op_reshape_teardown_test),
        cmocka_unit_test_teardown(test_op_reshape_backward_double, op_reshape_teardown_test),
    };
    
    return _cmocka_run_group_tests("cgrad_op_reshape", tests, sizeof(tests)/sizeof(tests[0]), NULL, NULL);
}
#endif
