#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdlib.h>
#include <math.h>

#include "autograd/cgrad_tensor.h"
#include "cgrad_errors.h"
#include "storage/backends/cgrad_storage_f32_cpu.h"

#define OP_GEMM_EPSILON 1e-4f

// ============================================================================
// Helper Functions
// ============================================================================

static float op_gemm_get_storage_value(cgrad_storage* storage, int idx) {
    cgrad_storage_f32_cpu* cpu_storage = (cgrad_storage_f32_cpu*)storage->data;
    return cpu_storage->data[idx];
}

static int op_gemm_approx_equal(float a, float b, float eps) {
    return fabsf(a - b) < eps;
}

// ============================================================================
// Setup and Teardown
// ============================================================================

static int op_gemm_teardown_test(void **state) {
    (void) state;
    cgrad_tensor_cleanup_global_graph();
    return 0;
}

// ============================================================================
// Test: GEMM forward
// ============================================================================

static void test_op_gemm_forward(void **state) {
    (void) state;
    
    // A: 2x3, B: 3x2, C = A @ B: 2x2
    uint32_t shape_a[] = {2, 3};
    uint32_t shape_b[] = {3, 2};
    cgrad_tensor a, b, c;
    
    cgrad_tensor_init(&a, shape_a, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_init(&b, shape_b, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_fill(&a, 1.0f);
    cgrad_tensor_fill(&b, 2.0f);
    
    int ret = cgrad_tensor_gemm(&a, &b, &c);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    ret = cgrad_tensor_execute(&c);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    cgrad_storage* result = cgrad_tensor_get_storage(&c);
    assert_non_null(result);
    
    // Each element = sum of 3 products of 1*2 = 6
    for (int i = 0; i < 4; i++) {
        assert_true(op_gemm_approx_equal(op_gemm_get_storage_value(result, i), 6.0f, OP_GEMM_EPSILON));
    }
}

// ============================================================================
// Test: GEMM backward - basic
// ============================================================================

static void test_op_gemm_backward_basic(void **state) {
    (void) state;
    
    // A: 2x3, B: 3x2, C = A @ B: 2x2
    uint32_t shape_a[] = {2, 3};
    uint32_t shape_b[] = {3, 2};
    cgrad_tensor a, b, c;
    
    cgrad_tensor_init(&a, shape_a, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_init(&b, shape_b, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_fill(&a, 1.0f);
    cgrad_tensor_fill(&b, 1.0f);
    
    cgrad_tensor_gemm(&a, &b, &c);
    
    int ret = cgrad_tensor_execute(&c);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    ret = cgrad_tensor_backward(&c);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    cgrad_storage* grad_a = cgrad_tensor_get_grad(&a);
    cgrad_storage* grad_b = cgrad_tensor_get_grad(&b);
    
    assert_non_null(grad_a);
    assert_non_null(grad_b);
    
    // For C = A @ B with all 1s:
    // grad_A = grad_C @ B^T, where grad_C is all 1s and B^T is 2x3 of 1s
    // So grad_A should be 2x3 with each element = 2
    for (int i = 0; i < 6; i++) {
        assert_true(op_gemm_approx_equal(op_gemm_get_storage_value(grad_a, i), 2.0f, OP_GEMM_EPSILON));
    }
    
    // grad_B = A^T @ grad_C, where A^T is 3x2 of 1s and grad_C is 2x2 of 1s
    // So grad_B should be 3x2 with each element = 2
    for (int i = 0; i < 6; i++) {
        assert_true(op_gemm_approx_equal(op_gemm_get_storage_value(grad_b, i), 2.0f, OP_GEMM_EPSILON));
    }
}

// ============================================================================
// Test: GEMM backward - one input no grad
// ============================================================================

static void test_op_gemm_backward_one_no_grad(void **state) {
    (void) state;
    
    uint32_t shape_a[] = {2, 3};
    uint32_t shape_b[] = {3, 2};
    cgrad_tensor a, b, c;
    
    cgrad_tensor_init(&a, shape_a, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_init(&b, shape_b, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    
    cgrad_tensor_set_requires_grad(&a, 0);
    
    cgrad_tensor_fill(&a, 1.0f);
    cgrad_tensor_fill(&b, 1.0f);
    
    cgrad_tensor_gemm(&a, &b, &c);
    
    int ret = cgrad_tensor_execute(&c);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    ret = cgrad_tensor_backward(&c);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // a should not have gradient
    cgrad_storage* grad_a = cgrad_tensor_get_grad(&a);
    assert_null(grad_a);
    
    // b should have gradient
    cgrad_storage* grad_b = cgrad_tensor_get_grad(&b);
    assert_non_null(grad_b);
}

// ============================================================================
// Test: GEMM backward - square matrices
// ============================================================================

static void test_op_gemm_backward_square(void **state) {
    (void) state;
    
    uint32_t shape[] = {3, 3};
    cgrad_tensor a, b, c;
    
    cgrad_tensor_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_init(&b, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_fill(&a, 1.0f);
    cgrad_tensor_fill(&b, 1.0f);
    
    cgrad_tensor_gemm(&a, &b, &c);
    
    int ret = cgrad_tensor_execute(&c);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    ret = cgrad_tensor_backward(&c);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    cgrad_storage* grad_a = cgrad_tensor_get_grad(&a);
    cgrad_storage* grad_b = cgrad_tensor_get_grad(&b);
    
    assert_non_null(grad_a);
    assert_non_null(grad_b);
    
    // For 3x3 matrices of 1s:
    // grad_A = grad_C @ B^T = 3x3 of 1s @ 3x3 of 1s = 3x3 of 3s
    for (int i = 0; i < 9; i++) {
        assert_true(op_gemm_approx_equal(op_gemm_get_storage_value(grad_a, i), 3.0f, OP_GEMM_EPSILON));
        assert_true(op_gemm_approx_equal(op_gemm_get_storage_value(grad_b, i), 3.0f, OP_GEMM_EPSILON));
    }
}

// ============================================================================
// Test Suite
// ============================================================================

int run_cgrad_op_gemm_tests(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_teardown(test_op_gemm_forward, op_gemm_teardown_test),
        cmocka_unit_test_teardown(test_op_gemm_backward_basic, op_gemm_teardown_test),
        cmocka_unit_test_teardown(test_op_gemm_backward_one_no_grad, op_gemm_teardown_test),
        cmocka_unit_test_teardown(test_op_gemm_backward_square, op_gemm_teardown_test),
    };
    
    return cmocka_run_group_tests_name("cgrad_op_gemm", tests, NULL, NULL);
}

#ifndef TEST_ALL_MAIN
int main(void) {
    return run_cgrad_op_gemm_tests();
}
#endif