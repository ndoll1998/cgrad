#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdlib.h>
#include <math.h>

#include "cgrad_tensor.h"
#include "cgrad_errors.h"
#include "storage_backends/cgrad_storage_f32_cpu.h"

#define OP_SUB_EPSILON 1e-4f

// ============================================================================
// Helper Functions
// ============================================================================

static float op_sub_get_storage_value(cgrad_storage* storage, int idx) {
    cgrad_storage_f32_cpu* cpu_storage = (cgrad_storage_f32_cpu*)storage->data;
    return cpu_storage->data[idx];
}

static int op_sub_approx_equal(float a, float b, float eps) {
    return fabsf(a - b) < eps;
}

// ============================================================================
// Setup and Teardown
// ============================================================================

static int op_sub_teardown_test(void **state) {
    (void) state;
    cgrad_tensor_cleanup_global_graph();
    return 0;
}

// ============================================================================
// Test: Sub forward
// ============================================================================

static void test_op_sub_forward(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 2};
    cgrad_tensor a, b, c;
    
    cgrad_tensor_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_init(&b, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_fill(&a, 5.0f);
    cgrad_tensor_fill(&b, 3.0f);
    
    int ret = cgrad_tensor_sub(&a, &b, &c);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    ret = cgrad_tensor_execute(&c);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    cgrad_storage* result = cgrad_tensor_get_storage(&c);
    assert_non_null(result);
    
    // 5 - 3 = 2
    for (int i = 0; i < 4; i++) {
        assert_true(op_sub_approx_equal(op_sub_get_storage_value(result, i), 2.0f, OP_SUB_EPSILON));
    }
}

// ============================================================================
// Test: Sub backward - basic
// ============================================================================

static void test_op_sub_backward_basic(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 2};
    cgrad_tensor a, b, c;
    
    cgrad_tensor_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_init(&b, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_fill(&a, 5.0f);
    cgrad_tensor_fill(&b, 3.0f);
    
    cgrad_tensor_sub(&a, &b, &c);
    
    int ret = cgrad_tensor_execute(&c);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    ret = cgrad_tensor_backward(&c);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // For c = a - b: dc/da = 1, dc/db = -1
    cgrad_storage* grad_a = cgrad_tensor_get_grad(&a);
    cgrad_storage* grad_b = cgrad_tensor_get_grad(&b);
    
    assert_non_null(grad_a);
    assert_non_null(grad_b);
    
    for (int i = 0; i < 4; i++) {
        assert_true(op_sub_approx_equal(op_sub_get_storage_value(grad_a, i), 1.0f, OP_SUB_EPSILON));
        assert_true(op_sub_approx_equal(op_sub_get_storage_value(grad_b, i), -1.0f, OP_SUB_EPSILON));
    }
}

// ============================================================================
// Test: Sub backward - same tensor (a - a)
// ============================================================================

static void test_op_sub_backward_same_tensor(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 2};
    cgrad_tensor a, b;
    
    cgrad_tensor_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_fill(&a, 5.0f);
    
    // b = a - a
    cgrad_tensor_sub(&a, &a, &b);
    
    int ret = cgrad_tensor_execute(&b);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    ret = cgrad_tensor_backward(&b);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // b = a - a = 0, db/da = 1 + (-1) = 0
    cgrad_storage* grad_a = cgrad_tensor_get_grad(&a);
    assert_non_null(grad_a);
    
    for (int i = 0; i < 4; i++) {
        assert_true(op_sub_approx_equal(op_sub_get_storage_value(grad_a, i), 0.0f, OP_SUB_EPSILON));
    }
}

// ============================================================================
// Test: Sub backward - chained with add
// ============================================================================

static void test_op_sub_backward_chained(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 2};
    cgrad_tensor a, b, c, d, e;
    
    cgrad_tensor_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_init(&b, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_init(&c, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_fill(&a, 1.0f);
    cgrad_tensor_fill(&b, 2.0f);
    cgrad_tensor_fill(&c, 3.0f);
    
    // d = a + b
    cgrad_tensor_add(&a, &b, &d);
    
    // e = d - c
    cgrad_tensor_sub(&d, &c, &e);
    
    int ret = cgrad_tensor_execute(&e);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    ret = cgrad_tensor_backward(&e);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // e = (a + b) - c
    // de/da = 1, de/db = 1, de/dc = -1
    cgrad_storage* grad_a = cgrad_tensor_get_grad(&a);
    cgrad_storage* grad_b = cgrad_tensor_get_grad(&b);
    cgrad_storage* grad_c = cgrad_tensor_get_grad(&c);
    
    assert_non_null(grad_a);
    assert_non_null(grad_b);
    assert_non_null(grad_c);
    
    for (int i = 0; i < 4; i++) {
        assert_true(op_sub_approx_equal(op_sub_get_storage_value(grad_a, i), 1.0f, OP_SUB_EPSILON));
        assert_true(op_sub_approx_equal(op_sub_get_storage_value(grad_b, i), 1.0f, OP_SUB_EPSILON));
        assert_true(op_sub_approx_equal(op_sub_get_storage_value(grad_c, i), -1.0f, OP_SUB_EPSILON));
    }
}

// ============================================================================
// Test Suite
// ============================================================================

#ifndef TEST_ALL_MAIN
int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_teardown(test_op_sub_forward, op_sub_teardown_test),
        cmocka_unit_test_teardown(test_op_sub_backward_basic, op_sub_teardown_test),
        cmocka_unit_test_teardown(test_op_sub_backward_same_tensor, op_sub_teardown_test),
        cmocka_unit_test_teardown(test_op_sub_backward_chained, op_sub_teardown_test),
    };
    
    return cmocka_run_group_tests(tests, NULL, NULL);
}
#else
int test_cgrad_op_sub_main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_teardown(test_op_sub_forward, op_sub_teardown_test),
        cmocka_unit_test_teardown(test_op_sub_backward_basic, op_sub_teardown_test),
        cmocka_unit_test_teardown(test_op_sub_backward_same_tensor, op_sub_teardown_test),
        cmocka_unit_test_teardown(test_op_sub_backward_chained, op_sub_teardown_test),
    };
    
    return _cmocka_run_group_tests("cgrad_op_sub", tests, sizeof(tests)/sizeof(tests[0]), NULL, NULL);
}
#endif
