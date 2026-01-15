#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdlib.h>
#include <math.h>

#include "autograd/cgrad_tensor.h"
#include "cgrad_errors.h"
#include "storage/backends/cgrad_storage_f32_cpu.h"

#define OP_ADD_EPSILON 1e-4f

// ============================================================================
// Helper Functions
// ============================================================================

static float op_add_get_storage_value(cgrad_storage* storage, int idx) {
    cgrad_storage_f32_cpu* cpu_storage = (cgrad_storage_f32_cpu*)storage->data;
    return cpu_storage->data[idx];
}

static int op_add_approx_equal(float a, float b, float eps) {
    return fabsf(a - b) < eps;
}

// ============================================================================
// Setup and Teardown
// ============================================================================

static int op_add_teardown_test(void **state) {
    (void) state;
    cgrad_tensor_cleanup_global_graph();
    return 0;
}

// ============================================================================
// Test: Add forward
// ============================================================================

static void test_op_add_forward(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 2};
    cgrad_tensor a, b, c;
    
    cgrad_tensor_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_init(&b, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_fill(&a, 2.0f);
    cgrad_tensor_fill(&b, 3.0f);
    
    int ret = cgrad_tensor_add(&a, &b, &c);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    ret = cgrad_tensor_execute(&c);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    cgrad_storage* result = cgrad_tensor_get_storage(&c);
    assert_non_null(result);
    
    // 2 + 3 = 5
    for (int i = 0; i < 4; i++) {
        assert_true(op_add_approx_equal(op_add_get_storage_value(result, i), 5.0f, OP_ADD_EPSILON));
    }
}

// ============================================================================
// Test: Add backward - basic
// ============================================================================

static void test_op_add_backward_basic(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 2};
    cgrad_tensor a, b, c;
    
    cgrad_tensor_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_init(&b, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_fill(&a, 2.0f);
    cgrad_tensor_fill(&b, 3.0f);
    
    cgrad_tensor_add(&a, &b, &c);
    
    int ret = cgrad_tensor_execute(&c);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    ret = cgrad_tensor_backward(&c);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // For c = a + b: dc/da = 1, dc/db = 1
    cgrad_storage* grad_a = cgrad_tensor_get_grad(&a);
    cgrad_storage* grad_b = cgrad_tensor_get_grad(&b);
    
    assert_non_null(grad_a);
    assert_non_null(grad_b);
    
    for (int i = 0; i < 4; i++) {
        assert_true(op_add_approx_equal(op_add_get_storage_value(grad_a, i), 1.0f, OP_ADD_EPSILON));
        assert_true(op_add_approx_equal(op_add_get_storage_value(grad_b, i), 1.0f, OP_ADD_EPSILON));
    }
}

// ============================================================================
// Test: Add backward - same tensor twice (a + a)
// ============================================================================

static void test_op_add_backward_same_tensor(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 2};
    cgrad_tensor a, b;
    
    cgrad_tensor_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_fill(&a, 1.0f);
    
    // b = a + a
    cgrad_tensor_add(&a, &a, &b);
    
    int ret = cgrad_tensor_execute(&b);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    ret = cgrad_tensor_backward(&b);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // b = a + a = 2a, db/da = 2
    cgrad_storage* grad_a = cgrad_tensor_get_grad(&a);
    assert_non_null(grad_a);
    
    for (int i = 0; i < 4; i++) {
        assert_true(op_add_approx_equal(op_add_get_storage_value(grad_a, i), 2.0f, OP_ADD_EPSILON));
    }
}

// ============================================================================
// Test: Add backward - one input no grad
// ============================================================================

static void test_op_add_backward_one_no_grad(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 2};
    cgrad_tensor a, b, c;
    
    cgrad_tensor_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_init(&b, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    
    cgrad_tensor_set_requires_grad(&a, 0);
    
    cgrad_tensor_fill(&a, 2.0f);
    cgrad_tensor_fill(&b, 3.0f);
    
    cgrad_tensor_add(&a, &b, &c);
    
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
    
    for (int i = 0; i < 4; i++) {
        assert_true(op_add_approx_equal(op_add_get_storage_value(grad_b, i), 1.0f, OP_ADD_EPSILON));
    }
}

// ============================================================================
// Test: Add backward - chained
// ============================================================================

static void test_op_add_backward_chained(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 2};
    cgrad_tensor a, b, c, d;
    
    cgrad_tensor_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_init(&b, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_fill(&a, 1.0f);
    cgrad_tensor_fill(&b, 2.0f);
    
    // c = a + b
    cgrad_tensor_add(&a, &b, &c);
    
    // d = c + a (a used twice)
    cgrad_tensor_add(&c, &a, &d);
    
    int ret = cgrad_tensor_execute(&d);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    ret = cgrad_tensor_backward(&d);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // d = (a + b) + a = 2a + b
    // dd/da = 2, dd/db = 1
    cgrad_storage* grad_a = cgrad_tensor_get_grad(&a);
    cgrad_storage* grad_b = cgrad_tensor_get_grad(&b);
    
    assert_non_null(grad_a);
    assert_non_null(grad_b);
    
    for (int i = 0; i < 4; i++) {
        assert_true(op_add_approx_equal(op_add_get_storage_value(grad_a, i), 2.0f, OP_ADD_EPSILON));
        assert_true(op_add_approx_equal(op_add_get_storage_value(grad_b, i), 1.0f, OP_ADD_EPSILON));
    }
}

// ============================================================================
// Test Suite
// ============================================================================

int run_cgrad_op_add_tests(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_teardown(test_op_add_forward, op_add_teardown_test),
        cmocka_unit_test_teardown(test_op_add_backward_basic, op_add_teardown_test),
        cmocka_unit_test_teardown(test_op_add_backward_same_tensor, op_add_teardown_test),
        cmocka_unit_test_teardown(test_op_add_backward_one_no_grad, op_add_teardown_test),
        cmocka_unit_test_teardown(test_op_add_backward_chained, op_add_teardown_test),
    };
    
    return cmocka_run_group_tests_name("cgrad_op_add", tests, NULL, NULL);
}

#ifndef TEST_ALL_MAIN
int main(void) {
    return run_cgrad_op_add_tests();
}
#endif