#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdlib.h>
#include <math.h>

#include "autograd/cgrad_ops.h"
#include "cgrad_errors.h"
#include "storage/cgrad_storage.h"
#include "storage/backends/cgrad_storage_f32_cpu.h"

#define OP_AXPY_EPSILON 1e-4f

// ============================================================================
// Helper Functions
// ============================================================================

static float op_axpy_get_storage_value(cgrad_storage* storage, int idx) {
    cgrad_storage_f32_cpu* cpu_storage = (cgrad_storage_f32_cpu*)storage->data;
    return cpu_storage->data[idx];
}

static int op_axpy_approx_equal(float a, float b, float eps) {
    return fabsf(a - b) < eps;
}

// ============================================================================
// Setup and Teardown
// ============================================================================

static int op_axpy_teardown_test(void **state) {
    (void) state;
    cgrad_storage_cleanup_global_registry();
    return 0;
}

// ============================================================================
// Test: AXPY forward
// ============================================================================

static void test_op_axpy_forward(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 2};
    cgrad_storage a, b, c;
    
    cgrad_storage_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_storage_init(&b, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_storage_init(&c, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    
    cgrad_storage_fill(&a, 2.0f);
    cgrad_storage_fill(&b, 3.0f);
    cgrad_storage_fill(&c, 0.0f);
    
    // Get the AXPY operation descriptor
    const cgrad_op_descriptor* op_desc = cgrad_get_op_descriptor(CGRAD_OP_AXPY);
    assert_non_null(op_desc);
    
    // Prepare inputs
    cgrad_storage* inputs[2] = {&a, &b};
    
    // Prepare metadata for addition (alpha = 1.0)
    cgrad_op_metadata metadata = {0};
    metadata.axpy.alpha = 1.0f;
    
    // Execute forward pass
    void* ctx = NULL;
    int ret = op_desc->forward(inputs, 2, &metadata, &c, &ctx);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // 2 + 3 = 5
    for (int i = 0; i < 4; i++) {
        assert_true(op_axpy_approx_equal(op_axpy_get_storage_value(&c, i), 5.0f, OP_AXPY_EPSILON));
    }
    
    cgrad_storage_free(&a);
    cgrad_storage_free(&b);
    cgrad_storage_free(&c);
}

// ============================================================================
// Test: AXPY backward - basic
// ============================================================================

static void test_op_axpy_backward_basic(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 2};
    cgrad_storage a, b, c;
    cgrad_storage grad_a, grad_b, grad_c;
    
    cgrad_storage_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_storage_init(&b, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_storage_init(&c, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_storage_init(&grad_a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_storage_init(&grad_b, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_storage_init(&grad_c, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    
    cgrad_storage_fill(&a, 2.0f);
    cgrad_storage_fill(&b, 3.0f);
    cgrad_storage_fill(&c, 0.0f);
    cgrad_storage_fill(&grad_a, 0.0f);
    cgrad_storage_fill(&grad_b, 0.0f);
    cgrad_storage_fill(&grad_c, 1.0f);  // Gradient from upstream
    
    // Get the AXPY operation descriptor
    const cgrad_op_descriptor* op_desc = cgrad_get_op_descriptor(CGRAD_OP_AXPY);
    assert_non_null(op_desc);
    
    // Prepare inputs
    cgrad_storage* inputs[2] = {&a, &b};
    cgrad_storage* grad_inputs[2] = {&grad_a, &grad_b};
    int input_requires_grad[2] = {1, 1};
    
    // Prepare metadata for addition (alpha = 1.0)
    cgrad_op_metadata metadata = {0};
    metadata.axpy.alpha = 1.0f;
    
    // Execute forward pass first
    void* ctx = NULL;
    int ret = op_desc->forward(inputs, 2, &metadata, &c, &ctx);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Execute backward pass
    ret = op_desc->backward(inputs, 2, &c, &grad_c, &metadata, ctx, grad_inputs, input_requires_grad);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // For c = a + b: dc/da = 1, dc/db = 1
    for (int i = 0; i < 4; i++) {
        assert_true(op_axpy_approx_equal(op_axpy_get_storage_value(&grad_a, i), 1.0f, OP_AXPY_EPSILON));
        assert_true(op_axpy_approx_equal(op_axpy_get_storage_value(&grad_b, i), 1.0f, OP_AXPY_EPSILON));
    }
    
    cgrad_storage_free(&a);
    cgrad_storage_free(&b);
    cgrad_storage_free(&c);
    cgrad_storage_free(&grad_a);
    cgrad_storage_free(&grad_b);
    cgrad_storage_free(&grad_c);
}

// ============================================================================
// Test: AXPY backward - one input no grad
// ============================================================================

static void test_op_axpy_backward_one_no_grad(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 2};
    cgrad_storage a, b, c;
    cgrad_storage grad_b, grad_c;
    
    cgrad_storage_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_storage_init(&b, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_storage_init(&c, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_storage_init(&grad_b, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_storage_init(&grad_c, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    
    cgrad_storage_fill(&a, 2.0f);
    cgrad_storage_fill(&b, 3.0f);
    cgrad_storage_fill(&c, 0.0f);
    cgrad_storage_fill(&grad_b, 0.0f);
    cgrad_storage_fill(&grad_c, 1.0f);
    
    // Get the AXPY operation descriptor
    const cgrad_op_descriptor* op_desc = cgrad_get_op_descriptor(CGRAD_OP_AXPY);
    assert_non_null(op_desc);
    
    // Prepare inputs (a does not require grad)
    cgrad_storage* inputs[2] = {&a, &b};
    cgrad_storage* grad_inputs[2] = {NULL, &grad_b};  // grad_a is NULL
    int input_requires_grad[2] = {0, 1};
    
    // Prepare metadata for addition (alpha = 1.0)
    cgrad_op_metadata metadata = {0};
    metadata.axpy.alpha = 1.0f;
    
    // Execute forward pass first
    void* ctx = NULL;
    int ret = op_desc->forward(inputs, 2, &metadata, &c, &ctx);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Execute backward pass
    ret = op_desc->backward(inputs, 2, &c, &grad_c, &metadata, ctx, grad_inputs, input_requires_grad);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // b should have gradient
    for (int i = 0; i < 4; i++) {
        assert_true(op_axpy_approx_equal(op_axpy_get_storage_value(&grad_b, i), 1.0f, OP_AXPY_EPSILON));
    }
    
    cgrad_storage_free(&a);
    cgrad_storage_free(&b);
    cgrad_storage_free(&c);
    cgrad_storage_free(&grad_b);
    cgrad_storage_free(&grad_c);
}

// ============================================================================
// Test: AXPY backward - gradient accumulation
// ============================================================================

static void test_op_axpy_backward_accumulation(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 2};
    cgrad_storage a, b, c;
    cgrad_storage grad_a, grad_b, grad_c;
    
    cgrad_storage_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_storage_init(&b, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_storage_init(&c, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_storage_init(&grad_a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_storage_init(&grad_b, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_storage_init(&grad_c, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    
    cgrad_storage_fill(&a, 1.0f);
    cgrad_storage_fill(&b, 2.0f);
    cgrad_storage_fill(&c, 0.0f);
    cgrad_storage_fill(&grad_a, 5.0f);  // Pre-existing gradient
    cgrad_storage_fill(&grad_b, 3.0f);  // Pre-existing gradient
    cgrad_storage_fill(&grad_c, 1.0f);
    
    // Get the AXPY operation descriptor
    const cgrad_op_descriptor* op_desc = cgrad_get_op_descriptor(CGRAD_OP_AXPY);
    assert_non_null(op_desc);
    
    // Prepare inputs
    cgrad_storage* inputs[2] = {&a, &b};
    cgrad_storage* grad_inputs[2] = {&grad_a, &grad_b};
    int input_requires_grad[2] = {1, 1};
    
    // Prepare metadata for addition (alpha = 1.0)
    cgrad_op_metadata metadata = {0};
    metadata.axpy.alpha = 1.0f;
    
    // Execute forward pass first
    void* ctx = NULL;
    int ret = op_desc->forward(inputs, 2, &metadata, &c, &ctx);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Execute backward pass - should accumulate gradients
    ret = op_desc->backward(inputs, 2, &c, &grad_c, &metadata, ctx, grad_inputs, input_requires_grad);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Gradients should be accumulated: grad_a = 5 + 1 = 6, grad_b = 3 + 1 = 4
    for (int i = 0; i < 4; i++) {
        assert_true(op_axpy_approx_equal(op_axpy_get_storage_value(&grad_a, i), 6.0f, OP_AXPY_EPSILON));
        assert_true(op_axpy_approx_equal(op_axpy_get_storage_value(&grad_b, i), 4.0f, OP_AXPY_EPSILON));
    }
    
    cgrad_storage_free(&a);
    cgrad_storage_free(&b);
    cgrad_storage_free(&c);
    cgrad_storage_free(&grad_a);
    cgrad_storage_free(&grad_b);
    cgrad_storage_free(&grad_c);
}

// ============================================================================
// Test Suite
// ============================================================================

int run_cgrad_op_axpy_tests(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_teardown(test_op_axpy_forward, op_axpy_teardown_test),
        cmocka_unit_test_teardown(test_op_axpy_backward_basic, op_axpy_teardown_test),
        cmocka_unit_test_teardown(test_op_axpy_backward_one_no_grad, op_axpy_teardown_test),
        cmocka_unit_test_teardown(test_op_axpy_backward_accumulation, op_axpy_teardown_test),
    };
    
    return cmocka_run_group_tests_name("cgrad_op_axpy", tests, NULL, NULL);
}

#ifndef TEST_ALL_MAIN
int main(void) {
    return run_cgrad_op_axpy_tests();
}
#endif
