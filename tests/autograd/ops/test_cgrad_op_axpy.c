#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdlib.h>
#include <math.h>

#include "cgrad.h"
#include "autograd/cgrad_ops.h"
#include "cgrad_status.h"
#include "storage/cgrad_storage.h"
#include "storage/cgrad_storage_layout.h"

#define OP_AXPY_EPSILON 1e-4f

// ============================================================================
// Setup and Teardown
// ============================================================================

static int axpy_setup_test(void **state) {
    (void) state;
    cgrad_init();
    return 0;
}

static int axpy_teardown_test(void **state) {
    (void) state;
    cgrad_cleanup();
    return 0;
}

// ============================================================================
// Test: AXPY forward
// ============================================================================

static void test_op_axpy_forward(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 2};
    cgrad_storage a, b, c;
    
    cgrad_storage_init(&a, shape, 2, "cpu_f32");
    cgrad_storage_init(&b, shape, 2, "cpu_f32");
    cgrad_storage_init(&c, shape, 2, "cpu_f32");
    
    cgrad_storage_fill(&a, 2.0f);
    cgrad_storage_fill(&b, 3.0f);
    cgrad_storage_fill(&c, 0.0f);
    
    // Get the AXPY operation descriptor
    const cgrad_op_descriptor* op_desc = &cgrad_op_axpy;
    assert_non_null(op_desc);
    
    // Prepare inputs
    cgrad_storage* inputs[2] = {&a, &b};
    
    // Prepare metadata for addition (alpha = 1.0)
    cgrad_op_metadata metadata = {0};
    metadata.axpy.alpha = 1.0f;
    
    // Execute forward pass
    void* ctx = NULL;
    int ret = op_desc->forward(inputs, 2, &metadata, &c, &ctx, 1);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // 2 + 3 = 5
    float value;
    uint32_t idx[2];
    for (uint32_t i = 0; i < 2; i++) {
        for (uint32_t j = 0; j < 2; j++) {
            idx[0] = i; idx[1] = j;
            cgrad_storage_get(&c, idx, 2, &value);
            assert_true(fabsf(value - 5.0f) < OP_AXPY_EPSILON);
        }
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
    
    cgrad_storage_init(&a, shape, 2, "cpu_f32");
    cgrad_storage_init(&b, shape, 2, "cpu_f32");
    cgrad_storage_init(&c, shape, 2, "cpu_f32");
    cgrad_storage_init(&grad_a, shape, 2, "cpu_f32");
    cgrad_storage_init(&grad_b, shape, 2, "cpu_f32");
    cgrad_storage_init(&grad_c, shape, 2, "cpu_f32");
    
    cgrad_storage_fill(&a, 2.0f);
    cgrad_storage_fill(&b, 3.0f);
    cgrad_storage_fill(&c, 0.0f);
    cgrad_storage_fill(&grad_a, 0.0f);
    cgrad_storage_fill(&grad_b, 0.0f);
    cgrad_storage_fill(&grad_c, 1.0f);  // Gradient from upstream
    
    // Get the AXPY operation descriptor
    const cgrad_op_descriptor* op_desc = &cgrad_op_axpy;
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
    int ret = op_desc->forward(inputs, 2, &metadata, &c, &ctx, 1);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Execute backward pass
    ret = op_desc->backward(inputs, 2, &c, &grad_c, &metadata, ctx, grad_inputs, input_requires_grad);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // For c = a + b: dc/da = 1, dc/db = 1
    float value;
    uint32_t idx[2];
    for (uint32_t i = 0; i < 2; i++) {
        for (uint32_t j = 0; j < 2; j++) {
            idx[0] = i; idx[1] = j;
            cgrad_storage_get(&grad_a, idx, 2, &value);
            assert_true(fabsf(value - 1.0f) < OP_AXPY_EPSILON);
            cgrad_storage_get(&grad_b, idx, 2, &value);
            assert_true(fabsf(value - 1.0f) < OP_AXPY_EPSILON);
        }
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
    
    cgrad_storage_init(&a, shape, 2, "cpu_f32");
    cgrad_storage_init(&b, shape, 2, "cpu_f32");
    cgrad_storage_init(&c, shape, 2, "cpu_f32");
    cgrad_storage_init(&grad_b, shape, 2, "cpu_f32");
    cgrad_storage_init(&grad_c, shape, 2, "cpu_f32");
    
    cgrad_storage_fill(&a, 2.0f);
    cgrad_storage_fill(&b, 3.0f);
    cgrad_storage_fill(&c, 0.0f);
    cgrad_storage_fill(&grad_b, 0.0f);
    cgrad_storage_fill(&grad_c, 1.0f);
    
    // Get the AXPY operation descriptor
    const cgrad_op_descriptor* op_desc = &cgrad_op_axpy;
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
    int ret = op_desc->forward(inputs, 2, &metadata, &c, &ctx, 1);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Execute backward pass
    ret = op_desc->backward(inputs, 2, &c, &grad_c, &metadata, ctx, grad_inputs, input_requires_grad);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // b should have gradient
    float value;
    uint32_t idx[2];
    for (uint32_t i = 0; i < 2; i++) {
        for (uint32_t j = 0; j < 2; j++) {
            idx[0] = i; idx[1] = j;
            cgrad_storage_get(&grad_b, idx, 2, &value);
            assert_true(fabsf(value - 1.0f) < OP_AXPY_EPSILON);
        }
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
    
    cgrad_storage_init(&a, shape, 2, "cpu_f32");
    cgrad_storage_init(&b, shape, 2, "cpu_f32");
    cgrad_storage_init(&c, shape, 2, "cpu_f32");
    cgrad_storage_init(&grad_a, shape, 2, "cpu_f32");
    cgrad_storage_init(&grad_b, shape, 2, "cpu_f32");
    cgrad_storage_init(&grad_c, shape, 2, "cpu_f32");
    
    cgrad_storage_fill(&a, 1.0f);
    cgrad_storage_fill(&b, 2.0f);
    cgrad_storage_fill(&c, 0.0f);
    cgrad_storage_fill(&grad_a, 0.0f);  // Start with zero gradient
    cgrad_storage_fill(&grad_b, 0.0f);  // Start with zero gradient
    cgrad_storage_fill(&grad_c, 1.0f);
    
    // Get the AXPY operation descriptor
    const cgrad_op_descriptor* op_desc = &cgrad_op_axpy;
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
    int ret = op_desc->forward(inputs, 2, &metadata, &c, &ctx, 1);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Execute backward pass - should accumulate gradients
    ret = op_desc->backward(inputs, 2, &c, &grad_c, &metadata, ctx, grad_inputs, input_requires_grad);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // For c = a + b: dc/da = 1, dc/db = 1
    float value;
    uint32_t idx[2];
    for (uint32_t i = 0; i < 2; i++) {
        for (uint32_t j = 0; j < 2; j++) {
            idx[0] = i; idx[1] = j;
            cgrad_storage_get(&grad_a, idx, 2, &value);
            assert_true(fabsf(value - 1.0f) < OP_AXPY_EPSILON);
            cgrad_storage_get(&grad_b, idx, 2, &value);
            assert_true(fabsf(value - 1.0f) < OP_AXPY_EPSILON);
        }
    }
    
    cgrad_storage_free(&a);
    cgrad_storage_free(&b);
    cgrad_storage_free(&c);
    cgrad_storage_free(&grad_a);
    cgrad_storage_free(&grad_b);
    cgrad_storage_free(&grad_c);
}

// ============================================================================
// Test: AXPY backward - broadcasting (3, 1) + (3, 4)
// ============================================================================

static void test_op_axpy_backward_broadcast_dim1(void **state) {
    (void) state;
    
    // Forward: x: (2, 1) + y: (2, 3) -> output: (2, 3)
    // x = [[1], [2]]
    // y = [[3, 4, 5], [6, 7, 8]]
    // output = [[4, 5, 6], [8, 9, 10]]
    
    // Backward: grad_output: (2, 3) = all ones
    // grad_x should be (2, 1) = [[3], [3]]  (sum across dim 1)
    // grad_y should be (2, 3) = all ones
    
    uint32_t x_shape[] = {2, 1};
    uint32_t y_shape[] = {2, 3};
    cgrad_storage x, y, output;
    cgrad_storage grad_x, grad_y, grad_output;
    
    cgrad_storage_init(&x, x_shape, 2, "cpu_f32");
    cgrad_storage_init(&y, y_shape, 2, "cpu_f32");
    cgrad_storage_init(&output, y_shape, 2, "cpu_f32");
    cgrad_storage_init(&grad_x, x_shape, 2, "cpu_f32");
    cgrad_storage_init(&grad_y, y_shape, 2, "cpu_f32");
    cgrad_storage_init(&grad_output, y_shape, 2, "cpu_f32");
    
    cgrad_storage_fill(&x, 1.0f);
    cgrad_storage_fill(&y, 2.0f);
    cgrad_storage_fill(&grad_x, 0.0f);
    cgrad_storage_fill(&grad_y, 0.0f);
    cgrad_storage_fill(&grad_output, 1.0f);
    
    // Get the AXPY operation descriptor
    const cgrad_op_descriptor* op_desc = &cgrad_op_axpy;
    
    // Prepare inputs
    cgrad_storage* inputs[2] = {&x, &y};
    cgrad_storage* grad_inputs[2] = {&grad_x, &grad_y};
    int input_requires_grad[2] = {1, 1};
    
    // Prepare metadata for addition (alpha = 1.0)
    cgrad_op_metadata metadata = {0};
    metadata.axpy.alpha = 1.0f;
    
    // Execute forward pass
    void* ctx = NULL;
    int ret = op_desc->forward(inputs, 2, &metadata, &output, &ctx, 1);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Execute backward pass
    ret = op_desc->backward(inputs, 2, &output, &grad_output, &metadata, ctx, grad_inputs, input_requires_grad);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // grad_x should be [[3], [3]] (sum of 3 ones across dim 1)
    float value;
    uint32_t idx[2];
    for (uint32_t i = 0; i < 2; i++) {
        idx[0] = i; idx[1] = 0;
        cgrad_storage_get(&grad_x, idx, 2, &value);
        assert_true(fabsf(value - 3.0f) < OP_AXPY_EPSILON);
    }
    
    // grad_y should be all ones
    for (uint32_t i = 0; i < 2; i++) {
        for (uint32_t j = 0; j < 3; j++) {
            idx[0] = i; idx[1] = j;
            cgrad_storage_get(&grad_y, idx, 2, &value);
            assert_true(fabsf(value - 1.0f) < OP_AXPY_EPSILON);
        }
    }
    
    cgrad_storage_free(&x);
    cgrad_storage_free(&y);
    cgrad_storage_free(&output);
    cgrad_storage_free(&grad_x);
    cgrad_storage_free(&grad_y);
    cgrad_storage_free(&grad_output);
}

// ============================================================================
// Test: AXPY backward - broadcasting (1, 4) + (3, 4)
// ============================================================================

static void test_op_axpy_backward_broadcast_dim0(void **state) {
    (void) state;
    
    // Forward: x: (1, 3) + y: (2, 3) -> output: (2, 3)
    // Backward: grad_output: (2, 3) = all ones
    // grad_x should be (1, 3) = [[2, 2, 2]]  (sum across dim 0)
    // grad_y should be (2, 3) = all ones
    
    uint32_t x_shape[] = {1, 3};
    uint32_t y_shape[] = {2, 3};
    cgrad_storage x, y, output;
    cgrad_storage grad_x, grad_y, grad_output;
    
    cgrad_storage_init(&x, x_shape, 2, "cpu_f32");
    cgrad_storage_init(&y, y_shape, 2, "cpu_f32");
    cgrad_storage_init(&output, y_shape, 2, "cpu_f32");
    cgrad_storage_init(&grad_x, x_shape, 2, "cpu_f32");
    cgrad_storage_init(&grad_y, y_shape, 2, "cpu_f32");
    cgrad_storage_init(&grad_output, y_shape, 2, "cpu_f32");
    
    cgrad_storage_fill(&x, 1.0f);
    cgrad_storage_fill(&y, 2.0f);
    cgrad_storage_fill(&grad_x, 0.0f);
    cgrad_storage_fill(&grad_y, 0.0f);
    cgrad_storage_fill(&grad_output, 1.0f);
    
    // Get the AXPY operation descriptor
    const cgrad_op_descriptor* op_desc = &cgrad_op_axpy;
    
    // Prepare inputs
    cgrad_storage* inputs[2] = {&x, &y};
    cgrad_storage* grad_inputs[2] = {&grad_x, &grad_y};
    int input_requires_grad[2] = {1, 1};
    
    // Prepare metadata for addition (alpha = 1.0)
    cgrad_op_metadata metadata = {0};
    metadata.axpy.alpha = 1.0f;
    
    // Execute forward pass
    void* ctx = NULL;
    int ret = op_desc->forward(inputs, 2, &metadata, &output, &ctx, 1);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Execute backward pass
    ret = op_desc->backward(inputs, 2, &output, &grad_output, &metadata, ctx, grad_inputs, input_requires_grad);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // grad_x should be [[2, 2, 2]] (sum of 2 ones across dim 0)
    float value;
    uint32_t idx[2];
    idx[0] = 0;
    for (uint32_t j = 0; j < 3; j++) {
        idx[1] = j;
        cgrad_storage_get(&grad_x, idx, 2, &value);
        assert_true(fabsf(value - 2.0f) < OP_AXPY_EPSILON);
    }
    
    // grad_y should be all ones
    for (uint32_t i = 0; i < 2; i++) {
        for (uint32_t j = 0; j < 3; j++) {
            idx[0] = i; idx[1] = j;
            cgrad_storage_get(&grad_y, idx, 2, &value);
            assert_true(fabsf(value - 1.0f) < OP_AXPY_EPSILON);
        }
    }
    
    cgrad_storage_free(&x);
    cgrad_storage_free(&y);
    cgrad_storage_free(&output);
    cgrad_storage_free(&grad_x);
    cgrad_storage_free(&grad_y);
    cgrad_storage_free(&grad_output);
}

// ============================================================================
// Test: AXPY backward - broadcasting both inputs (2, 1) + (1, 3)
// ============================================================================

static void test_op_axpy_backward_broadcast_both(void **state) {
    (void) state;
    
    // Forward: x: (2, 1) + y: (1, 3) -> output: (2, 3)
    // Backward: grad_output: (2, 3) = all ones
    // grad_x should be (2, 1) = [[3], [3]]  (sum across dim 1)
    // grad_y should be (1, 3) = [[2, 2, 2]]  (sum across dim 0)
    
    uint32_t x_shape[] = {2, 1};
    uint32_t y_shape[] = {1, 3};
    uint32_t out_shape[] = {2, 3};
    cgrad_storage x, y, output;
    cgrad_storage grad_x, grad_y, grad_output;
    
    cgrad_storage_init(&x, x_shape, 2, "cpu_f32");
    cgrad_storage_init(&y, y_shape, 2, "cpu_f32");
    cgrad_storage_init(&output, out_shape, 2, "cpu_f32");
    cgrad_storage_init(&grad_x, x_shape, 2, "cpu_f32");
    cgrad_storage_init(&grad_y, y_shape, 2, "cpu_f32");
    cgrad_storage_init(&grad_output, out_shape, 2, "cpu_f32");
    
    cgrad_storage_fill(&x, 1.0f);
    cgrad_storage_fill(&y, 2.0f);
    cgrad_storage_fill(&grad_x, 0.0f);
    cgrad_storage_fill(&grad_y, 0.0f);
    cgrad_storage_fill(&grad_output, 1.0f);
    
    // Get the AXPY operation descriptor
    const cgrad_op_descriptor* op_desc = &cgrad_op_axpy;
    
    // Prepare inputs
    cgrad_storage* inputs[2] = {&x, &y};
    cgrad_storage* grad_inputs[2] = {&grad_x, &grad_y};
    int input_requires_grad[2] = {1, 1};
    
    // Prepare metadata for addition (alpha = 1.0)
    cgrad_op_metadata metadata = {0};
    metadata.axpy.alpha = 1.0f;
    
    // Execute forward pass
    void* ctx = NULL;
    int ret = op_desc->forward(inputs, 2, &metadata, &output, &ctx, 1);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Execute backward pass
    ret = op_desc->backward(inputs, 2, &output, &grad_output, &metadata, ctx, grad_inputs, input_requires_grad);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // grad_x should be [[3], [3]] (sum of 3 ones across dim 1)
    float value;
    uint32_t idx[2];
    for (uint32_t i = 0; i < 2; i++) {
        idx[0] = i; idx[1] = 0;
        cgrad_storage_get(&grad_x, idx, 2, &value);
        assert_true(fabsf(value - 3.0f) < OP_AXPY_EPSILON);
    }
    
    // grad_y should be [[2, 2, 2]] (sum of 2 ones across dim 0)
    idx[0] = 0;
    for (uint32_t j = 0; j < 3; j++) {
        idx[1] = j;
        cgrad_storage_get(&grad_y, idx, 2, &value);
        assert_true(fabsf(value - 2.0f) < OP_AXPY_EPSILON);
    }
    
    cgrad_storage_free(&x);
    cgrad_storage_free(&y);
    cgrad_storage_free(&output);
    cgrad_storage_free(&grad_x);
    cgrad_storage_free(&grad_y);
    cgrad_storage_free(&grad_output);
}

// ============================================================================
// Test: AXPY backward - broadcasting with alpha != 1.0
// ============================================================================

static void test_op_axpy_backward_broadcast_with_alpha(void **state) {
    (void) state;
    
    // Forward: x: (2, 1) + y: (2, 3) with alpha = 2.0 -> output: (2, 3)
    // Backward: grad_output: (2, 3) = all ones
    // grad_x should be (2, 1) = [[6], [6]]  (alpha * sum of 3 ones across dim 1)
    // grad_y should be (2, 3) = all ones
    
    uint32_t x_shape[] = {2, 1};
    uint32_t y_shape[] = {2, 3};
    cgrad_storage x, y, output;
    cgrad_storage grad_x, grad_y, grad_output;
    
    cgrad_storage_init(&x, x_shape, 2, "cpu_f32");
    cgrad_storage_init(&y, y_shape, 2, "cpu_f32");
    cgrad_storage_init(&output, y_shape, 2, "cpu_f32");
    cgrad_storage_init(&grad_x, x_shape, 2, "cpu_f32");
    cgrad_storage_init(&grad_y, y_shape, 2, "cpu_f32");
    cgrad_storage_init(&grad_output, y_shape, 2, "cpu_f32");
    
    cgrad_storage_fill(&x, 1.0f);
    cgrad_storage_fill(&y, 2.0f);
    cgrad_storage_fill(&grad_x, 0.0f);
    cgrad_storage_fill(&grad_y, 0.0f);
    cgrad_storage_fill(&grad_output, 1.0f);
    
    // Get the AXPY operation descriptor
    const cgrad_op_descriptor* op_desc = &cgrad_op_axpy;
    
    // Prepare inputs
    cgrad_storage* inputs[2] = {&x, &y};
    cgrad_storage* grad_inputs[2] = {&grad_x, &grad_y};
    int input_requires_grad[2] = {1, 1};
    
    // Prepare metadata with alpha = 2.0
    cgrad_op_metadata metadata = {0};
    metadata.axpy.alpha = 2.0f;
    
    // Execute forward pass
    void* ctx = NULL;
    int ret = op_desc->forward(inputs, 2, &metadata, &output, &ctx, 1);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Execute backward pass
    ret = op_desc->backward(inputs, 2, &output, &grad_output, &metadata, ctx, grad_inputs, input_requires_grad);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // grad_x should be [[6], [6]] (alpha=2.0 * sum of 3 ones across dim 1)
    float value;
    uint32_t idx[2];
    for (uint32_t i = 0; i < 2; i++) {
        idx[0] = i; idx[1] = 0;
        cgrad_storage_get(&grad_x, idx, 2, &value);
        assert_true(fabsf(value - 6.0f) < OP_AXPY_EPSILON);
    }
    
    // grad_y should be all ones
    for (uint32_t i = 0; i < 2; i++) {
        for (uint32_t j = 0; j < 3; j++) {
            idx[0] = i; idx[1] = j;
            cgrad_storage_get(&grad_y, idx, 2, &value);
            assert_true(fabsf(value - 1.0f) < OP_AXPY_EPSILON);
        }
    }
    
    cgrad_storage_free(&x);
    cgrad_storage_free(&y);
    cgrad_storage_free(&output);
    cgrad_storage_free(&grad_x);
    cgrad_storage_free(&grad_y);
    cgrad_storage_free(&grad_output);
}

// ============================================================================
// Test Suite
// ============================================================================

int run_cgrad_op_axpy_tests(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_op_axpy_forward, axpy_setup_test, axpy_teardown_test),
        cmocka_unit_test_setup_teardown(test_op_axpy_backward_basic, axpy_setup_test, axpy_teardown_test),
        cmocka_unit_test_setup_teardown(test_op_axpy_backward_one_no_grad, axpy_setup_test, axpy_teardown_test),
        cmocka_unit_test_setup_teardown(test_op_axpy_backward_accumulation, axpy_setup_test, axpy_teardown_test),
        cmocka_unit_test_setup_teardown(test_op_axpy_backward_broadcast_dim1, axpy_setup_test, axpy_teardown_test),
        cmocka_unit_test_setup_teardown(test_op_axpy_backward_broadcast_dim0, axpy_setup_test, axpy_teardown_test),
        cmocka_unit_test_setup_teardown(test_op_axpy_backward_broadcast_both, axpy_setup_test, axpy_teardown_test),
        cmocka_unit_test_setup_teardown(test_op_axpy_backward_broadcast_with_alpha, axpy_setup_test, axpy_teardown_test),
    };
    
    return cmocka_run_group_tests_name("cgrad_op_axpy", tests, NULL, NULL);
}

#ifndef TEST_ALL_MAIN
int main(void) {
    return run_cgrad_op_axpy_tests();
}
#endif
