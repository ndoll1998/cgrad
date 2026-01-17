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

#define OP_RESHAPE_EPSILON 1e-4f

// ============================================================================
// Setup and Teardown
// ============================================================================

static int reshape_setup_test(void **state) {
    (void) state;
    cgrad_init();
    return 0;
}

static int reshape_teardown_test(void **state) {
    (void) state;
    cgrad_cleanup();
    return 0;
}

// ============================================================================
// Test: Reshape forward
// ============================================================================

static void test_op_reshape_forward(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 3};
    cgrad_storage a, b;
    
    cgrad_storage_init(&a, shape, 2, "cpu_f32");
    cgrad_storage_fill(&a, 1.0f);
    
    // Get the RESHAPE operation descriptor
    const cgrad_op_descriptor* op_desc = cgrad_get_op_descriptor(CGRAD_OP_RESHAPE);
    assert_non_null(op_desc);
    
    // Prepare inputs
    cgrad_storage* inputs[1] = {&a};
    
    // Prepare metadata for reshape (new_shape = {3, 2})
    cgrad_op_metadata metadata = {0};
    metadata.reshape.new_shape[0] = 3;
    metadata.reshape.new_shape[1] = 2;
    metadata.reshape.ndim = 2;
    
    // Execute forward pass - output will be initialized by reshape
    void* ctx = NULL;
    memset(&b, 0, sizeof(b));
    int ret = op_desc->forward(inputs, 1, &metadata, &b, &ctx, 1);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Get the layout to check shape
    cgrad_storage_layout* layout = b.backend->storage_get_layout(b.data);
    assert_non_null(layout);
    
    // Output should be (3, 2)
    assert_int_equal(layout->shape[TENSOR_DIM - 2], 3);
    assert_int_equal(layout->shape[TENSOR_DIM - 1], 2);
    
    cgrad_storage_free(&a);
    cgrad_storage_free(&b);
}

// ============================================================================
// Test: Reshape backward - basic
// ============================================================================

static void test_op_reshape_backward_basic(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 3};
    uint32_t shape_r[] = {3, 2};
    cgrad_storage a, b;
    cgrad_storage grad_a, grad_b;
    
    cgrad_storage_init(&a, shape, 2, "cpu_f32");
    cgrad_storage_init(&grad_a, shape, 2, "cpu_f32");
    cgrad_storage_fill(&a, 1.0f);
    cgrad_storage_fill(&grad_a, 0.0f);
    
    // Get the RESHAPE operation descriptor
    const cgrad_op_descriptor* op_desc = cgrad_get_op_descriptor(CGRAD_OP_RESHAPE);
    assert_non_null(op_desc);
    
    // Prepare inputs
    cgrad_storage* inputs[1] = {&a};
    
    // Prepare metadata for reshape (new_shape = {3, 2})
    cgrad_op_metadata metadata = {0};
    metadata.reshape.new_shape[0] = 3;
    metadata.reshape.new_shape[1] = 2;
    metadata.reshape.ndim = 2;
    
    // Execute forward pass
    void* ctx = NULL;
    memset(&b, 0, sizeof(b));
    int ret = op_desc->forward(inputs, 1, &metadata, &b, &ctx, 1);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Initialize gradient for output (reshaped shape)
    cgrad_storage_init(&grad_b, shape_r, 2, "cpu_f32");
    cgrad_storage_fill(&grad_b, 1.0f);
    
    // Execute backward pass
    cgrad_storage* grad_inputs[1] = {&grad_a};
    int input_requires_grad[1] = {1};
    ret = op_desc->backward(inputs, 1, &b, &grad_b, &metadata, ctx, grad_inputs, input_requires_grad);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Gradient should be reshaped back to original shape
    // All gradients should be 1.0
    float value;
    uint32_t idx[2];
    for (uint32_t i = 0; i < 2; i++) {
        for (uint32_t j = 0; j < 3; j++) {
            idx[0] = i; idx[1] = j;
            cgrad_storage_get(&grad_a, idx, 2, &value);
            assert_true(fabsf(value - 1.0f) < OP_RESHAPE_EPSILON);
        }
    }
    
    cgrad_storage_free(&a);
    cgrad_storage_free(&b);
    cgrad_storage_free(&grad_a);
    cgrad_storage_free(&grad_b);
}

// ============================================================================
// Test: Reshape backward - no grad
// ============================================================================

static void test_op_reshape_backward_no_grad(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 3};
    uint32_t shape_r[] = {3, 2};
    cgrad_storage a, b;
    cgrad_storage grad_b;
    
    cgrad_storage_init(&a, shape, 2, "cpu_f32");
    cgrad_storage_fill(&a, 1.0f);
    
    // Get the RESHAPE operation descriptor
    const cgrad_op_descriptor* op_desc = cgrad_get_op_descriptor(CGRAD_OP_RESHAPE);
    assert_non_null(op_desc);
    
    // Prepare inputs
    cgrad_storage* inputs[1] = {&a};
    
    // Prepare metadata for reshape (new_shape = {3, 2})
    cgrad_op_metadata metadata = {0};
    metadata.reshape.new_shape[0] = 3;
    metadata.reshape.new_shape[1] = 2;
    metadata.reshape.ndim = 2;
    
    // Execute forward pass
    void* ctx = NULL;
    memset(&b, 0, sizeof(b));
    int ret = op_desc->forward(inputs, 1, &metadata, &b, &ctx, 1);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Initialize gradient for output
    cgrad_storage_init(&grad_b, shape_r, 2, "cpu_f32");
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
// Test: Reshape backward - flatten
// ============================================================================

static void test_op_reshape_backward_flatten(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 3};
    uint32_t shape_flat[] = {6};
    cgrad_storage a, b;
    cgrad_storage grad_a, grad_b;
    
    cgrad_storage_init(&a, shape, 2, "cpu_f32");
    cgrad_storage_init(&grad_a, shape, 2, "cpu_f32");
    cgrad_storage_fill(&a, 1.0f);
    cgrad_storage_fill(&grad_a, 0.0f);
    
    // Get the RESHAPE operation descriptor
    const cgrad_op_descriptor* op_desc = cgrad_get_op_descriptor(CGRAD_OP_RESHAPE);
    assert_non_null(op_desc);
    
    // Prepare inputs
    cgrad_storage* inputs[1] = {&a};
    
    // Prepare metadata for reshape (flatten to 1D: new_shape = {6})
    cgrad_op_metadata metadata = {0};
    metadata.reshape.new_shape[0] = 6;
    metadata.reshape.ndim = 1;
    
    // Execute forward pass
    void* ctx = NULL;
    memset(&b, 0, sizeof(b));
    int ret = op_desc->forward(inputs, 1, &metadata, &b, &ctx, 1);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Initialize gradient for output (flattened shape)
    cgrad_storage_init(&grad_b, shape_flat, 1, "cpu_f32");
    cgrad_storage_fill(&grad_b, 1.0f);
    
    // Execute backward pass
    cgrad_storage* grad_inputs[1] = {&grad_a};
    int input_requires_grad[1] = {1};
    ret = op_desc->backward(inputs, 1, &b, &grad_b, &metadata, ctx, grad_inputs, input_requires_grad);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // All gradients should be 1.0
    float value;
    uint32_t idx[2];
    for (uint32_t i = 0; i < 2; i++) {
        for (uint32_t j = 0; j < 3; j++) {
            idx[0] = i; idx[1] = j;
            cgrad_storage_get(&grad_a, idx, 2, &value);
            assert_true(fabsf(value - 1.0f) < OP_RESHAPE_EPSILON);
        }
    }
    
    cgrad_storage_free(&a);
    cgrad_storage_free(&b);
    cgrad_storage_free(&grad_a);
    cgrad_storage_free(&grad_b);
}

// ============================================================================
// Test: Reshape backward - double reshape
// ============================================================================

static void test_op_reshape_backward_double(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 3};
    uint32_t shape1[] = {3, 2};
    uint32_t shape2[] = {6};
    cgrad_storage a, b, c;
    cgrad_storage grad_a, grad_b, grad_c;
    
    cgrad_storage_init(&a, shape, 2, "cpu_f32");
    cgrad_storage_init(&grad_a, shape, 2, "cpu_f32");
    cgrad_storage_fill(&a, 1.0f);
    cgrad_storage_fill(&grad_a, 0.0f);
    
    // Get the RESHAPE operation descriptor
    const cgrad_op_descriptor* op_desc = cgrad_get_op_descriptor(CGRAD_OP_RESHAPE);
    assert_non_null(op_desc);
    
    // First reshape: b = reshape(a, [3, 2])
    cgrad_storage* inputs1[1] = {&a};
    cgrad_op_metadata metadata1 = {0};
    metadata1.reshape.new_shape[0] = 3;
    metadata1.reshape.new_shape[1] = 2;
    metadata1.reshape.ndim = 2;
    
    void* ctx1 = NULL;
    memset(&b, 0, sizeof(b));
    int ret = op_desc->forward(inputs1, 1, &metadata1, &b, &ctx1, 1);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Second reshape: c = reshape(b, [6])
    cgrad_storage* inputs2[1] = {&b};
    cgrad_op_metadata metadata2 = {0};
    metadata2.reshape.new_shape[0] = 6;
    metadata2.reshape.ndim = 1;
    
    void* ctx2 = NULL;
    memset(&c, 0, sizeof(c));
    ret = op_desc->forward(inputs2, 1, &metadata2, &c, &ctx2, 1);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Initialize gradients
    cgrad_storage_init(&grad_b, shape1, 2, "cpu_f32");
    cgrad_storage_init(&grad_c, shape2, 1, "cpu_f32");
    cgrad_storage_fill(&grad_b, 0.0f);
    cgrad_storage_fill(&grad_c, 1.0f);
    
    // Backward through second reshape
    cgrad_storage* grad_inputs2[1] = {&grad_b};
    int input_requires_grad2[1] = {1};
    ret = op_desc->backward(inputs2, 1, &c, &grad_c, &metadata2, ctx2, grad_inputs2, input_requires_grad2);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Backward through first reshape
    cgrad_storage* grad_inputs1[1] = {&grad_a};
    int input_requires_grad1[1] = {1};
    ret = op_desc->backward(inputs1, 1, &b, &grad_b, &metadata1, ctx1, grad_inputs1, input_requires_grad1);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // All gradients should be 1.0
    float value;
    uint32_t idx[2];
    for (uint32_t i = 0; i < 2; i++) {
        for (uint32_t j = 0; j < 3; j++) {
            idx[0] = i; idx[1] = j;
            cgrad_storage_get(&grad_a, idx, 2, &value);
            assert_true(fabsf(value - 1.0f) < OP_RESHAPE_EPSILON);
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
// Test Suite
// ============================================================================

int run_cgrad_op_reshape_tests(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_op_reshape_forward, reshape_setup_test, reshape_teardown_test),
        cmocka_unit_test_setup_teardown(test_op_reshape_backward_basic, reshape_setup_test, reshape_teardown_test),
        cmocka_unit_test_setup_teardown(test_op_reshape_backward_no_grad, reshape_setup_test, reshape_teardown_test),
        cmocka_unit_test_setup_teardown(test_op_reshape_backward_flatten, reshape_setup_test, reshape_teardown_test),
        cmocka_unit_test_setup_teardown(test_op_reshape_backward_double, reshape_setup_test, reshape_teardown_test),
    };
    
    return cmocka_run_group_tests_name("cgrad_op_reshape", tests, NULL, NULL);
}

#ifndef TEST_ALL_MAIN
int main(void) {
    return run_cgrad_op_reshape_tests();
}
#endif
