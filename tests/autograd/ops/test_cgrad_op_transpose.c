#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdlib.h>
#include <math.h>

#include "autograd/cgrad_ops.h"
#include "cgrad_errors.h"
#include "storage/cgrad_storage.h"
#include "storage/cgrad_storage_layout.h"
#include "storage/backends/cgrad_storage_f32_cpu.h"

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
    cgrad_storage_cleanup_global_registry();
    return 0;
}

// ============================================================================
// Test: Transpose forward
// ============================================================================

static void test_op_transpose_forward(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 3};
    cgrad_storage a, b;
    
    cgrad_storage_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_storage_fill(&a, 1.0f);
    
    // Get the TRANSPOSE operation descriptor
    const cgrad_op_descriptor* op_desc = cgrad_get_op_descriptor(CGRAD_OP_TRANSPOSE);
    assert_non_null(op_desc);
    
    // Prepare inputs
    cgrad_storage* inputs[1] = {&a};
    
    // Prepare metadata for transpose (perm = {1, 0})
    cgrad_op_metadata metadata = {0};
    metadata.transpose.perm[0] = 1;
    metadata.transpose.perm[1] = 0;
    metadata.transpose.ndim = 2;
    
    // Execute forward pass - output will be initialized by transpose
    void* ctx = NULL;
    memset(&b, 0, sizeof(b));
    int ret = op_desc->forward(inputs, 1, &metadata, &b, &ctx, 1);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Get the layout to check shape
    cgrad_storage_f32_cpu* cpu_storage = (cgrad_storage_f32_cpu*)b.data;
    assert_non_null(cpu_storage);
    
    // Output should be (3, 2) - check via layout
    assert_int_equal(cpu_storage->layout.shape[TENSOR_DIM - 2], 3);
    assert_int_equal(cpu_storage->layout.shape[TENSOR_DIM - 1], 2);
    
    cgrad_storage_free(&a);
    cgrad_storage_free(&b);
}

// ============================================================================
// Test: Transpose backward - basic
// ============================================================================

static void test_op_transpose_backward_basic(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 3};
    uint32_t shape_t[] = {3, 2};
    cgrad_storage a, b;
    cgrad_storage grad_a, grad_b;
    
    cgrad_storage_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_storage_init(&grad_a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_storage_fill(&a, 1.0f);
    cgrad_storage_fill(&grad_a, 0.0f);
    
    // Get the TRANSPOSE operation descriptor
    const cgrad_op_descriptor* op_desc = cgrad_get_op_descriptor(CGRAD_OP_TRANSPOSE);
    assert_non_null(op_desc);
    
    // Prepare inputs
    cgrad_storage* inputs[1] = {&a};
    
    // Prepare metadata for transpose (perm = {1, 0})
    cgrad_op_metadata metadata = {0};
    metadata.transpose.perm[0] = 1;
    metadata.transpose.perm[1] = 0;
    metadata.transpose.ndim = 2;
    
    // Execute forward pass
    void* ctx = NULL;
    memset(&b, 0, sizeof(b));
    int ret = op_desc->forward(inputs, 1, &metadata, &b, &ctx, 1);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Initialize gradient for output (transposed shape)
    cgrad_storage_init(&grad_b, shape_t, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_storage_fill(&grad_b, 1.0f);
    
    // Execute backward pass
    cgrad_storage* grad_inputs[1] = {&grad_a};
    int input_requires_grad[1] = {1};
    ret = op_desc->backward(inputs, 1, &b, &grad_b, &metadata, ctx, grad_inputs, input_requires_grad);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Gradient should be transposed back to original shape
    // All gradients should be 1.0
    for (int i = 0; i < 6; i++) {
        assert_true(op_transpose_approx_equal(op_transpose_get_storage_value(&grad_a, i), 1.0f, OP_TRANSPOSE_EPSILON));
    }
    
    cgrad_storage_free(&a);
    cgrad_storage_free(&b);
    cgrad_storage_free(&grad_a);
    cgrad_storage_free(&grad_b);
}

// ============================================================================
// Test: Transpose backward - no grad
// ============================================================================

static void test_op_transpose_backward_no_grad(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 3};
    uint32_t shape_t[] = {3, 2};
    cgrad_storage a, b;
    cgrad_storage grad_b;
    
    cgrad_storage_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_storage_fill(&a, 1.0f);
    
    // Get the TRANSPOSE operation descriptor
    const cgrad_op_descriptor* op_desc = cgrad_get_op_descriptor(CGRAD_OP_TRANSPOSE);
    assert_non_null(op_desc);
    
    // Prepare inputs
    cgrad_storage* inputs[1] = {&a};
    
    // Prepare metadata for transpose (perm = {1, 0})
    cgrad_op_metadata metadata = {0};
    metadata.transpose.perm[0] = 1;
    metadata.transpose.perm[1] = 0;
    metadata.transpose.ndim = 2;
    
    // Execute forward pass
    void* ctx = NULL;
    memset(&b, 0, sizeof(b));
    int ret = op_desc->forward(inputs, 1, &metadata, &b, &ctx, 1);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Initialize gradient for output
    cgrad_storage_init(&grad_b, shape_t, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
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
// Test: Transpose backward - double transpose
// ============================================================================

static void test_op_transpose_backward_double(void **state) {
    (void) state;
    
    uint32_t shape[] = {2, 3};
    uint32_t shape_t[] = {3, 2};
    cgrad_storage a, b, c;
    cgrad_storage grad_a, grad_b, grad_c;
    
    cgrad_storage_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_storage_init(&grad_a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_storage_fill(&a, 1.0f);
    cgrad_storage_fill(&grad_a, 0.0f);
    
    // Get the TRANSPOSE operation descriptor
    const cgrad_op_descriptor* op_desc = cgrad_get_op_descriptor(CGRAD_OP_TRANSPOSE);
    assert_non_null(op_desc);
    
    // Prepare metadata for transpose (perm = {1, 0})
    cgrad_op_metadata metadata = {0};
    metadata.transpose.perm[0] = 1;
    metadata.transpose.perm[1] = 0;
    metadata.transpose.ndim = 2;
    
    // First transpose: b = transpose(a)
    cgrad_storage* inputs1[1] = {&a};
    void* ctx1 = NULL;
    memset(&b, 0, sizeof(b));
    int ret = op_desc->forward(inputs1, 1, &metadata, &b, &ctx1, 1);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Second transpose: c = transpose(b) = a (back to original shape)
    cgrad_storage* inputs2[1] = {&b};
    void* ctx2 = NULL;
    memset(&c, 0, sizeof(c));
    ret = op_desc->forward(inputs2, 1, &metadata, &c, &ctx2, 1);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Initialize gradients
    cgrad_storage_init(&grad_b, shape_t, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_storage_init(&grad_c, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_storage_fill(&grad_b, 0.0f);
    cgrad_storage_fill(&grad_c, 1.0f);
    
    // Backward through second transpose
    cgrad_storage* grad_inputs2[1] = {&grad_b};
    int input_requires_grad2[1] = {1};
    ret = op_desc->backward(inputs2, 1, &c, &grad_c, &metadata, ctx2, grad_inputs2, input_requires_grad2);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Backward through first transpose
    cgrad_storage* grad_inputs1[1] = {&grad_a};
    int input_requires_grad1[1] = {1};
    ret = op_desc->backward(inputs1, 1, &b, &grad_b, &metadata, ctx1, grad_inputs1, input_requires_grad1);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // All gradients should be 1.0
    for (int i = 0; i < 6; i++) {
        assert_true(op_transpose_approx_equal(op_transpose_get_storage_value(&grad_a, i), 1.0f, OP_TRANSPOSE_EPSILON));
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

int run_cgrad_op_transpose_tests(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_teardown(test_op_transpose_forward, op_transpose_teardown_test),
        cmocka_unit_test_teardown(test_op_transpose_backward_basic, op_transpose_teardown_test),
        cmocka_unit_test_teardown(test_op_transpose_backward_no_grad, op_transpose_teardown_test),
        cmocka_unit_test_teardown(test_op_transpose_backward_double, op_transpose_teardown_test),
    };
    
    return cmocka_run_group_tests_name("cgrad_op_transpose", tests, NULL, NULL);
}

#ifndef TEST_ALL_MAIN
int main(void) {
    return run_cgrad_op_transpose_tests();
}
#endif
