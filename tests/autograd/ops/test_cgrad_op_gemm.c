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

#define OP_GEMM_EPSILON 1e-4f

// ============================================================================
// Setup and Teardown
// ============================================================================

static int gemm_setup_test(void **state) {
    (void) state;
    cgrad_init();
    return 0;
}

static int gemm_teardown_test(void **state) {
    (void) state;
    cgrad_cleanup();
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
    uint32_t shape_c[] = {2, 2};
    cgrad_storage a, b, c;
    
    cgrad_storage_init(&a, shape_a, 2, "cpu_f32");
    cgrad_storage_init(&b, shape_b, 2, "cpu_f32");
    cgrad_storage_init(&c, shape_c, 2, "cpu_f32");
    
    cgrad_storage_fill(&a, 1.0f);
    cgrad_storage_fill(&b, 2.0f);
    cgrad_storage_fill(&c, 0.0f);
    
    // Get the GEMM operation descriptor
    const cgrad_op_descriptor* op_desc = &cgrad_op_gemm;
    assert_non_null(op_desc);
    
    // Prepare inputs
    cgrad_storage* inputs[2] = {&a, &b};
    
    // Prepare metadata for GEMM (alpha = 1.0, beta = 0.0)
    cgrad_op_metadata metadata = {0};
    metadata.gemm.alpha = 1.0f;
    metadata.gemm.beta = 0.0f;
    
    // Execute forward pass
    void* ctx = NULL;
    int ret = op_desc->forward(inputs, 2, &metadata, &c, &ctx, 1);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Each element = sum of 3 products of 1*2 = 6
    // Check all 4 elements of 2x2 result
    float value;
    uint32_t idx[2];
    for (uint32_t i = 0; i < 2; i++) {
        for (uint32_t j = 0; j < 2; j++) {
            idx[0] = i; idx[1] = j;
            cgrad_storage_get(&c, idx, 2, &value);
            assert_true(fabsf(value - 6.0f) < OP_GEMM_EPSILON);
        }
    }
    
    cgrad_storage_free(&a);
    cgrad_storage_free(&b);
    cgrad_storage_free(&c);
}

// ============================================================================
// Test: GEMM backward - basic
// ============================================================================

static void test_op_gemm_backward_basic(void **state) {
    (void) state;
    
    // A: 2x3, B: 3x2, C = A @ B: 2x2
    uint32_t shape_a[] = {2, 3};
    uint32_t shape_b[] = {3, 2};
    uint32_t shape_c[] = {2, 2};
    cgrad_storage a, b, c;
    cgrad_storage grad_a, grad_b, grad_c;
    
    cgrad_storage_init(&a, shape_a, 2, "cpu_f32");
    cgrad_storage_init(&b, shape_b, 2, "cpu_f32");
    cgrad_storage_init(&c, shape_c, 2, "cpu_f32");
    cgrad_storage_init(&grad_a, shape_a, 2, "cpu_f32");
    cgrad_storage_init(&grad_b, shape_b, 2, "cpu_f32");
    cgrad_storage_init(&grad_c, shape_c, 2, "cpu_f32");
    
    cgrad_storage_fill(&a, 1.0f);
    cgrad_storage_fill(&b, 1.0f);
    cgrad_storage_fill(&c, 0.0f);
    cgrad_storage_fill(&grad_a, 0.0f);
    cgrad_storage_fill(&grad_b, 0.0f);
    cgrad_storage_fill(&grad_c, 1.0f);
    
    // Get the GEMM operation descriptor
    const cgrad_op_descriptor* op_desc = &cgrad_op_gemm;
    assert_non_null(op_desc);
    
    // Prepare inputs
    cgrad_storage* inputs[2] = {&a, &b};
    cgrad_storage* grad_inputs[2] = {&grad_a, &grad_b};
    int input_requires_grad[2] = {1, 1};
    
    // Prepare metadata for GEMM
    cgrad_op_metadata metadata = {0};
    metadata.gemm.alpha = 1.0f;
    metadata.gemm.beta = 0.0f;
    
    // Execute forward pass first
    void* ctx = NULL;
    int ret = op_desc->forward(inputs, 2, &metadata, &c, &ctx, 1);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Execute backward pass
    ret = op_desc->backward(inputs, 2, &c, &grad_c, &metadata, ctx, grad_inputs, input_requires_grad);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // For C = A @ B with all 1s:
    // grad_A = grad_C @ B^T, where grad_C is all 1s and B^T is 2x3 of 1s
    // So grad_A should be 2x3 with each element = 2
    float value;
    uint32_t idx[2];
    for (uint32_t i = 0; i < 2; i++) {
        for (uint32_t j = 0; j < 3; j++) {
            idx[0] = i; idx[1] = j;
            cgrad_storage_get(&grad_a, idx, 2, &value);
            assert_true(fabsf(value - 2.0f) < OP_GEMM_EPSILON);
        }
    }
    
    // grad_B = A^T @ grad_C, where A^T is 3x2 of 1s and grad_C is 2x2 of 1s
    // So grad_B should be 3x2 with each element = 2
    for (uint32_t i = 0; i < 3; i++) {
        for (uint32_t j = 0; j < 2; j++) {
            idx[0] = i; idx[1] = j;
            cgrad_storage_get(&grad_b, idx, 2, &value);
            assert_true(fabsf(value - 2.0f) < OP_GEMM_EPSILON);
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
// Test: GEMM backward - one input no grad
// ============================================================================

static void test_op_gemm_backward_one_no_grad(void **state) {
    (void) state;
    
    uint32_t shape_a[] = {2, 3};
    uint32_t shape_b[] = {3, 2};
    uint32_t shape_c[] = {2, 2};
    cgrad_storage a, b, c;
    cgrad_storage grad_b, grad_c;
    
    cgrad_storage_init(&a, shape_a, 2, "cpu_f32");
    cgrad_storage_init(&b, shape_b, 2, "cpu_f32");
    cgrad_storage_init(&c, shape_c, 2, "cpu_f32");
    cgrad_storage_init(&grad_b, shape_b, 2, "cpu_f32");
    cgrad_storage_init(&grad_c, shape_c, 2, "cpu_f32");
    
    cgrad_storage_fill(&a, 1.0f);
    cgrad_storage_fill(&b, 1.0f);
    cgrad_storage_fill(&c, 0.0f);
    cgrad_storage_fill(&grad_b, 0.0f);
    cgrad_storage_fill(&grad_c, 1.0f);
    
    // Get the GEMM operation descriptor
    const cgrad_op_descriptor* op_desc = &cgrad_op_gemm;
    assert_non_null(op_desc);
    
    // Prepare inputs (a does not require grad)
    cgrad_storage* inputs[2] = {&a, &b};
    cgrad_storage* grad_inputs[2] = {NULL, &grad_b};
    int input_requires_grad[2] = {0, 1};
    
    // Prepare metadata for GEMM
    cgrad_op_metadata metadata = {0};
    metadata.gemm.alpha = 1.0f;
    metadata.gemm.beta = 0.0f;
    
    // Execute forward pass first
    void* ctx = NULL;
    int ret = op_desc->forward(inputs, 2, &metadata, &c, &ctx, 1);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Execute backward pass
    ret = op_desc->backward(inputs, 2, &c, &grad_c, &metadata, ctx, grad_inputs, input_requires_grad);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // b should have gradient (3x2 with each element = 2)
    float value;
    uint32_t idx[2];
    for (uint32_t i = 0; i < 3; i++) {
        for (uint32_t j = 0; j < 2; j++) {
            idx[0] = i; idx[1] = j;
            cgrad_storage_get(&grad_b, idx, 2, &value);
            assert_true(fabsf(value - 2.0f) < OP_GEMM_EPSILON);
        }
    }
    
    cgrad_storage_free(&a);
    cgrad_storage_free(&b);
    cgrad_storage_free(&c);
    cgrad_storage_free(&grad_b);
    cgrad_storage_free(&grad_c);
}

// ============================================================================
// Test: GEMM backward - square matrices
// ============================================================================

static void test_op_gemm_backward_square(void **state) {
    (void) state;
    
    uint32_t shape[] = {3, 3};
    cgrad_storage a, b, c;
    cgrad_storage grad_a, grad_b, grad_c;
    
    cgrad_storage_init(&a, shape, 2, "cpu_f32");
    cgrad_storage_init(&b, shape, 2, "cpu_f32");
    cgrad_storage_init(&c, shape, 2, "cpu_f32");
    cgrad_storage_init(&grad_a, shape, 2, "cpu_f32");
    cgrad_storage_init(&grad_b, shape, 2, "cpu_f32");
    cgrad_storage_init(&grad_c, shape, 2, "cpu_f32");
    
    cgrad_storage_fill(&a, 1.0f);
    cgrad_storage_fill(&b, 1.0f);
    cgrad_storage_fill(&c, 0.0f);
    cgrad_storage_fill(&grad_a, 0.0f);
    cgrad_storage_fill(&grad_b, 0.0f);
    cgrad_storage_fill(&grad_c, 1.0f);
    
    // Get the GEMM operation descriptor
    const cgrad_op_descriptor* op_desc = &cgrad_op_gemm;
    assert_non_null(op_desc);
    
    // Prepare inputs
    cgrad_storage* inputs[2] = {&a, &b};
    cgrad_storage* grad_inputs[2] = {&grad_a, &grad_b};
    int input_requires_grad[2] = {1, 1};
    
    // Prepare metadata for GEMM
    cgrad_op_metadata metadata = {0};
    metadata.gemm.alpha = 1.0f;
    metadata.gemm.beta = 0.0f;
    
    // Execute forward pass first
    void* ctx = NULL;
    int ret = op_desc->forward(inputs, 2, &metadata, &c, &ctx, 1);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Execute backward pass
    ret = op_desc->backward(inputs, 2, &c, &grad_c, &metadata, ctx, grad_inputs, input_requires_grad);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // For 3x3 matrices of 1s:
    // grad_A = grad_C @ B^T = 3x3 of 1s @ 3x3 of 1s = 3x3 of 3s
    float value;
    uint32_t idx[2];
    for (uint32_t i = 0; i < 3; i++) {
        for (uint32_t j = 0; j < 3; j++) {
            idx[0] = i; idx[1] = j;
            cgrad_storage_get(&grad_a, idx, 2, &value);
            assert_true(fabsf(value - 3.0f) < OP_GEMM_EPSILON);
            cgrad_storage_get(&grad_b, idx, 2, &value);
            assert_true(fabsf(value - 3.0f) < OP_GEMM_EPSILON);
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
// Test: GEMM backward - broadcasting batch dimension for A
// ============================================================================

static void test_op_gemm_backward_broadcast_batch_a(void **state) {
    (void) state;
    
    // Forward: A: (1, 2, 3) @ B: (2, 3, 2) -> C: (2, 2, 2)
    // A is broadcasted along batch dimension
    // Backward: grad_output: (2, 2, 2) = all ones
    // grad_A should be (1, 2, 3) = sum of gradients across batch dim
    // grad_B should be (2, 3, 2) = normal gradient
    
    uint32_t shape_a[] = {1, 2, 3};
    uint32_t shape_b[] = {2, 3, 2};
    uint32_t shape_c[] = {2, 2, 2};
    cgrad_storage a, b, c;
    cgrad_storage grad_a, grad_b, grad_c;
    
    cgrad_storage_init(&a, shape_a, 3, "cpu_f32");
    cgrad_storage_init(&b, shape_b, 3, "cpu_f32");
    cgrad_storage_init(&c, shape_c, 3, "cpu_f32");
    cgrad_storage_init(&grad_a, shape_a, 3, "cpu_f32");
    cgrad_storage_init(&grad_b, shape_b, 3, "cpu_f32");
    cgrad_storage_init(&grad_c, shape_c, 3, "cpu_f32");
    
    cgrad_storage_fill(&a, 1.0f);
    cgrad_storage_fill(&b, 1.0f);
    cgrad_storage_fill(&c, 0.0f);
    cgrad_storage_fill(&grad_a, 0.0f);
    cgrad_storage_fill(&grad_b, 0.0f);
    cgrad_storage_fill(&grad_c, 1.0f);
    
    // Get the GEMM operation descriptor
    const cgrad_op_descriptor* op_desc = &cgrad_op_gemm;
    assert_non_null(op_desc);
    
    // Prepare inputs
    cgrad_storage* inputs[2] = {&a, &b};
    cgrad_storage* grad_inputs[2] = {&grad_a, &grad_b};
    int input_requires_grad[2] = {1, 1};
    
    // Prepare metadata for GEMM
    cgrad_op_metadata metadata = {0};
    metadata.gemm.alpha = 1.0f;
    metadata.gemm.beta = 0.0f;
    
    // Execute forward pass first
    void* ctx = NULL;
    int ret = op_desc->forward(inputs, 2, &metadata, &c, &ctx, 1);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Execute backward pass
    ret = op_desc->backward(inputs, 2, &c, &grad_c, &metadata, ctx, grad_inputs, input_requires_grad);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // grad_A should be (1, 2, 3) with each element = 4
    // (sum of 2 batches, each contributing grad_C @ B^T where each element is 2)
    float value;
    uint32_t idx[3];
    idx[0] = 0;
    for (uint32_t i = 0; i < 2; i++) {
        for (uint32_t j = 0; j < 3; j++) {
            idx[1] = i; idx[2] = j;
            cgrad_storage_get(&grad_a, idx, 3, &value);
            assert_true(fabsf(value - 4.0f) < OP_GEMM_EPSILON);
        }
    }
    
    // grad_B should be (2, 3, 2) with each element = 2
    for (uint32_t b = 0; b < 2; b++) {
        for (uint32_t i = 0; i < 3; i++) {
            for (uint32_t j = 0; j < 2; j++) {
                idx[0] = b; idx[1] = i; idx[2] = j;
                cgrad_storage_get(&grad_b, idx, 3, &value);
                assert_true(fabsf(value - 2.0f) < OP_GEMM_EPSILON);
            }
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
// Test: GEMM backward - broadcasting batch dimension for B
// ============================================================================

static void test_op_gemm_backward_broadcast_batch_b(void **state) {
    (void) state;
    
    // Forward: A: (2, 2, 3) @ B: (1, 3, 2) -> C: (2, 2, 2)
    // B is broadcasted along batch dimension
    // Backward: grad_output: (2, 2, 2) = all ones
    // grad_A should be (2, 2, 3) = normal gradient
    // grad_B should be (1, 3, 2) = sum of gradients across batch dim
    
    uint32_t shape_a[] = {2, 2, 3};
    uint32_t shape_b[] = {1, 3, 2};
    uint32_t shape_c[] = {2, 2, 2};
    cgrad_storage a, b, c;
    cgrad_storage grad_a, grad_b, grad_c;
    
    cgrad_storage_init(&a, shape_a, 3, "cpu_f32");
    cgrad_storage_init(&b, shape_b, 3, "cpu_f32");
    cgrad_storage_init(&c, shape_c, 3, "cpu_f32");
    cgrad_storage_init(&grad_a, shape_a, 3, "cpu_f32");
    cgrad_storage_init(&grad_b, shape_b, 3, "cpu_f32");
    cgrad_storage_init(&grad_c, shape_c, 3, "cpu_f32");
    
    cgrad_storage_fill(&a, 1.0f);
    cgrad_storage_fill(&b, 1.0f);
    cgrad_storage_fill(&c, 0.0f);
    cgrad_storage_fill(&grad_a, 0.0f);
    cgrad_storage_fill(&grad_b, 0.0f);
    cgrad_storage_fill(&grad_c, 1.0f);
    
    // Get the GEMM operation descriptor
    const cgrad_op_descriptor* op_desc = &cgrad_op_gemm;
    assert_non_null(op_desc);
    
    // Prepare inputs
    cgrad_storage* inputs[2] = {&a, &b};
    cgrad_storage* grad_inputs[2] = {&grad_a, &grad_b};
    int input_requires_grad[2] = {1, 1};
    
    // Prepare metadata for GEMM
    cgrad_op_metadata metadata = {0};
    metadata.gemm.alpha = 1.0f;
    metadata.gemm.beta = 0.0f;
    
    // Execute forward pass first
    void* ctx = NULL;
    int ret = op_desc->forward(inputs, 2, &metadata, &c, &ctx, 1);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Execute backward pass
    ret = op_desc->backward(inputs, 2, &c, &grad_c, &metadata, ctx, grad_inputs, input_requires_grad);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // grad_A should be (2, 2, 3) with each element = 2
    float value;
    uint32_t idx[3];
    for (uint32_t b = 0; b < 2; b++) {
        for (uint32_t i = 0; i < 2; i++) {
            for (uint32_t j = 0; j < 3; j++) {
                idx[0] = b; idx[1] = i; idx[2] = j;
                cgrad_storage_get(&grad_a, idx, 3, &value);
                assert_true(fabsf(value - 2.0f) < OP_GEMM_EPSILON);
            }
        }
    }
    
    // grad_B should be (1, 3, 2) with each element = 4
    // (sum of 2 batches, each contributing A^T @ grad_C where each element is 2)
    idx[0] = 0;
    for (uint32_t i = 0; i < 3; i++) {
        for (uint32_t j = 0; j < 2; j++) {
            idx[1] = i; idx[2] = j;
            cgrad_storage_get(&grad_b, idx, 3, &value);
            assert_true(fabsf(value - 4.0f) < OP_GEMM_EPSILON);
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
// Test: GEMM backward - broadcasting both batch dimensions
// ============================================================================

static void test_op_gemm_backward_broadcast_both_batch(void **state) {
    (void) state;
    
    // Forward: A: (1, 2, 3) @ B: (1, 3, 2) -> C: (1, 2, 2)
    // Both A and B have batch size 1, output has batch size 1
    // This tests that broadcasting works when both inputs are broadcasted
    // (though in this case they're not actually broadcasted since output is also 1)
    
    uint32_t shape_a[] = {1, 2, 3};
    uint32_t shape_b[] = {1, 3, 2};
    uint32_t shape_c[] = {1, 2, 2};
    cgrad_storage a, b, c;
    cgrad_storage grad_a, grad_b, grad_c;
    
    cgrad_storage_init(&a, shape_a, 3, "cpu_f32");
    cgrad_storage_init(&b, shape_b, 3, "cpu_f32");
    cgrad_storage_init(&c, shape_c, 3, "cpu_f32");
    cgrad_storage_init(&grad_a, shape_a, 3, "cpu_f32");
    cgrad_storage_init(&grad_b, shape_b, 3, "cpu_f32");
    cgrad_storage_init(&grad_c, shape_c, 3, "cpu_f32");
    
    cgrad_storage_fill(&a, 1.0f);
    cgrad_storage_fill(&b, 1.0f);
    cgrad_storage_fill(&c, 0.0f);
    cgrad_storage_fill(&grad_a, 0.0f);
    cgrad_storage_fill(&grad_b, 0.0f);
    cgrad_storage_fill(&grad_c, 1.0f);
    
    // Get the GEMM operation descriptor
    const cgrad_op_descriptor* op_desc = &cgrad_op_gemm;
    assert_non_null(op_desc);
    
    // Prepare inputs
    cgrad_storage* inputs[2] = {&a, &b};
    cgrad_storage* grad_inputs[2] = {&grad_a, &grad_b};
    int input_requires_grad[2] = {1, 1};
    
    // Prepare metadata for GEMM
    cgrad_op_metadata metadata = {0};
    metadata.gemm.alpha = 1.0f;
    metadata.gemm.beta = 0.0f;
    
    // Execute forward pass first
    void* ctx = NULL;
    int ret = op_desc->forward(inputs, 2, &metadata, &c, &ctx, 1);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Execute backward pass
    ret = op_desc->backward(inputs, 2, &c, &grad_c, &metadata, ctx, grad_inputs, input_requires_grad);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // grad_A should be (1, 2, 3) with each element = 2
    float value;
    uint32_t idx[3];
    idx[0] = 0;
    for (uint32_t i = 0; i < 2; i++) {
        for (uint32_t j = 0; j < 3; j++) {
            idx[1] = i; idx[2] = j;
            cgrad_storage_get(&grad_a, idx, 3, &value);
            assert_true(fabsf(value - 2.0f) < OP_GEMM_EPSILON);
        }
    }
    
    // grad_B should be (1, 3, 2) with each element = 2
    idx[0] = 0;
    for (uint32_t i = 0; i < 3; i++) {
        for (uint32_t j = 0; j < 2; j++) {
            idx[1] = i; idx[2] = j;
            cgrad_storage_get(&grad_b, idx, 3, &value);
            assert_true(fabsf(value - 2.0f) < OP_GEMM_EPSILON);
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

int run_cgrad_op_gemm_tests(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_op_gemm_forward, gemm_setup_test, gemm_teardown_test),
        cmocka_unit_test_setup_teardown(test_op_gemm_backward_basic, gemm_setup_test, gemm_teardown_test),
        cmocka_unit_test_setup_teardown(test_op_gemm_backward_one_no_grad, gemm_setup_test, gemm_teardown_test),
        cmocka_unit_test_setup_teardown(test_op_gemm_backward_square, gemm_setup_test, gemm_teardown_test),
        cmocka_unit_test_setup_teardown(test_op_gemm_backward_broadcast_batch_a, gemm_setup_test, gemm_teardown_test),
        cmocka_unit_test_setup_teardown(test_op_gemm_backward_broadcast_batch_b, gemm_setup_test, gemm_teardown_test),
        cmocka_unit_test_setup_teardown(test_op_gemm_backward_broadcast_both_batch, gemm_setup_test, gemm_teardown_test),
    };
    
    return cmocka_run_group_tests_name("cgrad_op_gemm", tests, NULL, NULL);
}

#ifndef TEST_ALL_MAIN
int main(void) {
    return run_cgrad_op_gemm_tests();
}
#endif
