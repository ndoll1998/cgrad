#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdlib.h>
#include <math.h>

#include "cgrad.h"
#include "optim/cgrad_sgd.h"
#include "autograd/cgrad_tensor.h"
#include "backends/cgrad_backend.h"

/**
 * @file test_cgrad_sgd.c
 * @brief Unit tests for the SGD optimizer.
 */

#define EPSILON 1e-5

// ============================================================================
// Setup and Teardown
// ============================================================================

static int sgd_setup_test(void **state) {
    (void) state;
    cgrad_init();
    return 0;
}

static int sgd_teardown_test(void **state) {
    (void) state;
    cgrad_cleanup();
    return 0;
}

// ============================================================================
// Test Functions
// ============================================================================

static void test_sgd_init_no_momentum(void **state) {
    (void) state;
    
    // Create test parameters
    cgrad_tensor param1, param2;
    cgrad_tensor_init(&param1, (uint32_t[]){2, 3}, 2, "cpu_f32");
    cgrad_tensor_init(&param2, (uint32_t[]){3, 4}, 2, "cpu_f32");
    cgrad_tensor_fill(&param1, 1.0f);
    cgrad_tensor_fill(&param2, 2.0f);
    
    cgrad_tensor* params[] = {&param1, &param2};
    
    // Initialize SGD optimizer without momentum
    cgrad_optimizer optimizer;
    cgrad_status status = cgrad_sgd_init(&optimizer, params, 2, 0.01f, 0.0f);
    
    assert_int_equal(status, CGRAD_SUCCESS);
    assert_non_null(optimizer.state);
    assert_int_equal(cgrad_optimizer_num_parameters(&optimizer), 2);
    
    // Verify learning rate
    float lr;
    status = cgrad_sgd_get_learning_rate(&optimizer, &lr);
    assert_int_equal(status, CGRAD_SUCCESS);
    assert_true(fabs(lr - 0.01f) < EPSILON);
    
    // Verify momentum
    float momentum;
    status = cgrad_sgd_get_momentum(&optimizer, &momentum);
    assert_int_equal(status, CGRAD_SUCCESS);
    assert_true(fabs(momentum) < EPSILON);
    
    // Cleanup
    cgrad_optimizer_free(&optimizer);
    cgrad_tensor_free(&param2);
    cgrad_tensor_free(&param1);
}

static void test_sgd_init_with_momentum(void **state) {
    (void) state;
    
    // Create test parameters
    cgrad_tensor param1, param2;
    cgrad_tensor_init(&param1, (uint32_t[]){2, 2}, 2, "cpu_f32");
    cgrad_tensor_init(&param2, (uint32_t[]){3, 3}, 2, "cpu_f32");
    cgrad_tensor_fill(&param1, 1.0f);
    cgrad_tensor_fill(&param2, 2.0f);
    
    cgrad_tensor* params[] = {&param1, &param2};
    
    // Initialize SGD optimizer with momentum
    cgrad_optimizer optimizer;
    cgrad_status status = cgrad_sgd_init(&optimizer, params, 2, 0.1f, 0.9f);
    
    assert_int_equal(status, CGRAD_SUCCESS);
    assert_non_null(optimizer.state);
    
    // Verify momentum
    float momentum;
    status = cgrad_sgd_get_momentum(&optimizer, &momentum);
    assert_int_equal(status, CGRAD_SUCCESS);
    assert_true(fabs(momentum - 0.9f) < EPSILON);
    
    // Verify velocity buffers were created
    cgrad_sgd_state* sgd_state = (cgrad_sgd_state*)optimizer.state;
    assert_non_null(sgd_state->velocity_buffers);
    
    // Cleanup
    cgrad_optimizer_free(&optimizer);
    cgrad_tensor_free(&param2);
    cgrad_tensor_free(&param1);
}

static void test_sgd_step_no_momentum(void **state) {
    (void) state;
    
    // Create a simple parameter
    cgrad_tensor param;
    cgrad_tensor_init(&param, (uint32_t[]){2, 2}, 2, "cpu_f32");
    cgrad_tensor_fill(&param, 1.0f);
    cgrad_tensor_set_requires_grad(&param, 1);
    
    // Create a simple loss: sum of all elements
    uint8_t reduce_mask[] = {1, 1};
    cgrad_tensor loss;
    cgrad_tensor_reduce_sum(&param, reduce_mask, 2, &loss);
    
    // Backward pass to compute gradients
    cgrad_tensor_backward(&loss);
    
    // Get initial parameter value
    float initial_val;
    cgrad_tensor_get(&param, (uint32_t[]){0, 0}, 2, &initial_val);
    
    // Create SGD optimizer
    cgrad_tensor* params[] = {&param};
    cgrad_optimizer optimizer;
    float learning_rate = 0.1f;
    cgrad_sgd_init(&optimizer, params, 1, learning_rate, 0.0f);
    
    // Perform optimization step
    cgrad_status status = cgrad_optimizer_step(&optimizer);
    assert_int_equal(status, CGRAD_SUCCESS);
    
    // Get updated parameter value
    float updated_val;
    cgrad_tensor_get(&param, (uint32_t[]){0, 0}, 2, &updated_val);
    
    // Gradient should be 1.0 (sum reduction), so update should be:
    // param = param - lr * grad = 1.0 - 0.1 * 1.0 = 0.9
    assert_true(fabs(updated_val - 0.9f) < EPSILON);
    
    // Cleanup
    cgrad_optimizer_free(&optimizer);
    cgrad_tensor_free(&loss);
    cgrad_tensor_free(&param);
}

static void test_sgd_step_with_momentum(void **state) {
    (void) state;
    
    // Create a simple parameter
    cgrad_tensor param;
    cgrad_tensor_init(&param, (uint32_t[]){2, 2}, 2, "cpu_f32");
    cgrad_tensor_fill(&param, 1.0f);
    cgrad_tensor_set_requires_grad(&param, 1);
    
    // Create SGD optimizer with momentum
    cgrad_tensor* params[] = {&param};
    cgrad_optimizer optimizer;
    float learning_rate = 0.1f;
    float momentum = 0.9f;
    cgrad_sgd_init(&optimizer, params, 1, learning_rate, momentum);
    
    // First step
    uint8_t reduce_mask[] = {1, 1};
    cgrad_tensor loss1;
    cgrad_tensor_reduce_sum(&param, reduce_mask, 2, &loss1);
    cgrad_tensor_backward(&loss1);
    
    float val_before_step1;
    cgrad_tensor_get(&param, (uint32_t[]){0, 0}, 2, &val_before_step1);
    
    // update parameters
    cgrad_optimizer_step(&optimizer);
    
    float val_after_step1;
    cgrad_tensor_get(&param, (uint32_t[]){0, 0}, 2, &val_after_step1);
    
    // First step: velocity = 0 * 0.9 + 1.0 = 1.0
    //             param = 1.0 - 0.1 * 1.0 = 0.9
    assert_true(fabs(val_after_step1 - 0.81f) < EPSILON);
    
    // Second step (velocity should accumulate)
    cgrad_optimizer_zero_grad(&optimizer);
    cgrad_tensor_free(&loss1);
    
    cgrad_tensor loss2;
    cgrad_tensor_reduce_sum(&param, reduce_mask, 2, &loss2);
    cgrad_tensor_backward(&loss2);
    cgrad_optimizer_step(&optimizer);
    
    float val_after_step2;
    cgrad_tensor_get(&param, (uint32_t[]){0, 0}, 2, &val_after_step2);
    
    // Second step: velocity = 1.0 * 0.9 + 1.0 = 1.9
    //              param = 0.9 - 0.1 * 1.9 = 0.71
    assert_true(fabs(val_after_step2 - 0.62f) < EPSILON);
    
    // Cleanup
    cgrad_optimizer_free(&optimizer);
    cgrad_tensor_free(&loss2);
    cgrad_tensor_free(&param);
}

static void test_sgd_set_learning_rate(void **state) {
    (void) state;
    
    cgrad_tensor param;
    cgrad_tensor_init(&param, (uint32_t[]){2, 2}, 2, "cpu_f32");
    cgrad_tensor_fill(&param, 1.0f);
    
    cgrad_tensor* params[] = {&param};
    cgrad_optimizer optimizer;
    cgrad_sgd_init(&optimizer, params, 1, 0.01f, 0.0f);
    
    // Verify initial learning rate
    float lr;
    cgrad_sgd_get_learning_rate(&optimizer, &lr);
    assert_true(fabs(lr - 0.01f) < EPSILON);
    
    // Change learning rate
    cgrad_status status = cgrad_sgd_set_learning_rate(&optimizer, 0.001f);
    assert_int_equal(status, CGRAD_SUCCESS);
    
    // Verify new learning rate
    cgrad_sgd_get_learning_rate(&optimizer, &lr);
    assert_true(fabs(lr - 0.001f) < EPSILON);
    
    // Test invalid learning rate
    status = cgrad_sgd_set_learning_rate(&optimizer, -0.1f);
    assert_int_equal(status, CGRAD_ERR_INVALID_ARGUMENT);
    
    status = cgrad_sgd_set_learning_rate(&optimizer, 0.0f);
    assert_int_equal(status, CGRAD_ERR_INVALID_ARGUMENT);
    
    // Cleanup
    cgrad_optimizer_free(&optimizer);
    cgrad_tensor_free(&param);
}

static void test_sgd_invalid_arguments(void **state) {
    (void) state;
    
    cgrad_tensor param;
    cgrad_tensor_init(&param, (uint32_t[]){2, 2}, 2, "cpu_f32");
    cgrad_tensor* params[] = {&param};
    
    cgrad_optimizer optimizer;
    cgrad_status status;
    
    // Test NULL optimizer
    status = cgrad_sgd_init(NULL, params, 1, 0.01f, 0.0f);
    assert_int_equal(status, CGRAD_ERR_INVALID_ARGUMENT);
    
    // Test invalid learning rate
    status = cgrad_sgd_init(&optimizer, params, 1, 0.0f, 0.0f);
    assert_int_equal(status, CGRAD_ERR_INVALID_ARGUMENT);
    
    status = cgrad_sgd_init(&optimizer, params, 1, -0.1f, 0.0f);
    assert_int_equal(status, CGRAD_ERR_INVALID_ARGUMENT);
    
    // Test invalid momentum
    status = cgrad_sgd_init(&optimizer, params, 1, 0.01f, -0.1f);
    assert_int_equal(status, CGRAD_ERR_INVALID_ARGUMENT);
    
    status = cgrad_sgd_init(&optimizer, params, 1, 0.01f, 1.0f);
    assert_int_equal(status, CGRAD_ERR_INVALID_ARGUMENT);
    
    // Cleanup
    cgrad_tensor_free(&param);
}

static void test_sgd_multiple_parameters(void **state) {
    (void) state;
    
    // Create multiple parameters
    cgrad_tensor param1, param2, param3;
    cgrad_tensor_init(&param1, (uint32_t[]){2}, 1, "cpu_f32");
    cgrad_tensor_init(&param2, (uint32_t[]){2}, 1, "cpu_f32");
    cgrad_tensor_init(&param3, (uint32_t[]){2}, 1, "cpu_f32");
    
    cgrad_tensor_fill(&param1, 1.0f);
    cgrad_tensor_fill(&param2, 2.0f);
    cgrad_tensor_fill(&param3, 3.0f);
    
    cgrad_tensor_set_requires_grad(&param1, 1);
    cgrad_tensor_set_requires_grad(&param2, 1);
    cgrad_tensor_set_requires_grad(&param3, 1);
    
    // Create optimizer
    cgrad_tensor* params[] = {&param1, &param2, &param3};
    cgrad_optimizer optimizer;
    cgrad_sgd_init(&optimizer, params, 3, 0.1f, 0.0f);
    
    // Create simple loss (sum of all parameters)
    cgrad_tensor sum1 = {0}, sum2 = {0}, loss = {0};
    cgrad_tensor_add(&param1, &param2, &sum1);
    cgrad_tensor_add(&sum1, &param3, &sum2);
    
    uint8_t reduce_mask1[] = {1};
    uint8_t reduce_mask2[] = {1};
    cgrad_tensor reduced1, reduced2;
    cgrad_tensor_reduce_sum(&sum1, reduce_mask1, 1, &reduced1);
    cgrad_tensor_reduce_sum(&sum2, reduce_mask2, 1, &reduced2);
    cgrad_tensor_add(&reduced1, &reduced2, &loss);
    
    // Backward and step
    cgrad_tensor_backward(&loss);
    cgrad_optimizer_step(&optimizer);
    
    // Verify all parameters were updated
    float val1, val2, val3;
    cgrad_tensor_get(&param1, (uint32_t[]){0}, 1, &val1);
    cgrad_tensor_get(&param2, (uint32_t[]){0}, 1, &val2);
    cgrad_tensor_get(&param3, (uint32_t[]){0}, 1, &val3);
    
    // All gradients should be non-zero, so all parameters should have changed
    assert_true(fabs(val1 - 1.0f) > EPSILON);
    assert_true(fabs(val2 - 2.0f) > EPSILON);
    assert_true(fabs(val3 - 3.0f) > EPSILON);
    
    // Cleanup
    cgrad_optimizer_free(&optimizer);
    cgrad_tensor_free(&loss);
    cgrad_tensor_free(&reduced2);
    cgrad_tensor_free(&reduced1);
    cgrad_tensor_free(&sum2);
    cgrad_tensor_free(&sum1);
    cgrad_tensor_free(&param3);
    cgrad_tensor_free(&param2);
    cgrad_tensor_free(&param1);
}

// ============================================================================
// Test Suite
// ============================================================================

int run_cgrad_sgd_tests(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_sgd_init_no_momentum, sgd_setup_test, sgd_teardown_test),
        cmocka_unit_test_setup_teardown(test_sgd_init_with_momentum, sgd_setup_test, sgd_teardown_test),
        cmocka_unit_test_setup_teardown(test_sgd_step_no_momentum, sgd_setup_test, sgd_teardown_test),
        cmocka_unit_test_setup_teardown(test_sgd_step_with_momentum, sgd_setup_test, sgd_teardown_test),
        cmocka_unit_test_setup_teardown(test_sgd_set_learning_rate, sgd_setup_test, sgd_teardown_test),
        cmocka_unit_test_setup_teardown(test_sgd_invalid_arguments, sgd_setup_test, sgd_teardown_test),
        cmocka_unit_test_setup_teardown(test_sgd_multiple_parameters, sgd_setup_test, sgd_teardown_test),
    };
    
    return cmocka_run_group_tests_name("cgrad_sgd", tests, NULL, NULL);
}

#ifndef TEST_ALL_MAIN
int main(void) {
    return run_cgrad_sgd_tests();
}
#endif
