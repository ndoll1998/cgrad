#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdlib.h>
#include <math.h>

#include "cgrad.h"
#include "optim/cgrad_optimizer.h"
#include "autograd/cgrad_tensor.h"
#include "storage/cgrad_storage.h"
#include "backends/cgrad_backend.h"

/**
 * @file test_cgrad_optimizer.c
 * @brief Unit tests for the abstract optimizer interface.
 */

#define EPSILON 1e-5

// ============================================================================
// Setup and Teardown
// ============================================================================

static int optimizer_setup_test(void **state) {
    (void) state;
    cgrad_init();
    return 0;
}

static int optimizer_teardown_test(void **state) {
    (void) state;
    cgrad_cleanup();
    return 0;
}

// ============================================================================
// Test Functions
// ============================================================================

static void test_optimizer_init_and_free(void **state) {
    (void) state;
    
    // Create some test parameters
    cgrad_tensor param1, param2;
    cgrad_tensor_init(&param1, (uint32_t[]){2, 3}, 2, "cpu_f32");
    cgrad_tensor_init(&param2, (uint32_t[]){3, 4}, 2, "cpu_f32");
    cgrad_tensor_fill(&param1, 1.0f);
    cgrad_tensor_fill(&param2, 2.0f);
    
    // Create parameter array
    cgrad_tensor* params[] = {&param1, &param2};
    
    // Create a minimal vtable (no actual implementations needed for this test)
    cgrad_optim_vtable vtable = {
        .step = NULL,
        .zero_grad = cgrad_optimizer_zero_grad_default,
        .free_state = NULL
    };
    
    // Initialize optimizer
    cgrad_optimizer optimizer;
    cgrad_status status = cgrad_optimizer_init(
        &optimizer,
        params,
        2,
        NULL,  // No state for this test
        &vtable
    );
    
    assert_int_equal(status, CGRAD_SUCCESS);
    assert_non_null(optimizer.parameters);
    assert_ptr_equal(optimizer.vtable, &vtable);
    assert_int_equal(cgrad_optimizer_num_parameters(&optimizer), 2);
    
    // Test parameter access
    cgrad_tensor* retrieved_param1 = cgrad_optimizer_get_parameter(&optimizer, 0);
    cgrad_tensor* retrieved_param2 = cgrad_optimizer_get_parameter(&optimizer, 1);
    assert_ptr_equal(retrieved_param1, &param1);
    assert_ptr_equal(retrieved_param2, &param2);
    
    // Test out of bounds access
    cgrad_tensor* out_of_bounds = cgrad_optimizer_get_parameter(&optimizer, 2);
    assert_null(out_of_bounds);
    
    // Cleanup
    cgrad_optimizer_free(&optimizer);
    cgrad_tensor_free(&param1);
    cgrad_tensor_free(&param2);
}

static void test_optimizer_zero_grad_default(void **state) {
    (void) state;
    
    // Create parameters with gradients
    cgrad_tensor param1, param2;
    cgrad_tensor_init(&param1, (uint32_t[]){2, 2}, 2, "cpu_f32");
    cgrad_tensor_init(&param2, (uint32_t[]){2, 2}, 2, "cpu_f32");
    cgrad_tensor_fill(&param1, 1.0f);
    cgrad_tensor_fill(&param2, 2.0f);
    cgrad_tensor_set_requires_grad(&param1, 1);
    cgrad_tensor_set_requires_grad(&param2, 1);
    
    // Execute to materialize storage
    cgrad_tensor_execute(&param1);
    cgrad_tensor_execute(&param2);
    
    // Create a simple computation to generate gradients
    cgrad_tensor sum1, sum2, total;
    cgrad_tensor_add(&param1, &param2, &sum1);
    cgrad_tensor_add(&sum1, &param1, &sum2);
    
    // Reduce to scalar for backward
    uint8_t reduce_mask[] = {1, 1};
    cgrad_tensor_reduce_sum(&sum2, reduce_mask, 2, &total);
    
    // Backward pass
    cgrad_tensor_backward(&total);
    
    // Verify gradients exist
    cgrad_storage* grad1 = cgrad_tensor_get_grad_storage(&param1);
    cgrad_storage* grad2 = cgrad_tensor_get_grad_storage(&param2);
    assert_non_null(grad1);
    assert_non_null(grad2);
    
    // Create optimizer
    cgrad_tensor* params[] = {&param1, &param2};
    cgrad_optim_vtable vtable = {
        .step = NULL,
        .zero_grad = cgrad_optimizer_zero_grad_default,
        .free_state = NULL
    };
    
    cgrad_optimizer optimizer;
    cgrad_optimizer_init(&optimizer, params, 2, NULL, &vtable);
    
    // Zero gradients
    cgrad_status status = cgrad_optimizer_zero_grad(&optimizer);
    assert_int_equal(status, CGRAD_SUCCESS);
    
    // Verify gradients are zeroed (they should still exist but be zero)
    grad1 = cgrad_tensor_get_grad_storage(&param1);
    grad2 = cgrad_tensor_get_grad_storage(&param2);
    
    // Check that gradient values are zero
    if (grad1 != NULL) {
        float val;
        cgrad_storage_get(grad1, (uint32_t[]){0, 0}, 2, &val);
        assert_true(fabs(val) < EPSILON);
    }
    
    // Cleanup
    cgrad_optimizer_free(&optimizer);
    cgrad_tensor_free(&total);
    cgrad_tensor_free(&sum2);
    cgrad_tensor_free(&sum1);
    cgrad_tensor_free(&param2);
    cgrad_tensor_free(&param1);
}

static void test_optimizer_invalid_arguments(void **state) {
    (void) state;
    
    cgrad_tensor param;
    cgrad_tensor_init(&param, (uint32_t[]){2, 2}, 2, "cpu_f32");
    cgrad_tensor* params[] = {&param};
    
    cgrad_optim_vtable vtable = {
        .step = NULL,
        .zero_grad = cgrad_optimizer_zero_grad_default,
        .free_state = NULL
    };
    
    cgrad_optimizer optimizer;
    
    // Test NULL optimizer
    cgrad_status status = cgrad_optimizer_init(NULL, params, 1, NULL, &vtable);
    assert_int_equal(status, CGRAD_ERR_INVALID_ARGUMENT);
    
    // Test NULL vtable
    status = cgrad_optimizer_init(&optimizer, params, 1, NULL, NULL);
    assert_int_equal(status, CGRAD_ERR_INVALID_ARGUMENT);
    
    // Test NULL parameters with non-zero count
    status = cgrad_optimizer_init(&optimizer, NULL, 1, NULL, &vtable);
    assert_int_equal(status, CGRAD_ERR_INVALID_ARGUMENT);
    
    // Test valid initialization with zero parameters
    status = cgrad_optimizer_init(&optimizer, NULL, 0, NULL, &vtable);
    assert_int_equal(status, CGRAD_SUCCESS);
    assert_int_equal(cgrad_optimizer_num_parameters(&optimizer), 0);
    cgrad_optimizer_free(&optimizer);
    
    // Test NULL parameter in array
    cgrad_tensor* null_params[] = {NULL};
    status = cgrad_optimizer_init(&optimizer, null_params, 1, NULL, &vtable);
    assert_int_equal(status, CGRAD_ERR_INVALID_ARGUMENT);
    
    // Cleanup
    cgrad_tensor_free(&param);
}

static void test_optimizer_num_parameters(void **state) {
    (void) state;
    
    // Test with NULL optimizer
    size_t count = cgrad_optimizer_num_parameters(NULL);
    assert_int_equal(count, 0);
    
    // Test with valid optimizer
    cgrad_tensor param1, param2, param3;
    cgrad_tensor_init(&param1, (uint32_t[]){2}, 1, "cpu_f32");
    cgrad_tensor_init(&param2, (uint32_t[]){3}, 1, "cpu_f32");
    cgrad_tensor_init(&param3, (uint32_t[]){4}, 1, "cpu_f32");
    
    cgrad_tensor* params[] = {&param1, &param2, &param3};
    
    cgrad_optim_vtable vtable = {
        .step = NULL,
        .zero_grad = cgrad_optimizer_zero_grad_default,
        .free_state = NULL
    };
    
    cgrad_optimizer optimizer;
    cgrad_optimizer_init(&optimizer, params, 3, NULL, &vtable);
    
    count = cgrad_optimizer_num_parameters(&optimizer);
    assert_int_equal(count, 3);
    
    // Cleanup
    cgrad_optimizer_free(&optimizer);
    cgrad_tensor_free(&param3);
    cgrad_tensor_free(&param2);
    cgrad_tensor_free(&param1);
}

// ============================================================================
// Test Suite
// ============================================================================

int run_cgrad_optimizer_tests(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_optimizer_init_and_free, optimizer_setup_test, optimizer_teardown_test),
        cmocka_unit_test_setup_teardown(test_optimizer_zero_grad_default, optimizer_setup_test, optimizer_teardown_test),
        cmocka_unit_test_setup_teardown(test_optimizer_invalid_arguments, optimizer_setup_test, optimizer_teardown_test),
        cmocka_unit_test_setup_teardown(test_optimizer_num_parameters, optimizer_setup_test, optimizer_teardown_test),
    };
    
    return cmocka_run_group_tests_name("cgrad_optimizer", tests, NULL, NULL);
}

#ifndef TEST_ALL_MAIN
int main(void) {
    return run_cgrad_optimizer_tests();
}
#endif
