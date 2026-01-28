#include "cgrad.h"
#include "storage/cgrad_storage.h"
#include <stdio.h>
#include <stdint.h>
#include <math.h>

/**
 * @brief Example program demonstrating gradient descent optimization.
 * 
 * This program:
 * 1. Initializes two random matrices A and B
 * 2. Performs 20 iterations of gradient descent:
 *    - Computes loss = sum(A @ B)
 *    - Computes gradients via backpropagation
 *    - Updates A using gradient descent: A = A - learning_rate * dL/dA
 * 3. Prints iteration, loss, and gradient norm at each step
 */
int main() {
    
    printf("========================================\n");
    printf("Gradient Descent Optimization Demo\n");
    printf("========================================\n\n");
    
    // ========================================================================
    // Initialize matrices A and B
    // ========================================================================
    printf("--- Initializing Matrices ---\n");
    
    cgrad_tensor A, B;
    uint32_t shape_A[] = {3, 4};  // 3x4 matrix
    uint32_t shape_B[] = {4, 2};  // 4x2 matrix
    
    // Create tensors
    cgrad_tensor_init(&A, shape_A, 2, "cpu_f32");
    cgrad_tensor_init(&B, shape_B, 2, "cpu_f32");
    
    // Fill with random values
    cgrad_tensor_fill_rand(&A);
    cgrad_tensor_fill_rand(&B);
    
    // Set gradient requirements
    cgrad_tensor_set_requires_grad(&A, 1);  // Enable gradients for A
    cgrad_tensor_set_requires_grad(&B, 0);  // Disable gradients for B
    
    printf("Matrix A: 3x4, requires_grad=True\n");
    printf("Matrix B: 4x2, requires_grad=False\n\n");
    
    // ========================================================================
    // Gradient Descent Loop: 20 iterations
    // ========================================================================
    printf("--- Starting Gradient Descent (20 iterations) ---\n");
    printf("%-6s %-12s\n", "Iter", "Loss");
    printf("%-6s %-12s\n", "----", "----");
    
    const float learning_rate = 0.1f;
    const int num_iterations = 10;
    
    for (int iter = 0; iter < num_iterations; iter++) {

        // void* record = cgrad_storage_start_recording();
        // zero gradient of A
        cgrad_tensor_zero_grad(&A);

        // ====================================================================
        // Forward Pass: Compute loss = sum(A @ B)
        // ====================================================================
        cgrad_tensor C;
        cgrad_tensor_gemm(&A, &B, &C);
        
        cgrad_tensor loss;
        uint8_t reduce_mask[] = {1, 1};  // Reduce over all dimensions
        cgrad_tensor_reduce_sum(&C, reduce_mask, 2, &loss);
        
        // Execute to materialize loss value
        cgrad_tensor_execute(&loss);
        
        // Get loss value
        float loss_value = 0.0f;
        cgrad_tensor_get(&loss, (uint32_t[]){0}, 1, &loss_value);
        
        // ====================================================================
        // Backward Pass: Compute gradients
        // ====================================================================
        int ret = cgrad_tensor_backward(&loss);
        if (ret != 0) {
            printf("Error: Backward pass failed at iteration %d\n", iter);
            cgrad_cleanup();
            return 1;
        }
        
        // Print iteration info
        printf("%-6d %-12.6f\n", iter + 1, loss_value);
        
        // ====================================================================
        // Gradient Descent Update: A = A - learning_rate * dL/dA
        // ====================================================================
        cgrad_storage_axpy(
            -learning_rate,
            cgrad_tensor_get_grad_storage(&A),
            cgrad_tensor_get_storage(&A),
            cgrad_tensor_get_storage(&A)
        );

        // cgrad_storage_free_record(record);
        cgrad_tensor_free(&C);
        cgrad_tensor_free(&loss);
    }
    
    printf("\n");

    cgrad_tensor_print(&A);
    
    // ========================================================================
    // Cleanup
    // ========================================================================
    printf("--- Cleanup ---\n");
    cgrad_status cleanup_status = cgrad_cleanup();
    if (cleanup_status != CGRAD_SUCCESS) {
        printf("Warning: Cleanup returned error code %d\n", cleanup_status);
    } else {
        printf("All resources freed.\n");
    }

    printf("\n========================================\n");
    printf("Optimization Complete!\n");
    printf("========================================\n");
    
    return 0;
}
