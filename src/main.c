#include "cgrad.h"
#include "storage/cgrad_storage.h"
#include "optim/cgrad_sgd.h"
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
 *    - Updates A using stochastic gradient descent optimizer
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
    // Initialize SGD Optimizer
    // ========================================================================
    printf("--- Initializing SGD Optimizer ---\n");

    const float learning_rate = 0.1f;
    const float momentum = 0.0f;

    cgrad_tensor* parameters[] = {&A};
    cgrad_optimizer optimizer;

    cgrad_status status = cgrad_sgd_init(&optimizer, parameters, 1, learning_rate, momentum);
    if (status != CGRAD_SUCCESS) {
        printf("Error: Failed to initialize SGD optimizer\n");
        cgrad_cleanup();
        return 1;
    }
    
    printf("SGD Optimizer initialized with learning_rate=%.2f, momentum=%.2f\n\n", learning_rate, momentum);
    
    // ========================================================================
    // Gradient Descent Loop: 50 iterations
    // ========================================================================
    printf("--- Starting Gradient Descent (50 iterations) ---\n");
    printf("%-6s %-12s\n", "Iter", "Loss");
    printf("%-6s %-12s\n", "----", "----");
    
    const int num_iterations = 20;
    
    for (int iter = 0; iter < num_iterations; iter++) {

        // ====================================================================
        // Zero Gradients using Optimizer
        // ====================================================================
        cgrad_optimizer_zero_grad(&optimizer);

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
        cgrad_storage_get(cgrad_tensor_get_storage(&loss), (uint32_t[]){0}, 1, &loss_value);
        
        // ====================================================================
        // Backward Pass: Compute gradients
        // ====================================================================
        int ret = cgrad_tensor_backward(&loss);
        if (ret != 0) {
            printf("Error: Backward pass failed at iteration %d\n", iter);
            cgrad_optimizer_free(&optimizer);
            cgrad_cleanup();
            return 1;
        }
        
        // Print iteration info
        printf("%-6d %-12.6f\n", iter + 1, loss_value);
        
        // ====================================================================
        // Optimizer Step: Update parameters using SGD
        // ====================================================================
        status = cgrad_optimizer_step(&optimizer);
        if (status != CGRAD_SUCCESS) {
            printf("Error: Optimizer step failed at iteration %d\n", iter);
            cgrad_optimizer_free(&optimizer);
            cgrad_cleanup();
            return 1;
        }
    }
    
    printf("\n");

    cgrad_tensor_print(&A);
    
    // ========================================================================
    // Cleanup
    // ========================================================================
    printf("--- Cleanup ---\n");
    cgrad_optimizer_free(&optimizer);
    cgrad_cleanup();
    printf("All resources freed.\n");

    printf("\n========================================\n");
    printf("Optimization Complete!\n");
    printf("========================================\n");
    
    return 0;
}
