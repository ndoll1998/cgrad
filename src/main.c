#include "autograd/cgrad_tensor.h"
#include "storage/cgrad_storage.h"
#include "cgrad_errors.h"
#include <stdio.h>
#include <stdint.h>

/**
 * @brief Example program demonstrating autograd with matrix multiplication.
 * 
 * This program:
 * 1. Initializes two random matrices A and B
 * 2. Performs matrix multiplication: C = A @ B
 * 3. Sums the result: loss = sum(C)
 * 4. Computes gradients via backpropagation
 * 5. Prints the gradient of A
 */
int main() {
    printf("========================================\n");
    printf("Autograd Demo: Matrix Multiplication\n");
    printf("========================================\n\n");
    
    // ========================================================================
    // Initialize two random matrices A and B
    // ========================================================================
    printf("--- Initializing Matrices ---\n");
    
    cgrad_tensor A, B;
    uint32_t shape_A[] = {3, 4};  // 3x4 matrix
    uint32_t shape_B[] = {4, 2};  // 4x2 matrix
    
    // Create tensors
    cgrad_tensor_init(&A, shape_A, 2, "f32_cpu");
    cgrad_tensor_init(&B, shape_B, 2, "f32_cpu");
    
    // Fill with random values
    cgrad_tensor_fill_rand(&A);
    cgrad_tensor_fill_rand(&B);
    
    // Set gradient requirements: A requires gradients, B does not
    cgrad_tensor_set_requires_grad(&A, 1);  // Enable gradients for A
    cgrad_tensor_set_requires_grad(&B, 0);  // Disable gradients for B
    
    printf("Matrix A (3x4) - requires_grad=True:\n");
    cgrad_tensor_execute(&A);
    cgrad_tensor_print(&A);
    printf("\n");
    
    printf("Matrix B (4x2) - requires_grad=False:\n");
    cgrad_tensor_execute(&B);
    cgrad_tensor_print(&B);
    printf("\n");
    
    // ========================================================================
    // Matrix Multiplication: C = A @ B
    // ========================================================================
    printf("--- Matrix Multiplication: C = A @ B ---\n");
    
    cgrad_tensor C;
    cgrad_tensor_gemm(&A, &B, &C);
    
    printf("Result C (3x2):\n");
    cgrad_tensor_print(&C);
    printf("\n");
    
    // ========================================================================
    // Sum Reduction: loss = sum(C)
    // ========================================================================
    printf("--- Sum Reduction: loss = sum(C) ---\n");
    
    cgrad_tensor loss;
    uint8_t reduce_mask[] = {1, 1};  // Reduce over all dimensions
    cgrad_tensor_reduce_sum(&C, reduce_mask, 2, &loss);
    
    printf("Loss (scalar):\n");
    cgrad_tensor_print(&loss);
    printf("\n");
    
    // ========================================================================
    // Backward Pass: Compute Gradients
    // ========================================================================
    printf("--- Computing Gradients (Backward Pass) ---\n");
    
    int ret = cgrad_tensor_backward(&loss);
    if (ret != 0) {
        printf("Error: Backward pass failed with code %d\n", ret);
        cgrad_tensor_cleanup_global_graph();
        return 1;
    }
    
    printf("Backward pass completed successfully!\n\n");
    
    // ========================================================================
    // Print Gradient of A
    // ========================================================================
    printf("--- Gradient of A (dL/dA) ---\n");
    
    cgrad_tensor grad_A;
    ret = cgrad_tensor_get_gradient(&A, &grad_A);
    if (ret == CGRAD_SUCCESS) {
        printf("Gradient of A (3x4):\n");
        cgrad_tensor_print(&grad_A);
    } else {
        printf("Warning: Could not get gradient of A (error code: %d)\n", ret);
    }
    printf("\n");
    
    // Verify that B has no gradient (as expected)
    printf("--- Gradient of B (should be NULL) ---\n");
    cgrad_tensor grad_B;
    ret = cgrad_tensor_get_gradient(&B, &grad_B);
    if (ret != CGRAD_SUCCESS) {
        printf("Gradient of B: Not available (as expected, requires_grad=False)\n");
    } else {
        printf("Warning: Gradient of B exists (unexpected!)\n");
        cgrad_tensor_print(&grad_B);
    }
    printf("\n");
    
    // ========================================================================
    // Cleanup
    // ========================================================================
    printf("--- Cleanup ---\n");
    cgrad_tensor_cleanup_global_graph();
    printf("All resources freed.\n");

    printf("\n========================================\n");
    printf("Autograd Demo Complete!\n");
    printf("========================================\n");
    
    return 0;
}
