#include "autograd/cgrad_tensor.h"
#include "storage/cgrad_storage.h"
#include <stdio.h>
#include <stdint.h>

/**
 * @brief Example program demonstrating the lazy compute graph framework.
 * 
 * This program builds a computation graph lazily, then executes it on demand.
 * It also demonstrates graph visualization via DOT export.
 */
int main() {
    printf("========================================\n");
    printf("Lazy Compute Graph Demo for cgrad\n");
    printf("========================================\n\n");
    
    // ========================================================================
    // Example 1: Simple Addition
    // ========================================================================
    printf("--- Example 1: Simple Addition (c = a + b) ---\n");
    
    cgrad_tensor a, b, c;
    uint32_t shape[] = {2, 3};
    
    // Create input tensors (creates separate graphs)
    cgrad_tensor_init(&a, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_init(&b, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    
    // Fill with data
    cgrad_tensor_fill(&a, 2.0f);
    cgrad_tensor_fill(&b, 3.0f);
    
    // Build computation graph (lazy - no execution yet)
    cgrad_tensor_add(&a, &b, &c);
    printf("Graph built (lazy): c = a + b\n");
    
    // Execute the graph
    cgrad_tensor_execute(&c);
    printf("Graph executed!\n");
    
    // Print results
    printf("\nResult (c = a + b):\n");
    cgrad_tensor_print(&c);
    
    printf("\n");
    
    // ========================================================================
    // Example 2: Complex Graph
    // ========================================================================
    printf("--- Example 2: Complex Graph ((a + b) * c) ---\n");
    
    cgrad_tensor x, y, z, sum, result;
    uint32_t shape2[] = {2, 2};
    
    // Create inputs
    cgrad_tensor_init(&x, shape2, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_init(&y, shape2, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_init(&z, shape2, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    
    cgrad_tensor_fill(&x, 1.0f);
    cgrad_tensor_fill(&y, 2.0f);
    cgrad_tensor_fill(&z, 3.0f);
    
    // Build graph: sum = x + y, result = sum @ z (matrix multiply as approximation)
    cgrad_tensor_add(&x, &y, &sum);
    cgrad_tensor_gemm(&sum, &z, &result);
    printf("Graph built: result = (x + y) @ z\n");
    
    // Execute
    cgrad_tensor_execute(&result);
    printf("Graph executed!\n");
    
    printf("\nResult ((x + y) @ z):\n");
    cgrad_tensor_print(&result);
    printf("\n");
    
    // ========================================================================
    // Example 3: Transpose and Reshape
    // ========================================================================
    printf("--- Example 3: Transpose and Reshape ---\n");
    
    cgrad_tensor mat, transposed, reshaped;
    uint32_t mat_shape[] = {2, 3};
    
    cgrad_tensor_init(&mat, mat_shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_fill_rand(&mat);
    
    // Transpose
    uint32_t perm[] = {1, 0};
    cgrad_tensor_transpose(&mat, perm, 2, &transposed);
    
    // Reshape
    int32_t new_shape[] = {3, 2};
    cgrad_tensor_reshape(&transposed, new_shape, 2, &reshaped);
    
    printf("Graph built: reshape(transpose(mat))\n");
    
    // Execute
    cgrad_tensor_execute(&reshaped);
    printf("Graph executed!\n");
    
    printf("\nResult:\n");
    cgrad_tensor_print(&reshaped);
    
    // ========================================================================
    // Example 4: Execution Caching Demo
    // ========================================================================
    printf("\n--- Example 4: Execution Caching ---\n");
    
    cgrad_tensor p, q, cached;
    cgrad_tensor_init(&p, (uint32_t[]){2, 2}, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_tensor_init(&q, (uint32_t[]){2, 2}, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    
    cgrad_tensor_fill(&p, 5.0f);
    cgrad_tensor_fill(&q, 10.0f);
    
    cgrad_tensor_sub(&p, &q, &cached);
    
    printf("First execution...\n");
    cgrad_tensor_execute(&cached);
    
    printf("Second execution (uses cached result)...\n");
    cgrad_tensor_execute(&cached);
    
    printf("Result:\n");
    cgrad_tensor_print(&cached);
    
    // ========================================================================
    // Cleanup
    // ========================================================================
    printf("\n--- Cleanup ---\n");
    
    // Cleanup the global compute graph
    cgrad_tensor_cleanup_global_graph();
    
    printf("All resources freed.\n");

    printf("\n========================================\n");
    printf("Lazy Compute Graph Demo Complete!\n");
    printf("========================================\n");
    
    return 0;
}
