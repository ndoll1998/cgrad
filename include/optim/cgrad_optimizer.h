#ifndef CGRAD_OPTIMIZER_H
#define CGRAD_OPTIMIZER_H

#include "cgrad_status.h"
#include "autograd/cgrad_tensor.h"
#include "third_party/utarray.h"

/**
 * @file cgrad_optimizer.h
 * @brief Abstract optimizer interface for parameter optimization.
 * 
 * This module provides a unified interface for optimizers that update
 * model parameters based on their gradients. The design is inspired by
 * PyTorch's optimizer architecture and uses virtual function tables
 * for polymorphism.
 */

// Forward declarations
typedef struct cgrad_optimizer cgrad_optimizer;
typedef struct cgrad_optim_vtable cgrad_optim_vtable;

/**
 * @brief Virtual function table for optimizer operations.
 * 
 * This structure defines the interface that all optimizers must implement.
 * It enables polymorphism through function pointers.
 */
struct cgrad_optim_vtable {
    /**
     * @brief Perform a single optimization step.
     * 
     * This function updates all parameters based on their current gradients.
     * It should be called after backward() has computed gradients.
     * 
     * @param optimizer The optimizer instance.
     * @return CGRAD_SUCCESS on success, error code otherwise.
     */
    cgrad_status (*step)(cgrad_optimizer* optimizer);
    
    /**
     * @brief Zero out gradients for all parameters.
     * 
     * This function should be called before each backward pass to clear
     * accumulated gradients from the previous iteration.
     * 
     * @param optimizer The optimizer instance.
     * @return CGRAD_SUCCESS on success, error code otherwise.
     */
    cgrad_status (*zero_grad)(cgrad_optimizer* optimizer);
    
    /**
     * @brief Free optimizer-specific resources.
     * 
     * This function cleans up any optimizer-specific state (e.g., momentum
     * buffers, Adam moment estimates). It does not free the optimizer
     * structure itself or the parameter array.
     * 
     * @param optimizer The optimizer instance.
     */
    void (*free_state)(cgrad_optimizer* optimizer);
};

/**
 * @brief Abstract optimizer structure.
 * 
 * This structure represents a generic optimizer that can update a collection
 * of parameters. Specific optimizers (SGD, Adam, etc.) extend this by
 * providing their own state and vtable implementations.
 */
struct cgrad_optimizer {
    UT_array* parameters;           /**< Dynamic array of cgrad_tensor* */
    void* state;                    /**< Optimizer-specific state */
    cgrad_optim_vtable* vtable;     /**< Virtual function table */
};

// ============================================================================
// Optimizer Initialization and Management
// ============================================================================

/**
 * @brief Initialize an optimizer with the given parameters.
 * 
 * This function sets up the base optimizer structure by:
 * 1. Creating a utarray to store parameter pointers
 * 2. Copying all parameter pointers into the array
 * 3. Setting the state and vtable fields
 * 
 * Note: This is typically called by specific optimizer initialization
 * functions (e.g., cgrad_sgd_init) rather than directly by users.
 * 
 * @param optimizer Pointer to optimizer structure to initialize.
 * @param parameters Array of pointers to tensors to optimize.
 * @param num_parameters Number of parameters in the array.
 * @param state Optimizer-specific state.
 * @param vtable Virtual function table for this optimizer type.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
cgrad_status cgrad_optimizer_init(
    cgrad_optimizer* optimizer,
    cgrad_tensor** parameters,
    size_t num_parameters,
    void* state,
    cgrad_optim_vtable* vtable
);

/**
 * @brief Free optimizer resources.
 * 
 * This function:
 * 1. Calls the optimizer-specific free_state function
 * 2. Frees the parameter array
 * 3. Does NOT free the optimizer structure itself (stack-allocated)
 * 4. Does NOT free the parameter tensors (owned by caller)
 * 
 * @param optimizer Optimizer to free.
 */
void cgrad_optimizer_free(cgrad_optimizer* optimizer);

// ============================================================================
// Optimizer Operations
// ============================================================================

/**
 * @brief Perform a single optimization step.
 * 
 * This function delegates to the optimizer-specific step implementation
 * via the vtable. It updates all parameters based on their gradients.
 * 
 * @param optimizer The optimizer instance.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
cgrad_status cgrad_optimizer_step(cgrad_optimizer* optimizer);

/**
 * @brief Zero out gradients for all parameters.
 * 
 * This function delegates to the optimizer-specific zero_grad implementation
 * via the vtable. It should be called before each backward pass.
 * 
 * @param optimizer The optimizer instance.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
cgrad_status cgrad_optimizer_zero_grad(cgrad_optimizer* optimizer);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Default implementation of zero_grad.
 * 
 * This function iterates through all parameters and calls
 * cgrad_tensor_zero_grad on each one. Optimizers can use this
 * as their zero_grad implementation if they don't need custom behavior.
 * 
 * @param optimizer The optimizer instance.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
cgrad_status cgrad_optimizer_zero_grad_default(cgrad_optimizer* optimizer);

/**
 * @brief Get the number of parameters in the optimizer.
 * 
 * @param optimizer The optimizer instance.
 * @return Number of parameters, or 0 if optimizer is NULL.
 */
size_t cgrad_optimizer_num_parameters(const cgrad_optimizer* optimizer);

/**
 * @brief Get a parameter by index.
 * 
 * @param optimizer The optimizer instance.
 * @param index Index of the parameter (0-based).
 * @return Pointer to the parameter tensor, or NULL if index is out of bounds.
 */
cgrad_tensor* cgrad_optimizer_get_parameter(const cgrad_optimizer* optimizer, size_t index);

#endif // CGRAD_OPTIMIZER_H
