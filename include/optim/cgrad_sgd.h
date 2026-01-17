#ifndef CGRAD_SGD_H
#define CGRAD_SGD_H

#include "optim/cgrad_optimizer.h"

/**
 * @file cgrad_sgd.h
 * @brief Stochastic Gradient Descent (SGD) optimizer with optional momentum.
 * 
 * This module implements the SGD optimization algorithm:
 * 
 * Without momentum:
 *   param = param - learning_rate * gradient
 * 
 * With momentum:
 *   velocity = momentum * velocity + gradient
 *   param = param - learning_rate * velocity
 */

/**
 * @brief SGD optimizer state.
 * 
 * This structure holds the hyperparameters and momentum buffers
 * for the SGD optimizer.
 */
typedef struct {
    float learning_rate;        /**< Learning rate (step size) */
    float momentum;             /**< Momentum coefficient (0 = no momentum) */
    UT_array* velocity_buffers; /**< Velocity buffers for momentum (one per parameter) */
} cgrad_sgd_state;

// ============================================================================
// SGD Optimizer Initialization
// ============================================================================

/**
 * @brief Initialize an SGD optimizer.
 * 
 * This function creates an SGD optimizer with the specified hyperparameters.
 * If momentum > 0, velocity buffers are allocated for each parameter.
 * 
 * @param optimizer Pointer to optimizer structure to initialize.
 * @param parameters Array of pointers to tensors to optimize.
 * @param num_parameters Number of parameters in the array.
 * @param learning_rate Learning rate (step size). Must be > 0.
 * @param momentum Momentum coefficient. Use 0 for no momentum, typical values are 0.9 or 0.99.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
cgrad_status cgrad_sgd_init(
    cgrad_optimizer* optimizer,
    cgrad_tensor** parameters,
    size_t num_parameters,
    float learning_rate,
    float momentum
);

// ============================================================================
// SGD-specific Functions
// ============================================================================

/**
 * @brief Get the learning rate of an SGD optimizer.
 * 
 * @param optimizer The SGD optimizer instance.
 * @param out_lr Pointer to store the learning rate.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
cgrad_status cgrad_sgd_get_learning_rate(const cgrad_optimizer* optimizer, float* out_lr);

/**
 * @brief Set the learning rate of an SGD optimizer.
 * 
 * This is useful for implementing learning rate schedules.
 * 
 * @param optimizer The SGD optimizer instance.
 * @param learning_rate New learning rate. Must be > 0.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
cgrad_status cgrad_sgd_set_learning_rate(cgrad_optimizer* optimizer, float learning_rate);

/**
 * @brief Get the momentum coefficient of an SGD optimizer.
 * 
 * @param optimizer The SGD optimizer instance.
 * @param out_momentum Pointer to store the momentum coefficient.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
cgrad_status cgrad_sgd_get_momentum(const cgrad_optimizer* optimizer, float* out_momentum);

#endif // CGRAD_SGD_H
