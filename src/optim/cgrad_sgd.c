#include "optim/cgrad_sgd.h"
#include "storage/cgrad_storage.h"
#include "backends/cgrad_backend.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ============================================================================
// Forward Declarations
// ============================================================================

static cgrad_status cgrad_sgd_step_impl(cgrad_optimizer* optimizer);
static void cgrad_sgd_free_state_impl(cgrad_optimizer* optimizer);

// ============================================================================
// SGD Virtual Function Table
// ============================================================================

static cgrad_optim_vtable sgd_vtable = {
    .step = cgrad_sgd_step_impl,
    .zero_grad = cgrad_optimizer_zero_grad_default,
    .free_state = cgrad_sgd_free_state_impl
};

// ============================================================================
// utarray configuration for storage pointers
// ============================================================================

static UT_icd storage_ptr_icd = {sizeof(cgrad_storage*), NULL, NULL, NULL};

// ============================================================================
// SGD Optimizer Initialization
// ============================================================================

cgrad_status cgrad_sgd_init(
    cgrad_optimizer* optimizer,
    cgrad_tensor** parameters,
    size_t num_parameters,
    float learning_rate,
    float momentum
) {
    if (optimizer == NULL) {
        return CGRAD_ERR_INVALID_ARGUMENT;
    }
    
    if (learning_rate <= 0.0f) {
        return CGRAD_ERR_INVALID_ARGUMENT;
    }
    
    if (momentum < 0.0f || momentum >= 1.0f) {
        return CGRAD_ERR_INVALID_ARGUMENT;
    }
    
    // Allocate SGD state
    cgrad_sgd_state* state = (cgrad_sgd_state*)malloc(sizeof(cgrad_sgd_state));
    if (state == NULL) {
        return CGRAD_ERROR_OUT_OF_MEMORY;
    }
    
    state->learning_rate = learning_rate;
    state->momentum = momentum;
    state->velocity_buffers = NULL;
    
    // Initialize velocity buffers if using momentum
    if (momentum > 0.0f && num_parameters > 0) {
        utarray_new(state->velocity_buffers, &storage_ptr_icd);
        if (state->velocity_buffers == NULL) {
            free(state);
            return CGRAD_ERROR_OUT_OF_MEMORY;
        }
        
        // Create velocity buffer for each parameter
        for (size_t i = 0; i < num_parameters; i++) {
            if (parameters[i] == NULL) {
                utarray_free(state->velocity_buffers);
                free(state);
                return CGRAD_ERR_INVALID_ARGUMENT;
            }
            
            // For now, we'll create the velocity buffer lazily during first step
            cgrad_storage* null_storage = NULL;
            utarray_push_back(state->velocity_buffers, &null_storage);
        }
    }
    
    // Initialize base optimizer
    cgrad_status status = cgrad_optimizer_init(
        optimizer,
        parameters,
        num_parameters,
        state,
        &sgd_vtable
    );
    
    if (status != CGRAD_SUCCESS) {
        // Clean up on failure
        if (state->velocity_buffers != NULL) {
            // Free velocity buffers
            cgrad_storage** vel_ptr = NULL;
            while ((vel_ptr = (cgrad_storage**)utarray_next(state->velocity_buffers, vel_ptr)) != NULL) {
                if (*vel_ptr != NULL) {
                    cgrad_storage_free(*vel_ptr);
                }
            }
            utarray_free(state->velocity_buffers);
        }
        free(state);
        return status;
    }
    
    return CGRAD_SUCCESS;
}

// ============================================================================
// SGD Step Implementation
// ============================================================================

static cgrad_status cgrad_sgd_step_impl(cgrad_optimizer* optimizer) {
    if (optimizer == NULL || optimizer->state == NULL) {
        return CGRAD_ERR_INVALID_ARGUMENT;
    }
    
    cgrad_sgd_state* state = (cgrad_sgd_state*)optimizer->state;
    cgrad_tensor** param_ptr = NULL;
    size_t idx = 0;
    
    // Iterate through all parameters
    while ((param_ptr = (cgrad_tensor**)utarray_next(optimizer->parameters, param_ptr)) != NULL) {
        cgrad_tensor* param = *param_ptr;
        
        // Get gradient storage
        cgrad_storage* grad_storage = cgrad_tensor_get_grad_storage(param);
        if (grad_storage == NULL) {
            // No gradient for this parameter, skip
            idx++;
            continue;
        }
        
        // Get parameter storage
        cgrad_storage* param_storage = cgrad_tensor_get_storage(param);
        if (param_storage == NULL) {
            return CGRAD_ERROR_INVALID_STATE;
        }
        
        // Apply update based on momentum
        if (state->momentum > 0.0f) {
            // Get velocity buffer for this parameter
            cgrad_storage** velocity_ptr = (cgrad_storage**)utarray_eltptr(state->velocity_buffers, idx);
            if (velocity_ptr == NULL) {
                return CGRAD_ERROR_INVALID_STATE;
            }
            
            cgrad_storage* velocity = *velocity_ptr;
            
            // Lazy initialization of velocity buffer if needed
            if (velocity == NULL) {
                // Create velocity buffer using high-level storage API
                velocity = (cgrad_storage*)malloc(sizeof(cgrad_storage));
                if (velocity == NULL) {
                    return CGRAD_ERROR_OUT_OF_MEMORY;
                }
                
                // Initialize storage with same backend as parameter
                cgrad_status status = cgrad_storage_init(velocity, param->layout.shape, TENSOR_DIM, param_storage->backend->name);
                if (status != CGRAD_SUCCESS) {
                    free(velocity);
                    return status;
                }
                
                // Fill with zeros
                cgrad_storage_fill(velocity, 0.0f);
                *velocity_ptr = velocity;
            }
            
            // Update: velocity = momentum * velocity + gradient
            // Use high-level storage API
            cgrad_storage_axpy(state->momentum, velocity, grad_storage, velocity);
            
            // param = param - learning_rate * velocity
            cgrad_storage_axpy(-state->learning_rate, velocity, param_storage, param_storage);
        } else {
            // No momentum: param = param - learning_rate * gradient
            cgrad_storage_axpy(-state->learning_rate, grad_storage, param_storage, param_storage);
        }
        
        idx++;
    }
    
    return CGRAD_SUCCESS;
}

// ============================================================================
// SGD Free State Implementation
// ============================================================================

static void cgrad_sgd_free_state_impl(cgrad_optimizer* optimizer) {
    if (optimizer == NULL || optimizer->state == NULL) {
        return;
    }
    
    cgrad_sgd_state* state = (cgrad_sgd_state*)optimizer->state;
    
    // Free velocity buffers
    if (state->velocity_buffers != NULL) {
        cgrad_storage** vel_ptr = NULL;
        while ((vel_ptr = (cgrad_storage**)utarray_next(state->velocity_buffers, vel_ptr)) != NULL) {
            if (*vel_ptr != NULL) {
                cgrad_storage_free(*vel_ptr);
            }
        }
        utarray_free(state->velocity_buffers);
    }
    
    // Free state structure
    free(state);
}

// ============================================================================
// SGD-specific Functions
// ============================================================================

cgrad_status cgrad_sgd_get_learning_rate(const cgrad_optimizer* optimizer, float* out_lr) {
    if (optimizer == NULL || optimizer->state == NULL || out_lr == NULL) {
        return CGRAD_ERR_INVALID_ARGUMENT;
    }
    
    cgrad_sgd_state* state = (cgrad_sgd_state*)optimizer->state;
    *out_lr = state->learning_rate;
    
    return CGRAD_SUCCESS;
}

cgrad_status cgrad_sgd_set_learning_rate(cgrad_optimizer* optimizer, float learning_rate) {
    if (optimizer == NULL || optimizer->state == NULL) {
        return CGRAD_ERR_INVALID_ARGUMENT;
    }
    
    if (learning_rate <= 0.0f) {
        return CGRAD_ERR_INVALID_ARGUMENT;
    }
    
    cgrad_sgd_state* state = (cgrad_sgd_state*)optimizer->state;
    state->learning_rate = learning_rate;
    
    return CGRAD_SUCCESS;
}

cgrad_status cgrad_sgd_get_momentum(const cgrad_optimizer* optimizer, float* out_momentum) {
    if (optimizer == NULL || optimizer->state == NULL || out_momentum == NULL) {
        return CGRAD_ERR_INVALID_ARGUMENT;
    }
    
    cgrad_sgd_state* state = (cgrad_sgd_state*)optimizer->state;
    *out_momentum = state->momentum;
    
    return CGRAD_SUCCESS;
}
