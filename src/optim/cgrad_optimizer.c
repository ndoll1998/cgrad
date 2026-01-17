#include "optim/cgrad_optimizer.h"
#include <stdlib.h>
#include <string.h>

// ============================================================================
// utarray configuration for cgrad_tensor pointers
// ============================================================================

static UT_icd tensor_ptr_icd = {sizeof(cgrad_tensor*), NULL, NULL, NULL};

// ============================================================================
// Optimizer Initialization and Management
// ============================================================================

cgrad_status cgrad_optimizer_init(
    cgrad_optimizer* optimizer,
    cgrad_tensor** parameters,
    size_t num_parameters,
    void* state,
    cgrad_optim_vtable* vtable
) {
    if (optimizer == NULL || vtable == NULL) {
        return CGRAD_ERR_INVALID_ARGUMENT;
    }
    
    if (num_parameters > 0 && parameters == NULL) {
        return CGRAD_ERR_INVALID_ARGUMENT;
    }
    
    // Initialize the parameter array
    utarray_new(optimizer->parameters, &tensor_ptr_icd);
    if (optimizer->parameters == NULL) {
        return CGRAD_ERROR_OUT_OF_MEMORY;
    }
    
    // Copy parameter pointers into the array
    for (size_t i = 0; i < num_parameters; i++) {
        if (parameters[i] == NULL) {
            utarray_free(optimizer->parameters);
            return CGRAD_ERR_INVALID_ARGUMENT;
        }
        utarray_push_back(optimizer->parameters, &parameters[i]);
    }
    
    // Set state and vtable
    optimizer->state = state;
    optimizer->vtable = vtable;
    
    return CGRAD_SUCCESS;
}

void cgrad_optimizer_free(cgrad_optimizer* optimizer) {
    if (optimizer == NULL) {
        return;
    }
    
    // Call optimizer-specific cleanup
    if (optimizer->vtable && optimizer->vtable->free_state) {
        optimizer->vtable->free_state(optimizer);
    }
    
    // Free the parameter array
    if (optimizer->parameters) {
        utarray_free(optimizer->parameters);
        optimizer->parameters = NULL;
    }
    
    // Clear fields
    optimizer->state = NULL;
    optimizer->vtable = NULL;
}

// ============================================================================
// Optimizer Operations
// ============================================================================

cgrad_status cgrad_optimizer_step(cgrad_optimizer* optimizer) {
    if (optimizer == NULL || optimizer->vtable == NULL) {
        return CGRAD_ERR_INVALID_ARGUMENT;
    }
    
    if (optimizer->vtable->step == NULL) {
        return CGRAD_ERROR_NOT_IMPLEMENTED;
    }
    
    return optimizer->vtable->step(optimizer);
}

cgrad_status cgrad_optimizer_zero_grad(cgrad_optimizer* optimizer) {
    if (optimizer == NULL || optimizer->vtable == NULL) {
        return CGRAD_ERR_INVALID_ARGUMENT;
    }
    
    if (optimizer->vtable->zero_grad == NULL) {
        return CGRAD_ERROR_NOT_IMPLEMENTED;
    }
    
    return optimizer->vtable->zero_grad(optimizer);
}

// ============================================================================
// Utility Functions
// ============================================================================

cgrad_status cgrad_optimizer_zero_grad_default(cgrad_optimizer* optimizer) {
    if (optimizer == NULL || optimizer->parameters == NULL) {
        return CGRAD_ERR_INVALID_ARGUMENT;
    }
    
    cgrad_tensor** param_ptr = NULL;
    
    // Iterate through all parameters
    while ((param_ptr = (cgrad_tensor**)utarray_next(optimizer->parameters, param_ptr)) != NULL) {
        cgrad_tensor* param = *param_ptr;
        if (param != NULL) {
            cgrad_status status = cgrad_tensor_zero_grad(param);
            if (status != CGRAD_SUCCESS) {
                return status;
            }
        }
    }
    
    return CGRAD_SUCCESS;
}

size_t cgrad_optimizer_num_parameters(const cgrad_optimizer* optimizer) {
    if (optimizer == NULL || optimizer->parameters == NULL) {
        return 0;
    }
    
    return utarray_len(optimizer->parameters);
}

cgrad_tensor* cgrad_optimizer_get_parameter(const cgrad_optimizer* optimizer, size_t index) {
    if (optimizer == NULL || optimizer->parameters == NULL) {
        return NULL;
    }
    
    if (index >= utarray_len(optimizer->parameters)) {
        return NULL;
    }
    
    cgrad_tensor** param_ptr = (cgrad_tensor**)utarray_eltptr(optimizer->parameters, index);
    if (param_ptr == NULL) {
        return NULL;
    }
    
    return *param_ptr;
}
