#ifndef CGRAD_BACKEND_REGISTRY_H
#define CGRAD_BACKEND_REGISTRY_H

#include "backends/cgrad_backend.h"

/**
 * @file cgrad_backend_registry.h
 * @brief Backend registry interface for managing available backends.
 * 
 * This header provides the public API for registering and retrieving backends.
 * Backends can register themselves automatically via constructor attributes.
 */

/**
 * @brief Register a backend in the global registry.
 * @param backend Pointer to backend to register.
 * @return CGRAD_SUCCESS on success, CGRAD_ERR_BACKEND_REGISTRY_DUPLICATE if backend with same name already registered, other error code on other errors.
 */
cgrad_status cgrad_register_backend(cgrad_backend* backend);

/**
 * @brief Get the backend for a given backend name.
 * @param name Backend name (e.g., "cpu_f32").
 * @return Pointer to the backend, or NULL if not found.
 */
cgrad_backend* cgrad_get_backend(const char* name);

// ============================================================================
// Global Backend Registry Management
// ============================================================================

/**
 * @brief Initialize the global backend registry.
 * 
 * This function initializes the global backend registry.
 * It should be called once during library initialization.
 * The backend registry starts as NULL and is ready for backend registrations.
 * 
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
cgrad_status cgrad_backend_init_registry(void);

/**
 * @brief Cleanup the backend registry.
 * 
 * This function frees all resources associated with the backend registry.
 * It should be called once during library cleanup.
 */
void cgrad_backend_cleanup_registry(void);

#endif // CGRAD_BACKEND_REGISTRY_H
