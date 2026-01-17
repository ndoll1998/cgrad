#ifndef CGRAD_H
#define CGRAD_H

#include "autograd/cgrad_tensor.h"

/**
 * @file cgrad.h
 * @brief Main header file for the cgrad library.
 * 
 * This header provides the public API for the cgrad library, including
 * initialization, cleanup, and tensor operations.
 */

// ============================================================================
// Library Initialization and Cleanup
// ============================================================================

/**
 * @brief Initialize the cgrad library.
 * 
 * This function initializes all global state in the cgrad library:
 * 1. Backend registry (ready for backend registrations)
 * 2. Default backends (f32_cpu auto-registers via constructor)
 * 3. Global storage registry
 * 4. Global compute graph
 * 5. Gradient mode (enabled by default)
 * 
 * This function is marked with __attribute__((constructor)) and will be
 * called automatically when the library is loaded. It can also be called
 * explicitly if needed. Multiple calls are safe - the function will only
 * initialize once.
 * 
 * Note: Backends register themselves via constructor attributes before this
 * function runs, so they are available when initialization completes.
 * 
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_init(void) __attribute__((constructor));

/**
 * @brief Cleanup the cgrad library.
 * 
 * This function cleans up all global state in the cgrad library:
 * 1. Global compute graph
 * 2. Global storage registry
 * 3. Backend registry
 * 
 * This should be called when the library is no longer needed, typically
 * at program shutdown. After calling this function, the library must be
 * re-initialized with cgrad_init() before use.
 * 
 * Multiple calls are safe - the function will only cleanup once.
 */
void cgrad_cleanup(void);

/**
 * @brief Check if the cgrad library is initialized.
 * 
 * @return 1 if initialized, 0 otherwise.
 */
int cgrad_is_initialized(void);

#endif // CGRAD_H
