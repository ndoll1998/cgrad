#include "cgrad.h"
#include "cgrad_errors.h"
#include "autograd/cgrad_tensor.h"
#include "storage/cgrad_storage_registry.h"
#include "storage/cgrad_storage_backend.h"
#include <stdlib.h>
#include <stdio.h>

// ============================================================================
// Initialization State
// ============================================================================

static int g_cgrad_initialized = 0;

int cgrad_init(void) {
    // Prevent double initialization
    if (g_cgrad_initialized) {
        return CGRAD_SUCCESS;
    }

    int ret;

    // Step 1: Initialize backend registry
    ret = cgrad_backend_init_global_registry();
    if (ret != CGRAD_SUCCESS) {
        fprintf(stderr, "cgrad_init: Failed to initialize backend registry\n");
        return ret;
    }

    // Step 2: Initialize global storage registry
    ret = cgrad_storage_init_global_registry();
    if (ret != CGRAD_SUCCESS) {
        fprintf(stderr, "cgrad_init: Failed to initialize global storage registry\n");
        cgrad_backend_cleanup_global_registry();
        return ret;
    }

    // Step 3: Initialize global compute graph
    ret = cgrad_tensor_init_global_graph();
    if (ret != CGRAD_SUCCESS) {
        fprintf(stderr, "cgrad_init: Failed to initialize global compute graph\n");
        cgrad_storage_cleanup_global_registry();
        cgrad_backend_cleanup_global_registry();
        return ret;
    }

    // Step 4: Enable gradient mode (ensure it's enabled)
    ret = cgrad_enable_grad();
    if (ret != CGRAD_SUCCESS) {
        fprintf(stderr, "cgrad_init: Failed to enable gradient mode\n");
        cgrad_tensor_cleanup_global_graph();
        cgrad_storage_cleanup_global_registry();
        cgrad_backend_cleanup_global_registry();
        return ret;
    }

    // Mark as initialized
    g_cgrad_initialized = 1;

    return CGRAD_SUCCESS;
}

void cgrad_cleanup(void) {
    // Prevent double cleanup
    if (!g_cgrad_initialized) {
        return;
    }

    // Step 1: Cleanup global compute graph
    cgrad_tensor_cleanup_global_graph();

    // Step 2: Cleanup global storage registry
    cgrad_storage_cleanup_global_registry();

    // Mark as uninitialized
    g_cgrad_initialized = 0;
}

int cgrad_is_initialized(void) {
    return g_cgrad_initialized;
}
