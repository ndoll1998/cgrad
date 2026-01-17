#include "backends/cgrad_backend.h"
#include "cgrad_errors.h"
#include "uthash.h"
#include <string.h>
#include <stdlib.h>

// Global registry - backends are stored directly with their embedded uthash handle
static cgrad_backend* backend_registry = NULL;

int cgrad_backend_init_global_registry(void) {
    // Backend registry initialization - do NOT reset to NULL
    // Backends may have already registered themselves via constructor attributes
    // Just return success - the registry is ready
    return CGRAD_SUCCESS;
}

int cgrad_register_backend(cgrad_backend* backend) {
    if (!backend || !backend->name) return -1;
    
    // Check if already registered
    cgrad_backend* existing = NULL;
    HASH_FIND_STR(backend_registry, backend->name, existing);
    if (existing) {
        return -1;  // Already registered
    }
    
    // Add backend directly to hash table using its embedded uthash handle
    HASH_ADD_KEYPTR(hh, backend_registry, backend->name, strlen(backend->name), backend);
    
    return 0;
}

cgrad_backend* cgrad_get_backend(const char* name) {
    if (!name) return NULL;
    
    cgrad_backend* backend = NULL;
    HASH_FIND_STR(backend_registry, name, backend);
    
    return backend;
}

void cgrad_backend_cleanup_global_registry(void) {
    cgrad_backend* backend, *tmp;
    HASH_ITER(hh, backend_registry, backend, tmp) {
        HASH_DEL(backend_registry, backend);
    }
    backend_registry = NULL;
}
