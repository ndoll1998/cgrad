#include "storage/cgrad_storage_backend.h"
#include "cgrad_errors.h"
#include "uthash.h"
#include <string.h>
#include <stdlib.h>

// Registry entry for uthash
typedef struct backend_registry_entry {
    const char* name;              // Key for uthash
    cgrad_storage_backend* backend;
    UT_hash_handle hh;             // uthash handle
} backend_registry_entry;

// Global registry
static backend_registry_entry* backend_registry = NULL;

int cgrad_backend_init_global_registry(void) {
    // Backend registry initialization - do NOT reset to NULL
    // Backends may have already registered themselves via constructor attributes
    // Just return success - the registry is ready
    return CGRAD_SUCCESS;
}

int cgrad_register_backend(cgrad_storage_backend* backend) {
    if (!backend || !backend->name) return -1;
    
    // Check if already registered
    backend_registry_entry* existing = NULL;
    HASH_FIND_STR(backend_registry, backend->name, existing);
    if (existing) {
        return -1;  // Already registered
    }
    
    // Create new entry
    backend_registry_entry* entry = malloc(sizeof(backend_registry_entry));
    if (!entry) return -1;
    
    entry->name = backend->name;
    entry->backend = backend;
    
    // Add to hash table
    HASH_ADD_KEYPTR(hh, backend_registry, entry->name, strlen(entry->name), entry);
    
    return 0;
}

cgrad_storage_backend* cgrad_get_backend(const char* name) {
    if (!name) return NULL;
    
    backend_registry_entry* entry = NULL;
    HASH_FIND_STR(backend_registry, name, entry);
    
    return entry ? entry->backend : NULL;
}

void cgrad_backend_cleanup_global_registry(void) {
    backend_registry_entry* entry, *tmp;
    HASH_ITER(hh, backend_registry, entry, tmp) {
        HASH_DEL(backend_registry, entry);
        free(entry);
    }
    backend_registry = NULL;
}
