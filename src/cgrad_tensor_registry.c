#include "cgrad_tensor_registry.h"
#include "cgrad_errors.h"
#include <stdlib.h>
#include <stddef.h>

/* Global tensor registry instance */
cgrad_tensor_registry global_tensor_registry = {NULL};

/**
 * @brief Register a tensor in the global tensor registry.
 *        If parent is NULL, creates a new bucket with t as root.
 *        If parent is not NULL, adds t to the parent's bucket (if parent is registered).
 * @param t Pointer to tensor to register.
 * @param parent Pointer to parent tensor (or NULL).
 * @return CGRAD_SUCCESS on success, CGRAD_TENSOR_ERR_PARENT_NOT_REGISTERED if parent is not in registry.
 */
int cgrad_tensor_registry_register(cgrad_tensor* t, const cgrad_tensor* parent) {
    if (!t) return CGRAD_TENSOR_ERR_NULL_POINTER;

    cgrad_tensor_registry_entry* reg_entry = NULL;
    cgrad_tensor_registry_bucket* bucket = NULL;
    cgrad_tensor_registry_tensor_entry* entry = NULL;

    // Check if tensor is already registered
    HASH_FIND_PTR(global_tensor_registry.tensor_map, &t, reg_entry);
    if (reg_entry) {
        // Already registered, do nothing
        return CGRAD_SUCCESS;
    }

    if (parent == NULL) {
        // Create new bucket
        bucket = (cgrad_tensor_registry_bucket*)malloc(sizeof(cgrad_tensor_registry_bucket));
        if (!bucket) return CGRAD_TENSOR_F32_CPU_ERR_ALLOC_FAILED;
        bucket->root = t;
        bucket->tensor_map = NULL;

        // Add t to bucket's tensor_map
        entry = (cgrad_tensor_registry_tensor_entry*)malloc(sizeof(cgrad_tensor_registry_tensor_entry));
        if (!entry) {
            free(bucket);
            return CGRAD_TENSOR_F32_CPU_ERR_ALLOC_FAILED;
        }
        entry->tensor = t;
        HASH_ADD_PTR(bucket->tensor_map, tensor, entry);

        // Add t to registry (maps t to bucket)
        reg_entry = (cgrad_tensor_registry_entry*)malloc(sizeof(cgrad_tensor_registry_entry));
        if (!reg_entry) {
            // Clean up
            HASH_DEL(bucket->tensor_map, entry);
            free(entry);
            free(bucket);
            return CGRAD_TENSOR_F32_CPU_ERR_ALLOC_FAILED;
        }
        reg_entry->tensor = t;
        reg_entry->bucket = bucket;
        HASH_ADD_PTR(global_tensor_registry.tensor_map, tensor, reg_entry);
    } else {
        // Find parent's bucket
        cgrad_tensor_registry_entry* parent_entry = NULL;
        HASH_FIND_PTR(global_tensor_registry.tensor_map, &parent, parent_entry);
        if (!parent_entry) {
            return CGRAD_TENSOR_ERR_PARENT_NOT_REGISTERED;
        }
        bucket = parent_entry->bucket;

        // Add t to parent's bucket tensor_map
        entry = (cgrad_tensor_registry_tensor_entry*)malloc(sizeof(cgrad_tensor_registry_tensor_entry));
        if (!entry) return CGRAD_TENSOR_F32_CPU_ERR_ALLOC_FAILED;
        entry->tensor = t;
        HASH_ADD_PTR(bucket->tensor_map, tensor, entry);

        // Add t to registry (maps t to parent's bucket)
        reg_entry = (cgrad_tensor_registry_entry*)malloc(sizeof(cgrad_tensor_registry_entry));
        if (!reg_entry) {
            // Clean up
            HASH_DEL(bucket->tensor_map, entry);
            free(entry);
            return CGRAD_TENSOR_F32_CPU_ERR_ALLOC_FAILED;
        }
        reg_entry->tensor = t;
        reg_entry->bucket = bucket;
        HASH_ADD_PTR(global_tensor_registry.tensor_map, tensor, reg_entry);
    }

    return CGRAD_SUCCESS;
}

/**
 * @brief Get the number of tensors currently registered in the global tensor registry.
 * @return Number of registered tensors.
 */
size_t cgrad_tensor_registry_count(void) {
    return (size_t)HASH_COUNT(global_tensor_registry.tensor_map);
}

/**
 * @brief Deregister a tensor from the global tensor registry.
 *        Removes the tensor from its bucket and the registry.
 *        Writes the root tensor to *root if the bucket is empty after deregistering, NULL otherwise.
 * @param t Pointer to tensor to deregister.
 * @param root Output pointer to receive the root tensor if bucket is empty, NULL otherwise.
 * @return CGRAD_SUCCESS on success, CGRAD_TENSOR_ERR_PARENT_NOT_REGISTERED if tensor is not registered.
 */
int cgrad_tensor_registry_deregister(cgrad_tensor* t, cgrad_tensor** root) {
    if (root) *root = NULL;
    if (!t) return CGRAD_TENSOR_ERR_NULL_POINTER;

    cgrad_tensor_registry_entry* reg_entry = NULL;
    HASH_FIND_PTR(global_tensor_registry.tensor_map, &t, reg_entry);
    if (!reg_entry) return CGRAD_TENSOR_ERR_PARENT_NOT_REGISTERED;

    cgrad_tensor_registry_bucket* bucket = reg_entry->bucket;

    // Remove t from bucket's tensor_map
    cgrad_tensor_registry_tensor_entry* entry = NULL;
    HASH_FIND_PTR(bucket->tensor_map, &t, entry);
    if (entry) {
        HASH_DEL(bucket->tensor_map, entry);
        free(entry);
    }

    // Remove t from registry
    HASH_DEL(global_tensor_registry.tensor_map, reg_entry);
    free(reg_entry);

    // If bucket is now empty, free it and set *root
    if (bucket->tensor_map == NULL) {
        if (root) *root = bucket->root;
        free(bucket);
    } else {
        if (root) *root = NULL;
    }

    return CGRAD_SUCCESS;
}
