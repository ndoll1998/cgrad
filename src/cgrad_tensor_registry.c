#include "cgrad_tensor_registry.h"
#include "cgrad_errors.h"
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <uuid/uuid.h>

/* Global tensor registry instance */
cgrad_tensor_registry global_tensor_registry = {NULL, NULL};

/* --- Internal bucket management functions --- */

/* Create a new bucket with the given root tensor (by value). Returns pointer or NULL on failure. */
static cgrad_tensor_registry_bucket* create_new_bucket(const cgrad_tensor* root) {
    cgrad_tensor_registry_bucket* bucket = (cgrad_tensor_registry_bucket*)malloc(sizeof(cgrad_tensor_registry_bucket));
    if (!bucket) return NULL;
    bucket->root = *root;
    bucket->tensor_map = NULL;
    // Add to global bucket_map
    HASH_ADD_KEYPTR(hh, global_tensor_registry.bucket_map, bucket->root.uuid, sizeof(uuid_t), bucket);
    return bucket;
}

/* Add a tensor to a bucket's tensor_map. Returns 0 on success, nonzero on failure. */
static int add_to_bucket(cgrad_tensor_registry_bucket* bucket, cgrad_tensor* t) {
    cgrad_tensor_registry_tensor_entry* entry = (cgrad_tensor_registry_tensor_entry*)malloc(sizeof(cgrad_tensor_registry_tensor_entry));
    if (!entry) return -1;
    memcpy(entry->uuid, t->uuid, sizeof(uuid_t));
    entry->tensor = t;
    HASH_ADD_KEYPTR(hh, bucket->tensor_map, entry->uuid, sizeof(uuid_t), entry);
    return 0;
}

/* Remove a tensor from a bucket's tensor_map. Returns 0 on success, -1 if not found. */
static int remove_from_bucket(cgrad_tensor_registry_bucket* bucket, const cgrad_tensor* t) {
    cgrad_tensor_registry_tensor_entry* entry = NULL;
    HASH_FIND(hh, bucket->tensor_map, t->uuid, sizeof(uuid_t), entry);
    if (!entry) return -1;
    HASH_DEL(bucket->tensor_map, entry);
    free(entry);
    return 0;
}

/* Delete a bucket and free its memory. */
static void delete_bucket(cgrad_tensor_registry_bucket* bucket) {
    // Remove from global bucket_map
    HASH_DEL(global_tensor_registry.bucket_map, bucket);
    cgrad_tensor_registry_tensor_entry *entry, *tmp;
    HASH_ITER(hh, bucket->tensor_map, entry, tmp) {
        HASH_DEL(bucket->tensor_map, entry);
        free(entry);
    }
    free(bucket);
}

/* --- Public API --- */

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

    // Check if tensor is already registered
    HASH_FIND(hh, global_tensor_registry.tensor_map, t->uuid, sizeof(uuid_t), reg_entry);
    if (reg_entry) {
        // Already registered, do nothing
        return CGRAD_SUCCESS;
    }

    if (parent == NULL) {
        // Create new bucket and add t to it
        bucket = create_new_bucket(t);
        if (!bucket) return CGRAD_TENSOR_F32_CPU_ERR_ALLOC_FAILED;
        if (add_to_bucket(bucket, t) != 0) {
            delete_bucket(bucket);
            return CGRAD_TENSOR_F32_CPU_ERR_ALLOC_FAILED;
        }
    } else {
        // Find parent's bucket
        cgrad_tensor_registry_entry* parent_entry = NULL;
        HASH_FIND(hh, global_tensor_registry.tensor_map, parent->uuid, sizeof(uuid_t), parent_entry);
        if (!parent_entry) {
            return CGRAD_TENSOR_ERR_PARENT_NOT_REGISTERED;
        }
        bucket = parent_entry->bucket;
        if (add_to_bucket(bucket, t) != 0) {
            return CGRAD_TENSOR_F32_CPU_ERR_ALLOC_FAILED;
        }
    }

    // Add t to registry (maps t to bucket)
    reg_entry = (cgrad_tensor_registry_entry*)malloc(sizeof(cgrad_tensor_registry_entry));
    if (!reg_entry) {
        // Clean up: remove t from bucket if just added
        if (bucket) {
            cgrad_tensor_registry_tensor_entry* entry = NULL;
            HASH_FIND(hh, bucket->tensor_map, t->uuid, sizeof(uuid_t), entry);
            if (entry) {
                HASH_DEL(bucket->tensor_map, entry);
                free(entry);
            }
            // If this was a new bucket, delete it
            if (parent == NULL) {
                delete_bucket(bucket);
            }
        }
        return CGRAD_TENSOR_F32_CPU_ERR_ALLOC_FAILED;
    }
    memcpy(reg_entry->uuid, t->uuid, sizeof(uuid_t));
    reg_entry->tensor = t;
    reg_entry->bucket = bucket;
    HASH_ADD_KEYPTR(hh, global_tensor_registry.tensor_map, reg_entry->uuid, sizeof(uuid_t), reg_entry);

    return CGRAD_SUCCESS;
}

/**
 * @brief Deregister a tensor from the global tensor registry.
 * @param t Pointer to tensor to deregister.
 * @return CGRAD_SUCCESS on success, CGRAD_TENSOR_ERR_PARENT_NOT_REGISTERED if tensor is not registered.
 */
int cgrad_tensor_registry_deregister(cgrad_tensor* t) {
    if (!t) return CGRAD_TENSOR_ERR_NULL_POINTER;

    cgrad_tensor_registry_entry* reg_entry = NULL;
    HASH_FIND(hh, global_tensor_registry.tensor_map, t->uuid, sizeof(uuid_t), reg_entry);
    if (!reg_entry) return CGRAD_TENSOR_ERR_PARENT_NOT_REGISTERED;

    cgrad_tensor_registry_bucket* bucket = reg_entry->bucket;

    // Remove t from bucket's tensor_map
    remove_from_bucket(bucket, t);

    // Remove t from registry
    HASH_DEL(global_tensor_registry.tensor_map, reg_entry);
    free(reg_entry);

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
 * @brief Deregister all tensors in the bucket containing the given tensor and delete the bucket.
 *        Only succeeds if the bucket is empty.
 * @param t Pointer to any tensor in the bucket.
 * @return CGRAD_SUCCESS on success, CGRAD_TENSOR_ERR_PARENT_NOT_REGISTERED if tensor is not registered,
 *         CGRAD_TENSOR_ERR_BUCKET_NOT_EMPTY if the bucket is not empty.
 */
int cgrad_tensor_registry_deregister_and_delete_bucket(const cgrad_tensor* t) {
    if (!t) return CGRAD_TENSOR_ERR_NULL_POINTER;

    cgrad_tensor_registry_entry* reg_entry = NULL;
    HASH_FIND(hh, global_tensor_registry.tensor_map, t->uuid, sizeof(uuid_t), reg_entry);
    if (!reg_entry || !reg_entry->bucket) return CGRAD_TENSOR_ERR_PARENT_NOT_REGISTERED;

    cgrad_tensor_registry_bucket* bucket = reg_entry->bucket;

    // Remove t from bucket's tensor_map
    remove_from_bucket(bucket, t);

    // Remove t from registry
    HASH_DEL(global_tensor_registry.tensor_map, reg_entry);
    free(reg_entry);

    // If the bucket is not empty, return error
    if (HASH_COUNT(bucket->tensor_map) > 0) {
        return CGRAD_TENSOR_ERR_BUCKET_NOT_EMPTY;
    }

    // Remove all registry entries that point to this bucket
    cgrad_tensor_registry_entry *entry, *tmp;
    HASH_ITER(hh, global_tensor_registry.tensor_map, entry, tmp) {
        if (entry->bucket == bucket) {
            return CGRAD_TENSOR_ERR_BUCKET_NOT_EMPTY;
        }
    }

    // Delete the bucket
    delete_bucket(bucket);

    return CGRAD_SUCCESS;
}

/**
 * @brief Get the root tensor of the bucket containing the given tensor.
 *        Writes the root tensor to *root_out.
 * @param t Pointer to any tensor in the bucket.
 * @param root_out Output pointer to receive the root tensor (by value).
 * @return CGRAD_SUCCESS on success, CGRAD_TENSOR_ERR_PARENT_NOT_REGISTERED if tensor is not registered.
 */
int cgrad_tensor_registry_get_root(const cgrad_tensor* t, cgrad_tensor* root_out) {
    if (!t || !root_out) return CGRAD_TENSOR_ERR_NULL_POINTER;
    cgrad_tensor_registry_entry* reg_entry = NULL;
    HASH_FIND(hh, global_tensor_registry.tensor_map, t->uuid, sizeof(uuid_t), reg_entry);
    if (!reg_entry || !reg_entry->bucket) return CGRAD_TENSOR_ERR_PARENT_NOT_REGISTERED;
    *root_out = reg_entry->bucket->root;
    return CGRAD_SUCCESS;
}

/**
 * @brief Get the number of tensors in the bucket containing the given tensor.
 *        Returns 0 if the tensor is not registered.
 */
size_t cgrad_tensor_registry_get_bucket_size(const cgrad_tensor* t) {
    if (!t) return 0;
    cgrad_tensor_registry_entry* reg_entry = NULL;
    HASH_FIND(hh, global_tensor_registry.tensor_map, t->uuid, sizeof(uuid_t), reg_entry);
    if (!reg_entry || !reg_entry->bucket) return 0;
    return (size_t)HASH_COUNT(reg_entry->bucket->tensor_map);
}

/**
 * @brief Print the contents of the tensor registry to stdout.
 *        Each bucket is printed with its root tensor's uuid and shape, and all members indented below.
 */
void cgrad_tensor_registry_print(void) {
    cgrad_tensor_registry_bucket *bucket, *tmp;
    char uuid_str[37];
    HASH_ITER(hh, global_tensor_registry.bucket_map, bucket, tmp) {
        uuid_unparse(bucket->root.uuid, uuid_str);
        printf("Bucket root: %s  ", uuid_str);
        printf("Shape: ");
        cgrad_tensor_layout* layout = bucket->root.backend->tensor_get_layout(bucket->root.data);
        cgrad_tensor_layout_print_shape(layout, TENSOR_DIM);
        printf("  [bucket size: %zu]\n", (size_t)HASH_COUNT(bucket->tensor_map));

        // Print all members (excluding root)
        cgrad_tensor_registry_tensor_entry* tentry, *ttmp;
        HASH_ITER(hh, bucket->tensor_map, tentry, ttmp) {
            uuid_unparse(tentry->uuid, uuid_str);
            printf("  - %s  Shape: ", uuid_str);
            if (tentry->tensor) {
                layout = tentry->tensor->backend->tensor_get_layout(tentry->tensor->data);
                cgrad_tensor_layout_print_shape(layout, TENSOR_DIM);
            } else {
                printf("(null)\n");
            }
        }
    }
}
