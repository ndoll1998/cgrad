#include "storage/cgrad_storage_registry.h"
#include "cgrad_errors.h"
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <uuid/uuid.h>

/* --- Internal bucket management functions --- */

/* Create a new bucket with the given root tensor (by value). Returns pointer or NULL on failure. */
static cgrad_storage_registry_bucket* create_new_bucket(cgrad_storage_registry* registry, const cgrad_storage* root) {
    cgrad_storage_registry_bucket* bucket = (cgrad_storage_registry_bucket*)malloc(sizeof(cgrad_storage_registry_bucket));
    if (!bucket) return NULL;
    bucket->root = *root;
    bucket->storage_map = NULL;
    // Add to bucket_map
    HASH_ADD_KEYPTR(hh, registry->bucket_map, bucket->root.uuid, sizeof(uuid_t), bucket);
    return bucket;
}

/* Add a tensor to a bucket's tensor_map. Returns 0 on success, nonzero on failure. */
static int add_to_bucket(cgrad_storage_registry_bucket* bucket, cgrad_storage* t) {
    cgrad_storage_registry_node* entry = (cgrad_storage_registry_node*)malloc(sizeof(cgrad_storage_registry_node));
    if (!entry) return -1;
    memcpy(entry->uuid, t->uuid, sizeof(uuid_t));
    entry->storage = t;
    HASH_ADD_KEYPTR(hh, bucket->storage_map, entry->uuid, sizeof(uuid_t), entry);
    return 0;
}

/* Remove a tensor from a bucket's tensor_map. Returns 0 on success, -1 if not found. */
static int remove_from_bucket(cgrad_storage_registry_bucket* bucket, const cgrad_storage* t) {
    cgrad_storage_registry_node* entry = NULL;
    HASH_FIND(hh, bucket->storage_map, t->uuid, sizeof(uuid_t), entry);
    if (!entry) return -1;
    HASH_DEL(bucket->storage_map, entry);
    free(entry);
    return 0;
}

/* Delete a bucket and free its memory. */
static void delete_bucket(cgrad_storage_registry* registry, cgrad_storage_registry_bucket* bucket) {
    // Remove from bucket_map
    HASH_DEL(registry->bucket_map, bucket);
    cgrad_storage_registry_node *entry, *tmp;
    HASH_ITER(hh, bucket->storage_map, entry, tmp) {
        HASH_DEL(bucket->storage_map, entry);
        free(entry);
    }
    free(bucket);
}


/* --- Public API --- */

/**
 * @brief Initialize a storage registry.
 */
int cgrad_storage_registry_init(cgrad_storage_registry* registry) {
    if (!registry) return CGRAD_ERR_NULL_POINTER;
    registry->storage_map = NULL;
    registry->bucket_map = NULL;
    registry->active_records = NULL;
    return CGRAD_SUCCESS;
}

/**
 * @brief Free a storage registry and all its resources.
 */
void cgrad_storage_registry_free(cgrad_storage_registry* registry) {
    if (!registry) return;
    
    // Free all buckets
    cgrad_storage_registry_bucket *bucket, *tmp_bucket;
    HASH_ITER(hh, registry->bucket_map, bucket, tmp_bucket) {
        HASH_DEL(registry->bucket_map, bucket);
        
        // Free all entries in the bucket
        cgrad_storage_registry_node *entry, *tmp_entry;
        HASH_ITER(hh, bucket->storage_map, entry, tmp_entry) {
            HASH_DEL(bucket->storage_map, entry);
            free(entry);
        }
        free(bucket);
    }
    
    // Free all registry entries
    cgrad_storage_registry_entry *reg_entry, *tmp_reg;
    HASH_ITER(hh, registry->storage_map, reg_entry, tmp_reg) {
        HASH_DEL(registry->storage_map, reg_entry);
        free(reg_entry);
    }
    
    // Free all active records
    cgrad_storage_registry_record *record, *tmp_record;
    HASH_ITER(hh, registry->active_records, record, tmp_record) {
        HASH_DEL(registry->active_records, record);
        free(record);
    }
    
    registry->storage_map = NULL;
    registry->bucket_map = NULL;
    registry->active_records = NULL;
}

/**
 * @brief Register a tensor in the tensor registry.
 *        If parent is NULL, creates a new bucket with t as root.
 *        If parent is not NULL, adds t to the parent's bucket (if parent is registered).
 * @param registry Pointer to the registry.
 * @param t Pointer to tensor to register.
 * @param parent Pointer to parent tensor (or NULL).
 * @return CGRAD_SUCCESS on success, CGRAD_STORAGE_REGISTRY_PARENT_NOT_REGISTERED if parent is not in registry.
 */
int cgrad_storage_registry_register(cgrad_storage_registry* registry, cgrad_storage* t, const cgrad_storage* parent) {
    if (!registry || !t) return CGRAD_ERR_NULL_POINTER;

    cgrad_storage_registry_entry* reg_entry = NULL;
    cgrad_storage_registry_bucket* bucket = NULL;

    // Check if tensor is already registered
    HASH_FIND(hh, registry->storage_map, t->uuid, sizeof(uuid_t), reg_entry);
    if (reg_entry) {
        // Already registered, do nothing
        return CGRAD_SUCCESS;
    }

    if (parent == NULL) {
        // Create new bucket and add t to it
        bucket = create_new_bucket(registry, t);
        if (!bucket) return CGRAD_STORAGE_REGISTRY_ALLOC_FAILED;
        if (add_to_bucket(bucket, t) != 0) {
            delete_bucket(registry, bucket);
            return CGRAD_STORAGE_REGISTRY_ALLOC_FAILED;
        }
    } else {
        // Find parent's bucket
        cgrad_storage_registry_entry* parent_entry = NULL;
        HASH_FIND(hh, registry->storage_map, parent->uuid, sizeof(uuid_t), parent_entry);
        if (!parent_entry) {
            return CGRAD_STORAGE_REGISTRY_PARENT_NOT_REGISTERED;
        }
        bucket = parent_entry->bucket;
        if (add_to_bucket(bucket, t) != 0) {
            return CGRAD_STORAGE_REGISTRY_ALLOC_FAILED;
        }
    }

    // Add t to registry (maps t to bucket)
    reg_entry = (cgrad_storage_registry_entry*)malloc(sizeof(cgrad_storage_registry_entry));
    if (!reg_entry) {
        // Clean up: remove t from bucket if just added
        if (bucket) {
            cgrad_storage_registry_node* entry = NULL;
            HASH_FIND(hh, bucket->storage_map, t->uuid, sizeof(uuid_t), entry);
            if (entry) {
                HASH_DEL(bucket->storage_map, entry);
                free(entry);
            }
            // If this was a new bucket, delete it
            if (parent == NULL) {
                delete_bucket(registry, bucket);
            }
        }
        return CGRAD_STORAGE_REGISTRY_ALLOC_FAILED;
    }
    memcpy(reg_entry->uuid, t->uuid, sizeof(uuid_t));
    reg_entry->storage = t;
    reg_entry->bucket = bucket;
    HASH_ADD_KEYPTR(hh, registry->storage_map, reg_entry->uuid, sizeof(uuid_t), reg_entry);

    // Notify all active records
    cgrad_storage_registry_record *record, *tmp_record;
    HASH_ITER(hh, registry->active_records, record, tmp_record) {
        // Add storage to record's storage_map hashmap
        cgrad_storage_registry_node* entry = (cgrad_storage_registry_node*)malloc(sizeof(cgrad_storage_registry_node));
        if (entry) {
            uuid_copy(entry->uuid, t->uuid);
            entry->storage = t;
            HASH_ADD_KEYPTR(hh, record->storage_map, entry->uuid, sizeof(uuid_t), entry);
        }
        // Note: If malloc fails, we silently skip adding to this record
    }

    return CGRAD_SUCCESS;
}

/**
 * @brief Deregister a tensor from the tensor registry.
 * @param registry Pointer to the registry.
 * @param t Pointer to tensor to deregister.
 * @return CGRAD_SUCCESS on success, CGRAD_STORAGE_REGISTRY_PARENT_NOT_REGISTERED if tensor is not registered.
 */
int cgrad_storage_registry_deregister(cgrad_storage_registry* registry, cgrad_storage* t) {
    if (!registry || !t) return CGRAD_ERR_NULL_POINTER;

    cgrad_storage_registry_entry* reg_entry = NULL;
    HASH_FIND(hh, registry->storage_map, t->uuid, sizeof(uuid_t), reg_entry);
    if (!reg_entry) return CGRAD_STORAGE_REGISTRY_PARENT_NOT_REGISTERED;

    cgrad_storage_registry_bucket* bucket = reg_entry->bucket;

    // Remove t from bucket's tensor_map
    remove_from_bucket(bucket, t);

    // Remove t from all active records
    cgrad_storage_registry_record *record, *tmp_record;
    HASH_ITER(hh, registry->active_records, record, tmp_record) {
        cgrad_storage_registry_record_remove(record, t);
    }

    // Remove t from registry
    HASH_DEL(registry->storage_map, reg_entry);
    free(reg_entry);

    return CGRAD_SUCCESS;
}

/**
 * @brief Get the number of tensors currently registered in the tensor registry.
 * @param registry Pointer to the registry.
 * @return Number of registered tensors.
 */
size_t cgrad_storage_registry_count(cgrad_storage_registry* registry) {
    if (!registry) return 0;
    return (size_t)HASH_COUNT(registry->storage_map);
}

/**
 * @brief Deregister all tensors in the bucket containing the given tensor and delete the bucket.
 *        Only succeeds if the bucket is empty.
 * @param registry Pointer to the registry.
 * @param t Pointer to any tensor in the bucket.
 * @return CGRAD_SUCCESS on success, CGRAD_STORAGE_REGISTRY_PARENT_NOT_REGISTERED if tensor is not registered,
 *         CGRAD_STORAGE_REGISTRY_BUCKET_NOT_EMPTY if the bucket is not empty.
 */
int cgrad_storage_registry_deregister_and_delete_bucket(cgrad_storage_registry* registry, const cgrad_storage* t) {
    if (!registry || !t) return CGRAD_ERR_NULL_POINTER;

    cgrad_storage_registry_entry* reg_entry = NULL;
    HASH_FIND(hh, registry->storage_map, t->uuid, sizeof(uuid_t), reg_entry);
    if (!reg_entry || !reg_entry->bucket) return CGRAD_STORAGE_REGISTRY_PARENT_NOT_REGISTERED;

    cgrad_storage_registry_bucket* bucket = reg_entry->bucket;

    // Remove t from bucket's tensor_map
    remove_from_bucket(bucket, t);

    // Remove t from registry
    HASH_DEL(registry->storage_map, reg_entry);
    free(reg_entry);

    // If the bucket is not empty, return error
    if (HASH_COUNT(bucket->storage_map) > 0) {
        return CGRAD_STORAGE_REGISTRY_BUCKET_NOT_EMPTY;
    }

    // Remove all registry entries that point to this bucket
    cgrad_storage_registry_entry *entry, *tmp;
    HASH_ITER(hh, registry->storage_map, entry, tmp) {
        if (entry->bucket == bucket) {
            return CGRAD_STORAGE_REGISTRY_BUCKET_NOT_EMPTY;
        }
    }

    // Delete the bucket
    delete_bucket(registry, bucket);

    return CGRAD_SUCCESS;
}

/**
 * @brief Get the root tensor of the bucket containing the given tensor.
 *        Writes the root tensor to *root_out.
 * @param registry Pointer to the registry.
 * @param t Pointer to any tensor in the bucket.
 * @param root_out Output pointer to receive the root tensor (by value).
 * @return CGRAD_SUCCESS on success, CGRAD_STORAGE_REGISTRY_PARENT_NOT_REGISTERED if tensor is not registered.
 */
int cgrad_storage_registry_get_root(cgrad_storage_registry* registry, const cgrad_storage* t, cgrad_storage* root_out) {
    if (!registry || !t || !root_out) return CGRAD_ERR_NULL_POINTER;
    cgrad_storage_registry_entry* reg_entry = NULL;
    HASH_FIND(hh, registry->storage_map, t->uuid, sizeof(uuid_t), reg_entry);
    if (!reg_entry || !reg_entry->bucket) return CGRAD_STORAGE_REGISTRY_PARENT_NOT_REGISTERED;
    *root_out = reg_entry->bucket->root;
    return CGRAD_SUCCESS;
}

/**
 * @brief Get the number of tensors in the bucket containing the given tensor.
 *        Returns 0 if the tensor is not registered.
 * @param registry Pointer to the registry.
 * @param t Pointer to any tensor in the bucket.
 */
size_t cgrad_storage_registry_bucket_get_size(cgrad_storage_registry* registry, const cgrad_storage* t) {
    if (!registry || !t) return 0;
    cgrad_storage_registry_entry* reg_entry = NULL;
    HASH_FIND(hh, registry->storage_map, t->uuid, sizeof(uuid_t), reg_entry);
    if (!reg_entry || !reg_entry->bucket) return 0;
    return (size_t)HASH_COUNT(reg_entry->bucket->storage_map);
}

/**
 * @brief Print the contents of the tensor registry to stdout.
 *        Each bucket is printed with its root tensor's uuid and shape, and all members indented below.
 * @param registry Pointer to the registry.
 */
void cgrad_storage_registry_print(cgrad_storage_registry* registry) {
    if (!registry) return;
    cgrad_storage_registry_bucket *bucket, *tmp;
    char uuid_str[37];
    HASH_ITER(hh, registry->bucket_map, bucket, tmp) {
        uuid_unparse(bucket->root.uuid, uuid_str);
        printf("Bucket root: %s  ", uuid_str);
        printf("Shape: ");
        cgrad_storage_layout* layout = bucket->root.backend->storage_get_layout(bucket->root.data);
        cgrad_storage_layout_print_shape(layout, TENSOR_DIM);
        printf("  [bucket size: %zu]\n", (size_t)HASH_COUNT(bucket->storage_map));

        // Print all members (excluding root)
        cgrad_storage_registry_node* tentry, *ttmp;
        HASH_ITER(hh, bucket->storage_map, tentry, ttmp) {
            uuid_unparse(tentry->uuid, uuid_str);
            printf("  - %s  Shape: ", uuid_str);
            if (tentry->storage) {
                layout = tentry->storage->backend->storage_get_layout(tentry->storage->data);
                cgrad_storage_layout_print_shape(layout, TENSOR_DIM);
            } else {
                printf("(null)\n");
            }
        }
    }
}

// ============================================================================
// Storage Tracking API
// ============================================================================

/**
 * @brief Start tracking storage registrations.
 */
cgrad_storage_registry_record* cgrad_storage_registry_start_recording(cgrad_storage_registry* registry) {
    if (!registry) return NULL;
    
    // Allocate record
    cgrad_storage_registry_record* record = (cgrad_storage_registry_record*)malloc(sizeof(cgrad_storage_registry_record));
    if (!record) return NULL;
    
    // Initialize record
    uuid_generate(record->record_id);
    record->storage_map = NULL;  // Empty hashmap
    
    // Add to active records
    HASH_ADD_KEYPTR(hh, registry->active_records, record->record_id, sizeof(uuid_t), record);
    return record;
}

/**
 * @brief Stop tracking and deactivate the tracker.
 */
int cgrad_storage_registry_stop_recording(cgrad_storage_registry* registry, cgrad_storage_registry_record* record) {
    if (!registry || !record) return CGRAD_ERR_NULL_POINTER;
    
    // Find and remove from active records
    cgrad_storage_registry_record* found = NULL;
    HASH_FIND(hh, registry->active_records, record->record_id, sizeof(uuid_t), found);
    if (!found) {
        return CGRAD_STORAGE_REGISTRY_PARENT_NOT_REGISTERED; // Record not active
    }
    
    HASH_DEL(registry->active_records, found);
    cgrad_storage_registry_record_free(found);
    
    return CGRAD_SUCCESS;
}

/**
 * @brief Free a storage tracker and its resources.
 */
void cgrad_storage_registry_record_free(cgrad_storage_registry_record* record) {
    if (!record) return;
    
    // Free all record entries in the record
    cgrad_storage_registry_node *entry, *tmp;
    HASH_ITER(hh, record->storage_map, entry, tmp) {
        HASH_DEL(record->storage_map, entry);
        free(entry);
    }
    
    free(record);
}

/**
 * @brief Get the number of storages tracked by a tracker.
 */
size_t cgrad_storage_registry_record_count(const cgrad_storage_registry_record* record) {
    if (!record) return 0;
    return (size_t)HASH_COUNT(record->storage_map);
}

/**
 * @brief Remove a storage from a tracker.
 */
void cgrad_storage_registry_record_remove(cgrad_storage_registry_record* record, const cgrad_storage* t) {
    if (!record || !t) return;
    
    cgrad_storage_registry_node* entry = NULL;
    HASH_FIND(hh, record->storage_map, t->uuid, sizeof(uuid_t), entry);
    if (entry) {
        HASH_DEL(record->storage_map, entry);
        free(entry);
    }
}
