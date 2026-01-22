#ifndef CGRAD_STORAGE_REGISTRY_H
#define CGRAD_STORAGE_REGISTRY_H

#include "storage/cgrad_storage.h"
#include "third_party/uthash.h"

/**
 * @brief Generic storage node for hashmaps (used in buckets and records).
 */
typedef struct cgrad_storage_registry_node {
    uuid_t uuid; /**< UUID of the storage (key). */
    cgrad_storage* storage; /**< Pointer to the storage (value). */
    UT_hash_handle hh;    /**< uthash handle for hashmap. */
} cgrad_storage_registry_node;

/**
 * @brief Bucket structure for storage registry.
 *        Tracks storage sharing the same memory pool.
 */
typedef struct cgrad_storage_registry_bucket {
    cgrad_storage root; /**< Root storage of the bucket (first storage registered, stored by value). */
    cgrad_storage_registry_node* storage_map; /**< Hashmap of storage in this bucket. */
    UT_hash_handle hh; /**< uthash handle for bucket_map. */
} cgrad_storage_registry_bucket;

/**
 * @brief Registry entry mapping a storage to its bucket.
 */
typedef struct cgrad_storage_registry_entry {
    uuid_t uuid; /**< UUID of the storage (key). */
    cgrad_storage* storage; /**< Pointer to the storage (value). */
    cgrad_storage_registry_bucket* bucket; /**< Pointer to the bucket. */
    UT_hash_handle hh; /**< uthash handle for hashmap. */
} cgrad_storage_registry_entry;

/**
 * @brief Storage record for recording registrations within a scope.
 *        Allows nested recording of storage allocations.
 */
typedef struct cgrad_storage_registry_record {
    uuid_t record_id;             /**< Unique ID for this record. */
    cgrad_storage_registry_node* storage_map; /**< Hashmap mapping UUIDs to storages in this record. */
    UT_hash_handle hh;            /**< uthash handle for active_records. */
} cgrad_storage_registry_record;

/**
 * @brief Global storage registry structure.
 *        Maintains a hashmap of all registered storage to their buckets,
 *        and a hashmap of all buckets keyed by root uuid.
 */
typedef struct cgrad_storage_registry {
    cgrad_storage_registry_entry* storage_map;    /**< Hashmap of storage to buckets. */
    cgrad_storage_registry_bucket* bucket_map;    /**< Hashmap of all buckets, keyed by root uuid. */
    cgrad_storage_registry_record* active_records; /**< Hashmap of currently active records. */
} cgrad_storage_registry;

/**
 * @brief Initialize a storage registry.
 * @param registry Pointer to registry to initialize.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
cgrad_status cgrad_storage_registry_init(cgrad_storage_registry* registry);

/**
 * @brief Free a storage registry and all its resources.
 * @param registry Pointer to registry to free.
 */
void cgrad_storage_registry_free(cgrad_storage_registry* registry);

/**
 * @brief Register a tensor in the tensor registry.
 *        If parent is NULL, creates a new bucket with t as root.
 *        If parent is not NULL, adds t to the parent's bucket (if parent is registered).
 * @param registry Pointer to the registry.
 * @param t Pointer to tensor to register.
 * @param parent Pointer to parent tensor (or NULL).
 * @return CGRAD_SUCCESS on success, CGRAD_TENSOR_ERR_PARENT_NOT_REGISTERED if parent is not in registry.
 */
cgrad_status cgrad_storage_registry_register(cgrad_storage_registry* registry, cgrad_storage* t, const cgrad_storage* parent);

/**
 * @brief Deregister a tensor from the global tensor registry.
 *        Removes the tensor from its bucket and the registry.
 *        Writes the root tensor to *root if the bucket is empty after deregistering, NULL otherwise.
 * @param t Pointer to tensor to deregister.
 * @param root Output pointer to receive the root tensor if bucket is empty, NULL otherwise.
 * @return CGRAD_SUCCESS on success, CGRAD_TENSOR_ERR_PARENT_NOT_REGISTERED if tensor is not registered.
 */
/**
 * @brief Deregister a tensor from the tensor registry.
 * @param registry Pointer to the registry.
 * @param t Pointer to tensor to deregister.
 * @return CGRAD_SUCCESS on success, CGRAD_TENSOR_ERR_PARENT_NOT_REGISTERED if tensor is not registered.
 */
cgrad_status cgrad_storage_registry_deregister(cgrad_storage_registry* registry, cgrad_storage* t);

/**
 * @brief Get the number of tensors currently registered in the tensor registry.
 * @param registry Pointer to the registry.
 * @return Number of registered tensors.
 */
size_t cgrad_storage_registry_count(cgrad_storage_registry* registry);

/**
 * @brief Get the number of tensors in the bucket containing the given tensor.
 *        Returns 0 if the tensor is not registered.
 * @param registry Pointer to the registry.
 * @param t Pointer to any tensor in the bucket.
 * @return Number of tensors in the bucket, or 0 if not registered.
 */
size_t cgrad_storage_registry_bucket_get_size(cgrad_storage_registry* registry, const cgrad_storage* t);

/**
 * @brief Deregister all tensors in the bucket containing the given tensor and delete the bucket.
 *        Only succeeds if the bucket is empty.
 * @param registry Pointer to the registry.
 * @param t Pointer to any tensor in the bucket.
 * @return CGRAD_SUCCESS on success, CGRAD_TENSOR_ERR_PARENT_NOT_REGISTERED if tensor is not registered,
 *         CGRAD_TENSOR_ERR_BUCKET_NOT_EMPTY if the bucket is not empty.
 */
cgrad_status cgrad_storage_registry_deregister_and_delete_bucket(cgrad_storage_registry* registry, const cgrad_storage* t);

/**
 * @brief Get the root tensor of the bucket containing the given tensor.
 *        Writes the root tensor to *root_out.
 * @param registry Pointer to the registry.
 * @param t Pointer to any tensor in the bucket.
 * @param root_out Output pointer to receive the root tensor (by value).
 * @return CGRAD_SUCCESS on success, CGRAD_TENSOR_ERR_PARENT_NOT_REGISTERED if tensor is not registered.
 */
cgrad_status cgrad_storage_registry_get_root(cgrad_storage_registry* registry, const cgrad_storage* t, cgrad_storage* root_out);

/**
 * @brief Print the contents of the tensor registry to stdout.
 *        Each bucket is printed with its root tensor's uuid and shape, and all members indented below.
 * @param registry Pointer to the registry.
 */
void cgrad_storage_registry_print(cgrad_storage_registry* registry);

/**
 * @brief Start recording storage registrations.
 *        All storages registered after this call will be recorded in the returned record.
 *        Supports nesting - multiple records can be active simultaneously.
 * 
 * @param registry Pointer to the registry.
 * @return Pointer to a new record, or NULL on allocation failure.
 */
cgrad_storage_registry_record* cgrad_storage_registry_start_recording(cgrad_storage_registry* registry);

/**
 * @brief Stop recording and deactivate the record.
 *        The record remains valid and contains all recorded storage UUIDs.
 * 
 * @param registry Pointer to the registry.
 * @param record Pointer to the record to stop.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
cgrad_status cgrad_storage_registry_stop_recording(cgrad_storage_registry* registry, cgrad_storage_registry_record* record);

/**
 * @brief Free a storage record and its resources.
 *        The record must have been stopped before calling this.
 * 
 * @param record Pointer to the record to free.
 */
void cgrad_storage_registry_record_free(cgrad_storage_registry_record* record);

/**
 * @brief Get the number of storages recorded by a record.
 * 
 * @param record Pointer to the record.
 * @return Number of recorded storages.
 */
size_t cgrad_storage_registry_record_count(const cgrad_storage_registry_record* record);

/**
 * @brief Remove a storage from a record.
 *        This allows manual removal of specific storages from recording.
 * 
 * @param record Pointer to the record.
 * @param t Pointer to the storage to remove.
 */
void cgrad_storage_registry_record_remove(cgrad_storage_registry_record* record, const cgrad_storage* t);

// ============================================================================
// Global Storage Registry Management
// ============================================================================

/**
 * @brief Initialize the global storage registry.
 * 
 * This function allocates and initializes the global storage registry.
 * It should be called once during library initialization.
 * 
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
cgrad_status cgrad_storage_init_global_registry(void);

/**
 * @brief Cleanup the global storage registry.
 * 
 * This function frees all resources associated with the global storage registry.
 * It should be called once during library cleanup.
 * Returns an error if there are still tensors registered in the registry.
 * 
 * @return CGRAD_SUCCESS on success, CGRAD_ERR_STORAGE_REGISTRY_NOT_EMPTY if tensors are still registered.
 */
cgrad_status cgrad_storage_free_global_registry(void);

#endif // CGRAD_STORAGE_REGISTRY_H
