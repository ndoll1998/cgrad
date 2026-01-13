#ifndef CGRAD_STORAGE_REGISTRY_H
#define CGRAD_STORAGE_REGISTRY_H

#include "cgrad_storage.h"
#include "uthash.h"

/**
 * @brief Wrapper for tracking storage in a bucket (hashable by pointer).
 */
typedef struct cgrad_storage_registry_entry_tensor {
    uuid_t uuid; /**< UUID of the storage (key). */
    cgrad_storage* storage; /**< Pointer to the storage (value). */
    UT_hash_handle hh;    /**< uthash handle for hashmap. */
} cgrad_storage_registry_entry_tensor;

/**
 * @brief Bucket structure for storage registry.
 *        Tracks storage sharing the same memory pool.
 */
typedef struct cgrad_storage_registry_bucket {
    cgrad_storage root; /**< Root storage of the bucket (first storage registered, stored by value). */
    cgrad_storage_registry_entry_tensor* storage_map; /**< Hashmap of storage in this bucket. */
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
 * @brief Global storage registry structure.
 *        Maintains a hashmap of all registered storage to their buckets,
 *        and a hashmap of all buckets keyed by root uuid.
 */
typedef struct cgrad_storage_registry {
    cgrad_storage_registry_entry* storage_map; /**< Hashmap of storage to buckets. */
    cgrad_storage_registry_bucket* bucket_map; /**< Hashmap of all buckets, keyed by root uuid. */
} cgrad_storage_registry;

/**
 * @brief Global storage registry instance.
 */
extern struct cgrad_storage_registry global_storage_registry;

/**
 * @brief Register a tensor in the global tensor registry.
 *        If parent is NULL, creates a new bucket with t as root.
 *        If parent is not NULL, adds t to the parent's bucket (if parent is registered).
 * @param t Pointer to tensor to register.
 * @param parent Pointer to parent tensor (or NULL).
 * @return CGRAD_SUCCESS on success, CGRAD_TENSOR_ERR_PARENT_NOT_REGISTERED if parent is not in registry.
 */
int cgrad_storage_registry_register(cgrad_storage* t, const cgrad_storage* parent);

/**
 * @brief Deregister a tensor from the global tensor registry.
 *        Removes the tensor from its bucket and the registry.
 *        Writes the root tensor to *root if the bucket is empty after deregistering, NULL otherwise.
 * @param t Pointer to tensor to deregister.
 * @param root Output pointer to receive the root tensor if bucket is empty, NULL otherwise.
 * @return CGRAD_SUCCESS on success, CGRAD_TENSOR_ERR_PARENT_NOT_REGISTERED if tensor is not registered.
 */
/**
 * @brief Deregister a tensor from the global tensor registry.
 * @param t Pointer to tensor to deregister.
 * @return CGRAD_SUCCESS on success, CGRAD_TENSOR_ERR_PARENT_NOT_REGISTERED if tensor is not registered.
 */
int cgrad_storage_registry_deregister(cgrad_storage* t);

/**
 * @brief Get the number of tensors currently registered in the global tensor registry.
 * @return Number of registered tensors.
 */
size_t cgrad_storage_registry_count(void);

/**
 * @brief Get the number of tensors in the bucket containing the given tensor.
 *        Returns 0 if the tensor is not registered.
 * @param t Pointer to any tensor in the bucket.
 * @return Number of tensors in the bucket, or 0 if not registered.
 */
size_t cgrad_storage_registry_get_bucket_size(const cgrad_storage* t);

/**
 * @brief Deregister all tensors in the bucket containing the given tensor and delete the bucket.
 *        Only succeeds if the bucket is empty.
 * @param t Pointer to any tensor in the bucket.
 * @return CGRAD_SUCCESS on success, CGRAD_TENSOR_ERR_PARENT_NOT_REGISTERED if tensor is not registered,
 *         CGRAD_TENSOR_ERR_BUCKET_NOT_EMPTY if the bucket is not empty.
 */
int cgrad_storage_registry_deregister_and_delete_bucket(const cgrad_storage* t);

/**
 * @brief Get the root tensor of the bucket containing the given tensor.
 *        Writes the root tensor to *root_out.
 * @param t Pointer to any tensor in the bucket.
 * @param root_out Output pointer to receive the root tensor (by value).
 * @return CGRAD_SUCCESS on success, CGRAD_TENSOR_ERR_PARENT_NOT_REGISTERED if tensor is not registered.
 */
int cgrad_storage_registry_get_root(const cgrad_storage* t, cgrad_storage* root_out);

/**
 * @brief Print the contents of the tensor registry to stdout.
 *        Each bucket is printed with its root tensor's uuid and shape, and all members indented below.
 */
void cgrad_storage_registry_print(void);


#endif // CGRAD_STORAGE_REGISTRY_H
