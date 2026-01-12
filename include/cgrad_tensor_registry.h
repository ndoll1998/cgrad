#ifndef CGRAD_TENSOR_REGISTRY_H
#define CGRAD_TENSOR_REGISTRY_H

#include "cgrad_tensor.h"
#include "uthash.h"

/**
 * @brief Wrapper for tracking tensors in a bucket (hashable by pointer).
 */
typedef struct cgrad_tensor_registry_tensor_entry {
    uuid_t uuid; /**< UUID of the tensor (key). */
    cgrad_tensor* tensor; /**< Pointer to the tensor (value). */
    UT_hash_handle hh;    /**< uthash handle for hashmap. */
} cgrad_tensor_registry_tensor_entry;

/**
 * @brief Bucket structure for tensor registry.
 *        Tracks tensors sharing the same memory pool.
 */
typedef struct cgrad_tensor_registry_bucket {
    cgrad_tensor root; /**< Root tensor of the bucket (first tensor registered, stored by value). */
    cgrad_tensor_registry_tensor_entry* tensor_map; /**< Hashmap of tensors in this bucket. */
    UT_hash_handle hh; /**< uthash handle for bucket_map. */
} cgrad_tensor_registry_bucket;

/**
 * @brief Registry entry mapping a tensor to its bucket.
 */
typedef struct cgrad_tensor_registry_entry {
    uuid_t uuid; /**< UUID of the tensor (key). */
    cgrad_tensor* tensor; /**< Pointer to the tensor (value). */
    cgrad_tensor_registry_bucket* bucket; /**< Pointer to the bucket. */
    UT_hash_handle hh; /**< uthash handle for hashmap. */
} cgrad_tensor_registry_entry;

/**
 * @brief Global tensor registry structure.
 *        Maintains a hashmap of all registered tensors to their buckets,
 *        and a hashmap of all buckets keyed by root uuid.
 */
typedef struct cgrad_tensor_registry {
    cgrad_tensor_registry_entry* tensor_map; /**< Hashmap of tensors to buckets. */
    cgrad_tensor_registry_bucket* bucket_map; /**< Hashmap of all buckets, keyed by root uuid. */
} cgrad_tensor_registry;

/**
 * @brief Global tensor registry instance.
 */
extern struct cgrad_tensor_registry global_tensor_registry;

/**
 * @brief Register a tensor in the global tensor registry.
 *        If parent is NULL, creates a new bucket with t as root.
 *        If parent is not NULL, adds t to the parent's bucket (if parent is registered).
 * @param t Pointer to tensor to register.
 * @param parent Pointer to parent tensor (or NULL).
 * @return CGRAD_SUCCESS on success, CGRAD_TENSOR_ERR_PARENT_NOT_REGISTERED if parent is not in registry.
 */
int cgrad_tensor_registry_register(cgrad_tensor* t, const cgrad_tensor* parent);

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
int cgrad_tensor_registry_deregister(cgrad_tensor* t);

/**
 * @brief Get the number of tensors currently registered in the global tensor registry.
 * @return Number of registered tensors.
 */
size_t cgrad_tensor_registry_count(void);

/**
 * @brief Get the number of tensors in the bucket containing the given tensor.
 *        Returns 0 if the tensor is not registered.
 * @param t Pointer to any tensor in the bucket.
 * @return Number of tensors in the bucket, or 0 if not registered.
 */
size_t cgrad_tensor_registry_get_bucket_size(const cgrad_tensor* t);

/**
 * @brief Deregister all tensors in the bucket containing the given tensor and delete the bucket.
 *        Only succeeds if the bucket is empty.
 * @param t Pointer to any tensor in the bucket.
 * @return CGRAD_SUCCESS on success, CGRAD_TENSOR_ERR_PARENT_NOT_REGISTERED if tensor is not registered,
 *         CGRAD_TENSOR_ERR_BUCKET_NOT_EMPTY if the bucket is not empty.
 */
int cgrad_tensor_registry_deregister_and_delete_bucket(const cgrad_tensor* t);

/**
 * @brief Get the root tensor of the bucket containing the given tensor.
 *        Writes the root tensor to *root_out.
 * @param t Pointer to any tensor in the bucket.
 * @param root_out Output pointer to receive the root tensor (by value).
 * @return CGRAD_SUCCESS on success, CGRAD_TENSOR_ERR_PARENT_NOT_REGISTERED if tensor is not registered.
 */
int cgrad_tensor_registry_get_root(const cgrad_tensor* t, cgrad_tensor* root_out);

/**
 * @brief Print the contents of the tensor registry to stdout.
 *        Each bucket is printed with its root tensor's uuid and shape, and all members indented below.
 */
void cgrad_tensor_registry_print(void);


#endif // CGRAD_TENSOR_REGISTRY_H
