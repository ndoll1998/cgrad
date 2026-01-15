#ifndef CGRAD_TENSOR_F32_CPU_H
#define CGRAD_TENSOR_F32_CPU_H

#include "storage/cgrad_storage_backend.h"
#include "storage/cgrad_storage_layout.h"
#include <stdint.h>
#include <stddef.h>
#include <cblas.h>

/**
 * @brief Structure representing a float32 CPU tensor.
 */
typedef struct cgrad_storage_f32_cpu {
  cgrad_storage_layout layout;
  float* data;
} cgrad_storage_f32_cpu;

/**
 * @brief Initialize a float32 CPU tensor with the given shape and ndim.
 *        The user-specified shape (length ndim) is placed at the end; leading unspecified dims are set to 1.
 *        For example, shape={3,4}, ndim=2, MAX_TENSOR_DIM=4 => layout.shape={1,1,3,4}
 * @param t Pointer to tensor to initialize.
 * @param shape Array of dimensions (length ndim).
 * @param ndim Number of dimensions in shape (≤ MAX_TENSOR_DIM).
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_storage_f32_cpu_init(cgrad_storage_f32_cpu* t, const uint32_t* shape, int ndim);

/**
 * @brief Get the value at the given indices.
 * @param t Pointer to tensor.
 * @param indices Array of indices.
 * @param ndim Number of dimensions in indices.
 * @param out_value Pointer to float where the value will be written.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_storage_f32_cpu_get(const cgrad_storage_f32_cpu* t, const uint32_t* indices, int ndim, float* out_value);

/**
 * @brief Set the value at the given indices.
 * @param t Pointer to tensor.
 * @param indices Array of indices.
 * @param ndim Number of dimensions in indices.
 * @param value Value to set.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_storage_f32_cpu_set(cgrad_storage_f32_cpu* t, const uint32_t* indices, int ndim, float value);

/**
 * @brief Fill the tensor with a constant value.
 * @param t Pointer to tensor.
 * @param value The value to fill the tensor with.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_storage_f32_cpu_fill(cgrad_storage_f32_cpu* t, float value);

/**
 * @brief Fill the tensor with random values.
 * @param t Pointer to tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_storage_f32_cpu_fill_rand(cgrad_storage_f32_cpu* t);

/**
 * @brief Copy a tensor into a pre-initialized, contiguous destination tensor.
 *
 * The destination tensor (dst) must already be initialized with the same shape as src,
 * and its layout must be contiguous (see cgrad_storage_layout_is_contiguous).
 * No memory allocation or initialization is performed inside this function.
 *
 * @param src Source tensor.
 * @param dst Destination tensor (must be pre-initialized and contiguous).
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_storage_f32_cpu_contiguous(const cgrad_storage_f32_cpu* src, cgrad_storage_f32_cpu* dst);

/**
 * @brief Create a shallow copy of a tensor handle (deep copy layout, shallow copy data).
 * @param src Source tensor.
 * @param dst Destination tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_storage_f32_cpu_shallow_copy(const cgrad_storage_f32_cpu* src, cgrad_storage_f32_cpu* dst);

/**
 * @brief Free the memory associated with a tensor.
 * @param t Pointer to tensor.
 */
void cgrad_storage_f32_cpu_free(cgrad_storage_f32_cpu* t);

/**
 * @brief Add two tensors elementwise, modifying b in-place.
 *        Computes b = alpha * a + b.
 * @param alpha Scaling factor for a.
 * @param a First input tensor (read-only).
 * @param b Second input tensor (modified in-place).
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_storage_f32_cpu_add(
  float alpha,
  const cgrad_storage_f32_cpu* a,
  cgrad_storage_f32_cpu* b
);

/**
 * @brief Perform batched matrix multiplication (GEMM) on two tensors.
 *        Computes c = alpha * a * b + beta * c.
 * @param alpha Scaling factor for the matrix product.
 * @param a First input tensor.
 * @param b Second input tensor.
 * @param beta Scaling factor for the output tensor.
 * @param c Output tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_storage_f32_cpu_gemm(
  float alpha,
  const cgrad_storage_f32_cpu* a,
  const cgrad_storage_f32_cpu* b,
  float beta,
  cgrad_storage_f32_cpu* c
);

/**
 * @brief Get the layout of a tensor handle.
 * @param t Pointer to tensor.
 * @return Pointer to the tensor layout.
 */
cgrad_storage_layout* cgrad_storage_f32_cpu_get_layout(cgrad_storage_f32_cpu* t);

/**
 * @brief Print the tensor's shape and contents.
 * @param t Pointer to tensor.
 */
void cgrad_storage_f32_cpu_print(const cgrad_storage_f32_cpu* t);

extern cgrad_storage_backend cgrad_storage_backend_f32_cpu;

#endif // CGRAD_TENSOR_F32_CPU_H
