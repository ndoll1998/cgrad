#ifndef CGRAD_TENSOR_F32_CPU_H
#define CGRAD_TENSOR_F32_CPU_H

#include "cgrad_backend.h"
#include "cgrad_layout.h"
#include <stdint.h>
#include <stddef.h>
#include <cblas.h>

/**
 * @brief Structure representing a float32 CPU tensor.
 */
typedef struct cgrad_tensor_f32_cpu {
  cgrad_tensor_layout layout;
  float* data;
} cgrad_tensor_f32_cpu;

/**
 * @brief Initialize a float32 CPU tensor with the given shape and ndim.
 *        The user-specified shape (length ndim) is placed at the end; leading unspecified dims are set to 1.
 *        For example, shape={3,4}, ndim=2, MAX_TENSOR_DIM=4 => layout.shape={1,1,3,4}
 * @param t Pointer to tensor to initialize.
 * @param shape Array of dimensions (length ndim).
 * @param ndim Number of dimensions in shape (≤ MAX_TENSOR_DIM).
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_f32_cpu_init(cgrad_tensor_f32_cpu* t, const uint32_t* shape, int ndim);

/**
 * @brief Get the value at the given indices.
 * @param t Pointer to tensor.
 * @param indices Array of indices.
 * @param out_value Pointer to float where the value will be written.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_f32_cpu_get(const cgrad_tensor_f32_cpu* t, const uint32_t* indices, float* out_value);

/**
 * @brief Set the value at the given indices.
 * @param t Pointer to tensor.
 * @param indices Array of indices.
 * @param value Value to set.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_f32_cpu_set(cgrad_tensor_f32_cpu* t, const uint32_t* indices, float value);

/**
 * @brief Fill the tensor with a constant value.
 * @param t Pointer to tensor.
 * @param value The value to fill the tensor with.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_f32_cpu_fill(cgrad_tensor_f32_cpu* t, float value);

/**
 * @brief Fill the tensor with random values.
 * @param t Pointer to tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_f32_cpu_fill_rand(cgrad_tensor_f32_cpu* t);

/**
 * @brief Make a contiguous copy of a tensor.
 * @param src Source tensor.
 * @param dst Destination tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_f32_cpu_contiguous(const cgrad_tensor_f32_cpu* src, cgrad_tensor_f32_cpu* dst);

/**
 * @brief Create a shallow copy of a tensor handle (deep copy layout, shallow copy data).
 * @param src Source tensor.
 * @param dst Destination tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_f32_cpu_shallow_copy(const cgrad_tensor_f32_cpu* src, cgrad_tensor_f32_cpu* dst);

/**
 * @brief Free the memory associated with a tensor.
 * @param t Pointer to tensor.
 */
void cgrad_tensor_f32_cpu_free(cgrad_tensor_f32_cpu* t);

/**
 * @brief Add two tensors elementwise and store the result in a third tensor.
 *        Computes c = alpha * a + b.
 * @param a First input tensor.
 * @param b Second input tensor.
 * @param c Output tensor.
 * @param alpha Scaling factor for a.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_f32_cpu_add(
  float alpha,
  const cgrad_tensor_f32_cpu* a,
  const cgrad_tensor_f32_cpu* b,
  cgrad_tensor_f32_cpu* c
);

/**
 * @brief Perform batched matrix multiplication (GEMM) on two tensors.
 * @param a First input tensor.
 * @param b Second input tensor.
 * @param c Output tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_f32_cpu_gemm(
  const cgrad_tensor_f32_cpu* a,
  const cgrad_tensor_f32_cpu* b,
  cgrad_tensor_f32_cpu* c
);

/**
 * @brief Get the layout of a tensor handle.
 * @param t Pointer to tensor.
 * @return Pointer to the tensor layout.
 */
cgrad_tensor_layout* cgrad_tensor_f32_cpu_get_layout(cgrad_tensor_f32_cpu* t);

/**
 * @brief Print the tensor's shape and contents.
 * @param t Pointer to tensor.
 */
void cgrad_tensor_f32_cpu_print(const cgrad_tensor_f32_cpu* t);

extern cgrad_backend cgrad_backend_f32_cpu;

#endif // CGRAD_TENSOR_F32_CPU_H
