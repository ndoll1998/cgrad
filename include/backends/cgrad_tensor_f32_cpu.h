#ifndef CGRAD_TENSOR_F32_CPU_H
#define CGRAD_TENSOR_F32_CPU_H

#include "cgrad_layout.h"
#include <stdint.h>
#include <stddef.h>
#include <cblas.h>

typedef struct cgrad_tensor_f32 {
  cgrad_tensor_layout layout;
  float* data;
} cgrad_tensor_f32;

// Tensor initialization
int cgrad_tensor_f32_init(cgrad_tensor_f32* t, const uint32_t* shape);

// Fill
int cgrad_tensor_f32_fill_rand(cgrad_tensor_f32* t);

// Indexing
float* cgrad_tensor_f32_ptr(const cgrad_tensor_f32* t, const uint32_t* indices);
void cgrad_tensor_f32_set(cgrad_tensor_f32* t, const uint32_t* indices, float value);

// Make a contiguous copy of a tensor (arbitrary MAX_TENSOR_DIM)
int cgrad_tensor_f32_contiguous(const cgrad_tensor_f32* src, cgrad_tensor_f32* dst);

// GEMM
int cgrad_tensor_f32_gemm(
  cgrad_tensor_f32* a,
  cgrad_tensor_f32* b,
  cgrad_tensor_f32* c
);

// Free
void cgrad_tensor_f32_free(cgrad_tensor_f32* t);

// Print
void cgrad_tensor_f32_print(const cgrad_tensor_f32* t);

// Transpose: perm is an array of length MAX_TENSOR_DIM, giving the new order of axes
void cgrad_tensor_f32_transpose(cgrad_tensor_f32* t, const uint32_t* perm);

#endif // CGRAD_TENSOR_F32_CPU_H
// Transpose: perm is an array of length MAX_TENSOR_DIM, giving the new order of axes
