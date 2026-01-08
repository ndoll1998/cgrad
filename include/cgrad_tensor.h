#ifndef CGRAD_TENSOR_H
#define CGRAD_TENSOR_H

#include <stdint.h>
#include <stddef.h>
#include <cblas.h>

#define MAX_TENSOR_DIM 4

typedef struct cgrad_tensor_layout {
  uint32_t size;
  uint32_t shape[MAX_TENSOR_DIM];
  uint32_t strides[MAX_TENSOR_DIM];
} cgrad_tensor_layout;

typedef struct cgrad_tensor_f32 {
  cgrad_tensor_layout layout;
  float* data;
} cgrad_tensor_f32;

typedef struct cgrad_tensor_f64 {
  cgrad_tensor_layout layout;
  double* data;
} cgrad_tensor_f64;

// Layout/tensor initialization
int cgrad_tensor_layout_init(cgrad_tensor_layout* l, const uint32_t* shape);
int cgrad_tensor_f32_init(cgrad_tensor_f32* t, const uint32_t* shape);
int cgrad_tensor_f64_init(cgrad_tensor_f64* t, const uint32_t* shape);

// Fill
int cgrad_tensor_f32_fill_rand(cgrad_tensor_f32* t);
int cgrad_tensor_f64_fill_rand(cgrad_tensor_f64* t);

// Indexing
size_t cgrad_tensor_flat_index(const uint32_t* indices, const uint32_t* strides, int ndim);
float* cgrad_tensor_f32_ptr(cgrad_tensor_f32* t, const uint32_t* indices);
void cgrad_tensor_f32_set(cgrad_tensor_f32* t, const uint32_t* indices, float value);

// Make a contiguous copy of a tensor (arbitrary MAX_TENSOR_DIM)
int cgrad_tensor_f32_make_contiguous(const cgrad_tensor_f32* src, cgrad_tensor_f32* dst);

// GEMM
int cgrad_tensor_f32_gemm(
  cgrad_tensor_f32* a,
  cgrad_tensor_f32* b,
  cgrad_tensor_f32* c
);

// Free
void cgrad_tensor_f32_free(cgrad_tensor_f32* t);
void cgrad_tensor_f64_free(cgrad_tensor_f64* t);

// Print
void cgrad_tensor_f32_print(const cgrad_tensor_f32* t);

// Transpose: perm is an array of length MAX_TENSOR_DIM, giving the new order of axes
void cgrad_tensor_f32_transpose(cgrad_tensor_f32* t, const uint32_t* perm);

#endif // CGRAD_TENSOR_H
