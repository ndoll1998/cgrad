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

#endif // CGRAD_TENSOR_H