#ifndef CGRAD_TENSOR_F32_CPU_H
#define CGRAD_TENSOR_F32_CPU_H

#include "cgrad_layout.h"
#include <stdint.h>
#include <stddef.h>
#include <cblas.h>

typedef struct cgrad_tensor_f32_cpu {
  cgrad_tensor_layout layout;
  float* data;
} cgrad_tensor_f32_cpu;

int cgrad_tensor_f32_cpu_init(cgrad_tensor_f32_cpu* t, const uint32_t* shape);
int cgrad_tensor_f32_cpu_fill_rand(cgrad_tensor_f32_cpu* t);
float* cgrad_tensor_f32_cpu_ptr(const cgrad_tensor_f32_cpu* t, const uint32_t* indices);
void cgrad_tensor_f32_cpu_set(cgrad_tensor_f32_cpu* t, const uint32_t* indices, float value);
int cgrad_tensor_f32_cpu_contiguous(const cgrad_tensor_f32_cpu* src, cgrad_tensor_f32_cpu* dst);
int cgrad_tensor_f32_cpu_gemm(
  const cgrad_tensor_f32_cpu* a,
  const cgrad_tensor_f32_cpu* b,
  cgrad_tensor_f32_cpu* c
);

int cgrad_tensor_f32_cpu_add(
  const cgrad_tensor_f32_cpu* a,
  const cgrad_tensor_f32_cpu* b,
  cgrad_tensor_f32_cpu* c
);
void cgrad_tensor_f32_cpu_free(cgrad_tensor_f32_cpu* t);
void cgrad_tensor_f32_cpu_print(const cgrad_tensor_f32_cpu* t);
void cgrad_tensor_f32_cpu_transpose(cgrad_tensor_f32_cpu* t, const uint32_t* perm);

// Create a shallow copy of a tensor handle (deep copy layout, shallow copy data)
int cgrad_tensor_f32_cpu_shallow_copy(const cgrad_tensor_f32_cpu* src, cgrad_tensor_f32_cpu* dst);

// Get the layout of a tensor handle
cgrad_tensor_layout* cgrad_tensor_f32_cpu_get_layout(cgrad_tensor_f32_cpu* t);

#endif // CGRAD_TENSOR_F32_CPU_H
