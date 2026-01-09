#ifndef CGRAD_LAYOUT_H
#define CGRAD_LAYOUT_H

#include <stdint.h>
#include <stddef.h>

#define MAX_TENSOR_DIM 4

typedef struct cgrad_tensor_layout {
  uint32_t size;
  uint32_t shape[MAX_TENSOR_DIM];
  uint32_t strides[MAX_TENSOR_DIM];
} cgrad_tensor_layout;

// Layout initialization
int cgrad_tensor_layout_init(cgrad_tensor_layout* l, const uint32_t* shape);

// Indexing
size_t cgrad_tensor_flat_index(const uint32_t* indices, const uint32_t* strides);

// Transpose: perm is an array of length MAX_TENSOR_DIM, giving the new order of axes
void cgrad_tensor_layout_transpose(cgrad_tensor_layout* layout, const uint32_t* perm);

#endif // CGRAD_LAYOUT_H
