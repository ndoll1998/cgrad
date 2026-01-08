#include "cgrad_tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Layout/tensor initialization
int cgrad_tensor_layout_init(cgrad_tensor_layout* l, const uint32_t* shape) {
  uint32_t cur_stride = 1;
  for (int i = MAX_TENSOR_DIM - 1; i > -1; i--) {
      l->strides[i] = cur_stride;
      l->shape[i] = shape[i];
      cur_stride *= shape[i];
  }
  l->size = cur_stride;
  return 0;
}

int cgrad_tensor_f32_init(cgrad_tensor_f32* t, const uint32_t* shape) {
  if (!t || !shape) return -1;
  if (cgrad_tensor_layout_init(&t->layout, shape)) return -1;
  t->data = (float*)calloc(t->layout.size, sizeof(float));
  if (!t->data) return -1;
  return 0;
}

int cgrad_tensor_f64_init(cgrad_tensor_f64* t, const uint32_t* shape) {
  if (!t || !shape) return -1;
  if (cgrad_tensor_layout_init(&t->layout, shape)) return -1;
  t->data = (double*)calloc(t->layout.size, sizeof(double));
  if (!t->data) return -1;
  return 0;
}

// Fill
int cgrad_tensor_f32_fill_rand(cgrad_tensor_f32* t) {
  if (!t || !t->data) return -1;
  for (int i = 0; i < t->layout.size; i++)
    t->data[i] = (float)rand()/(float)(RAND_MAX);
  return 0;
}

int cgrad_tensor_f64_fill_rand(cgrad_tensor_f64* t) {
  if (!t || !t->data) return -1;
  for (int i = 0; i < t->layout.size; i++)
    t->data[i] = (double)rand()/(double)(RAND_MAX);
  return 0;
}

// Indexing
size_t cgrad_tensor_flat_index(const uint32_t* indices, const uint32_t* strides, int ndim) {
  size_t idx = 0;
  for (int i = 0; i < ndim; i++) {
    idx += indices[i] * strides[i];
  }
  return idx;
}

float* cgrad_tensor_f32_ptr(cgrad_tensor_f32* t, const uint32_t* indices) {
  size_t idx = cgrad_tensor_flat_index(indices, t->layout.strides, MAX_TENSOR_DIM);
  return t->data + idx;
}

// GEMM
int cgrad_tensor_f32_gemm(
  cgrad_tensor_f32* a,
  cgrad_tensor_f32* b,
  cgrad_tensor_f32* c
) {
  if (!a || !b || !c) return -1;
  for (int i = 0; i < MAX_TENSOR_DIM; i++) {
    if (a->layout.shape[i] == 0 || b->layout.shape[i] == 0) return -1;
  }

  int m = a->layout.shape[MAX_TENSOR_DIM-2];
  int n = b->layout.shape[MAX_TENSOR_DIM-1];
  int k = b->layout.shape[MAX_TENSOR_DIM-2];

  int bs = 1;
  uint32_t c_shape[MAX_TENSOR_DIM];
  for (int i = 0; i < MAX_TENSOR_DIM - 2; i++) {
    if (a->layout.shape[i] != b->layout.shape[i]) return -1;
    c_shape[i] = a->layout.shape[i];
    bs *= a->layout.shape[i];
  }
  c_shape[MAX_TENSOR_DIM-1] = n;
  c_shape[MAX_TENSOR_DIM-2] = m;

  if (cgrad_tensor_f32_init(c, c_shape)) return -1;

  float alpha = 1.0f;
  float beta = 0.0f;

  int lda = k;
  int ldb = n;
  int ldc = n;

  float** A_array = (float**)malloc(bs * sizeof(float*));
  float** B_array = (float**)malloc(bs * sizeof(float*));
  float** C_array = (float**)malloc(bs * sizeof(float*));
  if (!A_array || !B_array || !C_array) {
    if (A_array) free(A_array);
    if (B_array) free(B_array);
    if (C_array) free(C_array);
    return -1;
  }

  uint32_t batch_indices[MAX_TENSOR_DIM] = {0};
  int batch_dim0 = a->layout.shape[0];
  int batch_dim1 = a->layout.shape[1];
  int batch_idx = 0;
  for (int i = 0; i < batch_dim0; i++) {
    for (int j = 0; j < batch_dim1; j++) {
      batch_indices[0] = i;
      batch_indices[1] = j;
      batch_indices[2] = 0;
      batch_indices[3] = 0;
      A_array[batch_idx] = cgrad_tensor_f32_ptr(a, batch_indices);
      B_array[batch_idx] = cgrad_tensor_f32_ptr(b, batch_indices);
      C_array[batch_idx] = cgrad_tensor_f32_ptr(c, batch_indices);
      batch_idx++;
    }
  }

  CBLAS_TRANSPOSE transA = CblasNoTrans;
  CBLAS_TRANSPOSE transB = CblasNoTrans;

  cblas_sgemm_batch(
    CblasRowMajor,
    &transA,
    &transB,
    &m, &n, &k,
    &alpha,
    (const float**)A_array, &lda,
    (const float**)B_array, &ldb,
    &beta,
    C_array, &ldc,
    1,
    &bs
  );

  free(A_array);
  free(B_array);
  free(C_array);

  return 0;
}

// Free
void cgrad_tensor_f32_free(cgrad_tensor_f32* t) {
    if (t && t->data) {
        free(t->data);
        t->data = NULL;
    }
}

void cgrad_tensor_f64_free(cgrad_tensor_f64* t) {
    if (t && t->data) {
        free(t->data);
        t->data = NULL;
    }
}

// Print
void cgrad_tensor_f32_print(const cgrad_tensor_f32* t) {
  printf("Shape: (");
  for (int i = 0; i < MAX_TENSOR_DIM; i++) printf(" %i ", t->layout.shape[i]);
  printf(")\n");

  for (int i = 0; i < t->layout.strides[0]; i++) {
    for (int j = 0; j < MAX_TENSOR_DIM-1; j++) {
      if ((i % t->layout.strides[j]) == 0) printf("\n");
    }
    printf("%f ", t->data[i]);
  }
  printf("\n");
}