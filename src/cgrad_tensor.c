#include "cgrad_tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Layout/tensor initialization
int cgrad_tensor_layout_init(cgrad_tensor_layout* l, const uint32_t* shape) {
  uint32_t cur_stride = 1;
  # pragma unroll
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

// Make a contiguous copy of a tensor (arbitrary MAX_TENSOR_DIM)
int cgrad_tensor_f32_make_contiguous(const cgrad_tensor_f32* src, cgrad_tensor_f32* dst) {
  if (!src || !dst) return -1;
  if (cgrad_tensor_f32_init(dst, src->layout.shape)) return -1;

  // Multi-dimensional index
  uint32_t idx[MAX_TENSOR_DIM] = {0};
  for (size_t flat = 0; flat < src->layout.size; flat++) {
    // Compute multi-dimensional index from flat index in row-major order
    size_t rem = flat;
    for (int d = MAX_TENSOR_DIM - 1; d >= 0; d--) {
      idx[d] = rem % src->layout.shape[d];
      rem /= src->layout.shape[d];
    }
    dst->data[flat] = *cgrad_tensor_f32_ptr((cgrad_tensor_f32*)src, idx);
  }
  return 0;
}

void cgrad_tensor_f32_set(cgrad_tensor_f32* t, const uint32_t* indices, float value) {
  size_t idx = cgrad_tensor_flat_index(indices, t->layout.strides, MAX_TENSOR_DIM);
  t->data[idx] = value;
}

int cgrad_tensor_f32_build_batch_array(cgrad_tensor_f32* t, float*** array) {

  // compute batch size
  int batch_size = 1;
  # pragma unroll
  for (int i = 0; i < MAX_TENSOR_DIM - 2; i++) {
    batch_size *= t->layout.shape[i];
  }
  
  // allocate memory
  *array = (float**)malloc(batch_size * sizeof(float*));
  if (!(*array)) return -1;

  // fill array
  uint32_t indices[MAX_TENSOR_DIM] = {0};
  for (int i = 0; i < batch_size; i++) {
    size_t rem = i;
    # pragma unroll
    for (int d = MAX_TENSOR_DIM - 3; d >= 0; d--) {
      indices[d] = rem % t->layout.shape[d];
      rem /= t->layout.shape[d];
    }
    (*array)[i] = cgrad_tensor_f32_ptr(t, indices);
  }

  return 0;
}

// GEMM
int cgrad_tensor_f32_gemm(
  cgrad_tensor_f32* a,
  cgrad_tensor_f32* b,
  cgrad_tensor_f32* c
) {
  if (!a || !b || !c) return -1;
  # pragma unroll
  for (int i = 0; i < MAX_TENSOR_DIM; i++) {
    if (a->layout.shape[i] == 0 || b->layout.shape[i] == 0) return -1;
  }

  int m = a->layout.shape[MAX_TENSOR_DIM-2];
  int n = b->layout.shape[MAX_TENSOR_DIM-1];
  int k = b->layout.shape[MAX_TENSOR_DIM-2];

  int bs = 1;
  uint32_t c_shape[MAX_TENSOR_DIM];
  # pragma unroll
  for (int i = 0; i < MAX_TENSOR_DIM - 2; i++) {
    if (a->layout.shape[i] != b->layout.shape[i]) return -1;
    c_shape[i] = a->layout.shape[i];
    bs *= a->layout.shape[i];
  }
  c_shape[MAX_TENSOR_DIM-1] = n;
  c_shape[MAX_TENSOR_DIM-2] = m;

  if (cgrad_tensor_f32_init(c, c_shape)) return -1;
  // TODO: make sure C has the expected shape
  
  // Only require last stride == 1 for row-major matrix
  cgrad_tensor_f32 a_contig;
  cgrad_tensor_f32 b_contig;
  int is_a_contiguous = (a->layout.strides[MAX_TENSOR_DIM-1] == 1);
  int is_b_contiguous = (b->layout.strides[MAX_TENSOR_DIM-1] == 1);
  if (!is_a_contiguous) {
    cgrad_tensor_f32_make_contiguous(a, &a_contig);
    a = &a_contig;
  }
  if (!is_b_contiguous) {
    cgrad_tensor_f32_make_contiguous(b, &b_contig);
    b = &b_contig;
  }

  // build batch arrays
  float **A_array, **B_array, **C_array;
  if (
    cgrad_tensor_f32_build_batch_array(a, &A_array)
    || cgrad_tensor_f32_build_batch_array(b, &B_array)
    || cgrad_tensor_f32_build_batch_array(c, &C_array)
  ) return -1;

  CBLAS_TRANSPOSE transA = CblasNoTrans;
  CBLAS_TRANSPOSE transB = CblasNoTrans;

  float alpha = 1.0f;
  float beta = 0.0f;

  int lda = a->layout.strides[MAX_TENSOR_DIM-2];
  int ldb = b->layout.strides[MAX_TENSOR_DIM-2];
  int ldc = c->layout.strides[MAX_TENSOR_DIM-2];

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
  
  if (!is_a_contiguous) cgrad_tensor_f32_free(&a_contig);
  if (!is_b_contiguous) cgrad_tensor_f32_free(&b_contig);

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

// Transpose: perm is an array of length MAX_TENSOR_DIM, giving the new order of axes
void cgrad_tensor_f32_transpose(cgrad_tensor_f32* t, const uint32_t* perm) {
  uint32_t new_shape[MAX_TENSOR_DIM];
  uint32_t new_strides[MAX_TENSOR_DIM];
  for (int i = 0; i < MAX_TENSOR_DIM; i++) {
    new_shape[i] = t->layout.shape[perm[i]];
    new_strides[i] = t->layout.strides[perm[i]];
  }
  for (int i = 0; i < MAX_TENSOR_DIM; i++) {
    t->layout.shape[i] = new_shape[i];
    t->layout.strides[i] = new_strides[i];
  }
}
