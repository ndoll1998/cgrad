#include "cgrad_tensor.h"
#include "backends/cgrad_tensor_f32_cpu.h"
#include "cgrad_layout.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Init
int cgrad_tensor_f32_init(cgrad_tensor_f32* t, const uint32_t* shape) {
  if (!t || !shape) return -1;
  if (cgrad_tensor_layout_init(&t->layout, shape)) return -1;
  t->data = (float*)calloc(t->layout.size, sizeof(float));
  if (!t->data) return -1;
  return 0;
}

int cgrad_tensor_f32_fill_rand(cgrad_tensor_f32* t) {
  if (!t || !t->data) return -1;
  for (int i = 0; i < t->layout.size; i++)
    t->data[i] = (float)rand()/(float)(RAND_MAX);
  return 0;
}

float* cgrad_tensor_f32_ptr(const cgrad_tensor_f32* t, const uint32_t* indices) {
  size_t idx = cgrad_tensor_flat_index(indices, t->layout.strides);
  return t->data + idx;
}

int cgrad_tensor_f32_contiguous(const cgrad_tensor_f32* src, cgrad_tensor_f32* dst) {
  if (!src || !dst) return -1;
  if (cgrad_tensor_f32_init(dst, src->layout.shape)) return -1;

  uint32_t block_size = src->layout.shape[MAX_TENSOR_DIM-1];
  uint32_t block_ndim = 1;

  while (
    block_ndim < MAX_TENSOR_DIM
    && (
      src->layout.strides[MAX_TENSOR_DIM - block_ndim - 1]
      == src->layout.shape[MAX_TENSOR_DIM - block_ndim] * src->layout.strides[MAX_TENSOR_DIM - block_ndim]
    )
  ) {
    block_size *= src->layout.shape[MAX_TENSOR_DIM - block_ndim - 1];
    block_ndim++;
  }

  uint32_t idx[MAX_TENSOR_DIM] = {0};
  for (size_t offset = 0; offset < dst->layout.size; offset += block_size) {
    // update indexes for non-contiguous dims
    for (uint32_t d = 0; d < MAX_TENSOR_DIM - block_ndim; d++) {
      idx[d] = (offset / dst->layout.strides[d]) % dst->layout.shape[d];
    }
    // compute source and destination starting pointers
    cblas_scopy(
      block_size,
      cgrad_tensor_f32_ptr(src, idx), src->layout.strides[MAX_TENSOR_DIM-1],
      cgrad_tensor_f32_ptr(dst, idx), 1
    );
  }

  return 0;
}


void cgrad_tensor_f32_set(cgrad_tensor_f32* t, const uint32_t* indices, float value) {
  size_t idx = cgrad_tensor_flat_index(indices, t->layout.strides);
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
    cgrad_tensor_f32_contiguous(a, &a_contig);
    a = &a_contig;
  }
  if (!is_b_contiguous) {
    cgrad_tensor_f32_contiguous(b, &b_contig);
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

void cgrad_tensor_f32_print(const cgrad_tensor_f32* t) {
  printf("Shape: (");
  for (int i = 0; i < MAX_TENSOR_DIM; i++) printf(" %i ", t->layout.shape[i]);
  printf(")\n");

  cgrad_tensor_layout l;
  cgrad_tensor_layout_init(&l, t->layout.shape);
  uint32_t idx[MAX_TENSOR_DIM] = {0};

  for (int i = 0; i < l.size; i++) {
    #pragma unroll
    for (int j = 0; j < MAX_TENSOR_DIM-1; j++) {
      idx[j] = (i / l.strides[j]) % l.shape[j];
      if ((i > 0) && ((i % l.strides[j]) == 0)) printf("\n");
    }
    idx[MAX_TENSOR_DIM-1] = (i / l.strides[MAX_TENSOR_DIM-1]) % l.shape[MAX_TENSOR_DIM-1];
    printf("%f ", *cgrad_tensor_f32_ptr(t, idx));
  }
  printf("\n");
}

void cgrad_tensor_f32_transpose(cgrad_tensor_f32* t, const uint32_t* perm) {
  cgrad_tensor_layout_transpose(&t->layout, perm);
}
