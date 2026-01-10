#include "cgrad_tensor.h"
#include "backends/cgrad_tensor_f32_cpu.h"
#include "cgrad_layout.h"
#include "cgrad_errors.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Init
int cgrad_tensor_f32_cpu_init(cgrad_tensor_f32_cpu* t, const uint32_t* shape) {
  if (!t || !shape) return CGRAD_TENSOR_ERR_NULL_POINTER;
  int layout_err = cgrad_tensor_layout_init(&t->layout, shape);
  if (layout_err != CGRAD_SUCCESS) return layout_err;
  t->data = (float*)calloc(t->layout.size, sizeof(float));
  if (!t->data) return CGRAD_TENSOR_F32_CPU_ERR_ALLOC_FAILED;
  return CGRAD_SUCCESS;
}

int cgrad_tensor_f32_cpu_fill_rand(cgrad_tensor_f32_cpu* t) {
  if (!t || !t->data) return CGRAD_TENSOR_ERR_NULL_POINTER;
  for (int i = 0; i < t->layout.size; i++)
    t->data[i] = (float)rand()/(float)(RAND_MAX);
  return CGRAD_SUCCESS;
}

float* cgrad_tensor_f32_cpu_ptr(const cgrad_tensor_f32_cpu* t, const uint32_t* indices) {
  size_t idx = cgrad_tensor_flat_index(indices, t->layout.strides);
  return t->data + idx;
}

int cgrad_tensor_f32_cpu_contiguous(const cgrad_tensor_f32_cpu* src, cgrad_tensor_f32_cpu* dst) {
  if (!src || !dst) return CGRAD_TENSOR_ERR_NULL_POINTER;
  int init_err = cgrad_tensor_f32_cpu_init(dst, src->layout.shape);
  if (init_err != CGRAD_SUCCESS) return init_err;

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
      cgrad_tensor_f32_cpu_ptr(src, idx), src->layout.strides[MAX_TENSOR_DIM-1],
      cgrad_tensor_f32_cpu_ptr(dst, idx), 1
    );
  }

  return CGRAD_SUCCESS;
}


void cgrad_tensor_f32_cpu_set(cgrad_tensor_f32_cpu* t, const uint32_t* indices, float value) {
  size_t idx = cgrad_tensor_flat_index(indices, t->layout.strides);
  t->data[idx] = value;
}

int cgrad_tensor_f32_cpu_build_batch_array(cgrad_tensor_f32_cpu* t, float*** array) {

  // compute batch size
  int batch_size = 1;
  # pragma unroll
  for (int i = 0; i < MAX_TENSOR_DIM - 2; i++) {
    batch_size *= t->layout.shape[i];
  }
  
  // allocate memory
  *array = (float**)malloc(batch_size * sizeof(float*));
  if (!(*array)) return CGRAD_TENSOR_F32_CPU_ERR_BATCH_ALLOC_FAILED;

  // fill array
  uint32_t indices[MAX_TENSOR_DIM] = {0};
  #pragma omp parallel for
  for (int i = 0; i < batch_size; i++) {
    size_t rem = i;
    # pragma unroll
    for (int d = MAX_TENSOR_DIM - 3; d >= 0; d--) {
      indices[d] = rem % t->layout.shape[d];
      rem /= t->layout.shape[d];
    }
    (*array)[i] = cgrad_tensor_f32_cpu_ptr(t, indices);
  }

  return CGRAD_SUCCESS;
}

int cgrad_tensor_f32_cpu_add(
  const cgrad_tensor_f32_cpu* a,
  const cgrad_tensor_f32_cpu* b,
  cgrad_tensor_f32_cpu* c
) {
  if (!a || !b || !c) return CGRAD_TENSOR_ERR_NULL_POINTER;

  // Make a contiguous if needed
  cgrad_tensor_f32_cpu a_contig;
  int is_a_contiguous = cgrad_tensor_layout_is_contiguous(&a->layout);
  const cgrad_tensor_f32_cpu* a_used = a;
  if (!is_a_contiguous) {
    int contig_err = cgrad_tensor_f32_cpu_contiguous(a, &a_contig);
    if (contig_err != CGRAD_SUCCESS) return contig_err;
    a_used = &a_contig;
  }

  // Allocate c with broadcasted shape and write b into c
  int contig_err = cgrad_tensor_f32_cpu_contiguous(b, c);
  if (contig_err != CGRAD_SUCCESS) return contig_err;

  // c = a + c (elementwise addition)
  cblas_saxpy(
    c->layout.size,
    1.0f,
    a_used->data, 1,
    c->data, 1
  );

  if (!is_a_contiguous) cgrad_tensor_f32_cpu_free(&a_contig);

  return CGRAD_SUCCESS;
}

// GEMM
int cgrad_tensor_f32_cpu_gemm(
  const cgrad_tensor_f32_cpu* a,
  const cgrad_tensor_f32_cpu* b,
  cgrad_tensor_f32_cpu* c
) {
  if (!a || !b || !c) return CGRAD_TENSOR_ERR_NULL_POINTER;
 
  // get dimensions
  int m = a->layout.shape[MAX_TENSOR_DIM-2];
  int n = b->layout.shape[MAX_TENSOR_DIM-1];
  int k = b->layout.shape[MAX_TENSOR_DIM-2];

  // compute batch size
  int bs = a->layout.size / (m * k);

  // Only require last stride == 1 for row-major matrix
  cgrad_tensor_f32_cpu a_contig;
  cgrad_tensor_f32_cpu b_contig;
  int is_a_contiguous = (a->layout.strides[MAX_TENSOR_DIM-1] == 1);
  int is_b_contiguous = (b->layout.strides[MAX_TENSOR_DIM-1] == 1);
  if (!is_a_contiguous) {
    int contig_err = cgrad_tensor_f32_cpu_contiguous(a, &a_contig);
    if (contig_err != CGRAD_SUCCESS) return contig_err;
    a = &a_contig;
  }
  if (!is_b_contiguous) {
    int contig_err = cgrad_tensor_f32_cpu_contiguous(b, &b_contig);
    if (contig_err != CGRAD_SUCCESS) return contig_err;
    b = &b_contig;
  }

  // build batch arrays
  float **A_array, **B_array, **C_array;
  int batch_err = cgrad_tensor_f32_cpu_build_batch_array((cgrad_tensor_f32_cpu*)a, &A_array);
  if (batch_err != CGRAD_SUCCESS) return batch_err;
  batch_err = cgrad_tensor_f32_cpu_build_batch_array((cgrad_tensor_f32_cpu*)b, &B_array);
  if (batch_err != CGRAD_SUCCESS) { free(A_array); return batch_err; }
  batch_err = cgrad_tensor_f32_cpu_build_batch_array(c, &C_array);
  if (batch_err != CGRAD_SUCCESS) { free(A_array); free(B_array); return batch_err; }

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
  
  if (!is_a_contiguous) cgrad_tensor_f32_cpu_free(&a_contig);
  if (!is_b_contiguous) cgrad_tensor_f32_cpu_free(&b_contig);

  return CGRAD_SUCCESS;
}

// Free
void cgrad_tensor_f32_cpu_free(cgrad_tensor_f32_cpu* t) {
    if (t && t->data) {
        free(t->data);
        t->data = NULL;
    }
}

void cgrad_tensor_f32_cpu_print(const cgrad_tensor_f32_cpu* t) {
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
    printf("%f ", *cgrad_tensor_f32_cpu_ptr(t, idx));
  }
  printf("\n");
}

void cgrad_tensor_f32_cpu_transpose(cgrad_tensor_f32_cpu* t, const uint32_t* perm) {
  cgrad_tensor_layout_transpose(&t->layout, perm);
}

// Create a shallow copy of a tensor handle (deep copy layout, shallow copy data)
int cgrad_tensor_f32_cpu_shallow_copy(const cgrad_tensor_f32_cpu* src, cgrad_tensor_f32_cpu* dst) {
  if (!src || !dst) return CGRAD_TENSOR_ERR_NULL_POINTER;
  cgrad_tensor_layout_copy(&dst->layout, &src->layout);
  dst->data = src->data;
  return CGRAD_SUCCESS;
}

// Get the layout of a tensor handle
cgrad_tensor_layout* cgrad_tensor_f32_cpu_get_layout(cgrad_tensor_f32_cpu* t) {
  if (!t) return NULL;
  return &t->layout;
}
