#include "backends/cgrad_tensor_f32_cpu.h"
#include "cgrad_backend.h"
#include "cgrad_errors.h"
#include "cgrad_layout.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


/**
 * @brief Build a batch array of pointers for batched operations.
 * @param t Pointer to tensor.
 * @param array Output: pointer to array of float pointers.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int helper_cgrad_tensor_f32_cpu_build_batch_array(cgrad_tensor_f32_cpu* t, float*** array) {
  int batch_size = 1;
  # pragma unroll
  for (int i = 0; i < TENSOR_DIM - 2; i++) {
    batch_size *= t->layout.shape[i];
  }
  *array = (float**)malloc(batch_size * sizeof(float*));
  if (!(*array)) return CGRAD_TENSOR_F32_CPU_ERR_BATCH_ALLOC_FAILED;
  uint32_t indices[TENSOR_DIM] = {0};
  #pragma omp parallel for
  for (int i = 0; i < batch_size; i++) {
    size_t rem = i;
    # pragma unroll
    for (int d = TENSOR_DIM - 3; d >= 0; d--) {
      indices[d] = rem % t->layout.shape[d];
      rem /= t->layout.shape[d];
    }
    size_t idx = 0;
    int err = cgrad_tensor_layout_flat_index(&t->layout, indices, TENSOR_DIM, &idx);
    if (err != CGRAD_SUCCESS) {
      (*array)[i] = NULL;
    } else {
      (*array)[i] = t->data + idx;
    }
  }
  return CGRAD_SUCCESS;
}


/**
 * @brief Initialize a float32 CPU tensor with the given shape.
 * @param t Pointer to tensor to initialize.
 * @param shape Array of dimensions.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_f32_cpu_init(cgrad_tensor_f32_cpu* t, const uint32_t* shape, int ndim) {
  if (!t || !shape) return CGRAD_TENSOR_ERR_NULL_POINTER;
  int layout_err = cgrad_tensor_layout_init(&t->layout, shape, ndim);
  if (layout_err != CGRAD_SUCCESS) return layout_err;
  t->data = (float*)calloc(t->layout.size, sizeof(float));
  if (!t->data) return CGRAD_TENSOR_F32_CPU_ERR_ALLOC_FAILED;
  return CGRAD_SUCCESS;
}

static int backend_cgrad_tensor_f32_cpu_tensor_init(void* t, const uint32_t* shape, int ndim) {
    return cgrad_tensor_f32_cpu_init((cgrad_tensor_f32_cpu*)t, shape, ndim);
}

/**
 * @brief Get the value at the given indices.
 * @param t Pointer to tensor.
 * @param indices Array of indices.
 * @param out_value Pointer to float where the value will be written.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_f32_cpu_get(const cgrad_tensor_f32_cpu* t, const uint32_t* indices, float* out_value) {
  if (!t || !indices || !out_value) return CGRAD_TENSOR_ERR_NULL_POINTER;
  size_t idx = 0;
  int err = cgrad_tensor_layout_flat_index(&t->layout, indices, TENSOR_DIM, &idx);
  if (err != CGRAD_SUCCESS) return err;
  *out_value = t->data[idx];
  return CGRAD_SUCCESS;
}

static int backend_cgrad_tensor_f32_cpu_tensor_get(const void* t, const uint32_t* indices, float* out_value) {
    return cgrad_tensor_f32_cpu_get((const cgrad_tensor_f32_cpu*)t, indices, out_value);
}

/**
 * @brief Set the value at the given indices.
 * @param t Pointer to tensor.
 * @param indices Array of indices.
 * @param value Value to set.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_f32_cpu_set(cgrad_tensor_f32_cpu* t, const uint32_t* indices, float value) {
  if (!t || !indices) return CGRAD_TENSOR_ERR_NULL_POINTER;
  size_t idx = 0;
  int err = cgrad_tensor_layout_flat_index(&t->layout, indices, TENSOR_DIM, &idx);
  if (err != CGRAD_SUCCESS) return err;
  t->data[idx] = value;
  return CGRAD_SUCCESS;
}

static int backend_cgrad_tensor_f32_cpu_tensor_set(void* t, const uint32_t* indices, float value) {
    return cgrad_tensor_f32_cpu_set((cgrad_tensor_f32_cpu*)t, indices, value);
}

/**
 * @brief Fill the tensor with a constant value.
 * @param t Pointer to tensor.
 * @param value The value to fill the tensor with.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_f32_cpu_fill(cgrad_tensor_f32_cpu* t, float value) {
  if (!t || !t->data) return CGRAD_TENSOR_ERR_NULL_POINTER;

  // Find the minimum nonzero stride in the tensor layout
  int min_stride = 0;
  for (int i = 0; i < TENSOR_DIM; i++) {
    int stride = t->layout.strides[i];
    if (stride > 0 && (min_stride == 0 || stride < min_stride)) {
      min_stride = stride;
    }
  }
  if (min_stride == 0) min_stride = 1; // fallback for degenerate case

  // fill the tensor
  cblas_scopy(
    t->layout.size,
    &value, 0,
    t->data, min_stride
  );

  return CGRAD_SUCCESS;
}

static int backend_cgrad_tensor_f32_cpu_tensor_fill(void* t, float value) {
    return cgrad_tensor_f32_cpu_fill((cgrad_tensor_f32_cpu*)t, value);
}

/**
 * @brief Fill the tensor with random values.
 * @param t Pointer to tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_f32_cpu_fill_rand(cgrad_tensor_f32_cpu* t) {
  if (!t || !t->data) return CGRAD_TENSOR_ERR_NULL_POINTER;
  for (int i = 0; i < t->layout.size; i++)
    t->data[i] = (float)rand()/(float)(RAND_MAX);
  return CGRAD_SUCCESS;
}

static int backend_cgrad_tensor_f32_cpu_tensor_fill_rand(void* t) {
    return cgrad_tensor_f32_cpu_fill_rand((cgrad_tensor_f32_cpu*)t);
}

/**
 * @brief Create a shallow copy of a tensor handle (deep copy layout, shallow copy data).
 * @param src Source tensor.
 * @param dst Destination tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_f32_cpu_shallow_copy(const cgrad_tensor_f32_cpu* src, cgrad_tensor_f32_cpu* dst) {
  if (!src || !dst) return CGRAD_TENSOR_ERR_NULL_POINTER;
  cgrad_tensor_layout_copy(&dst->layout, &src->layout);
  dst->data = src->data;
  return CGRAD_SUCCESS;
}

static int backend_cgrad_tensor_f32_cpu_tensor_shallow_copy(const void* src, void* dst) {
    return cgrad_tensor_f32_cpu_shallow_copy(
        (const cgrad_tensor_f32_cpu*)src,
        (cgrad_tensor_f32_cpu*)dst
    );
}

/**
 * @brief Make a contiguous copy of a tensor.
 * @param src Source tensor.
 * @param dst Destination tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_f32_cpu_contiguous(const cgrad_tensor_f32_cpu* src, cgrad_tensor_f32_cpu* dst) {
  if (!src || !dst) return CGRAD_TENSOR_ERR_NULL_POINTER;
  int init_err = cgrad_tensor_f32_cpu_init(dst, src->layout.shape, TENSOR_DIM);
  if (init_err != CGRAD_SUCCESS) return init_err;

  uint32_t block_size = src->layout.shape[TENSOR_DIM-1];
  uint32_t block_ndim = 1;

  while (
    block_ndim < TENSOR_DIM
    && (
      src->layout.strides[TENSOR_DIM - block_ndim - 1]
      == src->layout.shape[TENSOR_DIM - block_ndim] * src->layout.strides[TENSOR_DIM - block_ndim]
    )
  ) {
    block_size *= src->layout.shape[TENSOR_DIM - block_ndim - 1];
    block_ndim++;
  }

  uint32_t idx[TENSOR_DIM] = {0};
  for (size_t offset = 0; offset < dst->layout.size; offset += block_size) {
    for (uint32_t d = 0; d < TENSOR_DIM - block_ndim; d++) {
      idx[d] = (offset / dst->layout.strides[d]) % dst->layout.shape[d];
    }
    size_t src_idx = 0, dst_idx = 0;
    int src_err = cgrad_tensor_layout_flat_index(&src->layout, idx, TENSOR_DIM, &src_idx);
    int dst_err = cgrad_tensor_layout_flat_index(&dst->layout, idx, TENSOR_DIM, &dst_idx);
    if (src_err == CGRAD_SUCCESS && dst_err == CGRAD_SUCCESS) {
      cblas_scopy(
        block_size,
        src->data + src_idx, src->layout.strides[TENSOR_DIM-1],
        dst->data + dst_idx, 1
      );
    }
  }
  return CGRAD_SUCCESS;
}

static int backend_cgrad_tensor_f32_cpu_tensor_contiguous(const void* src, void* dst) {
    return cgrad_tensor_f32_cpu_contiguous(
        (const cgrad_tensor_f32_cpu*)src,
        (cgrad_tensor_f32_cpu*)dst
    );
}

/**
 * @brief Free the memory associated with a tensor.
 * @param t Pointer to tensor.
 */
void cgrad_tensor_f32_cpu_free(cgrad_tensor_f32_cpu* t) {
    if (t && t->data) {
        free(t->data);
        t->data = NULL;
    }
}

static void backend_cgrad_tensor_f32_cpu_tensor_free(void* t) {
    cgrad_tensor_f32_cpu_free((cgrad_tensor_f32_cpu*)t);
}

/**
 * @brief Add two tensors elementwise and store the result in a third tensor.
 * @param a First input tensor.
 * @param b Second input tensor.
 * @param c Output tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_f32_cpu_add(
  float alpha,
  const cgrad_tensor_f32_cpu* a,
  const cgrad_tensor_f32_cpu* b,
  cgrad_tensor_f32_cpu* c
) {
  if (!a || !b || !c) return CGRAD_TENSOR_ERR_NULL_POINTER;
  
  for (int d = 0; d < TENSOR_DIM; d++) {
    if (a->layout.shape[d] != b->layout.shape[d]) {
      return CGRAD_TENSOR_F32_CPU_ERR_SHAPE_MISMATCH;
    }
  }
  
  cgrad_tensor_f32_cpu a_contig;
  int is_a_regular = cgrad_tensor_layout_is_regular(&a->layout);
  const cgrad_tensor_f32_cpu* a_used = a;
  if (!is_a_regular) {
    int contig_err = cgrad_tensor_f32_cpu_contiguous(a, &a_contig);
    if (contig_err != CGRAD_SUCCESS) return contig_err;
    a_used = &a_contig;
  }
  
  int contig_err = cgrad_tensor_f32_cpu_contiguous(b, c);
  if (contig_err != CGRAD_SUCCESS) return contig_err;
  
  cblas_saxpy(
    c->layout.size,
    alpha,
    a_used->data, a_used->layout.strides[TENSOR_DIM-1],
    c->data, 1
  );
  
  if (!is_a_regular) cgrad_tensor_f32_cpu_free(&a_contig);
  return CGRAD_SUCCESS;
}

static int backend_cgrad_tensor_f32_cpu_tensor_add(float alpha, void* a, void* b, void* c) {
    return cgrad_tensor_f32_cpu_add(
        alpha,
        (const cgrad_tensor_f32_cpu*)a,
        (const cgrad_tensor_f32_cpu*)b,
        (cgrad_tensor_f32_cpu*)c
    );
}

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
) {
  if (!a || !b || !c) return CGRAD_TENSOR_ERR_NULL_POINTER;
  
  for (int d = 0; d < TENSOR_DIM - 2; d++) {
    if (a->layout.shape[d] != b->layout.shape[d]) {
      return CGRAD_TENSOR_F32_CPU_ERR_SHAPE_MISMATCH;
    }
  }
  
  int a_m = a->layout.shape[TENSOR_DIM-2];
  int a_k = a->layout.shape[TENSOR_DIM-1];
  int b_k = b->layout.shape[TENSOR_DIM-2];
  int b_n = b->layout.shape[TENSOR_DIM-1];
  if (a_k != b_k) {
    return CGRAD_TENSOR_F32_CPU_ERR_SHAPE_MISMATCH;
  }
  
  int m = a->layout.shape[TENSOR_DIM-2];
  int n = b->layout.shape[TENSOR_DIM-1];
  int k = b->layout.shape[TENSOR_DIM-2];
  int bs = a->layout.size / (m * k);
  
  cgrad_tensor_f32_cpu a_contig;
  cgrad_tensor_f32_cpu b_contig;
  int is_a_contiguous = (a->layout.strides[TENSOR_DIM-1] == 1);
  int is_b_contiguous = (b->layout.strides[TENSOR_DIM-1] == 1);
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
  
  float **A_array, **B_array, **C_array;
  int batch_err = helper_cgrad_tensor_f32_cpu_build_batch_array((cgrad_tensor_f32_cpu*)a, &A_array);
  if (batch_err != CGRAD_SUCCESS) return batch_err;
  batch_err = helper_cgrad_tensor_f32_cpu_build_batch_array((cgrad_tensor_f32_cpu*)b, &B_array);
  if (batch_err != CGRAD_SUCCESS) { free(A_array); return batch_err; }
  batch_err = helper_cgrad_tensor_f32_cpu_build_batch_array(c, &C_array);
  if (batch_err != CGRAD_SUCCESS) { free(A_array); free(B_array); return batch_err; }
  
  CBLAS_TRANSPOSE transA = CblasNoTrans;
  CBLAS_TRANSPOSE transB = CblasNoTrans;
  float alpha = 1.0f;
  float beta = 0.0f;
  
  int lda = a->layout.strides[TENSOR_DIM-2];
  int ldb = b->layout.strides[TENSOR_DIM-2];
  int ldc = c->layout.strides[TENSOR_DIM-2];
  
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

static int backend_cgrad_tensor_f32_cpu_tensor_gemm(void* a, void* b, void* c) {
    return cgrad_tensor_f32_cpu_gemm((cgrad_tensor_f32_cpu*)a, (cgrad_tensor_f32_cpu*)b, (cgrad_tensor_f32_cpu*)c);
}

/**
 * @brief Get the layout of a tensor handle.
 * @param t Pointer to tensor.
 * @return Pointer to the tensor layout.
 */
cgrad_tensor_layout* cgrad_tensor_f32_cpu_get_layout(cgrad_tensor_f32_cpu* t) {
  if (!t) return NULL;
  return &t->layout;
}

static cgrad_tensor_layout* backend_cgrad_tensor_f32_cpu_tensor_get_layout(void* t) {
    return cgrad_tensor_f32_cpu_get_layout((cgrad_tensor_f32_cpu*)t);
}

/**
 * @brief Print the tensor's shape and contents.
 * @param t Pointer to tensor.
 */
void cgrad_tensor_f32_cpu_print(const cgrad_tensor_f32_cpu* t) {
  printf("Shape: (");
  for (int i = 0; i < TENSOR_DIM; i++) printf(" %i ", t->layout.shape[i]);
  printf(")\n");
  cgrad_tensor_layout l;
  cgrad_tensor_layout_init(&l, t->layout.shape, TENSOR_DIM);
  uint32_t idx[TENSOR_DIM] = {0};
  for (int i = 0; i < l.size; i++) {
    #pragma unroll
    for (int j = 0; j < TENSOR_DIM-1; j++) {
      idx[j] = (i / l.strides[j]) % l.shape[j];
      if ((i > 0) && ((i % l.strides[j]) == 0)) printf("\n");
    }
    idx[TENSOR_DIM-1] = (i / l.strides[TENSOR_DIM-1]) % l.shape[TENSOR_DIM-1];
    float value = 0.0f;
    int err = cgrad_tensor_f32_cpu_get(t, idx, &value);
    if (err == CGRAD_SUCCESS) {
      printf("%f ", value);
    } else {
      printf("ERR ");
    }
  }
  printf("\n");
}

static void backend_cgrad_tensor_f32_cpu_tensor_print(const void* t) {
    cgrad_tensor_f32_cpu_print((const cgrad_tensor_f32_cpu*)t);
}


static void* backend_alloc_tensor_f32_cpu_handle(void) {
    return malloc(sizeof(struct cgrad_tensor_f32_cpu));
}

cgrad_backend cgrad_backend_f32_cpu = {
    .type = CGRAD_BACKEND_F32_CPU,
    .alloc_tensor_handle = backend_alloc_tensor_f32_cpu_handle,
    .tensor_init      = backend_cgrad_tensor_f32_cpu_tensor_init,
    .tensor_fill      = backend_cgrad_tensor_f32_cpu_tensor_fill,
    .tensor_fill_rand = backend_cgrad_tensor_f32_cpu_tensor_fill_rand,
    .tensor_shallow_copy = backend_cgrad_tensor_f32_cpu_tensor_shallow_copy,
    .tensor_contiguous   = backend_cgrad_tensor_f32_cpu_tensor_contiguous,
    .tensor_free      = backend_cgrad_tensor_f32_cpu_tensor_free,
    .tensor_add       = backend_cgrad_tensor_f32_cpu_tensor_add,
    .tensor_gemm      = backend_cgrad_tensor_f32_cpu_tensor_gemm,
    .tensor_get       = backend_cgrad_tensor_f32_cpu_tensor_get,
    .tensor_set       = backend_cgrad_tensor_f32_cpu_tensor_set,
    .tensor_get_layout   = backend_cgrad_tensor_f32_cpu_tensor_get_layout,
    .tensor_print     = backend_cgrad_tensor_f32_cpu_tensor_print,
};
