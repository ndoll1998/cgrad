#include "cgrad_status.h"
#include "backends/cgrad_backend.h"
#include "backends/cgrad_backend_registry.h"
#include "storage/cgrad_storage_layout.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cblas.h>

// Struct definition
struct cgrad_backend_cpu_f32 {
    cgrad_storage_layout layout;
    float* data;
};

typedef struct cgrad_backend_cpu_f32 cgrad_backend_cpu_f32;

static cgrad_status cgrad_backend_cpu_f32_init(void* t, const uint32_t* shape, int ndim);
static cgrad_status cgrad_backend_cpu_f32_get(const void* t, const uint32_t* indices, int ndim, float* out_value);
static cgrad_status cgrad_backend_cpu_f32_set(void* t, const uint32_t* indices, int ndim, float value);
static cgrad_status cgrad_backend_cpu_f32_fill(void* t, float value);
static cgrad_status cgrad_backend_cpu_f32_fill_rand(void* t);
static cgrad_status cgrad_backend_cpu_f32_view(const void* src, void* dst, const cgrad_storage_layout* target_layout);
static cgrad_status cgrad_backend_cpu_f32_contiguous(const void* src, void* dst);
static void cgrad_backend_cpu_f32_free(void* t);
static cgrad_status cgrad_backend_cpu_f32_axpy(float alpha, void* x, void* y);
static cgrad_status cgrad_backend_cpu_f32_gemm(float alpha, void* a, void* b, float beta, void* c);
static cgrad_storage_layout* cgrad_backend_cpu_f32_get_layout(void* t);
static void cgrad_backend_cpu_f32_print_data(const void* t);

// Backend struct definition
static cgrad_backend backend_f32_cpu = {
    .name = "cpu_f32",
    .storage_handle_size = sizeof(struct cgrad_backend_cpu_f32),
    .storage_init = cgrad_backend_cpu_f32_init,
    .storage_fill = cgrad_backend_cpu_f32_fill,
    .storage_fill_rand = cgrad_backend_cpu_f32_fill_rand,
    .storage_view = cgrad_backend_cpu_f32_view,
    .storage_contiguous = cgrad_backend_cpu_f32_contiguous,
    .storage_free = cgrad_backend_cpu_f32_free,
    .storage_axpy = cgrad_backend_cpu_f32_axpy,
    .storage_gemm = cgrad_backend_cpu_f32_gemm,
    .storage_get = cgrad_backend_cpu_f32_get,
    .storage_set = cgrad_backend_cpu_f32_set,
    .storage_get_layout = cgrad_backend_cpu_f32_get_layout,
    .storage_print_data = cgrad_backend_cpu_f32_print_data,
};

// Auto-registration using constructor attribute
__attribute__((constructor))
static void register_f32_cpu_backend(void) {
    cgrad_register_backend(&backend_f32_cpu);
}

// Helper function for batch operations
static int helper_cgrad_backend_cpu_f32_build_batch_array(const cgrad_backend_cpu_f32* t, float*** array, uint32_t ndim) {
    // compute number of arrays in the batch
    int batch_size = 1;
    for (int i = 0; i < TENSOR_DIM - ndim; i++) {
        batch_size *= t->layout.shape[i];
    }
    
    // allocate memory for array of pointers
    *array = (float**)malloc(batch_size * sizeof(float*));
    if (!(*array)) return CGRAD_ERR_ALLOC_FAILED;
    
    // fill array
    uint32_t indices[TENSOR_DIM] = {0};
    #pragma omp parallel for
    for (size_t i = 0; i < batch_size; i++) {
        size_t rem = i;
        // Compute multi-dimensional index for the batch dims in the current layout
        // The remaining ndims are set to 0
        for (int d = TENSOR_DIM - ndim - 1; d >= 0; d--) {
            indices[d] = rem % t->layout.shape[d];
            rem /= t->layout.shape[d];
        }
        // compute the flat index of to find the data pointer
        size_t idx = 0;
        int err = cgrad_storage_layout_flat_index(&t->layout, indices, TENSOR_DIM, &idx);
        if (err != CGRAD_SUCCESS) {
            return err;
        }

        // set pointer in array
        (*array)[i] = t->data + idx;
    }
    return CGRAD_SUCCESS;
}

// Function implementations
static cgrad_status cgrad_backend_cpu_f32_init(void* t, const uint32_t* shape, int ndim) {
    cgrad_backend_cpu_f32* tensor = (cgrad_backend_cpu_f32*)t;
    if (!tensor || !shape) return CGRAD_ERR_NULL_POINTER;
    
    int layout_err = cgrad_storage_layout_init(&tensor->layout, shape, ndim);
    if (layout_err != CGRAD_SUCCESS) return layout_err;
    
    tensor->data = (float*)calloc(tensor->layout.size, sizeof(float));
    if (!tensor->data) return CGRAD_ERR_ALLOC_FAILED;
    
    return CGRAD_SUCCESS;
}

static cgrad_status cgrad_backend_cpu_f32_get(const void* t, const uint32_t* indices, int ndim, float* out_value) {
    const cgrad_backend_cpu_f32* tensor = (const cgrad_backend_cpu_f32*)t;
    if (!tensor || !indices || !out_value) return CGRAD_ERR_NULL_POINTER;
    
    size_t idx = 0;
    int err = cgrad_storage_layout_flat_index(&tensor->layout, indices, ndim, &idx);
    if (err != CGRAD_SUCCESS) return err;
    
    *out_value = tensor->data[idx];
    return CGRAD_SUCCESS;
}

static cgrad_status cgrad_backend_cpu_f32_set(void* t, const uint32_t* indices, int ndim, float value) {
    cgrad_backend_cpu_f32* tensor = (cgrad_backend_cpu_f32*)t;
    if (!tensor || !indices) return CGRAD_ERR_NULL_POINTER;
    
    size_t idx = 0;
    int err = cgrad_storage_layout_flat_index(&tensor->layout, indices, ndim, &idx);
    if (err != CGRAD_SUCCESS) return err;
    
    tensor->data[idx] = value;
    return CGRAD_SUCCESS;
}

static cgrad_status cgrad_backend_cpu_f32_fill(void* t, float value) {
    cgrad_backend_cpu_f32* tensor = (cgrad_backend_cpu_f32*)t;
    if (!tensor || !tensor->data) return CGRAD_ERR_NULL_POINTER;

    // Find the minimum nonzero stride in the tensor layout
    int min_stride = 0;
    for (int i = 0; i < TENSOR_DIM; i++) {
        int stride = tensor->layout.strides[i];
        if (stride > 0 && (min_stride == 0 || stride < min_stride)) {
            min_stride = stride;
        }
    }
    if (min_stride == 0) min_stride = 1; // fallback for degenerate case

    // fill the tensor
    cblas_scopy(
        tensor->layout.size,
        &value, 0,
        tensor->data, min_stride
    );

    return CGRAD_SUCCESS;
}

static cgrad_status cgrad_backend_cpu_f32_fill_rand(void* t) {
    cgrad_backend_cpu_f32* tensor = (cgrad_backend_cpu_f32*)t;
    if (!tensor || !tensor->data) return CGRAD_ERR_NULL_POINTER;
    
    for (int i = 0; i < tensor->layout.size; i++)
        tensor->data[i] = (float)rand()/(float)(RAND_MAX);
    
    return CGRAD_SUCCESS;
}

static cgrad_status cgrad_backend_cpu_f32_view(const void* src, void* dst, const cgrad_storage_layout* target_layout) {
    const cgrad_backend_cpu_f32* src_tensor = (const cgrad_backend_cpu_f32*)src;
    cgrad_backend_cpu_f32* dst_tensor = (cgrad_backend_cpu_f32*)dst;
    
    // Validate inputs
    if (!src_tensor || !dst_tensor || !target_layout) {
        return CGRAD_ERR_NULL_POINTER;
    }
    
    // Check if target_layout is contained in src's layout
    if (!cgrad_storage_layout_is_contained_in(&src_tensor->layout, target_layout)) {
        return CGRAD_ERR_STORAGE_LAYOUT_INDEX_OUT_OF_BOUNDS;
    }
    
    // Copy target_layout to dst
    cgrad_storage_layout_copy(&dst_tensor->layout, target_layout);
    
    // Share data pointer
    dst_tensor->data = src_tensor->data;
    
    return CGRAD_SUCCESS;
}

static cgrad_status cgrad_backend_cpu_f32_contiguous(const void* src, void* dst) {
    const cgrad_backend_cpu_f32* src_tensor = (const cgrad_backend_cpu_f32*)src;
    cgrad_backend_cpu_f32* dst_tensor = (cgrad_backend_cpu_f32*)dst;
    
    if (!src_tensor || !dst_tensor) return CGRAD_ERR_NULL_POINTER;

    // Check shape
    for (int d = 0; d < TENSOR_DIM; d++) {
        if (src_tensor->layout.shape[d] != dst_tensor->layout.shape[d]) {
            return CGRAD_ERR_STORAGE_LAYOUT_SHAPE_MISMATCH;
        }
    }

    // Check that dst is contiguous
    if (!cgrad_storage_layout_is_contiguous(&dst_tensor->layout)) {
        return CGRAD_ERR_STORAGE_LAYOUT_NOT_CONTIGUOUS;
    }

    uint32_t block_size = src_tensor->layout.shape[TENSOR_DIM-1];
    uint32_t block_ndim = 1;

    while (
        block_ndim < TENSOR_DIM
        && (
            src_tensor->layout.strides[TENSOR_DIM - block_ndim - 1]
            == src_tensor->layout.shape[TENSOR_DIM - block_ndim] * src_tensor->layout.strides[TENSOR_DIM - block_ndim]
        )
    ) {
        block_size *= src_tensor->layout.shape[TENSOR_DIM - block_ndim - 1];
        block_ndim++;
    }

    uint32_t idx[TENSOR_DIM] = {0};
    for (size_t offset = 0; offset < dst_tensor->layout.size; offset += block_size) {
        for (uint32_t d = 0; d < TENSOR_DIM - block_ndim; d++) {
            idx[d] = (offset / dst_tensor->layout.strides[d]) % dst_tensor->layout.shape[d];
        }
        size_t src_idx = 0, dst_idx = 0;
        int src_err = cgrad_storage_layout_flat_index(&src_tensor->layout, idx, TENSOR_DIM, &src_idx);
        int dst_err = cgrad_storage_layout_flat_index(&dst_tensor->layout, idx, TENSOR_DIM, &dst_idx);
        if (src_err == CGRAD_SUCCESS && dst_err == CGRAD_SUCCESS) {
            cblas_scopy(
                block_size,
                src_tensor->data + src_idx, src_tensor->layout.strides[TENSOR_DIM-1],
                dst_tensor->data + dst_idx, 1
            );
        }
    }
    return CGRAD_SUCCESS;
}

static void cgrad_backend_cpu_f32_free(void* t) {
    cgrad_backend_cpu_f32* tensor = (cgrad_backend_cpu_f32*)t;
    if (tensor && tensor->data) {
        free(tensor->data);
        tensor->data = NULL;
    }
}

static cgrad_status cgrad_backend_cpu_f32_axpy(float alpha, void* x, void* y) {
    const cgrad_backend_cpu_f32* x_tensor = (const cgrad_backend_cpu_f32*)x;
    cgrad_backend_cpu_f32* y_tensor = (cgrad_backend_cpu_f32*)y;
    
    if (!x_tensor || !y_tensor) return CGRAD_ERR_NULL_POINTER;
    
    // Check shapes match
    for (int d = 0; d < TENSOR_DIM; d++) {
        if (x_tensor->layout.shape[d] != y_tensor->layout.shape[d]) {
            return CGRAD_ERR_STORAGE_LAYOUT_SHAPE_MISMATCH;
        }
    }

    // Check if y is contiguous (required for in-place modification)
    if (!cgrad_storage_layout_is_contiguous(&y_tensor->layout)) {
        return CGRAD_ERR_NOT_IMPLEMENTED;
    }

    // Make a contiguous copy of x if needed
    cgrad_backend_cpu_f32 x_contig;
    const cgrad_backend_cpu_f32* x_used = x_tensor;
    int is_x_contiguous = cgrad_storage_layout_is_contiguous(&x_tensor->layout);
    
    if (!is_x_contiguous) {
        int init_err = cgrad_backend_cpu_f32_init(&x_contig, x_tensor->layout.shape, TENSOR_DIM);
        if (init_err != CGRAD_SUCCESS) return init_err;
        
        int contig_err = cgrad_backend_cpu_f32_contiguous(x_tensor, &x_contig);
        if (contig_err != CGRAD_SUCCESS) {
            cgrad_backend_cpu_f32_free(&x_contig);
            return contig_err;
        }
        x_used = &x_contig;
    }

    // Use cblas_saxpy to compute y = alpha * x + y
    // Both tensors are now contiguous, so stride is 1
    cblas_saxpy(
        y_tensor->layout.size,
        alpha,
        x_used->data, 1,
        y_tensor->data, 1
    );

    if (!is_x_contiguous) {
        cgrad_backend_cpu_f32_free(&x_contig);
    }
    
    return CGRAD_SUCCESS;
}

static cgrad_status cgrad_backend_cpu_f32_gemm(float alpha, void* a, void* b, float beta, void* c) {
    const cgrad_backend_cpu_f32* a_tensor = (const cgrad_backend_cpu_f32*)a;
    const cgrad_backend_cpu_f32* b_tensor = (const cgrad_backend_cpu_f32*)b;
    cgrad_backend_cpu_f32* c_tensor = (cgrad_backend_cpu_f32*)c;
    
    if (!a_tensor || !b_tensor || !c_tensor) return CGRAD_ERR_NULL_POINTER;
    
    for (int d = 0; d < TENSOR_DIM - 2; d++) {
        if (a_tensor->layout.shape[d] != b_tensor->layout.shape[d]) {
            return CGRAD_ERR_STORAGE_LAYOUT_SHAPE_MISMATCH;
        }
    }
    
    int a_m = a_tensor->layout.shape[TENSOR_DIM-2];
    int a_k = a_tensor->layout.shape[TENSOR_DIM-1];
    int b_k = b_tensor->layout.shape[TENSOR_DIM-2];
    int b_n = b_tensor->layout.shape[TENSOR_DIM-1];
    if (a_k != b_k) {
        return CGRAD_ERR_STORAGE_LAYOUT_SHAPE_MISMATCH;
    }
    
    int m = a_tensor->layout.shape[TENSOR_DIM-2];
    int n = b_tensor->layout.shape[TENSOR_DIM-1];
    int k = b_tensor->layout.shape[TENSOR_DIM-2];
    int bs = a_tensor->layout.size / (m * k);
    
    cgrad_backend_cpu_f32 a_contig, b_contig;
    int is_a_regular = cgrad_storage_layout_is_regular(&a_tensor->layout);
    int is_b_regular = cgrad_storage_layout_is_regular(&b_tensor->layout);
    
    if (!is_a_regular) {
        cgrad_backend_cpu_f32_init(&a_contig, a_tensor->layout.shape, TENSOR_DIM);
        int contig_err = cgrad_backend_cpu_f32_contiguous(a_tensor, &a_contig);
        if (contig_err != CGRAD_SUCCESS) return contig_err;
        a_tensor = &a_contig;
    }
    if (!is_b_regular) {
        cgrad_backend_cpu_f32_init(&b_contig, b_tensor->layout.shape, TENSOR_DIM);
        int contig_err = cgrad_backend_cpu_f32_contiguous(b_tensor, &b_contig);
        if (contig_err != CGRAD_SUCCESS) {
            if (!is_a_regular) cgrad_backend_cpu_f32_free(&a_contig);
            return contig_err;
        }
        b_tensor = &b_contig;
    }
    
    float **A_array, **B_array, **C_array;
    int batch_err = helper_cgrad_backend_cpu_f32_build_batch_array(a_tensor, &A_array, 2);
    if (batch_err != CGRAD_SUCCESS) return batch_err;
    batch_err = helper_cgrad_backend_cpu_f32_build_batch_array(b_tensor, &B_array, 2);
    if (batch_err != CGRAD_SUCCESS) { free(A_array); return batch_err; }
    batch_err = helper_cgrad_backend_cpu_f32_build_batch_array(c_tensor, &C_array, 2);
    if (batch_err != CGRAD_SUCCESS) { free(A_array); free(B_array); return batch_err; }

    CBLAS_TRANSPOSE transA = CblasNoTrans;
    CBLAS_TRANSPOSE transB = CblasNoTrans;
    
    int lda = a_tensor->layout.strides[TENSOR_DIM-2];
    int ldb = b_tensor->layout.strides[TENSOR_DIM-2];
    int ldc = c_tensor->layout.strides[TENSOR_DIM-2];

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
    
    if (!is_a_regular) cgrad_backend_cpu_f32_free(&a_contig);
    if (!is_b_regular) cgrad_backend_cpu_f32_free(&b_contig);
    
    return CGRAD_SUCCESS;
}

static cgrad_storage_layout* cgrad_backend_cpu_f32_get_layout(void* t) {
    cgrad_backend_cpu_f32* tensor = (cgrad_backend_cpu_f32*)t;
    if (!tensor) return NULL;
    return &tensor->layout;
}

static void cgrad_backend_cpu_f32_print_data(const void* t) {
    const cgrad_backend_cpu_f32* tensor = (const cgrad_backend_cpu_f32*)t;
    
    cgrad_storage_layout l;
    cgrad_storage_layout_init(&l, tensor->layout.shape, TENSOR_DIM);
    uint32_t idx[TENSOR_DIM] = {0};
    for (int i = 0; i < l.size; i++) {
        #pragma unroll
        for (int j = 0; j < TENSOR_DIM-1; j++) {
            idx[j] = (i / l.strides[j]) % l.shape[j];
            if ((i > 0) && ((i % l.strides[j]) == 0)) printf("\n");
        }
        idx[TENSOR_DIM-1] = (i / l.strides[TENSOR_DIM-1]) % l.shape[TENSOR_DIM-1];
        float value = 0.0f;
        int err = cgrad_backend_cpu_f32_get(tensor, idx, TENSOR_DIM, &value);
        if (err == CGRAD_SUCCESS) {
            printf("%f ", value);
        } else {
            printf("ERR ");
        }
    }
    printf("\n");
}
