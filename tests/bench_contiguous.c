#include "cgrad_tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static inline double elapsed_sec(struct timespec start,
                                 struct timespec end) {
    return (end.tv_sec - start.tv_sec) +
           (end.tv_nsec - start.tv_nsec) / 1e9;
}

int main() {
    uint32_t shape[MAX_TENSOR_DIM] = {512, 64, 64, 32};
    cgrad_tensor_f32 t, t_trans, t_contig;
    int ret = 0;

    // Initialize tensor
    if (cgrad_tensor_f32_init(&t, shape)) {
        printf("Failed to initialize tensor\n");
        return 1;
    }

    struct timespec start, end;

    // Benchmark random fill
    clock_gettime(CLOCK_MONOTONIC, &start);
    cgrad_tensor_f32_fill_rand(&t);
    clock_gettime(CLOCK_MONOTONIC, &end);

    printf("rand fill: %.9f seconds\n", elapsed_sec(start, end));

    // Make tensor non-contiguous by transposing axes 0 and 1
    uint32_t perm[MAX_TENSOR_DIM] = {3, 1, 2, 0};
    t_trans = t;
    cgrad_tensor_f32_transpose(&t_trans, perm);

    // Benchmark make_contiguous
    clock_gettime(CLOCK_MONOTONIC, &start);
    ret = cgrad_tensor_f32_contiguous(&t_trans, &t_contig);
    clock_gettime(CLOCK_MONOTONIC, &end);

    if (ret) {
        printf("cgrad_tensor_f32_make_contiguous failed\n");
        cgrad_tensor_f32_free(&t);
        return 1;
    }

    printf("Time to make contiguous: %.9f seconds\n",
           elapsed_sec(start, end));

    // Cleanup
    cgrad_tensor_f32_free(&t);
    cgrad_tensor_f32_free(&t_contig);

    return 0;
}
