// Google Benchmark version of contiguous benchmark
#include <benchmark/benchmark.h>

extern "C" {
#include "cgrad_errors.h"
#include "cgrad_tensor.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
}

#define CGRAD_BACKEND CGRAD_BACKEND_F32_CPU

static void BM_MakeContiguous(benchmark::State& state) {

    // create tensor instances
    cgrad_tensor t, t_contig;

    // initialize tensor and fill with random values
    uint32_t shape[4] = {512, 32, 32, 32};
    if (cgrad_tensor_init(&t, shape, 4, CGRAD_BACKEND)) {
        state.SkipWithError("Failed to initialize tensor");
        return;
    }
    
    if (cgrad_tensor_fill_rand(&t)) {
        state.SkipWithError("Failed to fill tensor with random values");
        cgrad_tensor_free(&t);
        return;
    }

    uint32_t perm[4] = {2, 1, 3, 0};
    int err = cgrad_tensor_transpose(&t, perm, 4);
    if (err != CGRAD_SUCCESS) {
        // skip error with error code
        state.SkipWithError("Failed to transpose tensor");
        cgrad_tensor_free(&t);
        return;
    }

    for (auto _ : state) {
        int _err = cgrad_tensor_contiguous(&t, &t_contig);

        state.PauseTiming();

        if (_err != CGRAD_SUCCESS) {
            state.SkipWithError("Failed to make tensor contiguous");
            break;
        }
        // free contiguous tensor
        cgrad_tensor_free(&t_contig);

        state.ResumeTiming();
    }

    // free tensors
    cgrad_tensor_free(&t);
}
BENCHMARK(BM_MakeContiguous)
    ->Unit(benchmark::kSecond)
    ->MinTime(1.0);

static void BM_TensorGEMM(benchmark::State& state) {
    int B = state.range(0);
    int M = state.range(0);
    int K = state.range(1);
    int N = state.range(2);

    uint32_t shape_a[3] = {static_cast<uint32_t>(B), static_cast<uint32_t>(M), static_cast<uint32_t>(K)};
    uint32_t shape_b[3] = {static_cast<uint32_t>(B), static_cast<uint32_t>(K), static_cast<uint32_t>(N)};
    cgrad_tensor a, b, r;

    if (
        cgrad_tensor_init(&a, shape_a, 2, CGRAD_BACKEND_F32_CPU)
        || cgrad_tensor_init(&b, shape_b, 2, CGRAD_BACKEND_F32_CPU)
    ) {
        state.SkipWithError("Failed to initialize tensors for GEMM");
        return;
    }
    cgrad_tensor_fill_rand(&a);
    cgrad_tensor_fill_rand(&b);

    for (auto _ : state) {
        r.data = NULL; r.backend = NULL;
        int err = cgrad_tensor_gemm(&a, &b, &r);
        if (err != CGRAD_SUCCESS) {
            state.SkipWithError("GEMM failed");
            break;
        }
        cgrad_tensor_free(&r);
    }
    cgrad_tensor_free(&a);
    cgrad_tensor_free(&b);
}
// Register GEMM for several shape setups (M, K, N)
BENCHMARK(BM_TensorGEMM)
    ->Args({1, 256, 256, 256})
    ->Args({1, 512, 512, 512})
    ->Args({1, 1024, 1024, 1024})
    ->Args({1, 512, 1024, 256})
    ->MinTime(1.0);

static void BM_TensorAdd(benchmark::State& state) {
    // state.range(0): total elements = shape[0] * shape[1]
    int dim0 = state.range(0);
    int dim1 = state.range(1);
    uint32_t shape[2] = {static_cast<uint32_t>(dim0), static_cast<uint32_t>(dim1)};
    cgrad_tensor a, b, r;

    if (
        cgrad_tensor_init(&a, shape, 2, CGRAD_BACKEND_F32_CPU)
        || cgrad_tensor_init(&b, shape, 2, CGRAD_BACKEND_F32_CPU)
    ) {
        state.SkipWithError("Failed to initialize tensors for add");
        return;
    }
    cgrad_tensor_fill_rand(&a);
    cgrad_tensor_fill_rand(&b);

    for (auto _ : state) {
        r.data = NULL; r.backend = NULL;
        int err = cgrad_tensor_add(&a, &b, &r);
        if (err != CGRAD_SUCCESS) {
            state.SkipWithError("Add failed");
            break;
        }
        cgrad_tensor_free(&r);
    }
    cgrad_tensor_free(&a);
    cgrad_tensor_free(&b);
}
// Register add for several shape setups (dim0, dim1)
BENCHMARK(BM_TensorAdd)
    ->Args({256, 256})
    ->Args({512, 512})
    ->Args({1024, 1024})
    ->MinTime(1.0);

BENCHMARK_MAIN();
