#ifndef CGRAD_ERRORS_H
#define CGRAD_ERRORS_H

/**
 * @file cgrad_errors.h
 * @brief Error code definitions for the cgrad tensor library.
 *
 * Error codes are grouped by subsystem:
 * - General tensor errors: -1000s
 * - Layout/broadcasting errors: -1500s
 * - Backend-specific errors: -2000s and above
 * 
 * All error codes are negative; CGRAD_SUCCESS is 0.
 */

// Success
#define CGRAD_SUCCESS 0

// General tensor errors
#define CGRAD_TENSOR_ERR_NULL_POINTER                -1001
#define CGRAD_TENSOR_ERR_BACKEND_MISMATCH            -1002
#define CGRAD_TENSOR_ERR_HANDLE_UNINITIALIZED        -1003
#define CGRAD_TENSOR_ERR_SHAPE_MISMATCH              -1004
#define CGRAD_TENSOR_ERR_NOT_IMPLEMENTED             -1005

// Layout/broadcasting errors
#define CGRAD_LAYOUT_ERR_BROADCAST                   -1501
#define CGRAD_LAYOUT_ERR_NULL_POINTER                -1502
#define CGRAD_LAYOUT_ERR_SHAPE_MISMATCH              -1503
#define CGRAD_LAYOUT_ERR_DUPLICATE_DIM               -1504  // Duplicate dimension in permutation array

// Backend: F32 CPU specific errors
#define CGRAD_TENSOR_F32_CPU_ERR_ALLOC_FAILED        -2001
#define CGRAD_TENSOR_F32_CPU_ERR_SHAPE_MISMATCH      -2002
#define CGRAD_TENSOR_F32_CPU_ERR_BATCH_ALLOC_FAILED  -2003
#define CGRAD_TENSOR_F32_CPU_ERR_CONTIGUOUS_FAILED   -2004

#endif // CGRAD_ERRORS_H