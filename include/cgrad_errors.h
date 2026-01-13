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

// General errors
#define CGRAD_ERR_NOT_IMPLEMENTED                       -1001
#define CGRAD_ERR_NULL_POINTER                          -1002

// Storage errors
#define CGRAD_STORAGE_ERR_BACKEND_MISMATCH              -1101
#define CGRAD_STORAGE_ERR_HANDLE_UNINITIALIZED          -1102
#define CGRAD_STORAGE_ERR_SHAPE_MISMATCH                -1103

// Storage registry errors
#define CGRAD_STORAGE_REGISTRY_ALLOC_FAILED             -1201
#define CGRAD_STORAGE_REGISTRY_PARENT_NOT_REGISTERED    -1202
#define CGRAD_STORAGE_REGISTRY_BUCKET_NOT_EMPTY         -1203

// Layout/broadcasting errors
#define CGRAD_STORAGE_LAYOUT_ERR_BROADCAST              -1301
#define CGRAD_STORAGE_LAYOUT_ERR_NULL_POINTER           -1302
#define CGRAD_STORAGE_LAYOUT_ERR_SHAPE_MISMATCH         -1303
#define CGRAD_STORAGE_LAYOUT_ERR_DUPLICATE_DIM          -1304
#define CGRAD_STORAGE_LAYOUT_ERR_INDEX_OUT_OF_BOUNDS    -1305
#define CGRAD_STORAGE_LAYOUT_ERR_RESHAPE_INVALID_SHAPE  -1306
#define CGRAD_STORAGE_LAYOUT_ERR_NOT_REGULAR            -1307

// Backend: F32 CPU specific errors
#define CGRAD_STORAGE_F32_CPU_ERR_ALLOC_FAILED          -1401
#define CGRAD_STORAGE_F32_CPU_ERR_SHAPE_MISMATCH        -1402
#define CGRAD_STORAGE_F32_CPU_ERR_BATCH_ALLOC_FAILED    -1403
#define CGRAD_STORAGE_F32_CPU_ERR_CONTIGUOUS_FAILED     -1404
#define CGRAD_STORAGE_F32_CPU_ERR_LAYOUT_NOT_CONTIGUOUS -1405

#endif // CGRAD_ERRORS_H
