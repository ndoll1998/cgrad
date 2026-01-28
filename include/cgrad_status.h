#ifndef CGRAD_ERRORS_H
#define CGRAD_ERRORS_H

/**
 * @file cgrad_status.h
 * @brief Status code definitions for the cgrad tensor library.
 * 
 * All error codes are negative; CGRAD_SUCCESS is 0.
 */

// Success
#define CGRAD_SUCCESS 0

// General errors
#define CGRAD_ERR_NOT_IMPLEMENTED                           -1001
#define CGRAD_ERR_NULL_POINTER                              -1002
#define CGRAD_ERR_ALLOC_FAILED                              -1003
#define CGRAD_ERR_NOT_INITIALIZED                           -1004

// Storage errors
#define CGRAD_ERR_STORAGE_BACKEND_MISMATCH                  -1101
#define CGRAD_ERR_STORAGE_HANDLE_UNINITIALIZED              -1102
#define CGRAD_ERR_STORAGE_SHAPE_MISMATCH                    -1103
#define CGRAD_ERR_STORAGE_INVALID_BACKEND                   -1104

// Storage registry errors
#define CGRAD_ERR_STORAGE_REGISTRY_PARENT_NOT_REGISTERED    -1201
#define CGRAD_ERR_STORAGE_REGISTRY_BUCKET_NOT_EMPTY         -1202
#define CGRAD_ERR_STORAGE_REGISTRY_NOT_EMPTY                -1203
#define CGRAD_ERR_STORAGE_REGISTRY_RECORD_NOT_FOUND         -1204

// Layout/broadcasting errors
#define CGRAD_ERR_STORAGE_LAYOUT_BROADCAST                  -1301
#define CGRAD_ERR_STORAGE_LAYOUT_NULL_POINTER               -1302
#define CGRAD_ERR_STORAGE_LAYOUT_SHAPE_MISMATCH             -1303
#define CGRAD_ERR_STORAGE_LAYOUT_DUPLICATE_DIM              -1304
#define CGRAD_ERR_STORAGE_LAYOUT_INDEX_OUT_OF_BOUNDS        -1305
#define CGRAD_ERR_STORAGE_LAYOUT_RESHAPE_INVALID_SHAPE      -1306
#define CGRAD_ERR_STORAGE_LAYOUT_NOT_REGULAR                -1307
#define CGRAD_ERR_STORAGE_LAYOUT_NOT_CONTIGUOUS             -1308

// Compute graph errors
#define CGRAD_ERR_COMPUTE_GRAPH_INVALID_OPERATION           -1501
#define CGRAD_ERR_COMPUTE_GRAPH_TOPOLOGICAL_SORT_FAILED     -1502
#define CGRAD_ERR_COMPUTE_GRAPH_EXECUTION_FAILED            -1503
#define CGRAD_ERR_COMPUTE_GRAPH_INVALID_NODE                -1504
#define CGRAD_ERR_COMPUTE_GRAPH_TOO_MANY_INPUTS             -1505
#define CGRAD_ERR_COMPUTE_GRAPH_BACKEND_MISMATCH            -1506
#define CGRAD_ERR_COMPUTE_GRAPH_BACKWARD_NOT_IMPLEMENTED    -1507
#define CGRAD_ERR_COMPUTE_GRAPH_GRADIENT_NOT_AVAILABLE      -1508
#define CGRAD_ERR_COMPUTE_GRAPH_FORWARD_NOT_EXECUTED        -1509
#define CGRAD_ERR_COMPUTE_GRAPH_REQUIRES_GRAD_FALSE         -1510

/**
 * @typedef cgrad_status
 * @brief Represents the result of a cgrad operation.
 *
 * All operations return a `cgrad_status` indicating success or failure.
 * `CGRAD_SUCCESS` (0) indicates success; negative values indicate errors.
 */
typedef int cgrad_status;

#endif // CGRAD_ERRORS_H
