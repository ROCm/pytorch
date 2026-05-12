#pragma once

// Drop-in shim so non-hipified files including <cuda_runtime_api.h> on a
// HIP build compile. Forwards to hip_runtime_api.h and aliases the cuda*
// types/enums/functions used by c10/cuda/* headers to their hip*
// equivalents — mirrors the source-level rewrites that hipify performs.

#include <hip/hip_runtime_api.h>

using cudaStream_t = hipStream_t;
using cudaError_t = hipError_t;
using cudaMemcpyKind = hipMemcpyKind;
using cudaStreamCaptureMode = hipStreamCaptureMode;
using cudaStreamCaptureStatus = hipStreamCaptureStatus;

// Enum values are accessed via `cudaStreamCaptureStatus::cudaStreamCaptureStatusNone`
// (C++ scope resolution into the enum), which becomes
// `hipStreamCaptureStatus::cudaStreamCaptureStatusNone` after the type alias.
// The inner identifier needs to be a macro that substitutes to the hip-named
// value so the lookup hits the enum's actual member.
#define cudaSuccess hipSuccess
#define cudaStreamCaptureStatusNone hipStreamCaptureStatusNone
#define cudaStreamCaptureStatusActive hipStreamCaptureStatusActive
#define cudaStreamCaptureStatusInvalidated hipStreamCaptureStatusInvalidated

#define cudaMemGetInfo hipMemGetInfo
#define cudaMallocAsync hipMallocAsync
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamGetPriority hipStreamGetPriority
#define cudaStreamIsCapturing hipStreamIsCapturing
#define cudaStreamGetCaptureInfo hipStreamGetCaptureInfo
#define cudaThreadExchangeStreamCaptureMode hipThreadExchangeStreamCaptureMode
#define cudaGetLastError hipGetLastError
#define cudaGetErrorString hipGetErrorString
#define cudaDeviceGetStreamPriorityRange hipDeviceGetStreamPriorityRange
