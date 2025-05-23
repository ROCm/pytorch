typedef enum rocblas_status_
{
    rocblas_status_success         = 0, /**< Success */
    rocblas_status_invalid_handle  = 1, /**< Handle not initialized, invalid or null */
    rocblas_status_not_implemented = 2, /**< Function is not implemented */
    rocblas_status_invalid_pointer = 3, /**< Invalid pointer argument */
    rocblas_status_invalid_size    = 4, /**< Invalid size argument */
    rocblas_status_memory_error    = 5, /**< Failed internal memory allocation, copy or dealloc */
    rocblas_status_internal_error  = 6, /**< Other internal library failure */
    rocblas_status_perf_degraded   = 7, /**< Performance degraded due to low device memory */
    rocblas_status_size_query_mismatch = 8, /**< Unmatched start/stop size query */
    rocblas_status_size_increased      = 9, /**< Queried device memory size increased */
    rocblas_status_size_unchanged      = 10, /**< Queried device memory size unchanged */
    rocblas_status_invalid_value       = 11, /**< Passed argument not valid */
    rocblas_status_continue            = 12, /**< Nothing preventing function to proceed */
    rocblas_status_check_numerics_fail  = 13, /**< Will be set if the vector/matrix has a NaN/Infinity/denormal value */
    rocblas_status_excluded_from_build   = 14, /**< Function is not available in build, likely a function requiring Tensile built without Tensile */
    rocblas_status_arch_mismatch   = 15, /**< The function requires a feature absent from the device architecture */
} rocblas_status;
rocblas_status rocsolver_log_begin();
rocblas_status rocsolver_log_end();
