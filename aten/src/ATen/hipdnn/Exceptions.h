#pragma once

// #include <ATen/miopen/miopen-wrapper.h>
#include <string>
#include <stdexcept>
#include <sstream>
#include <hipdnn_frontend.hpp>

namespace at { namespace native {

class hipdnn_exception : public std::runtime_error {
public:
  hipdnnStatus_t status;
  hipdnn_exception(hipdnnStatus_t status, const char* msg)
      : std::runtime_error(msg)
      , status(status) {}
  hipdnn_exception(hipdnnStatus_t status, const std::string& msg)
      : std::runtime_error(msg)
      , status(status) {}
};

inline void HIPDNN_CHECK(hipdnnStatus_t status)
{
  if (status != HIPDNN_STATUS_SUCCESS ) {
    if (status == HIPDNN_STATUS_NOT_SUPPORTED) {
        throw hipdnn_exception(status, std::string(hipdnnGetErrorString(status)) +
                ". This error may appear if you passed in a non-contiguous input.");
    }
    throw hipdnn_exception(status, hipdnnGetErrorString(status));
  }
}

inline void HIP_CHECK(hipError_t error)
{
  if (error != hipSuccess) {
    std::string msg("HIP error: ");
    msg += hipGetErrorString(error);
    throw std::runtime_error(msg);
  }
}

}} // namespace at::native
