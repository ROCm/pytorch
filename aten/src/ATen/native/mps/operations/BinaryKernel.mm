#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/ExpandUtils.h>
#include <ATen/TensorIndexing.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/BinaryOps.h>
<<<<<<< HEAD
=======
#include <ATen/native/Lerp.h>
#include <ATen/native/TensorFactories.h>
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
#include <ATen/native/TensorIterator.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/operations/BinaryKernel.h>
// For MTLLanguageVersion_3_1
#include <ATen/native/mps/MPSGraphSonomaOps.h>
#include <fmt/format.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/complex_native.h>
#include <ATen/ops/maximum.h>
#include <ATen/ops/minimum.h>
#include <ATen/ops/nextafter_native.h>
#include <ATen/ops/polar_native.h>
#include <ATen/ops/view_as_real.h>
#endif

namespace at::native {
<<<<<<< HEAD
namespace mps {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
=======
#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
#else
#include <ATen/native/mps/BinaryKernel_metallib.h>
#endif

<<<<<<< HEAD
static void binary_mps_impl(TensorIteratorBase& iter, const std::string func_name, bool supports_dense = true) {
  TORCH_CHECK(iter.common_dtype() != at::kDouble, "float64 is not supported on MPS");

  auto convert_double_scalar = [](Tensor& t) {
    if (t.dim() != 0) {
      return;
    }
    if (t.scalar_type() == kDouble) {
      t = t.to(kFloat);
    } else if (t.scalar_type() == kComplexDouble) {
      t = t.to(kComplexFloat);
    }
  };

  Tensor input = iter.input(0);
  Tensor other = iter.input(1);
  Tensor out = iter.output();

  convert_double_scalar(input);
  convert_double_scalar(other);

  id<MTLDevice> device = MPSDevice::getInstance()->device();
  MPSStream* mpsStream = getCurrentMPSStream();
  const uint32_t nDim = iter.ndim();
  constexpr uint32_t nOffsets = 3;
  const uint32_t numThreads = iter.numel();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      auto computeEncoder = mpsStream->commandEncoder();
      if (supports_dense && iter.is_contiguous()) {
        const auto kernel_name = fmt::format("{}_dense_{}", func_name, scalarToMetalTypeString(input));
        auto binaryPSO = lib.getPipelineStateForFunc(kernel_name);
        [computeEncoder setComputePipelineState:binaryPSO];
        mtl_setArgs(computeEncoder, input, other, out);
        mtl_dispatch1DJob(computeEncoder, binaryPSO, numThreads);
        return;
      }
      const std::string kernel = func_name + "_" + scalarToMetalTypeString(input);
      auto kernelDataOffsets = generateKernelDataOffsets(computeEncoder, iter);

      id<MTLComputePipelineState> binaryPSO = lib.getPipelineStateForFunc(kernel);

      // this function call is a no-op if MPS Profiler is not enabled
      getMPSProfiler().beginProfileKernel(binaryPSO, kernel, {input, other});

      [computeEncoder setComputePipelineState:binaryPSO];
      mtl_setArgs(computeEncoder, input, other, out);
      [computeEncoder setBuffer:kernelDataOffsets offset:0 atIndex:3];
      mtl_dispatch1DJob(computeEncoder, binaryPSO, numThreads);

      getMPSProfiler().endProfileKernel(binaryPSO);
    }
  });
}

void complex_mul_out(const Tensor& input, const Tensor& other, const Tensor& output) {
  TORCH_INTERNAL_ASSERT(c10::isComplexType(input.scalar_type()) || c10::isComplexType(other.scalar_type()));
=======
namespace mps {

void binary_op_kernel(const std::string func_name,
                      const Tensor& input,
                      const Tensor& other,
                      const Tensor& output,
                      const std::optional<Scalar> alpha) {
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  auto new_size = at::infer_size(input.sizes(), other.sizes());
  if (!output.sizes().equals(new_size)) {
    output.resize_(new_size);
  }
  uint32_t length = output.numel();
  if (length == 0) {
    return;
  }
<<<<<<< HEAD
  auto common_dtype = output.scalar_type();
  auto output_as_real = at::view_as_real(output).select(output.dim(), 0);
  auto input_as_real = at::view_as_real(input.to(kMPS, common_dtype)).select(input.dim(), 0);
  auto other_as_real = at::view_as_real(other.to(kMPS, common_dtype)).select(other.dim(), 0);
  auto iter =
      TensorIteratorConfig().add_output(output_as_real).add_input(input_as_real).add_input(other_as_real).build();

  mps::binary_mps_impl(iter, "complex_mul", false);
=======

  auto iter = TensorIteratorConfig()
                  .allow_cpu_scalars(true)
                  .add_output(output)
                  .add_input(input)
                  .add_input(other)
                  .check_all_same_dtype(false)
                  .build();

  lib.exec_binary_kernel(iter, func_name, alpha);
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
}

} // namespace mps

static void fmax_mps_kernel(TensorIteratorBase& iter) {
  if (isFloatingType(iter.common_dtype())) {
<<<<<<< HEAD
    mps::binary_mps_impl(iter, "fmax");
=======
    lib.exec_binary_kernel(iter, "fmax");
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  } else {
    at::maximum_out(const_cast<Tensor&>(iter.output()), iter.input(0), iter.input(1));
  }
}
<<<<<<< HEAD
static void fmin_mps_kernel(TensorIteratorBase& iter) {
  if (isFloatingType(iter.common_dtype())) {
    mps::binary_mps_impl(iter, "fmin");
=======

static void fmin_mps_kernel(TensorIteratorBase& iter) {
  if (isFloatingType(iter.common_dtype())) {
    lib.exec_binary_kernel(iter, "fmin");
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  } else {
    at::minimum_out(const_cast<Tensor&>(iter.output()), iter.input(0), iter.input(1));
  }
}

static void copysign_mps_kernel(TensorIteratorBase& iter) {
<<<<<<< HEAD
  mps::binary_mps_impl(iter, "copysign");
=======
  lib.exec_binary_kernel(iter, "copysign");
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
}

static void nextafter_mps_kernel(TensorIteratorBase& iter) {
  TORCH_CHECK_TYPE(isFloatingType(iter.common_dtype()), "nextafter_mps not implemented for non-floating types");
<<<<<<< HEAD
  mps::binary_mps_impl(iter, "nextafter");
=======
  lib.exec_binary_kernel(iter, "nextafter");
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
}

static void zeta_mps_kernel(TensorIteratorBase& iter) {
  TORCH_CHECK_TYPE(isFloatingType(iter.common_dtype()), "zeta_mps not implemented for non-floating types");
<<<<<<< HEAD
  mps::binary_mps_impl(iter, "zeta");
=======
  lib.exec_binary_kernel(iter, "zeta");
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
}

static void xlog1py_mps_kernel(TensorIteratorBase& iter) {
  TORCH_CHECK_TYPE(isFloatingType(iter.common_dtype()), "xlog1py_mps not implemented for non-floating types");
<<<<<<< HEAD
  mps::binary_mps_impl(iter, "xlog1py");
=======
  lib.exec_binary_kernel(iter, "xlog1py");
}

static void chebyshev_polynomial_t_mps_kernel(TensorIteratorBase& iter) {
  TORCH_CHECK_TYPE(isFloatingType(iter.common_dtype()),
                   "chebyshev_polynomial_t_mps not implemented for non-floating types");
  lib.exec_binary_kernel(iter, "chebyshev_polynomial_t");
}

static void chebyshev_polynomial_u_mps_kernel(TensorIteratorBase& iter) {
  TORCH_CHECK_TYPE(isFloatingType(iter.common_dtype()),
                   "chebyshev_polynomial_u_mps not implemented for non-floating types");
  lib.exec_binary_kernel(iter, "chebyshev_polynomial_u");
}

static void chebyshev_polynomial_v_mps_kernel(TensorIteratorBase& iter) {
  TORCH_CHECK_TYPE(isFloatingType(iter.common_dtype()),
                   "chebyshev_polynomial_v_mps not implemented for non-floating types");
  lib.exec_binary_kernel(iter, "chebyshev_polynomial_v");
}

static void chebyshev_polynomial_w_mps_kernel(TensorIteratorBase& iter) {
  TORCH_CHECK_TYPE(isFloatingType(iter.common_dtype()),
                   "chebyshev_polynomial_w_mps not implemented for non-floating types");
  lib.exec_binary_kernel(iter, "chebyshev_polynomial_w");
}

static void hermite_polynomial_h_mps_kernel(TensorIteratorBase& iter) {
  TORCH_CHECK_TYPE(isFloatingType(iter.common_dtype()),
                   "hermite_polynomial_h_mps not implemented for non-floating types");
  lib.exec_binary_kernel(iter, "hermite_polynomial_h");
}

static void hermite_polynomial_he_mps_kernel(TensorIteratorBase& iter) {
  TORCH_CHECK_TYPE(isFloatingType(iter.common_dtype()),
                   "hermite_polynomial_he_mps not implemented for non-floating types");
  lib.exec_binary_kernel(iter, "hermite_polynomial_he");
}

static void polar_mps_kernel(TensorIterator& iter) {
  lib.exec_binary_kernel(iter, "polar");
}

static void complex_mps_kernel(TensorIterator& iter) {
  lib.exec_binary_kernel(iter, "make_complex");
}

static void lerp_scalar_mps_kernel(at::TensorIteratorBase& iter, const Scalar& weight) {
  lib.exec_binary_kernel(iter, "lerp_alpha", weight);
}

static void mul_mps_kernel(TensorIteratorBase& iter) {
  lib.exec_binary_kernel(iter, "mul");
}

static void div_true_mps_kernel(TensorIteratorBase& iter) {
  lib.exec_binary_kernel(iter, "div_true");
}

static void div_floor_mps_kernel(TensorIteratorBase& iter) {
  lib.exec_binary_kernel(iter, "div_floor");
}

static void div_trunc_mps_kernel(TensorIteratorBase& iter) {
  lib.exec_binary_kernel(iter, "div_trunc");
}

static void remainder_mps_kernel(TensorIteratorBase& iter) {
  lib.exec_binary_kernel(iter, "remainder");
}

static void fmod_mps_kernel(TensorIteratorBase& iter) {
  lib.exec_binary_kernel(iter, "fmod");
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
}

REGISTER_DISPATCH(fmax_stub, &fmax_mps_kernel)
REGISTER_DISPATCH(fmin_stub, &fmin_mps_kernel)
REGISTER_DISPATCH(copysign_stub, &copysign_mps_kernel)
REGISTER_DISPATCH(nextafter_stub, &nextafter_mps_kernel)
REGISTER_DISPATCH(zeta_stub, &zeta_mps_kernel)
REGISTER_DISPATCH(xlog1py_stub, &xlog1py_mps_kernel)
<<<<<<< HEAD

Tensor& polar_out_mps(const Tensor& abs, const Tensor& angle, Tensor& output) {
  auto new_size = at::infer_size(abs.sizes(), angle.sizes());
  if (!output.sizes().equals(new_size)) {
    output.resize_(new_size);
  }
  uint32_t length = output.numel();
  if (length == 0) {
    return output;
  }
  auto output_as_real = at::view_as_real(output).select(output.dim(), 0);
  auto iter = TensorIteratorConfig().add_output(output_as_real).add_input(abs).add_input(angle).build();

  mps::binary_mps_impl(iter, "polar");
  return output;
}

Tensor& complex_out_mps(const Tensor& real, const Tensor& imag, Tensor& output) {
  auto new_size = at::infer_size(real.sizes(), imag.sizes());
  if (!output.sizes().equals(new_size)) {
    output.resize_(new_size);
  }
  uint32_t length = output.numel();
  if (length == 0) {
    return output;
  }
  auto output_as_real = at::view_as_real(output).select(output.dim(), 0);
  auto iter = TensorIteratorConfig().add_output(output_as_real).add_input(real).add_input(imag).build();

  mps::binary_mps_impl(iter, "complex_kernel", false);
  return output;
}
=======
REGISTER_DISPATCH(chebyshev_polynomial_t_stub, &chebyshev_polynomial_t_mps_kernel)
REGISTER_DISPATCH(chebyshev_polynomial_u_stub, &chebyshev_polynomial_u_mps_kernel)
REGISTER_DISPATCH(chebyshev_polynomial_v_stub, &chebyshev_polynomial_v_mps_kernel)
REGISTER_DISPATCH(chebyshev_polynomial_w_stub, &chebyshev_polynomial_w_mps_kernel)
REGISTER_DISPATCH(hermite_polynomial_h_stub, &hermite_polynomial_h_mps_kernel)
REGISTER_DISPATCH(hermite_polynomial_he_stub, &hermite_polynomial_he_mps_kernel)
REGISTER_DISPATCH(polar_stub, &polar_mps_kernel);
REGISTER_DISPATCH(complex_stub, &complex_mps_kernel);
REGISTER_DISPATCH(lerp_kernel_scalar_weight, &lerp_scalar_mps_kernel)
REGISTER_DISPATCH(mul_stub, &mul_mps_kernel)
REGISTER_DISPATCH(div_true_stub, &div_true_mps_kernel)
REGISTER_DISPATCH(div_floor_stub, &div_floor_mps_kernel)
REGISTER_DISPATCH(div_trunc_stub, &div_trunc_mps_kernel)
REGISTER_DISPATCH(fmod_stub, &fmod_mps_kernel)
REGISTER_DISPATCH(remainder_stub, &remainder_mps_kernel)
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
} // namespace at::native
