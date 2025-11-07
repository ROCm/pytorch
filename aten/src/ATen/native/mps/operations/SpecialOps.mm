#include <ATen/native/mps/OperationUtils.h>

#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/UnaryOps.h>

#include <ATen/native/TensorIterator.h>

namespace at::native {
#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/SpecialOps_metallib.h>
#endif

static void i0_kernel_mps(TensorIteratorBase& iter) {
  lib.exec_unary_kernel(iter, "i0");
}

<<<<<<< HEAD
=======
static void i0e_kernel_mps(TensorIteratorBase& iter) {
  lib.exec_unary_kernel(iter, "i0e");
}

>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
static void i1_kernel_mps(TensorIteratorBase& iter) {
  lib.exec_unary_kernel(iter, "i1");
}

<<<<<<< HEAD
=======
static void i1e_kernel_mps(TensorIteratorBase& iter) {
  lib.exec_unary_kernel(iter, "i1e");
}

>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
static void spherical_bessel_j0_kernel_mps(TensorIteratorBase& iter) {
  lib.exec_unary_kernel(iter, "spherical_bessel_j0");
}

static void entr_kernel_mps(TensorIteratorBase& iter) {
  lib.exec_unary_kernel(iter, "entr");
}

<<<<<<< HEAD
REGISTER_DISPATCH(i0_stub, &i0_kernel_mps)
REGISTER_DISPATCH(special_i1_stub, &i1_kernel_mps)
=======
static void bessel_j0_kernel_mps(TensorIteratorBase& iter) {
  lib.exec_unary_kernel(iter, "bessel_j0_forward");
}

static void bessel_j1_kernel_mps(TensorIteratorBase& iter) {
  lib.exec_unary_kernel(iter, "bessel_j1_forward");
}

static void modified_bessel_i0_kernel_mps(TensorIteratorBase& iter) {
  lib.exec_unary_kernel(iter, "modified_bessel_i0_forward");
}

static void modified_bessel_i1_kernel_mps(TensorIteratorBase& iter) {
  lib.exec_unary_kernel(iter, "modified_bessel_i1_forward");
}

static void modified_bessel_k0_kernel_mps(TensorIteratorBase& iter) {
  lib.exec_unary_kernel(iter, "modified_bessel_k0_forward");
}

static void modified_bessel_k1_kernel_mps(TensorIteratorBase& iter) {
  lib.exec_unary_kernel(iter, "modified_bessel_k1_forward");
}

static void scaled_modified_bessel_k0_kernel_mps(TensorIteratorBase& iter) {
  lib.exec_unary_kernel(iter, "scaled_modified_bessel_k0_forward");
}

static void scaled_modified_bessel_k1_kernel_mps(TensorIteratorBase& iter) {
  lib.exec_unary_kernel(iter, "scaled_modified_bessel_k1_forward");
}

static void bessel_y0_kernel_mps(TensorIteratorBase& iter) {
  lib.exec_unary_kernel(iter, "bessel_y0_forward");
}

static void bessel_y1_kernel_mps(TensorIteratorBase& iter) {
  lib.exec_unary_kernel(iter, "bessel_y1_forward");
}

REGISTER_DISPATCH(i0_stub, &i0_kernel_mps)
REGISTER_DISPATCH(special_i0e_stub, &i0e_kernel_mps)
REGISTER_DISPATCH(special_i1_stub, &i1_kernel_mps)
REGISTER_DISPATCH(special_i1e_stub, &i1e_kernel_mps)
REGISTER_DISPATCH(special_bessel_j0_stub, &bessel_j0_kernel_mps)
REGISTER_DISPATCH(special_bessel_j1_stub, &bessel_j1_kernel_mps)
REGISTER_DISPATCH(special_modified_bessel_i0_stub, &modified_bessel_i0_kernel_mps)
REGISTER_DISPATCH(special_modified_bessel_i1_stub, &modified_bessel_i1_kernel_mps)
REGISTER_DISPATCH(special_modified_bessel_k0_stub, &modified_bessel_k0_kernel_mps)
REGISTER_DISPATCH(special_modified_bessel_k1_stub, &modified_bessel_k1_kernel_mps)
REGISTER_DISPATCH(special_scaled_modified_bessel_k0_stub, &scaled_modified_bessel_k0_kernel_mps)
REGISTER_DISPATCH(special_scaled_modified_bessel_k1_stub, &scaled_modified_bessel_k1_kernel_mps)
REGISTER_DISPATCH(special_bessel_y0_stub, &bessel_y0_kernel_mps)
REGISTER_DISPATCH(special_bessel_y1_stub, &bessel_y1_kernel_mps)
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
REGISTER_DISPATCH(special_spherical_bessel_j0_stub, &spherical_bessel_j0_kernel_mps)
REGISTER_DISPATCH(special_entr_stub, &entr_kernel_mps)
} // namespace at::native
