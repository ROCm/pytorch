/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <numeric>

#undef __HIP_NO_HALF_CONVERSIONS__

#include <ATen/ATen.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <torch/torch.h>
#include <ATen/native/hip/ck_gemm.h>


#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/utility/loop_scheduler.hpp"

#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_abd_xdl_cshuffle.hpp"


// Define commonly used types.
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
//using Add = ck::tensor_operation::element_wise::Add; // don't need?

namespace at::native {

template <>
void gemm_internal_ck<double>(CUDABLAS_GEMM_ARGTYPES(double)) {
  return;
}

template <>
void gemm_internal_ck<at::Half>(CUDABLAS_GEMM_ARGTYPES(at::Half)) {
  return;
}

template <>
void gemm_internal_ck<at::BFloat16>(CUDABLAS_GEMM_ARGTYPES(at::BFloat16)) {
  return;
}

template <typename T>
struct CkMathType {
  using dtype = T;
};

template <>
struct CkMathType<at::BFloat16> {
  using dtype = ck::bhalf_t;
};

template <>
struct CkMathType<at::Half> {
  using dtype = ck::half_t;
};


// Elementwise Operators
struct AddScale
{
  __host__ __device__ constexpr void
  operator()(float& a, const float& a0) const
  {
    a = scale * a0;
  }

  float scale = 1.0;
};

struct AlphaBetaAdd
{
  AlphaBetaAdd(float alpha, float beta) : alpha_(alpha), beta_(beta){};

  template <typename E, typename C, typename D>
  __host__ __device__ constexpr void operator()(E& e, const C& c, const D& d) const;

  template<>
  __host__ __device__ constexpr void operator()<float, float, float>
    (float& e, const float& c, const float& d) const
    {
      e = alpha_ * c + beta_ * d;
    };

    float alpha_;
    float beta_;
};




template <
    typename Dtype,
    int BLOCK_SIZE,
    int MBLOCK,
    int NBLOCK,
    int KBLOCK,
    int MPER_XDL,
    int NPER_XDL,
    int MPER_WAVE,
    int NPER_WAVE,
    int CNPER_WAVE = 1,
    bool PADDING = false>
void float_gemm_impl(CUDABLAS_GEMM_ARGTYPES(Dtype)) {
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << "INSIDE FLOAT_GEMM_IMPL" << std::endl;
  // Get input information.
  int M = m;
  int N = n;
  int K = k;

  int StrideA = lda;
  int StrideB = ldb;
  int StrideC = ldc;
  int StrideD = N;

  float falpha = alpha;
  float fbeta = beta;


  using ADataType = typename CkMathType<Dtype>::dtype;
  using BDataType = typename CkMathType<Dtype>::dtype;
  using CDataType = typename CkMathType<Dtype>::dtype;
  using DDataType = typename CkMathType<Dtype>::dtype;

  using AccDataType = float;
  using CShuffleDataType = typename CkMathType<Dtype>::dtype;



#if 0
  using ALayout = Row;
  using BLayout = Col;
  using DLayout = Row;
  using CLayout = Row;
#else
  using ALayout = Col;
  using BLayout = Col;
  using CLayout = Col;
  using DLayout = Col;
#endif


  // These element ops will probably need to change
  using AElementOp = PassThrough;
  using BElementOp = PassThrough;
  using CDEElementOp = AlphaBetaAdd;


  static constexpr auto GemmDefault =
      ck::tensor_operation::device::GemmSpecialization::Default;
  static constexpr auto GemmMNKPadding =
      ck::tensor_operation::device::GemmSpecialization::MNKPadding;
  //static constexpr auto GemmSpec = PADDING ? GemmMNKPadding : GemmDefault;
  static constexpr auto GemmSpec = GemmMNKPadding;
  // Define derivative constants based on template parameters.
  static constexpr int BLOCK_CLUSTER = BLOCK_SIZE / 4;
  static constexpr int CBLOCK_N = NBLOCK / 16;
  static constexpr int CBLOCK_M = BLOCK_SIZE / CBLOCK_N;


  static constexpr auto LoopSched = ck::make_default_loop_scheduler();



  using DeviceGemmV2Instance =
      ck::tensor_operation::device::DeviceGemmMultipleABD_Xdl_CShuffle<
          ck::Tuple<ALayout>,// ALayout
          ck::Tuple<BLayout>,		// BLayout
          ck::Tuple<DLayout>,		// DLayout
          CLayout,					// CLayout
          ck::Tuple<ADataType>,	// ADataType
          ck::Tuple<BDataType>,				// BDataType
          AccDataType,				// AccDataType
          CShuffleDataType,			// CShuffleDataType
          ck::Tuple<DDataType>,		// DDataType
          CDataType,				// CDataType
          AElementOp,				// AElementOp
          BElementOp,				// BElementOp
          CDEElementOp,				// CDEElementOp
          GemmSpec,					// GemmSpec
          1,						// Prefetch stage
          BLOCK_SIZE,				// BLOCK_SIZE
          MBLOCK,					// MBLOCK
          NBLOCK,					// NBLOCK
          KBLOCK,					// KBLOCK
          8, 						// AK1
          8, 						// BK1
          MPER_XDL,					// MPerXDL
          NPER_XDL,					// NPerXDL
          MPER_WAVE, 				// MXdlPerWave
          NPER_WAVE, 				// NXdlPerWave
          S<4, BLOCK_CLUSTER, 1>, 	// ABlockTransferThreadClusterLengths_AK0_M_AK1
          S<1, 0, 2>, 				// ABlockTransferThreadClusterOrder
          S<1, 0, 2>, 				// ABlockTransferSrcAccessOrder
          2,						// ABlockTransferSrcVectorDim
          8,						// ABlockTransferSrcScalarPerVector
          8,						// ABlockTransferDstScalarPerVector_AK1
          1,						// ABlockLdsExtraM
          S<4, BLOCK_CLUSTER, 1>,	// BBlockTransferThreadClusterLengths_BK0_N_BK1
          S<1, 0, 2>,				// BBlockTransferThreadClusterArrangeOrder
          S<1, 0, 2>,				// BBlockTransferSrcAccessOrder
          1,						// BBlockTransferSrcVectorDim
          2,						// BBlockTransferSrcScalarPerVector
          8,						// BBlockTransferDstScalarPerVector_BK1
          1,						// BBlockLdsExtraN
          1,						// CShuffleMXdlPerWavePerShuffle
          1,						// CShuffleNXdlPerWavePerShuffle
          S<1, CBLOCK_M, 1, CBLOCK_N>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
          8>;						// CShuffleBlockTransferScalarPerVector_NPerBlock


  // Create gemm launcher and arguments.
  auto gemm = DeviceGemmV2Instance{};
  auto invoker = gemm.MakeInvoker();

  auto a_element_op = AElementOp{};
  auto b_element_op = BElementOp{};
  auto cde_element_op = CDEElementOp{falpha, fbeta};

  using DDataArrayType = std::array<const void*, 1>;
  using DStrideArrayType = std::array<ck::index_t, 0>;

  DDataArrayType DDataArray;
  DStrideArrayType DStrideArray;
  auto arguments = gemm.MakeArgument(
      std::array<const void*, 1>{reinterpret_cast<const void*>(a)},
      std::array<const void*, 1>{reinterpret_cast<const void*>(b)},
      DDataArray,
      reinterpret_cast<void*>(c),
      M,
      N,
      K,
      std::array<ck::index_t, 1>{StrideA},
      std::array<ck::index_t, 1>{StrideB},
      std::array<ck::index_t, 1>{StrideD},
      StrideC,
      a_element_op,
      b_element_op,
      cde_element_op);

  auto stream = at::cuda::getCurrentHIPStream().stream();
  invoker.Run(arguments, StreamConfig{stream, false});

}

#if 0
    int BLOCK_SIZE,
    int MBLOCK,
    int NBLOCK,
    int KBLOCK,
    int MPER_XDL,
    int NPER_XDL,
    int MPER_WAVE,
    int NPER_WAVE,
    int CNPER_WAVE = 1,
    bool PADDING = false>
#endif
void dispatch_float_gemm(CUDABLAS_GEMM_ARGTYPES(float)) {
  std::cout << "CHECKPOINT 0" << std::endl;
  // If any of the shapes cant be tiled, we must use padding.
  bool use_padding = ((m % 256 != 0) || (n % 128 != 0) || (k % 64 != 0));
  std::cout << "M: " << m << std::endl;
  std::cout << "N: " << n << std::endl;
  std::cout << "K: " << k << std::endl;
  // Dispatch to best implementation. TODO add more configurations.
  if (use_padding) {
    if (m <= 128) {
	  std::cout << "CHECKPOINT ALPHA" << std::endl;
      float_gemm_impl<float, 256, 128, 64, 32, 32, 32, 2, 1, 1, true>(CUDABLAS_GEMM_ARGS(float));
#if 0
    // as float type, get error: local memory (98304) exceeds limit (65536)
    } else if ((m >= 8192 && n >= 4096) || (n >= 8192 && m >= 4096)) {
      float_gemm_impl<float, 256, 256, 128, 64, 16, 16, 8, 4, 2, true>(CUDABLAS_GEMM_ARGS(float));
#endif
    } else {
	  std::cout << "CHECKPOINT BETA" << std::endl;

	  // local memory exceeds error
      //float_gemm_impl<float, 256, 256, 128, 64, 16, 16, 8, 4, 2, true>(CUDABLAS_GEMM_ARGS(float));

	  float_gemm_impl<float, 256, 256, 128, 32, 32, 32, 4, 2, 1, true>(CUDABLAS_GEMM_ARGS(float));
	  std::cout << "BACK FROM BETA" << std::endl;
    }
  } else {
#if 0
    // as float type, get error: local memory (98304) exceeds limit (65536)
    if ((m >= 8192 && n >= 4096) || (n >= 8192 && m >= 4096)) {
      float_gemm_impl<float, 256, 256, 128, 64, 16, 16, 8, 4, 2, false>(CUDABLAS_GEMM_ARGS(float));
    } else
#endif
    {
	  std::cout << "CHECKPOINT GAMMA" << std::endl;
      float_gemm_impl<float, 256, 128, 128, 64, 16, 16, 4, 4, 2, false>(CUDABLAS_GEMM_ARGS(float));
    }
  }
}

template <>
void gemm_internal_ck<float>(CUDABLAS_GEMM_ARGTYPES(float)) {
  std::cout << "ENTRY CHECKPOINT" << std::endl;
  dispatch_float_gemm(CUDABLAS_GEMM_ARGS(float));
}

} // namespace at::native

