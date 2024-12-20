
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <ck_tile/core.hpp>
#include "layernorm2d_fwd.hpp"


// this is used to pattern-match internl kernel implementation, not to instantiate kernel
template <typename XDataType_,
          typename YDataType_,
          typename XScaleDataType_,
          typename YScaleDataType_,
          ck_tile::index_t Repeat_M_,         // each thread repeat along M
          ck_tile::index_t Repeat_N_,         // each thread repeat along N
          ck_tile::index_t ThreadPerBlock_M_, // num threads along M
          ck_tile::index_t ThreadPerBlock_N_, // num threads along N
          ck_tile::index_t Vector_N_,         // vector size along N
          bool kPadN_,
          bool kSaveMeanInvStd_,
          bool kFastFDiv_,
          bool kTwoPass_,
          ck_tile::index_t kFusedAdd_ = 0,
          ck_tile::index_t kFusedQuant_ = 0>
struct layernorm2d_fwd_traits_
{
    using XDataType = ck_tile::remove_cvref_t<XDataType_>;
    using YDataType = ck_tile::remove_cvref_t<YDataType_>;
    using XScaleDataType = ck_tile::remove_cvref_t<XScaleDataType_>;
    using YScaleDataType = ck_tile::remove_cvref_t<YScaleDataType_>;

    static constexpr bool is_warp_per_row = ThreadPerBlock_N_ <= warpSize;
    static_assert((ThreadPerBlock_M_ * ThreadPerBlock_N_) % warpSize == 0);
    static constexpr ck_tile::index_t total_warps =
        (ThreadPerBlock_M_ * ThreadPerBlock_N_) / warpSize;

    // num of warps along m
    static constexpr ck_tile::index_t BlockWarps_M = []() {
        if constexpr(is_warp_per_row)
        {
            static_assert(warpSize % ThreadPerBlock_N_ == 0);
            return total_warps * (warpSize / ThreadPerBlock_N_);
        }
        else
        {
            // static_assert(warpSize % ThreadPerBlock_M_ == 0);
            return total_warps / (ThreadPerBlock_N_ / warpSize);
        }
    }();

    // num of warps along n
    static constexpr ck_tile::index_t BlockWarps_N = []() {
        if constexpr(is_warp_per_row)
        {
            static_assert(warpSize % ThreadPerBlock_N_ == 0);
            return 1;
        }
        else
        {
            static_assert(ThreadPerBlock_N_ % warpSize == 0);
            return ThreadPerBlock_N_ / warpSize;
        }
    }();

    static constexpr ck_tile::index_t Repeat_M = Repeat_M_;
    static constexpr ck_tile::index_t Repeat_N = Repeat_N_;

    static constexpr ck_tile::index_t Block_M = Repeat_M_ * ThreadPerBlock_M_;
    static constexpr ck_tile::index_t Block_N = Repeat_N_ * ThreadPerBlock_N_ * Vector_N_;

    static constexpr ck_tile::index_t Warp_M = ThreadPerBlock_M_ / BlockWarps_M;
    static constexpr ck_tile::index_t Warp_N = ThreadPerBlock_N_ / BlockWarps_N * Vector_N_;

    using BlockTile  = ck_tile::sequence<Block_M, Block_N>;
    using BlockWarps = ck_tile::sequence<BlockWarps_M, BlockWarps_N>;
    using WarpTile   = ck_tile::sequence<Warp_M, Warp_N>;
    using Vector     = ck_tile::sequence<1, Vector_N_>;

    using Shape = ck_tile::Generic2dBlockShape<BlockTile, BlockWarps, WarpTile, Vector>;

    static constexpr bool kPadN           = kPadN_;
    static constexpr bool kSaveMeanInvStd = kSaveMeanInvStd_;
    static constexpr bool kFastFDiv       = kFastFDiv_;
    static constexpr bool kTwoPass        = kTwoPass_;
    static constexpr ck_tile::index_t kFusedAdd = kFusedAdd_;
    static constexpr ck_tile::index_t kFusedQuant = kFusedQuant_;
};

template <typename XDataType_,
          typename YDataType_,
          typename XScaleDataType_,
          typename YScaleDataType_,
          ck_tile::index_t Repeat_M_,         // each thread repeat along M
          ck_tile::index_t Repeat_N_,         // each thread repeat along N
          ck_tile::index_t ThreadPerBlock_M_, // num threads along M
          ck_tile::index_t ThreadPerBlock_N_, // num threads along N
          ck_tile::index_t Vector_N_,         // vector size along N
          bool kPadN_,
          bool kSaveMeanInvStd_,
          bool kFastFDiv_,
          bool kTwoPass_,
          int  kFusedAdd_,
          int  kFusedQuant_>
using traits_ = layernorm2d_fwd_traits_<XDataType_,
                                       YDataType_,
                                       XScaleDataType_,
                                       YScaleDataType_,
                                       Repeat_M_,
                                       Repeat_N_,
                                       ThreadPerBlock_M_,
                                       ThreadPerBlock_N_,
                                       Vector_N_,
                                       kPadN_,
                                       kSaveMeanInvStd_,
                                       kFastFDiv_,
                                       kTwoPass_,
                                       kFusedAdd_,
                                       kFusedQuant_>;


// Note: this internal API only declare, not define here, otherwise will block `make -j`
template <typename Traits_>
float layernorm2d_fwd_(const ck_tile::stream_config& s, layernorm2d_fwd_args a);

float layernorm2d_fwd(layernorm2d_fwd_traits t,
                      layernorm2d_fwd_args a,
                      const ck_tile::stream_config& s)
{
    float r = -1;
    if(t.prec_i == "fp16" && t.prec_o == "fp16"){
        if (a.n <= 64) {
            if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  1,  8,    8,  8, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  1,  4,   16,  4, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  1,  4,   64,  1, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  1,  8,    8,  8, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  1,  4,   16,  4, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  1,  4,   64,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  1,  8,    8,  8, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  1,  4,   16,  4, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  1,  4,   64,  1, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  1,  8,    8,  8, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  1,  4,   16,  4, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  1,  4,   64,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else if (a.n <= 128) {
            if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  1,  4,   16,  8, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  1,  4,   64,  2, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  2,  4,   64,  1, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  1,  4,   16,  8, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  1,  4,   64,  2, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  2,  4,   64,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  1,  4,   16,  8, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  1,  4,   64,  2, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  2,  4,   64,  1, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  1,  4,   16,  8, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  1,  4,   64,  2, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  2,  4,   64,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else if (a.n <= 256) {
            if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  1,  4,   64,  4, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  2,  4,   64,  2, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  4,   64,  1, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  1,  4,   64,  4, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  2,  4,   64,  2, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  4,   64,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  1,  4,   64,  4, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  2,  4,   64,  2, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  4,   64,  1, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  1,  4,   64,  4, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  2,  4,   64,  2, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  4,   64,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else if (a.n <= 512) {
            if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  1,  4,   64,  8, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  2,  4,   64,  4, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  4,   64,  2, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  8,  4,   64,  1, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  1,  4,   64,  8, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  2,  4,   64,  4, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  4,   64,  2, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  8,  4,   64,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  1,  4,   64,  8, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  2,  4,   64,  4, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  4,   64,  2, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  8,  4,   64,  1, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  1,  4,   64,  8, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  2,  4,   64,  4, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  4,   64,  2, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  8,  4,   64,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else if (a.n <= 768) {
            if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  3,  4,   64,  4, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  6,  4,   64,  2, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1, 12,  4,   64,  1, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  3,  4,   64,  4, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  6,  4,   64,  2, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1, 12,  4,   64,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  3,  4,   64,  4, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  6,  4,   64,  2, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1, 12,  4,   64,  1, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  3,  4,   64,  4, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  6,  4,   64,  2, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1, 12,  4,   64,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else if (a.n <= 1024) {
            if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  1,  2,  128,  8, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  2,  2,  128,  4, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  2,  128,  2, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  1,  256,  1, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  1,  2,  128,  8, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  2,  2,  128,  4, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  2,  128,  2, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  1,  256,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  1,  2,  128,  8, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  2,  2,  128,  4, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  2,  128,  2, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  1,  256,  1, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  1,  2,  128,  8, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  2,  2,  128,  4, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  2,  128,  2, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  1,  256,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else if (a.n <= 1536) {
            if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  3,  4,   64,  8, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  3,  2,  128,  4, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  3,  1,  256,  2, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  6,  1,  256,  1, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  3,  4,   64,  8, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  3,  2,  128,  4, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  3,  1,  256,  2, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  6,  1,  256,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  3,  4,   64,  8, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  3,  2,  128,  4, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  3,  1,  256,  2, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  6,  1,  256,  1, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  3,  4,   64,  8, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  3,  2,  128,  4, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  3,  1,  256,  2, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  6,  1,  256,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else if (a.n <= 2048) {
            if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  1,  1,  256,  8, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  2,  1,  256,  4, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  1,  256,  2, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  8,  1,  256,  1, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  1,  1,  256,  8, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  2,  1,  256,  4, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  1,  256,  2, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  8,  1,  256,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  1,  1,  256,  8, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  2,  1,  256,  4, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  1,  256,  2, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  8,  1,  256,  1, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  1,  1,  256,  8, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  2,  1,  256,  4, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  1,  256,  2, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  8,  1,  256,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else if (a.n <= 3072) {
            if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  3,  1,  128,  8, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  3,  1,  256,  4, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  6,  1,  256,  2, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  3,  1, 1024,  1, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  3,  1,  128,  8, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  3,  1,  256,  4, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  6,  1,  256,  2, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  3,  1, 1024,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  3,  1,  128,  8, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  3,  1,  256,  4, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  6,  1,  256,  2, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  3,  1, 1024,  1, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  3,  1,  128,  8, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  3,  1,  256,  4, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  6,  1,  256,  2, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  3,  1, 1024,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else if (a.n <= 4096) {
            if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  2,  1,  256,  8, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  1,  256,  4, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  2,  1, 1024,  2, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  1, 1024,  1, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  2,  1,  256,  8, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  1,  256,  4, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  2,  1, 1024,  2, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  1, 1024,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  2,  1,  256,  8, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  1,  256,  4, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  2,  1, 1024,  2, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  1, 1024,  1, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  2,  1,  256,  8, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  1,  256,  4, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  2,  1, 1024,  2, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  1, 1024,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else if (a.n <= 6144) {
            if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  3,  1,  256,  8, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  3,  1,  512,  4, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  3,  1, 1024,  2, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  6,  1, 1024,  1, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  3,  1,  256,  8, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  3,  1,  512,  4, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  3,  1, 1024,  2, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  6,  1, 1024,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  3,  1,  256,  8, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  3,  1,  512,  4, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  3,  1, 1024,  2, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  6,  1, 1024,  1, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  3,  1,  256,  8, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  3,  1,  512,  4, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  3,  1, 1024,  2, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  6,  1, 1024,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else if (a.n <= 8192) {
            if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  1,  256,  8, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  1,  512,  4, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  1, 1024,  2, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  8,  1, 1024,  1, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  1,  256,  8, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  1,  512,  4, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  1, 1024,  2, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  8,  1, 1024,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  1,  256,  8, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  1,  512,  4, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  1, 1024,  2, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  8,  1, 1024,  1, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  1,  256,  8, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  1,  512,  4, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  1, 1024,  2, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  8,  1, 1024,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else  {
            if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  2,  1,  256,  8, true , false, true , true ,    0,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  1,  256,  4, true , false, true , true ,    0,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  2,  1, 1024,  2, true , false, true , true ,    0,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  1, 1024,  1, true , false, true , true ,    0,    0>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  2,  1,  256,  8, true , false, true , true ,    1,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  1,  256,  4, true , false, true , true ,    1,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  2,  1, 1024,  2, true , false, true , true ,    1,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  1, 1024,  1, true , false, true , true ,    1,    0>>(s, a);

        }

    }
    else if(t.prec_i == "bf16" && t.prec_o == "bf16"){
        if (a.n <= 64) {
            if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  1,  8,    8,  8, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  1,  4,   16,  4, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  1,  4,   64,  1, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  1,  8,    8,  8, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  1,  4,   16,  4, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  1,  4,   64,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  1,  8,    8,  8, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  1,  4,   16,  4, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  1,  4,   64,  1, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  1,  8,    8,  8, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  1,  4,   16,  4, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  1,  4,   64,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else if (a.n <= 128) {
            if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  1,  4,   16,  8, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  1,  4,   64,  2, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  2,  4,   64,  1, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  1,  4,   16,  8, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  1,  4,   64,  2, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  2,  4,   64,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  1,  4,   16,  8, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  1,  4,   64,  2, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  2,  4,   64,  1, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  1,  4,   16,  8, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  1,  4,   64,  2, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  2,  4,   64,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else if (a.n <= 256) {
            if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  1,  4,   64,  4, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  2,  4,   64,  2, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  4,  4,   64,  1, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  1,  4,   64,  4, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  2,  4,   64,  2, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  4,  4,   64,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  1,  4,   64,  4, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  2,  4,   64,  2, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  4,  4,   64,  1, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  1,  4,   64,  4, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  2,  4,   64,  2, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  4,  4,   64,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else if (a.n <= 512) {
            if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  1,  4,   64,  8, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  2,  4,   64,  4, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  4,  4,   64,  2, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  8,  4,   64,  1, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  1,  4,   64,  8, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  2,  4,   64,  4, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  4,  4,   64,  2, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  8,  4,   64,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  1,  4,   64,  8, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  2,  4,   64,  4, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  4,  4,   64,  2, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  8,  4,   64,  1, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  1,  4,   64,  8, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  2,  4,   64,  4, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  4,  4,   64,  2, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  8,  4,   64,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else if (a.n <= 768) {
            if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  3,  4,   64,  4, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  6,  4,   64,  2, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1, 12,  4,   64,  1, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  3,  4,   64,  4, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  6,  4,   64,  2, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1, 12,  4,   64,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  3,  4,   64,  4, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  6,  4,   64,  2, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1, 12,  4,   64,  1, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  3,  4,   64,  4, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  6,  4,   64,  2, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1, 12,  4,   64,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else if (a.n <= 1024) {
            if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  1,  2,  128,  8, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  2,  2,  128,  4, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  4,  2,  128,  2, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  4,  1,  256,  1, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  1,  2,  128,  8, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  2,  2,  128,  4, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  4,  2,  128,  2, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  4,  1,  256,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  1,  2,  128,  8, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  2,  2,  128,  4, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  4,  2,  128,  2, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  4,  1,  256,  1, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  1,  2,  128,  8, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  2,  2,  128,  4, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  4,  2,  128,  2, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  4,  1,  256,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else if (a.n <= 1536) {
            if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  3,  4,   64,  8, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  3,  2,  128,  4, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  3,  1,  256,  2, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  6,  1,  256,  1, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  3,  4,   64,  8, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  3,  2,  128,  4, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  3,  1,  256,  2, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  6,  1,  256,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  3,  4,   64,  8, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  3,  2,  128,  4, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  3,  1,  256,  2, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  6,  1,  256,  1, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  3,  4,   64,  8, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  3,  2,  128,  4, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  3,  1,  256,  2, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  6,  1,  256,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else if (a.n <= 2048) {
            if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  1,  1,  256,  8, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  2,  1,  256,  4, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  4,  1,  256,  2, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  8,  1,  256,  1, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  1,  1,  256,  8, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  2,  1,  256,  4, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  4,  1,  256,  2, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  8,  1,  256,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  1,  1,  256,  8, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  2,  1,  256,  4, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  4,  1,  256,  2, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  8,  1,  256,  1, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  1,  1,  256,  8, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  2,  1,  256,  4, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  4,  1,  256,  2, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  8,  1,  256,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else if (a.n <= 3072) {
            if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  3,  1,  128,  8, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  3,  1,  256,  4, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  6,  1,  256,  2, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  3,  1, 1024,  1, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  3,  1,  128,  8, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  3,  1,  256,  4, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  6,  1,  256,  2, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  3,  1, 1024,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  3,  1,  128,  8, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  3,  1,  256,  4, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  6,  1,  256,  2, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  3,  1, 1024,  1, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  3,  1,  128,  8, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  3,  1,  256,  4, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  6,  1,  256,  2, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  3,  1, 1024,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else if (a.n <= 4096) {
            if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  2,  1,  256,  8, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  4,  1,  256,  4, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  2,  1, 1024,  2, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  4,  1, 1024,  1, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  2,  1,  256,  8, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  4,  1,  256,  4, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  2,  1, 1024,  2, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  4,  1, 1024,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  2,  1,  256,  8, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  4,  1,  256,  4, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  2,  1, 1024,  2, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  4,  1, 1024,  1, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  2,  1,  256,  8, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  4,  1,  256,  4, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  2,  1, 1024,  2, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  4,  1, 1024,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else if (a.n <= 6144) {
            if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  3,  1,  256,  8, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  3,  1,  512,  4, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  3,  1, 1024,  2, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  6,  1, 1024,  1, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  3,  1,  256,  8, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  3,  1,  512,  4, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  3,  1, 1024,  2, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  6,  1, 1024,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  3,  1,  256,  8, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  3,  1,  512,  4, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  3,  1, 1024,  2, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  6,  1, 1024,  1, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  3,  1,  256,  8, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  3,  1,  512,  4, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  3,  1, 1024,  2, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  6,  1, 1024,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else if (a.n <= 8192) {
            if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  4,  1,  256,  8, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  4,  1,  512,  4, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  4,  1, 1024,  2, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  8,  1, 1024,  1, true , false, true , false,    0,    0>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  4,  1,  256,  8, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  4,  1,  512,  4, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  4,  1, 1024,  2, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  8,  1, 1024,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  4,  1,  256,  8, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  4,  1,  512,  4, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  4,  1, 1024,  2, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  8,  1, 1024,  1, true , false, true , false,    1,    0>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  4,  1,  256,  8, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  4,  1,  512,  4, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  4,  1, 1024,  2, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  8,  1, 1024,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else  {
            if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  2,  1,  256,  8, true , false, true , true ,    0,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  4,  1,  256,  4, true , false, true , true ,    0,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  2,  1, 1024,  2, true , false, true , true ,    0,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  4,  1, 1024,  1, true , false, true , true ,    0,    0>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  2,  1,  256,  8, true , false, true , true ,    1,    0>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  4,  1,  256,  4, true , false, true , true ,    1,    0>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  2,  1, 1024,  2, true , false, true , true ,    1,    0>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 0))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float,  1,  4,  1, 1024,  1, true , false, true , true ,    1,    0>>(s, a);

        }

    }
    else if(t.prec_i == "fp16" && t.prec_o == "int8"){
        if (a.n <= 64) {
            if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  1,  8,    8,  8, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  1,  4,   16,  4, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  1,  4,   64,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  1,  8,    8,  8, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  1,  4,   16,  4, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  1,  4,   64,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else if (a.n <= 128) {
            if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  1,  4,   16,  8, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  1,  4,   64,  2, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  2,  4,   64,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  1,  4,   16,  8, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  1,  4,   64,  2, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  2,  4,   64,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else if (a.n <= 256) {
            if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  1,  4,   64,  4, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  2,  4,   64,  2, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  4,  4,   64,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  1,  4,   64,  4, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  2,  4,   64,  2, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  4,  4,   64,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else if (a.n <= 512) {
            if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  1,  4,   64,  8, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  2,  4,   64,  4, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  4,  4,   64,  2, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  8,  4,   64,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  1,  4,   64,  8, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  2,  4,   64,  4, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  4,  4,   64,  2, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  8,  4,   64,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else if (a.n <= 768) {
            if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  3,  4,   64,  4, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  6,  4,   64,  2, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1, 12,  4,   64,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  3,  4,   64,  4, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  6,  4,   64,  2, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1, 12,  4,   64,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else if (a.n <= 1024) {
            if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  1,  2,  128,  8, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  2,  2,  128,  4, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  4,  2,  128,  2, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  4,  1,  256,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  1,  2,  128,  8, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  2,  2,  128,  4, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  4,  2,  128,  2, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  4,  1,  256,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else if (a.n <= 1536) {
            if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  3,  4,   64,  8, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  3,  2,  128,  4, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  3,  1,  256,  2, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  6,  1,  256,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  3,  4,   64,  8, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  3,  2,  128,  4, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  3,  1,  256,  2, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  6,  1,  256,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else if (a.n <= 2048) {
            if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  1,  1,  256,  8, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  2,  1,  256,  4, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  4,  1,  256,  2, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  8,  1,  256,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  1,  1,  256,  8, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  2,  1,  256,  4, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  4,  1,  256,  2, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  8,  1,  256,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else if (a.n <= 3072) {
            if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  3,  1,  128,  8, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  3,  1,  256,  4, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  6,  1,  256,  2, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  3,  1, 1024,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  3,  1,  128,  8, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  3,  1,  256,  4, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  6,  1,  256,  2, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  3,  1, 1024,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else if (a.n <= 4096) {
            if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  2,  1,  256,  8, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  4,  1,  256,  4, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  2,  1, 1024,  2, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  4,  1, 1024,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  2,  1,  256,  8, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  4,  1,  256,  4, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  2,  1, 1024,  2, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  4,  1, 1024,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else if (a.n <= 6144) {
            if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  3,  1,  256,  8, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  3,  1,  512,  4, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  3,  1, 1024,  2, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  6,  1, 1024,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  3,  1,  256,  8, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  3,  1,  512,  4, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  3,  1, 1024,  2, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  6,  1, 1024,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else  {
            if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  4,  1,  256,  8, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  4,  1,  512,  4, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  4,  1, 1024,  2, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  8,  1, 1024,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  4,  1,  256,  8, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  4,  1,  512,  4, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  4,  1, 1024,  2, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  8,  1, 1024,  1, true , false, true , false,    1,    1>>(s, a);

        }

    }
    else if(t.prec_i == "bf16" && t.prec_o == "int8"){
        if (a.n <= 64) {
            if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  1,  8,    8,  8, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  1,  4,   16,  4, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  1,  4,   64,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  1,  8,    8,  8, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  1,  4,   16,  4, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  1,  4,   64,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else if (a.n <= 128) {
            if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  1,  4,   16,  8, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  1,  4,   64,  2, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  2,  4,   64,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  1,  4,   16,  8, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  1,  4,   64,  2, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  2,  4,   64,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else if (a.n <= 256) {
            if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  1,  4,   64,  4, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  2,  4,   64,  2, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  4,  4,   64,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  1,  4,   64,  4, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  2,  4,   64,  2, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  4,  4,   64,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else if (a.n <= 512) {
            if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  1,  4,   64,  8, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  2,  4,   64,  4, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  4,  4,   64,  2, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  8,  4,   64,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  1,  4,   64,  8, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  2,  4,   64,  4, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  4,  4,   64,  2, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  8,  4,   64,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else if (a.n <= 768) {
            if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  3,  4,   64,  4, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  6,  4,   64,  2, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1, 12,  4,   64,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  3,  4,   64,  4, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  6,  4,   64,  2, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1, 12,  4,   64,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else if (a.n <= 1024) {
            if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  1,  2,  128,  8, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  2,  2,  128,  4, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  4,  2,  128,  2, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  4,  1,  256,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  1,  2,  128,  8, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  2,  2,  128,  4, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  4,  2,  128,  2, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  4,  1,  256,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else if (a.n <= 1536) {
            if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  3,  4,   64,  8, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  3,  2,  128,  4, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  3,  1,  256,  2, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  6,  1,  256,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  3,  4,   64,  8, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  3,  2,  128,  4, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  3,  1,  256,  2, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  6,  1,  256,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else if (a.n <= 2048) {
            if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  1,  1,  256,  8, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  2,  1,  256,  4, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  4,  1,  256,  2, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  8,  1,  256,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  1,  1,  256,  8, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  2,  1,  256,  4, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  4,  1,  256,  2, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  8,  1,  256,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else if (a.n <= 3072) {
            if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  3,  1,  128,  8, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  3,  1,  256,  4, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  6,  1,  256,  2, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  3,  1, 1024,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  3,  1,  128,  8, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  3,  1,  256,  4, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  6,  1,  256,  2, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  3,  1, 1024,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else if (a.n <= 4096) {
            if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  2,  1,  256,  8, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  4,  1,  256,  4, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  2,  1, 1024,  2, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  4,  1, 1024,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  2,  1,  256,  8, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  4,  1,  256,  4, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  2,  1, 1024,  2, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  4,  1, 1024,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else if (a.n <= 6144) {
            if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  3,  1,  256,  8, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  3,  1,  512,  4, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  3,  1, 1024,  2, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  6,  1, 1024,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  3,  1,  256,  8, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  3,  1,  512,  4, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  3,  1, 1024,  2, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  6,  1, 1024,  1, true , false, true , false,    1,    1>>(s, a);

        }
        else  {
            if ((a.n % 8 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  4,  1,  256,  8, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  4,  1,  512,  4, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  4,  1, 1024,  2, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 0) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  8,  1, 1024,  1, true , false, true , false,    0,    1>>(s, a);
            else if ((a.n % 8 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  4,  1,  256,  8, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 4 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  4,  1,  512,  4, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 2 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  4,  1, 1024,  2, true , false, true , false,    1,    1>>(s, a);
            else if ((a.n % 1 == 0) && (t.fused_add == 1) && (t.fused_quant == 1 && (t.prec_sx == "fp32" && t.prec_sy == "fp32")))
                r=layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  8,  1, 1024,  1, true , false, true , false,    1,    1>>(s, a);

        }

    }

    return r;
}

