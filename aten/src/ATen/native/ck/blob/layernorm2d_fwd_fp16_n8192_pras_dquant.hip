
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "layernorm2d_fwd_api_common.hpp"

// clang-format off
//                                      prec_i           prec_o           prec_sy           rm  rn  tm    tn  vn  pd     mv    rpcf    2p      add  sweep
template float layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  1,  256,  8, true , false, true , false,    1,    1>>(const S&, A);
template float layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  1,  512,  4, true , false, true , false,    1,    1>>(const S&, A);
template float layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  4,  1, 1024,  2, true , false, true , false,    1,    1>>(const S&, A);
template float layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float,  1,  8,  1, 1024,  1, true , false, true , false,    1,    1>>(const S&, A);

// clang-format on

