
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "layernorm2d_fwd_api_common.hpp"

// clang-format off
//                                      prec_i           prec_o           prec_sy           rm  rn  tm    tn  vn  pd     mv    rpcf    2p      add  sweep
template float layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  1,  2,  128,  8, true , false, true , false,    1,    1>>(const S&, A);
template float layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  2,  2,  128,  4, true , false, true , false,    1,    1>>(const S&, A);
template float layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  4,  2,  128,  2, true , false, true , false,    1,    1>>(const S&, A);
template float layernorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::int8_t, float, float,  1,  4,  1,  256,  1, true , false, true , false,    1,    1>>(const S&, A);

// clang-format on

