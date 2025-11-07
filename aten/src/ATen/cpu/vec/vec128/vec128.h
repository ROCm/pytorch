#pragma once
// ARM NEON uses 128-bit vector registers.

#include <ATen/cpu/vec/intrinsics.h>

#ifdef __aarch64__
#if !defined(CPU_CAPABILITY_SVE)
#include <ATen/cpu/vec/vec128/vec128_bfloat16_neon.h>
<<<<<<< HEAD
#include <ATen/cpu/vec/vec128/vec128_double_neon.h>
#include <ATen/cpu/vec/vec128/vec128_float_neon.h>
#include <ATen/cpu/vec/vec128/vec128_half_neon.h>
#include <ATen/cpu/vec/vec128/vec128_int_aarch64.h>
#include <ATen/cpu/vec/vec128/vec128_uint_aarch64.h>
=======
#include <ATen/cpu/vec/vec128/vec128_float_neon.h>
#include <ATen/cpu/vec/vec128/vec128_half_neon.h>
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
#endif

#include <ATen/cpu/vec/vec128/vec128_convert.h>
#endif
