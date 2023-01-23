#pragma once

template <typename T, int we, int wm, bool PRINT_KERNEL_INFO> __global__ void Quant8_inplace(T* _p, int32_t count, bool stoch, uint32_t seed);

void Quant8_inplace_host(__half* _p, int32_t count, uint32_t seed, hipStream_t stream, bool f152);
void Quant8_inplace_host(float* _p, int32_t count, uint32_t seed, hipStream_t stream, bool f152);
