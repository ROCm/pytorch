# MIT License

# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import re

def gemm_name_parser(kernel_name):
    """
    Parse the kernel name to identify GEMM op details.
    Args:
        kernel_name (str): The name of the kernel.
    Returns:
        dict: A dictionary containing the GEMM operation details.
    """
    if is_rocm_gemm(kernel_name):
        return parse_rocm_gemm(kernel_name)
    # NVIDIA trace: check if any of the known NVIDIA markers exist in the kernel name
    elif is_cuda_gemm(kernel_name):
        return parse_cuda_gemm(kernel_name)

def is_rocm_gemm(kernel_name):
    """
    Check if a kernel name matches the more general ROCm GEMM naming pattern.
    Allows an arbitrary prefix before 'Cijk_Alik_Bljk_...' where each of C/A/B
    is followed by exactly three axis letters.
    Example matches:
      - 'Cijk_Alik_Bljk_...'
      - 'Custom_Cijk_Alik_Bljk_BBS_BH_Bias_AS_SAV_User...'
    """
    pattern = r'^.*C[a-z]{3}_A[a-z]{3}_B[a-z]{3}.*$'
    return bool(re.match(pattern, kernel_name))

def parse_rocm_gemm(kernel_name):

    # 1. Parse the transpose flags from the kernel name
    trans_a, trans_b = None, None
    if "_Ailk_" in kernel_name:
        trans_a = False
    elif "_Alik_" in kernel_name:
        trans_a = True
    if "_Bljk_" in kernel_name:
        trans_b = False
    elif "_Bjlk_" in kernel_name:
        trans_b = True
    
    # 2. Parse the macro tile size from the kernel name
    # Example: ''Cijk_Ailk_Bjlk_BBS_BH_Bias_HAS_SAV_UserArgs_MT64x16x64_MI16x16x1_SN_LDSB0_AFC...'
    # The macro tile size is usually represented by 'MT' followed by the tile dimensions.
    # In this example, the macro tile size is 'MT64x16x64'.
    # 64 is M tile, 16 is N tile, 64 is K loop unroll called DepthU
    macro_tile_match = re.search(r'MT(\d+)x(\d+)x(\d+)', kernel_name)
    if macro_tile_match:
        mt_m = int(macro_tile_match.group(1))
        mt_n = int(macro_tile_match.group(2))
        depth_u = int(macro_tile_match.group(3))
    else:
        mt_m, mt_n, depth_u = None, None, None  # Fallback in case pattern is not found

    # Feel free to add more details as needed.
    # https://github.com/ROCm/Tensile/wiki/Kernel-Parameters#kernel-names

    return {
        'transpose': (trans_a, trans_b),
        'mt_m': mt_m,
        'mt_n': mt_n,
        'depth_u': depth_u,
    }
    
def is_cuda_gemm(kernel_name):
    """
    Check if a kernel name matches the NVIDIA GEMM naming pattern:
    """
    # Right now, we only check if the kernel name starts with 'nvjet'.
    # This is a temporary solution and will be expanded in the future.
    return kernel_name.startswith('nvjet')

def parse_cuda_gemm(kernel_name):
    transpose_chars = kernel_name.split('_')[-1]
    transpose = transpose_chars[0] == 'T', transpose_chars[1] == 'T'
    return {'transpose': transpose}

