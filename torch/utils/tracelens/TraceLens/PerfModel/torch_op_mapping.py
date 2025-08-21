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

from . import perf_model
from collections import defaultdict

op_to_perf_model_class_map = {
    'aten::mm': perf_model.aten_mm,
    'aten::addmm': perf_model.aten_addmm,
    'aten::_scaled_mm': perf_model.aten_scaled_mm,

    # TEv2 pseudo ops
    '_Linear_yfwd_mm': perf_model.tev2_pseudo_gemm,
    '_LinearBackward_xgrad_mm': perf_model.tev2_pseudo_gemm,
    '_LinearBackward_wgrad_mm': perf_model.tev2_pseudo_gemm,
    '_LayerNormLinear_yfwd_mm': perf_model.tev2_pseudo_gemm,
    '_LayerNormLinearBackward_xgrad_mm': perf_model.tev2_pseudo_gemm,
    '_LayerNormLinearBackward_wgrad_mm': perf_model.tev2_pseudo_gemm,

    'aten::bmm': perf_model.aten_bmm,
    'tex_ts::te_gemm_ts': perf_model.tex_ts_te_gemm_ts,
    'aten::baddbmm': perf_model.aten_baddbmm,
    'FlashAttnFunc': perf_model.flash_attention,
    'flash_attn::_flash_attn_forward': perf_model.flash_attention,
    'aten::_scaled_dot_product_cudnn_attention': perf_model.aten__scaled_dot_product_cudnn_attention,
    'aten::_scaled_dot_product_efficient_attention': perf_model.aten__scaled_dot_product_efficient_attention,
    'aten::_scaled_dot_product_flash_attention': perf_model.aten__scaled_dot_product_flash_attention,
    'aten::convolution': perf_model.aten_conv,
    "aiter::_flash_attn_forward": perf_model.aiter__flash_attn_forward,
    "aiter::_flash_attn_backward": perf_model.aiter__flash_attn_backward,
}

unary_elemwise_ops = [
    'aten::copy', 'aten::copy_',
    'aten::clamp_min', 'aten::clamp_min_',
    'aten::clamp_max', 'aten::clamp_max_',
    'aten::sigmoid',
]

binary_elemwise_ops = [
    'aten::div', 'aten::div_',
    'aten::mul', 'aten::mul_',
    'aten::add', 'aten::add_',
    'aten::sigmoid_backward',
    'aten::threshold_backward',
]

for op in unary_elemwise_ops:
    op_to_perf_model_class_map[op] = perf_model.aten_unary_elementwise
for op in binary_elemwise_ops:
    op_to_perf_model_class_map[op] = perf_model.aten_binary_elementwise

dict_base_class2category = {
    perf_model.GEMM: 'GEMM',
    perf_model.CONV: 'CONV',
    perf_model.SDPA: 'SDPA',
    perf_model.UnaryElementwise: 'UnaryElementwise',
    perf_model.BinaryElementwise: 'BinaryElementwise',
}
dict_cat2names = defaultdict(list)
for op_name, perf_model_class in op_to_perf_model_class_map.items():
    base_classes = perf_model_class.__bases__
    assert len(base_classes) == 1, f"op_name: {op_name}, perf_model_class: {perf_model_class}, base_classes: {base_classes}"
    base_class = base_classes[0]
    cat =  dict_base_class2category.get(base_class)
    if cat is None:
        raise ValueError(f"op_name: {op_name}, perf_model_class: {perf_model_class}, base_class: {base_classes}")
    dict_cat2names[cat].append(op_name)

def categorize_torch_op(row):
    """
    Categorizes a row based on the 'name' and 'kernel_names' fields.
    Args:
        row (dict): A dictionary representing a row with 'name' and 'kernel_names' keys.
    Returns:
        str: The category of the row, which can be one of 'GEMM', 'CONV_fwd', 'CONV_bwd', 'BN_fwd', 'BN_bwd',
             'SDPA_fwd', 'SDPA_bwd', 'triton', 'elementwise', 'reduce', 'multi_tensor_apply', or 'other'.
    """

    debug = False
    if row['name'] in dict_cat2names['GEMM']:
        return 'GEMM'
    elif row['name'] in ['aten::convolution', 
                         'aten::miopen_convolution', 'aten::cudnn_convolution']:
        return 'CONV_fwd'
    elif row['name'] == 'aten::convolution_backward':
        return 'CONV_bwd'
    elif row['name'] in ['aten::batch_norm',
                         'aten::native_batch_norm', 
                         'aten::miopen_batch_norm', 'aten::cudnn_batch_norm']:
        return 'BN_fwd'
    elif row['name'] in ['aten::native_batch_norm_backward', 
                         'aten::miopen_batch_norm_backward', 'aten::cudnn_batch_norm_backward']:
        return 'BN_bwd'
    # SDPA ops: distinguish forward and backward
    sdpa_bwd_names = [
        "FlashAttnFuncBackward",
        "flash_attn::_flash_attn_backward",
        "aten::_scaled_dot_product_cudnn_attention_backward",
        "aten::_scaled_dot_product_efficient_attention_backward",
        "aten::_scaled_dot_product_flash_attention_backward",
        "aiter::_flash_attn_backward",
    ]
    if row["name"] in dict_cat2names["SDPA"]:
        if row["name"].endswith("_backward") or row["name"] in sdpa_bwd_names:
            return "SDPA_bwd"
        else:
            return "SDPA_fwd"
    elif row['name'].startswith('triton'):
        return 'triton'
    if 'kernel_details' in row and len(row['kernel_details']) > 0:
        kernel_name = row['kernel_details'][0]['name']
    else:
        raise ValueError("Row does not contain 'kernel_names' or 'kernel_details' with a valid name.")
    if kernel_name.startswith('void at::native'):
        if debug:
            print("Found ATen native kernel:", kernel_name[:64])
        if 'elementwise' in kernel_name:
            return 'elementwise'
        elif 'reduce' in kernel_name:
            return 'reduce'
        elif 'multi_tensor_apply' in kernel_name:
            return 'multi_tensor_apply'
    # if none of the above cases match, return 'other'
    return 'other'