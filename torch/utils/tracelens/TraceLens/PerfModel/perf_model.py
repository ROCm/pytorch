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

from math import prod
import math
import os
import re
import subprocess
import warnings
from .kernel_name_parser import gemm_name_parser

def name2bpe(name):
    """
    This function maps a data type name to the number of bytes per element.
    Args:
        name (str): The name of the data type.
    Returns:
        int: The number of bytes per element.
    """
    dict_bpe2dtype = {
        8: ['double', 'long int'],
        4: ['float', 'scalar'],
        2: ['c10::half', 'c10::bfloat16'],
        1: ['c10::float8_e4m3fnuz', 'c10::float8_e4m3fn', 'c10::float8_e5m2',
            'unsigned char', 'fp8'],
    }
    dict_dtype2bpe = {dtype: bpe for bpe, dtypes in dict_bpe2dtype.items() for dtype in dtypes}
    return dict_dtype2bpe.get(name.lower(), None)

def gemmologist_dtype_map(dtype):
    """
    This function maps a PyTorch data type to a gemmologist data type.
    Args:
        dtype (str): The name of the pytorch data type.
    Returns:
        str: The name of the PyTorch data type.
    """
    dict_dtype2gemmologist = {
        'fp32':'float',
        'fp64':'double',
        'fp16':'c10::half',
        'bf16':'c10::bfloat16',
        'fp8':'c10::float8_e4m3fnuz'
    }
    return dict_dtype2gemmologist.get(dtype.lower(), None)

def torch_dtype_map(dtype):
    """
    This function maps a PyTorch data type to a gemmologist data type.
    Args:
        dtype (str): The name of the PyTorch data type.
    Returns:
        str: The name of the gemmologist data type.
    """
    dict_dtype2gemmologist = {
        'float': 'fp32',
        'double': 'fp64',
        'c10::half': 'fp16',
        'c10::bfloat16': 'bf16',
        'c10::float8_e4m3fnuz': 'fp8',
        'unsigned char': 'fp8',
        'fp8': 'fp8',
    }
    return dict_dtype2gemmologist.get(dtype.lower(), None)

# 1. GEMM
class GEMM:
    """
    This is the base class for all GEMM operations.
    If you want to add a new GEMM operation, you should inherit from this class.
    """
    cache_gemm_results = {}  # This is used to cache gemm results
    def __init__(self, event, arch=None, python_path=None):
        self.event = event
        # parse kernel info (e.g. transpose) before kernel params since it can be needed
        self.parsed_kernel_info = None
        self.arch = arch
        self.python_path = python_path
        if 'kernel_names' in event and len(event['kernel_names']) > 0:
            kernel_names = event['kernel_names']
        elif 'kernel_details' in event and len(event['kernel_details']) > 0:
            kernel_names = [kernel['name'] for kernel in event['kernel_details']]
        for kernel_name in kernel_names:
            # TODO: think you really wanna pass around dicts instead of objects?
            self.parsed_kernel_info = gemm_name_parser(kernel_name)
            if self.parsed_kernel_info is not None:
                break
        self.param_details = self.get_param_details(event)
        if not hasattr(self.param_details, 'B'):
            self.param_details['B'] = 1

        if self.parsed_kernel_info is not None:
            self.param_details['transpose'] = self.parsed_kernel_info['transpose']

        self.B, self.M, self.N, self.K = self.param_details['B'], self.param_details['M'], self.param_details['N'], self.param_details['K']
        self.bias = self.param_details['bias']


        if arch is not None:
            if os.environ.get('GEMMOLOGIST_PATH') is not None:
                if not os.path.exists(os.environ.get('GEMMOLOGIST_PATH')):
                    raise ValueError(f"GEMMOLOGIST_PATH does not exist: {os.environ.get('GEMMOLOGIST_PATH')}")
                dtype = self.param_details.get("gemmologist_dtype")
                if dtype is None:
                    dtype = torch_dtype_map(self.param_details['dtype_A_B'][0])
                self.gemmologist_time, self.gemmologist_cmd = GEMM.get_simulation_time_func(arch, self.M, self.N, self.K, self.B, dtype, self.python_path)
            else:
                # TODO: use naive roofline model
                pass

    @staticmethod
    def get_param_details(event):
        # to be implemented in the child class
        raise NotImplementedError

    @staticmethod
    def flops_func(M, N, K, bias):
        flops_matmul = 2 * M * N * K
        flops_bias = M * N if bias else 0
        return flops_matmul + flops_bias

    def flops(self):
        return self.flops_func(self.M, self.N, self.K, self.bias)

    @staticmethod
    def bytes_func(M, N, K, bias, bpe_mat1, bpe_mat2, bpe_bias, bpe_output):
        #if any of the bpe is None, we will return None
        if None in {bpe_mat1, bpe_mat2, bpe_bias, bpe_output}:
            return None
        bytes_mat1 = M * K * bpe_mat1
        bytes_mat2 = K * N * bpe_mat2
        bytes_output = M * N * bpe_output
        # to be totally accurate we should use the bias shape from profile info
        # but we just assume bias shape as 1xN
        #TODO: use profile info to get the bias shape
        bytes_bias = (N if bias else 0) * bpe_bias
        return bytes_mat1 + bytes_mat2 + bytes_output + bytes_bias
    def bytes(self, bpe_mat1, bpe_mat2, bpe_bias, bpe_output):
        return self.bytes_func(self.M, self.N, self.K, self.bias, bpe_mat1, bpe_mat2, bpe_bias, bpe_output)

    """
    bwd pass for Y = X.matmul(W^T) + B
    X_grad = Y_grad.matmul(W)
    W_grad = Y_grad^T.matmul(X)
    B_grad = Y_grad.sum(dim=0)
    """

    def flops_bwd(self):
        flops_input_grad = self.flops_func(M=self.M, N=self.K, K=self.N, bias=False)
        flops_weight_grad = self.flops_func(M=self.N, N=self.K, K=self.M, bias=False)
        flops_bias_grad = self.M * self.N if self.bias else 0
        return flops_input_grad + flops_weight_grad + flops_bias_grad

    def bytes_bwd(self, bytes_per_element):
        bytes_input_grad = self.bytes_func(M=self.M, N=self.K, K=self.N, bias=False, bytes_per_element=bytes_per_element)
        bytes_weight_grad = self.bytes_func(M=self.N, N=self.K, K=self.M, bias=False, bytes_per_element=bytes_per_element)
        bytes_bias_grad = self.M * self.N if self.bias else 0
        return bytes_input_grad + bytes_weight_grad + bytes_bias_grad

    @staticmethod
    def get_simulation_time_func(arch, M, N, K, B, dtype, python_path=None, force_to_l1=False, num_cus=None):
        missing_inputs = []
        if M is None:
            missing_inputs.append("M")
        if N is None:
            missing_inputs.append("N")
        if K is None:
            missing_inputs.append("K")
        if B is None:
            B = 1
        if dtype is None:
            missing_inputs.append("dtype")
        if "name" not in arch:
            missing_inputs.append("arch['name']")
        assert not missing_inputs, f"Invalid inputs: {', '.join(missing_inputs)} are missing or None"
        # assume that gemmologist path is given in the environment variable GEMMOLOGIST_PATH
        gemmologist_path = os.environ.get('GEMMOLOGIST_PATH')
    
        cmd = [
            "./bin/gemmologist.py",
            "-b", str(B),
            "-m", str(M),
            "-n", str(N),
            "-k", str(K),
            "--dtype", dtype,
            "-d", "1",
            "-a", arch["name"]
        ]

        # Windows does need a python executable for running gemmologist
        if not python_path and os.name == 'nt':
            raise AssertionError("Python executable path need to be specified in Windows for running Gemmologist.")
        # Add the python executable path if it is given
        if python_path:
            cmd.insert(0, python_path)

        if "freq_mhz" in arch:
            cmd.append("--freq_mhz")
            cmd.append(str(arch["freq_mhz"]))

        if num_cus:
            cmd.append("--cus")
            cmd.append(str(num_cus))

        if "mem_bw_gbps" in arch:
            cmd.append("--hbm_bw")
            # In case of flash attention when everything happens in cache, we change the
            # memory bw to l1 bandwidth so as to simulate the same
            mem_bw = arch["mem_bw_gbps"] if not force_to_l1 else arch["l1_bw_gbps"]
            if num_cus and num_cus != arch['num_cus']:
                mem_bw = round(mem_bw / arch['num_cus'] * num_cus)
            cmd.append(str(mem_bw))

        # Check if the result is already in the cache
        cache_key = tuple(cmd)
        if cache_key in GEMM.cache_gemm_results:
            return GEMM.cache_gemm_results[cache_key], " ".join(cmd)

        # Run the command
        result = subprocess.run(cmd, cwd=gemmologist_path, capture_output=True, text=True)
        stdout = result.stdout
        stderr = result.stderr
        log = re.findall(r"Time=\d+\.\d+", stdout)
        if len(log) > 0:
            gemmologist_time = float(re.sub("Time=", "", str(log[0])))
            # Cache the result
            GEMM.cache_gemm_results[cache_key] = gemmologist_time
            return gemmologist_time, " ".join(cmd)
        else:
            raise AssertionError("Not able to simulate in gemmologist", cmd, stdout, stderr)

    def get_simulation_time(self):
        simulation_time = None
        if self.arch is not None:
            if os.environ.get('GEMMOLOGIST_PATH') is not None:
                if not os.path.exists(os.environ.get('GEMMOLOGIST_PATH')):
                    raise ValueError(f"GEMMOLOGIST_PATH does not exist: {os.environ.get('GEMMOLOGIST_PATH')}")
                dtype = self.param_details.get("gemmologist_dtype")
                if dtype is None:
                    dtype = torch_dtype_map(self.param_details['dtype_A_B'][0])
                simulation_time, self.gemmologist_cmd = GEMM.get_simulation_time_func(self.arch, self.M, self.N, self.K, self.B,
                                                                                  dtype, self.python_path)
            else:
                # TODO: use naive roofline model
                pass
        return simulation_time


class aten_mm(GEMM):
    """
    aten::mm the matrix multiplication primitive in PyTorch
    A.matmul(B)
    """
    @staticmethod
    def get_param_details(event):
        input_dims = event['args']['Input Dims']
        A_shape, B_shape = input_dims[0], input_dims[1]
        M = A_shape[0]
        N = B_shape[1]
        K = A_shape[1]

        dtype_A_B = tuple(event['args']['Input type'][:2])
        try:
            stride_A = tuple(event['args']['Input Strides'][0])
            stride_B = tuple(event['args']['Input Strides'][1])
        except KeyError:
            stride_A = stride_B = None

        return {"M": M, "N": N, "K": K, "bias": False,
                "stride_A": stride_A, "stride_B": stride_B,
                "dtype_A_B": dtype_A_B}

    def bytes(self):
        dtype_A_B = self.param_details['dtype_A_B']
        if dtype_A_B[0] != dtype_A_B[1]:
            # raise ValueError(f"Data types of A and B are different: {dtype_A_B}")
            warnings.warn(f"Data types of A and B are different: {dtype_A_B} for aten_mm. ")
        self.bpe = name2bpe(dtype_A_B[0])
        return super().bytes(bpe_mat1=self.bpe, bpe_mat2=self.bpe,
                             bpe_bias=self.bpe, # does not matter
                             bpe_output=self.bpe) # out dtype is not always provided. #TODO: use out dtype if provided
    def flops_bwd(self):
        raise NotImplementedError("Backward pass for aten::mm is not defined.")
    def bytes_bwd(self, bytes_per_element):
        raise NotImplementedError("Backward pass for aten::mm is not defined.")


class aten_addmm(GEMM):
    """
    aten::addmm is the A.matmul(B) + C operation in PyTorch
    """
    @staticmethod
    def get_param_details(event):
        input_dims = event['args']['Input Dims']
        C_shape, A_shape, B_shape = input_dims[0], input_dims[1], input_dims[2]
        M = A_shape[0]
        N = B_shape[1]
        K = A_shape[1]

        dtype_A_B = tuple(event['args']['Input type'][1:3])
        try:
            stride_A = tuple(event['args']['Input Strides'][1])
            stride_B = tuple(event['args']['Input Strides'][2])
        except KeyError:
            stride_A = stride_B = None

        return {"M": M, "N": N, "K": K, "bias": True,
                "stride_A": stride_A, "stride_B": stride_B,
                "dtype_A_B": dtype_A_B}

    def bytes(self):
        dtype_A_B = self.param_details['dtype_A_B']
        if dtype_A_B[0] != dtype_A_B[1]:
            # raise ValueError(f"Data types of A and B are different: {dtype_A_B}")
            warnings.warn(f"Data types of A and B are different: {dtype_A_B} for aten_addmm. ")
        self.bpe = name2bpe(dtype_A_B[0])
        # setting bias bpe to be the same as the input matrices is not totally correct
        # TODO: correct later
        # TODO: similar to aten_mm, we need to use the output dtype if provided
        return super().bytes(bpe_mat1=self.bpe, bpe_mat2=self.bpe,
                             bpe_bias=self.bpe,
                             bpe_output=self.bpe)

    def flops_bwd(self):
        raise NotImplementedError("Backward pass for aten::addmm is not defined.")
    def bytes_bwd(self, bytes_per_element):
        raise NotImplementedError("Backward pass for aten::addmm is not defined.")

class aten_scaled_mm(GEMM):
    """
    aten::scaled_mm is the scale_result(scale_a*A.matmul(scale_b*B) + bias)
    """
    @staticmethod
    def get_param_details(event):
        # ref: https://pytorch.org/cppdocs/api/function_namespaceat_1a2902105d8aed3fa448a0da42f90e2cbf.html
        input_dims = event['args']['Input Dims']
        A_shape, B_shape = input_dims[0], input_dims[1]
        M = A_shape[0]
        N = B_shape[1]
        K = A_shape[1]
        bias = len(input_dims) == 3

        dtype_A_B = tuple(event['args']['Input type'][:2])
        try:
            stride_A = tuple(event['args']['Input Strides'][0])
            stride_B = tuple(event['args']['Input Strides'][1])
        except KeyError:
            stride_A = stride_B = None

        return {"M": M, "N": N, "K": K, "bias": bias,
                "stride_A": stride_A, "stride_B": stride_B,
                "dtype_A_B": dtype_A_B}

    def bytes(self):
        dtype_A_B = self.param_details['dtype_A_B']
        bpeA = name2bpe(dtype_A_B[0])
        bpeB = name2bpe(dtype_A_B[1])
        assert bpeA is not None and bpeB is not None, \
            f"Data types of A and B are not supported: {dtype_A_B}"
        if bpeA != bpeB:
            # raise ValueError(f"Data types of A and B are different: {dtype_A_B}")
            warnings.warn(f"Data sizes of A and B are different: {dtype_A_B} for aten_scaled_mm. ")
        self.bpe = bpeA  # or bpeB, they are the same
        # assumption:
        # for fp8 the output dtype is fp16
        # for fp16, bf16, fp32 the output dtype is the same as the input dtype
        if self.bpe == 1:
            out_bpe = 2
        elif self.bpe in [2, 4]:
            out_bpe = self.bpe
        else:
            out_bpe = None
        return super().bytes(bpe_mat1=self.bpe, bpe_mat2=self.bpe,
                             bpe_bias=self.bpe, # does not matter
                             bpe_output=out_bpe)

    def flops_bwd(self):
        raise NotImplementedError("Backward pass for aten::addmm is not defined.")
    def bytes_bwd(self, bytes_per_element):
        raise NotImplementedError("Backward pass for aten::addmm is not defined.")

class aten_bmm(GEMM):
    """
    aten::bmm — batch matrix multiplication
    (B, M, K) × (B, K, N) → (B, M, N)
    Inherits FLOP/byte analytics from GEMM and scales them by the batch size.
    """

    @staticmethod
    def get_param_details(event):
        """Extract B, M, N, K and metadata from the profiler event."""
        input_dims = event['args']['Input Dims']
        A_shape, B_shape = input_dims[0], input_dims[1]

        B_dim, M, K = A_shape  # (B, M, K)
        _,      _, N = B_shape # (B, K, N)

        dtype_A_B = tuple(event['args']['Input type'][:2])
        try:
            stride_A = tuple(event['args']['Input Strides'][0])
            stride_B = tuple(event['args']['Input Strides'][1])
        except KeyError:
            stride_A = stride_B = None

        return {
            "B": B_dim,
            "M": M,
            "N": N,
            "K": K,
            "bias": False,            # aten::bmm has no implicit bias term
            "stride_A": stride_A,
            "stride_B": stride_B,
            "dtype_A_B": dtype_A_B,
        }

    # ---------------------- FLOPs / Bytes ----------------------
    def flops(self):
        """Total FLOPs for the entire batch."""
        return self.param_details["B"] * super().flops()

    def bytes(self):
        """Total DRAM traffic for the entire batch (read+write)."""
        dtype_A_B = self.param_details['dtype_A_B']
        if dtype_A_B[0] != dtype_A_B[1]:
            # raise ValueError(f"Data types of A and B are different: {dtype_A_B}")
            warnings.warn(f"Data types of A and B are different: {dtype_A_B} for aten_bmm. ")

        bpe = name2bpe(dtype_A_B[0])
        per_batch = super().bytes(bpe_mat1=bpe, bpe_mat2=bpe,
                                   bpe_bias=bpe,   # not used, but keeps call signature
                                   bpe_output=bpe)
        return None if per_batch is None else self.param_details['B'] * per_batch

    # ---------------------- Backward placeholders ----------------------
    def flops_bwd(self):
        raise NotImplementedError("Backward pass for aten::bmm is not defined.")

    def bytes_bwd(self, bytes_per_element):
        raise NotImplementedError("Backward pass for aten::bmm is not defined.")

class aten_baddbmm(GEMM):
    """
    aten::baddbmm — batch matrix multiplication with bias
    (B, M, K) × (B, K, N) + (B, M, N) → (B, M, N)
    Inherits FLOP/byte analytics from GEMM and scales them by the batch size.
    """
    @staticmethod
    def get_param_details(event):
        """Extract B, M, N, K and metadata from the profiler event."""
        input_dims = event['args']['Input Dims']
        C_shape, A_shape, B_shape = input_dims[0], input_dims[1], input_dims[2]

        B_dim, M, K = A_shape  # (B, M, K)
        _,      _, N = B_shape # (B, K, N)

        dtype_A_B = tuple(event['args']['Input type'][1:3])
        try:
            stride_A = tuple(event['args']['Input Strides'][1])
            stride_B = tuple(event['args']['Input Strides'][2])
        except KeyError:
            stride_A = stride_B = None

        return {
            "B": B_dim,
            "M": M,
            "N": N,
            "K": K,
            "bias": True,
            "stride_A": stride_A,
            "stride_B": stride_B,
            "dtype_A_B": dtype_A_B,
        }
    # ---------------------- FLOPs / Bytes ----------------------
    def flops(self):
        """Total FLOPs for the entire batch."""
        return self.param_details["B"] * super().flops()
    def bytes(self):
        """Total DRAM traffic for the entire batch (read+write)."""
        dtype_A_B = self.param_details['dtype_A_B']
        if dtype_A_B[0] != dtype_A_B[1]:
            # raise ValueError(f"Data types of A and B are different: {dtype_A_B}")
            warnings.warn(f"Data types of A and B are different: {dtype_A_B} for aten_baddbmm. ")

        bpe = name2bpe(dtype_A_B[0])
        per_batch = super().bytes(bpe_mat1=bpe, bpe_mat2=bpe,
                                   bpe_bias=bpe,   # not used, but keeps call signature
                                   bpe_output=bpe)
        return None if per_batch is None else self.param_details['B'] * per_batch
    # ---------------------- Backward placeholders ----------------------
    def flops_bwd(self):
        raise NotImplementedError("Backward pass for aten::baddbmm is not defined.")
    def bytes_bwd(self, bytes_per_element):
        raise NotImplementedError("Backward pass for aten::baddbmm is not defined.")


class tex_ts_te_gemm_ts(GEMM):
    """
    tex_ts::te_gemm_ts is a matmul op in TransformerEngine

    https://github.com/ROCm/TransformerEngine/blob/e9772d4d18b2980e8e0643c94591a94cad9bb8b7/transformer_engine/pytorch/csrc/ts_fp8_op.cpp#L244
    https://github.com/ROCm/TransformerEngine/blob/e9772d4d18b2980e8e0643c94591a94cad9bb8b7/transformer_engine/pytorch/csrc/extensions/gemm.cpp#L10

    """

    def __init__(self, event, arch=None, python_path=None):
        super().__init__(event, arch)

    def get_param_details(self, event):
        input_dims = event['args']['Input Dims']
    
        C_shape, A_shape, B_shape = input_dims[10], input_dims[0], input_dims[5]

        # index 4 and 9 are transa and transb respectively
        # https://github.com/ROCm/TransformerEngine/blob/e9772d4d18b2980e8e0643c94591a94cad9bb8b7/transformer_engine/pytorch/cpp_extensions/gemm.py#L248
        concrete_inputs = event['args']['Concrete Inputs']
        trans_a = concrete_inputs[4] == '1'
        trans_b = concrete_inputs[9] == '1'


        # https://github.com/ROCm/TransformerEngine/blob/e9772d4d18b2980e8e0643c94591a94cad9bb8b7/transformer_engine/common/gemm/cublaslt_gemm.cu#L330C17-L330C23
        if trans_a:
            M = A_shape[0]
            K = A_shape[1]
        else:
            M = A_shape[1]
            K = A_shape[0]

        if trans_b:
            N = B_shape[1]
        else:
            N = B_shape[0]

        bias_term = event['args']['Concrete Inputs'][14]

        if bias_term == '':
            bias = False
        else:
            bias = True

        # dtype A, B, output, bias
        dtype_A_B = (event['args']['Input type'][0], event['args']['Input type'][5], event['args']['Input type'][10], event['args']['Input type'][18])
        try:
            stride_A = tuple(event['args']['Input Strides'][0])
            stride_B = tuple(event['args']['Input Strides'][5])
        except KeyError:
            stride_A = stride_B = None

        return {"M": M, "N": N, "K": K, "bias": bias,
                "stride_A": stride_A, "stride_B": stride_B,
                "dtype_A_B": dtype_A_B}

    def bytes(self):
        dtype_A_B = self.param_details['dtype_A_B']
        self.bpe_mat1 = name2bpe(dtype_A_B[0])
        self.bpe_mat2 = name2bpe(dtype_A_B[1])
        self.bpe_output = name2bpe(dtype_A_B[2])
        self.bpe_bias = name2bpe(dtype_A_B[3])

        return super().bytes(bpe_mat1=self.bpe_mat1, bpe_mat2=self.bpe_mat2,
                             bpe_bias=self.bpe_bias,
                             bpe_output=self.bpe_output)

    def flops_bwd(self):
        raise NotImplementedError("Backward pass for tex_ts::te_gemm_ts is not defined.")
    def bytes_bwd(self, bytes_per_element):
        raise NotImplementedError("Backward pass for tex_ts::te_gemm_ts is not defined.")

class tev2_pseudo_gemm(GEMM):
    # TODO: need to cleanup and reuse perf models better
    @staticmethod
    def get_param_details(event):
        input_dims = event['args']['Input Dims']
        A_shape, B_shape = input_dims[0], input_dims[1]
        M = A_shape[0]
        N = B_shape[1]
        K = A_shape[1]

        dtype_A_B = tuple(event['args']['Input type'][:2])
        try:
            stride_A = tuple(event['args']['Input Strides'][0])
            stride_B = tuple(event['args']['Input Strides'][1])
        except KeyError:
            stride_A = stride_B = None

        return {"M": M, "N": N, "K": K, "bias": False,
                "stride_A": stride_A, "stride_B": stride_B,
                "dtype_A_B": dtype_A_B}

    def bytes(self):
        dtype_A_B = self.param_details['dtype_A_B']
        if dtype_A_B[0] != dtype_A_B[1]:
            raise ValueError(f"Data types of A and B are different: {dtype_A_B}")
        self.bpe_in = name2bpe(dtype_A_B[0])

        # irrespective of the input dtype, the output dtype is always fp16/bf16
        self.bpe_out = 2 
        return super().bytes(bpe_mat1=self.bpe_in, bpe_mat2=self.bpe_in,
                             bpe_bias=self.bpe_in, # does not matter
                             bpe_output=self.bpe_out) # out dtype is not always provided. #TODO: use out dtype if provided
    def flops_bwd(self):
        raise NotImplementedError("Backward pass for tev2_pseudo_gemm is not defined.")
    def bytes_bwd(self, bytes_per_element):
        raise NotImplementedError("Backward pass for tev2_pseudo_gemm is not defined.")


# 2. Convolution
class CONV:
    # Conv perf model is based on: https://github.com/pytorch/pytorch/blob/main/torch/utils/flop_counter.py
    # we will make stuff reusiable across conv1d, conv2d, and conv3d
    def __init__(self, event, arch=None, python_path=None):
        self.event = event
        self.param_details = self.get_param_details(event)
        self.x_shape, self.w_shape = self.param_details['input_shape'], self.param_details['filter_shape']
        self.stride, self.padding, self.dilation, self.groups = (self.param_details[key] for key in ['stride', 'padding', 'dilation', 'groups'])
        self.bias = self.param_details['bias']
        self.transposed_conv = self.param_details['transposed_conv']
        self.output_padding = self.param_details['output_padding'] if self.transposed_conv else None
        self.out_shape = CONV.get_output_shape(self.x_shape, self.w_shape, self.stride, self.padding, self.dilation, self.transposed_conv, self.output_padding)

    @staticmethod
    def get_output_shape(input_shape, filter_shape, stride, padding, dilation, transposed_conv, output_padding):
        x_spatial_shape, w_spatial_shape = input_shape[2:], filter_shape[2:]
        conv_ndims = len(x_spatial_shape)
        spatial_out_fn = CONV.get_conv_out_dim if not transposed_conv else CONV.get_transposed_conv_out_dim
        out_filters = filter_shape[0] if not transposed_conv else filter_shape[1]

        if not transposed_conv:
            output_padding = (None,) * conv_ndims
        out_spatial_shape = tuple(spatial_out_fn(x_spatial_shape[i], w_spatial_shape[i],
                                                 stride[i], padding[i], dilation[i], output_padding[i]) for i in range(conv_ndims))
        return (input_shape[0], out_filters) + tuple(out_spatial_shape)

    @staticmethod
    def t(shape):
        return (shape[1], shape[0]) + shape[2:]

    @staticmethod
    def get_conv_out_dim(input_dim, kernel_size, stride, padding, dilation, output_padding=None):
        return int(((input_dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1)

    @staticmethod
    def get_transposed_conv_out_dim(input_dim, kernel_size, stride, padding, dilation, output_padding):
        return (input_dim - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1

    @staticmethod
    def flops_func(x_shape, w_shape, out_shape, bias, transposed_conv=False):
        # c_in =filter[1] already accounts for grouped convolutions
        flops_per_element = 2 * prod(w_shape[1:])
        if transposed_conv:
            flops_conv = prod(x_shape) * flops_per_element
        else:
            flops_conv = prod(out_shape) * flops_per_element
        flops_bias = prod(out_shape) if bias else 0
        return flops_conv + flops_bias
    def flops(self):
        return self.flops_func(self.x_shape, self.w_shape, self.out_shape,
                                self.bias, self.transposed_conv)

    @staticmethod
    # we assume same bytes per element for all tensors
    # TODO: make it more general later
    def bytes_func(x_shape, w_shape, out_shape, bias, bytes_per_element):
        if bytes_per_element is None:
            return None
        elems_input_read = prod(x_shape)
        elems_weight_read = prod(w_shape)
        elems_bias_read = out_shape[1] if bias else 0
        elems_output_write = prod(out_shape)
        total_elems_moved = elems_input_read + elems_weight_read + elems_bias_read + elems_output_write
        return total_elems_moved * bytes_per_element
    def bytes(self, bytes_per_element):
        return self.bytes_func(self.x_shape, self.w_shape, self.out_shape, self.bias, bytes_per_element)

    @staticmethod
    def flops_bwd_func(out_shape, x_shape, w_shape, bias, transposed_conv=False):
        flops_input_grad = CONV.flops_func(out_shape, w_shape, x_shape, False, not transposed_conv)
        if not transposed_conv:
            flops_weight_grad = CONV.flops_func(CONV.t(x_shape), CONV.t(out_shape), CONV.t(w_shape), False, False)
        else:
            flops_weight_grad = CONV.flops_func(CONV.t(out_shape), CONV.t(x_shape), CONV.t(w_shape), False, False)

        flops_bias_grad = prod(out_shape) if bias else 0
        return flops_input_grad + flops_weight_grad + flops_bias_grad
    def flops_bwd(self):
        return self.flops_bwd_func(self.out_shape, self.x_shape, self.w_shape, self.bias, self.transposed_conv)

    @staticmethod
    def bytes_bwd_func(x_shape, w_shape, out_shape, bias, bytes_per_element):
        if bytes_per_element is None:
            return None
        bytes_input_grad = CONV.bytes_func(out_shape, w_shape, x_shape, False, bytes_per_element)
        bytes_weight_grad = CONV.bytes_func(out_shape, x_shape, w_shape, False, bytes_per_element)
        # for bias we read the output gradient and write the bias gradient
        bytes_bias_grad = prod(out_shape) + out_shape[1] if bias else 0
        return bytes_input_grad + bytes_weight_grad + bytes_bias_grad
    def bytes_bwd(self, bytes_per_element):
        return self.bytes_bwd_func(self.x_shape, self.w_shape, self.out_shape, self.bias, bytes_per_element)

    @staticmethod
    def get_param_details(event):
        # to be implemented in the child class
        raise NotImplementedError

class aten_conv(CONV):

    @staticmethod
    def str_to_tuple(s):
        return tuple(int(x) for x in s[1:-1].split(','))
    @staticmethod
    def get_param_details(event):
        # 0 input tensor
        # 1 weight tensor
        # 2 bias tensor (optional)
        # 3 stride
        # 4 padding
        # 5 dilation
        # 6 transposed (boolean)
        # 7 output_padding
        # 8 groups
        input_dims = event['args']['Input Dims']
        concrete_inputs = event['args']['Concrete Inputs']

        input_shape = tuple(input_dims[0])
        ndims = len(input_shape) - 2 #first two dimensions are batch and channel
        filter_shape = tuple(input_dims[1])
        bias = len(input_dims) == 3


        stride_arg = concrete_inputs[3]
        stride = aten_conv.str_to_tuple(stride_arg) if stride_arg != '' else (1,) * ndims
        padding_arg = concrete_inputs[4]
        padding = aten_conv.str_to_tuple(padding_arg) if padding_arg != '' else (0,) * ndims
        dilation_arg = concrete_inputs[5]
        dilation = aten_conv.str_to_tuple(dilation_arg) if dilation_arg != '' else (1,) * ndims
        transposed_conv = eval(concrete_inputs[6])
        output_padding_arg = concrete_inputs[7]
        output_padding = aten_conv.str_to_tuple(output_padding_arg) if output_padding_arg != '' else (0,) * ndims
        groups = int(concrete_inputs[8])

        # if its a length 1 tuple then we broadcast it to the number of spatial dimensions
        stride, padding, dilation, output_padding = [
            param * ndims if len(param) == 1 else param
            for param in [stride, padding, dilation, output_padding]
        ]

        dtype_input_weight = tuple(event['args']['Input type'][:2])
        # check no mixed precision
        if dtype_input_weight[0] != dtype_input_weight[1]:
            raise ValueError(f"Data types of input and weight are different: {dtype_input_weight}")
        try:
            input_stride = tuple(event['args']['Input Strides'][0])
            weight_stride = tuple(event['args']['Input Strides'][1])
        except KeyError:
            input_stride = weight_stride = None

        if len(input_shape) == 3:
            convNd = 'conv1d'
        elif len(input_shape) == 4:
            convNd = 'conv2d'
        elif len(input_shape) == 5:
            convNd = 'conv3d'
        else:
            raise ValueError(f"Unknown convolution dimension: {len(input_shape)}")

        return {"convNd": convNd, "input_shape": input_shape, "filter_shape": filter_shape, "dtype_input_weight": dtype_input_weight,
                "input_stride": input_stride, "weight_stride": weight_stride,
                "bias": bias, "stride": stride, "padding": padding, "dilation": dilation,
                "transposed_conv": transposed_conv, "output_padding": output_padding,
                "groups": groups}

    def bytes(self):
        dtype_input_weight = self.param_details['dtype_input_weight']
        if dtype_input_weight[0] != dtype_input_weight[1]:
            raise ValueError(f"Data types of input and weight are different: {dtype_input_weight}")
        self.bpe = name2bpe(dtype_input_weight[0])
        return super().bytes(self.bpe)

    def bytes_bwd(self):
        dtype_input_weight = self.param_details['dtype_input_weight']
        if dtype_input_weight[0] != dtype_input_weight[1]:
            raise ValueError(f"Data types of input and weight are different: {dtype_input_weight}")
        self.bpe = name2bpe(dtype_input_weight[0])
        return super().bytes_bwd(self.bpe)


class aten_conv_bwd(aten_conv):
    def __init__(self, event):
        super().__init__(event)

    def flops(self):
        return self.flops_bwd()

    def bytes(self, bytes_per_element):
        return self.bytes_bwd(bytes_per_element)

class Softmax:
    """
    Softmax operation
    """

    @staticmethod
    def flops_func(M, N, B=1):
        # softmax is O(N) for each element, so total is O(N^2)
        # 3 ops : Exponent, Division and Addition
        # Let's assume for now exponent and division are implement by LUT and hence
        # take 1 cycle
        return B * M * (3 * N)

    @staticmethod
    def bytes_func(M, N, bytes_per_element, B=1):
        # Input and output sizes are same
        return 2 * B * M * N * bytes_per_element

    @staticmethod
    def flops(M, N, B=1):
        return Softmax.flops_func(M, N, B)

    @staticmethod
    def bytes(M, N, bytes_per_element, B=1):
        return Softmax.bytes_func(M, N, bytes_per_element, B)

    @staticmethod
    def flops_bwd(M, N, B=1):
        # Forward pass of softmax is recomputed for backward
        return 2 * Softmax.flops_func(M, N, B)

    @staticmethod
    def bytes_bwd(M, N, bytes_per_element, B=1):
        # Forward pass of softmax is recomputed for backward
        return 2 * Softmax.bytes_func(M, N, bytes_per_element, B)

    @staticmethod
    def get_time(arch, M, N, bytes_per_element, B=1, force_to_l1=False, bwd=False, num_cus=None):
        flops = Softmax.flops(M, N, B) if not bwd else Softmax.flops_bwd(M, N, B)
        bytes_moved = Softmax.bytes(M, N, bytes_per_element, B) if not bwd else Softmax.bytes_bwd(M, N, bytes_per_element, B)
        ew_ops_per_cycle_simd = 256

        if not arch:
            raise ValueError("Architecture information is required.")

        if not num_cus:
            num_cus = arch['num_cus']

        compute_time = flops / ( num_cus * arch['gemm_units_per_cu'] *\
                                        ew_ops_per_cycle_simd) / \
                                        arch['freq_mhz']
        # This is in case of flash attention
        if force_to_l1:
            memory_time = bytes_moved / (arch['l1_bw_gbps'] * 1000)
        else:
            memory_time = bytes_moved / (arch['mem_bw_gbps'] * 1000)
        return max(compute_time, memory_time)

class SDPA:

    def __init__(self, event, arch=None, python_path=None):
        # S = QK^T
        # P = softmax(S)
        # O = PV
        self.event = event
        self.param_details = self.get_param_details(event)
        self.arch = arch
        self.python_path = python_path
        self.B, self.N_Q, self.H_Q, self.N_KV, self.H_KV, self.d_h_qk, self.d_h_v = (self.param_details[key] for key in ['B', 'N_Q', 'H_Q', 'N_KV', 'H_KV', 'd_h_qk', 'd_h_v'])

    def get_param_details(event):
        # to be implemented in the child class
        raise NotImplementedError

    @staticmethod
    def flops_func(B, N_Q, H_Q, N_KV, H_KV, d_h_qk, d_h_v, causal):
        # ref: https://github.com/Dao-AILab/flash-attention/blob/main/benchmarks/benchmark_flash_attention.py#L29
        flops_qk = B * H_Q * (2 * N_Q * N_KV * d_h_qk)
        # not including softmax for now as flops are order of d_k smaller
        flops_pv = B * H_Q * (2 * N_Q * d_h_v * N_KV)
        total_flops = flops_qk + flops_pv
        if causal:
            if N_Q == N_KV:
                total_flops /= 2
            else:
                raise ValueError(f"causal=True but N_Q != N_K: {N_Q} != {N_KV}")
        return total_flops
    def flops(self):
        return self.flops_func(self.B, self.N_Q, self.H_Q, self.N_KV, self.H_KV, self.d_h_qk, self.d_h_v,
                                self.param_details['causal'])

    @staticmethod
    def bytes_func(B, N_Q, H_Q, N_KV, H_KV, d_h_qk, d_h_v, causal, bytes_per_element):
        elems_q_read = B * N_Q * H_Q * d_h_qk
        elems_k_read = B * N_KV * H_KV * d_h_qk
        elems_v_read = B * N_KV * H_KV * d_h_v
        elems_out_write = B * N_Q * H_Q * d_h_v
        total_elems_moved = elems_q_read + elems_k_read + elems_v_read + elems_out_write
        return total_elems_moved * bytes_per_element
    #TODO make bytes_per_element based on profile info
    def bytes(self, bytes_per_element=2):
        return self.bytes_func(self.B, self.N_Q, self.H_Q, self.N_KV, self.H_KV, self.d_h_qk, self.d_h_v,
                                self.param_details['causal'], bytes_per_element)

    @staticmethod
    def flops_bwd_func(B, N_Q, H_Q, N_KV, H_KV, d_h_qk, d_h_v, causal, flash_impl):
        total_flops = 0
        if flash_impl:
            # 0. recompute qk
            flops_recompute_qk = B * H_Q * (2 * N_Q * N_KV * d_h_qk)
            total_flops += flops_recompute_qk
            # 0.1 recompute softmax P - ignored as it is small

        # not including softmax for now
        # 1. V_grad = P_grad^T.matmul(O)
        flops_vgrad = B * H_Q * (2 * N_KV * N_Q  * d_h_v)
        flops_vgrad +=  B * N_KV * d_h_v * (H_Q//H_KV -1 ) # reduce from H_Q to H_KV for GQA
        total_flops += flops_vgrad
        # 2. P_grad = O_grad.matmul(V^T)
        flops_pgrad = B * H_Q * (2 * N_Q * N_KV * d_h_v)
        total_flops += flops_pgrad
        # 3. S_grad = f(P_grad, P) -  ignored as it is small
        # 4. Q_grad = S_grad.matmul(K)
        flops_q_grad = B * H_Q * (2 * N_Q * N_KV * d_h_qk)
        total_flops += flops_q_grad
        # 5. K_grad = S_grad^T.matmul(Q)
        flops_k_grad = B * H_Q * (2 * N_KV * N_Q * d_h_qk)
        flops_k_grad += B * N_KV * d_h_qk * (H_Q//H_KV -1 ) # reduce from H_Q to H_KV for GQA
        total_flops += flops_k_grad

        if causal:
            if N_Q == N_KV:
                total_flops /= 2
            else:
                raise ValueError(f"causal=True but N_Q != N_K: {N_Q} != {N_KV}")
        return total_flops

    @staticmethod
    def bytes_bwd_func(B, N_Q, H_Q, N_KV, H_KV, d_h_qk, d_h_v, causal, bytes_per_element):
        # This will be done for recompute in flash attention
        elems_q_read = B * N_Q * H_Q * d_h_qk
        elems_k_read = B * N_KV * H_KV * d_h_qk
        elems_v_read = B * N_KV * H_KV * d_h_v

        elems_o_grad_read = B * N_Q * H_Q * d_h_v
        # grad for q, k and v
        elems_q_grad_write = B * N_Q * H_Q * d_h_qk
        elems_k_grad_write = B * N_KV * H_KV * d_h_qk
        elems_v_grad_write = B * N_KV * H_KV * d_h_v

        total_elems_moved = (elems_q_read + elems_k_read + elems_v_read + elems_o_grad_read +
                             elems_q_grad_write + elems_k_grad_write + elems_v_grad_write)
        return total_elems_moved * bytes_per_element

    def flops_bwd(self):
        return self.flops_bwd_func(self.B, self.N_Q, self.H_Q, self.N_KV, self.H_KV, self.d_h_qk, self.d_h_v,
                                    self.param_details['causal'],
                                    self.param_details['flash_impl'])

    # @staticmethod
    # def bytes_bwd_func(B, N_Q, H, d_k, N_K, dropout, causal, flash_impl, bytes_per_element):
    def bytes_bwd(self, bytes_per_element=2):
        # Same as forward for now
        return self.bytes_bwd_func(self.B, self.N_Q, self.H_Q, self.N_KV, self.H_KV, self.d_h_qk, self.d_h_v,
                                   self.param_details['causal'], bytes_per_element)

    @staticmethod
    def get_simulation_time_func(arch, dtype, python_path,dtype_A_B, bytes,
                                          B, H_Q, N_Q, N_KV, d_h, fa=True):
        force_to_l1 = False
        block_N_Q = N_Q
        block_N_KV = N_KV

        if fa:
            force_to_l1 = True
            # Every Q tile block goes through full K and V, so we keep block_N_KV same
            # and Q tile size is 128 for all the cases observed
            block_N_Q = min(128, N_Q)
            # block_N_KV = min(self.N_KV, self.N_KV)

        num_blocks_N_Q = math.ceil(N_Q / block_N_Q)
        # num_blocks_N_KV = math.ceil(N_KV / block_N_KV)
        total_num_blocks = num_blocks_N_Q * B * H_Q
        num_waves = math.ceil(total_num_blocks / arch['num_cus'])

        qkt_time, _ = GEMM.get_simulation_time_func(arch, M=block_N_Q, K=d_h, N=block_N_KV,
                                                B=1, dtype=dtype,
                                                python_path=python_path, force_to_l1=force_to_l1, num_cus=1)
        qkt_time = num_waves * qkt_time

        softmax_time = num_waves * Softmax.get_time(arch, block_N_Q, block_N_KV,
                                                    name2bpe(dtype_A_B),
                                                    1, force_to_l1=force_to_l1, num_cus=1)
        pv_time, _ = GEMM.get_simulation_time_func(arch, M=block_N_Q, K=block_N_KV, N=d_h,
                                               B=1, dtype=dtype,
                                               python_path=python_path, force_to_l1=force_to_l1, num_cus=1)
        pv_time = num_waves * pv_time

        mem_time = bytes / N_Q / N_KV * block_N_Q * block_N_KV / \
                   (arch['mem_bw_gbps'] * 1000) * num_waves
        return qkt_time + softmax_time + pv_time + mem_time

    def get_simulation_time(self):
        simulated_time = None
        if self.arch is not None:
            if os.environ.get('GEMMOLOGIST_PATH') is not None:
                if not os.path.exists(os.environ.get('GEMMOLOGIST_PATH')):
                    raise ValueError(f"GEMMOLOGIST_PATH does not exist: {os.environ.get('GEMMOLOGIST_PATH')}")
                dtype = self.param_details.get("gemmologist_dtype")
                if dtype is None:
                    dtype = torch_dtype_map(self.param_details['dtype_A_B'][0])
                bytes = self.bytes(name2bpe(self.param_details['dtype_A_B'][0]))
                fa = True if type(self).__name__ == "flash_attention" else False
                simulated_time = SDPA.get_simulation_time_func(self.arch, dtype,
                    self.python_path, self.param_details['dtype_A_B'][0], bytes, self.B,
                    self.H_Q, self.N_Q, self.N_KV, self.d_h, fa)
            else:
                # TODO: use naive roofline model
                pass
        return simulated_time

    @staticmethod
    def get_simulation_time_bwd_func(arch, dtype, python_path, dtype_A_B, bytes,
                                          B, H_Q, N_Q, N_KV, d_h, fa=True):
        force_to_l1 = False
        block_N_Q = N_Q
        block_N_KV = N_KV
        qkt_time = 0
        pv_time = 0

        if fa:
            force_to_l1 = True
            # ∇Q is tiled — but it is not partitioned exclusively across thread blocks the same way ∇K and ∇V are.
            # Instead, multiple thread blocks may contribute to the same ∇Q tile, which is why atomics are needed on ∇Q
            block_N_Q = min(N_Q, N_Q)
            block_N_KV = min(128, N_KV)

        num_blocks_N_Q = math.ceil(N_Q / block_N_Q)
        num_blocks_N_KV = math.ceil(N_KV / block_N_KV)
        # Partition happens on ∇K and ∇V and not ∇Q
        total_num_blocks = num_blocks_N_KV * B * H_Q
        num_waves = math.ceil(total_num_blocks / arch['num_cus'])

        qkt_fwd_time, _ = GEMM.get_simulation_time_func(arch, M=block_N_Q, K=d_h, N=block_N_KV,
                                                    B=1,
                                                    dtype=dtype,
                                                    python_path=python_path, force_to_l1=force_to_l1, num_cus=1)

        qkt_fwd_time = num_waves * qkt_fwd_time

        # B = B * H_Q, M = N_Q, N = d_H, K = N_KV
        pv_fwd_time, _ = GEMM.get_simulation_time_func(arch, M=block_N_Q, K=block_N_KV, N=d_h,
                                                   B=1, dtype=dtype,
                                                   python_path=python_path, force_to_l1=force_to_l1, num_cus=1)
        pv_fwd_time = num_waves * pv_fwd_time
        # pv_fwd_time, _ =  math.ceil(self.N_Q / 512) * math.ceil(self.N_KV / 512) * GEMM.get_gemmologist_time(self.arch, M=512, K=512, N=self.d_h,
        #                                       B=self.B * self.H_Q, dtype=dtype,
        #                                       python_path=self.python_path, force_to_l1=force_to_l1)

        if fa:
            # In case of flash attention we have to recompute
            # B = B * H_Q, M = N_Q, N = N_KV, K = d_H
            qkt_time = qkt_fwd_time
            pv_time = pv_fwd_time

        # We don't need to go to gemmologist to calculate these,
        # as we already have the times
        p_grad_time = qkt_fwd_time
        v_grad_time = pv_fwd_time
        q_grad_time = pv_fwd_time
        k_grad_time = pv_fwd_time

        # p_grad_time = pv_fwd_time
        # v_grad_time = qkt_fwd_time
        # q_grad_time = qkt_fwd_time
        # k_grad_time = qkt_fwd_time

        softmax_time = num_waves * Softmax.get_time(arch, block_N_Q, block_N_KV,
                                                    name2bpe(dtype_A_B), 1, force_to_l1=force_to_l1, num_cus=1)

        # We assume that we use atomics for adding up the gradients together
        atomic_latency_global_ns = 400  # ns for global memory
        atomic_latency_local_ns = 40  # ns for shared memory/ L1
        # This is the tile size for ∇K. For every tile of ∇Q, we need to accumulate the contributions
        # from all the ∇K blocks
        k_tile = block_N_KV
        warp_size = 64

        # Shared-memory tile reduction:
        # Each block uses atomics only once per (k_tile × d)
        # This optimization won't be there for now possibly?
        num_k_tiles = math.ceil(block_N_KV / k_tile)

        # Warp-level reduction:
        # Each warp atomics once per d vector
        # warps_per_block = (block_N_Q * self.d_h) // warp_size
        warp_reduction_updates_per_block_global = math.ceil(num_k_tiles * math.ceil(d_h / warp_size))
        total_updates_global = warp_reduction_updates_per_block_global * num_waves

        warp_reduction_updates_per_block_local = math.ceil(k_tile * math.ceil(d_h / warp_size))
        total_updates_local = warp_reduction_updates_per_block_local * num_waves

        # Total atomic time (serialized across all blocks)
        total_atomic_time_us = (atomic_latency_global_ns * total_updates_global +
                                atomic_latency_local_ns * total_updates_local) / 1e3

        # We have to read the first block and write the last block
        mem_time = bytes / N_Q / N_KV * block_N_Q * block_N_KV / \
                   (arch['mem_bw_gbps'] * 1000) * num_waves
        simulated_time = qkt_time + pv_time + p_grad_time + v_grad_time + q_grad_time + \
                         k_grad_time + softmax_time + total_atomic_time_us + mem_time
        return simulated_time

    def get_simulation_time_bwd(self):
        simulated_time = None
        if self.arch is not None:
            if os.environ.get('GEMMOLOGIST_PATH') is not None:
                if not os.path.exists(os.environ.get('GEMMOLOGIST_PATH')):
                    raise ValueError(f"GEMMOLOGIST_PATH does not exist: {os.environ.get('GEMMOLOGIST_PATH')}")
                dtype = self.param_details.get("gemmologist_dtype")
                if dtype is None:
                    dtype = torch_dtype_map(self.param_details['dtype_A_B'][0])

                bytes = self.bytes_bwd(name2bpe(self.param_details['dtype_A_B'][0]))
                fa = True if type(self).__name__ == "flash_attention" else False
                simulated_time = SDPA.get_simulation_time_bwd_func(self.arch, dtype,
                    self.python_path, self.param_details['dtype_A_B'][0], bytes, self.B,
                    self.H_Q, self.N_Q, self.N_KV, self.d_h, fa)
            else:
                # TODO: use naive roofline model
                pass
        return simulated_time

def extract_sdpa_cfg(q_shape, k_shape, v_shape, bhnd_idx):
    B_q, H_Q, N_Q, d_h_Q = tuple(q_shape[i] for i in bhnd_idx)
    B_k, H_K, N_K, d_h_K = tuple(k_shape[i] for i in bhnd_idx)
    B_v, H_V, N_V, d_h_V = tuple(v_shape[i] for i in bhnd_idx)
    if B_q != B_k or B_q != B_v:
        raise ValueError(f"Batch sizes do not match: {B_q} != {B_k} != {B_v}")
    if H_K != H_V:
        raise ValueError(f"Head sizes do not match for K and V: {H_K} != {H_V}")
    if N_K != N_V:
        raise ValueError(f"Length sizes do not match for K and V: {N_K} != {N_V}")
    if d_h_Q != d_h_K:
        raise ValueError(f"Head dimensions do not match for Q and K: {d_h_Q} != {d_h_K}")
    return {
        "B": B_q,
        "N_Q": N_Q,
        "H_Q": H_Q,
        "N_KV": N_K,
        "H_KV": H_K,
        "d_h_qk": d_h_Q,
        "d_h_v": d_h_V,
    }

class flash_attention(SDPA):

    @staticmethod
    def get_param_details(event):
        input_dims = event['args']['Input Dims']
        q_idx, k_idx, v_idx = 0, 1, 2
        q_shape, k_shape, v_shape = input_dims[q_idx], input_dims[k_idx], input_dims[v_idx]
        bhnd_idx = 0, 2, 1, 3
        sdpa_cfg = extract_sdpa_cfg(q_shape, k_shape, v_shape, bhnd_idx)
        B, N_Q, H_Q, N_KV, H_KV, d_h_qk, d_h_v = (sdpa_cfg[key] for key in ['B', 'N_Q', 'H_Q', 'N_KV', 'H_KV', 'd_h_qk', 'd_h_v'])
        
        dtype_A_B = tuple(event['args']['Input type'][:2])
        strides = event['args']['Input Strides']
        q_stride, k_stride, v_stride = tuple(strides[q_idx]), tuple(strides[k_idx]), tuple(strides[v_idx])        
        dropout = float(event['args']['Concrete Inputs'][3])
        causal = eval(event['args']['Concrete Inputs'][5])
        return {"B": B, "N_Q": N_Q, "H_Q": H_Q, "N_KV": N_KV, "H_KV": H_KV, "d_h_qk": d_h_qk, "d_h_v": d_h_v,
                "q_stride": q_stride, "k_stride": k_stride, "v_stride": v_stride,
                "dropout": dropout, "causal": causal, "flash_impl": True, "dtype_A_B": dtype_A_B}

class flash_attention_backward(flash_attention):

    def __init__(self, event):
        super().__init__(event)

    def flops(self):
        return self.flops_bwd()

    def bytes(self, bytes_per_element):
        return self.bytes_bwd(bytes_per_element)

class aten__scaled_dot_product_cudnn_attention(SDPA):

    @staticmethod
    def get_param_details(event):
        # the order of arguments for aten::_scaled_dot_product_cudnn_attention is:

        # query: Tensor
        # key: Tensor
        # value: Tensor
        # attn_bias: Optional[Tensor]
        # compute_log_sumexp: bool
        # dropout_p: float
        # is_causal: bool
        # return_debug_mask: bool
        # scale: Optional[float]
        input_dims = event['args']['Input Dims']
        concrete_inputs = event['args']['Concrete Inputs']
        q_shape, k_shape, v_shape = input_dims[0], input_dims[1], input_dims[2]
        bhnd_idx = 0, 1, 2, 3
        sdpa_cfg = extract_sdpa_cfg(q_shape, k_shape, v_shape, bhnd_idx)
        B, N_Q, H_Q, N_KV, H_KV, d_h_qk, d_h_v = (sdpa_cfg[key] for key in ['B', 'N_Q', 'H_Q', 'N_KV', 'H_KV', 'd_h_qk', 'd_h_v'])

        dropout_p = 0.0
        if concrete_inputs[5] not in ('', 'None'):
            try:
                dropout_p = float(concrete_inputs[5])
            except (ValueError, TypeError):
                pass

        is_causal = concrete_inputs[6].lower() == 'true' if concrete_inputs[6] not in ('', 'None') else False

        return {"B": B, "N_Q": N_Q, "H_Q": H_Q, "N_KV": N_KV, "H_KV": H_KV, "d_h_qk": d_h_qk, "d_h_v": d_h_v,
                "dropout": dropout_p, "causal": is_causal, "flash_impl": False}    

class aten__scaled_dot_product_efficient_attention(SDPA):
    # Seems to have the exact same signature as aten::_scaled_dot_product_cudnn_attention
    # Tensor query, 
    # Tensor key, 
    # Tensor value, 
    # Tensor? attn_bias, 
    # bool compute_log_sumexp, 
    # float dropout_p=0., 
    # bool is_causal=False, 
    # *, 
    # float? scale=None

    @staticmethod
    def get_param_details(event):
        input_dims = event['args']['Input Dims']
        concrete_inputs = event['args']['Concrete Inputs']
        q_shape, k_shape, v_shape = input_dims[0], input_dims[1], input_dims[2]
        bhnd_idx = 0, 1, 2, 3
        sdpa_cfg = extract_sdpa_cfg(q_shape, k_shape, v_shape, bhnd_idx)
        B, N_Q, H_Q, N_KV, H_KV, d_h_qk, d_h_v = (sdpa_cfg[key] for key in ['B', 'N_Q', 'H_Q', 'N_KV', 'H_KV', 'd_h_qk', 'd_h_v'])

        dropout_p = 0.0
        if concrete_inputs[5] not in ('', 'None'):
            try:
                dropout_p = float(concrete_inputs[5])
            except (ValueError, TypeError):
                pass

        is_causal = concrete_inputs[6].lower() == 'true' if concrete_inputs[6] not in ('', 'None') else False
        # scale = float(concrete_inputs[7]) if concrete_inputs[7] not in ('', 'None') else None

        return {"B": B, "N_Q": N_Q, "H_Q": H_Q, "N_KV": N_KV, "H_KV": H_KV, "d_h_qk": d_h_qk, "d_h_v": d_h_v,
                "dropout": dropout_p, "causal": is_causal, "flash_impl": False}    

class aten__scaled_dot_product_flash_attention(SDPA):

    @staticmethod
    def get_param_details(event):
        # the order of arguments for aten::_scaled_dot_product_flash_attention is:
        # query: Tensor
        # key: Tensor
        # value: Tensor
        # dropout_p: float
        # is_causal: bool
        # return_debug_mask: bool
        # *
        # scale: Optional[float]
        input_dims = event['args']['Input Dims']
        concrete_inputs = event['args']['Concrete Inputs']
        q_shape, k_shape, v_shape = input_dims[0], input_dims[1], input_dims[2]
        bhnd_idx = 0, 1, 2, 3
        sdpa_cfg = extract_sdpa_cfg(q_shape, k_shape, v_shape, bhnd_idx)
        B, N_Q, H_Q, N_KV, H_KV, d_h_qk, d_h_v = (sdpa_cfg[key] for key in ['B', 'N_Q', 'H_Q', 'N_KV', 'H_KV', 'd_h_qk', 'd_h_v'])
        dropout_p = 0.0
        if concrete_inputs[3] not in ('', 'None'):
            try:
                dropout_p = float(concrete_inputs[3])
            except (ValueError, TypeError):
                pass
        is_causal = concrete_inputs[4].lower() == 'true' if concrete_inputs[4] not in ('', 'None') else False
        # scale = float(concrete_inputs[5]) if concrete_inputs[5] not in ('', 'None') else None

        return {"B": B, "N_Q": N_Q, "H_Q": H_Q, "N_KV": N_KV, "H_KV": H_KV, "d_h_qk": d_h_qk, "d_h_v": d_h_v,
                "dropout": dropout_p, "causal": is_causal, "flash_impl": True}

class aiter__flash_attn_forward(SDPA):
    
    @staticmethod
    def get_param_details(event):
        # the order of arguments for aiter::_flash_attn_forward is:
        # q: torch.Tensor
        # k: torch.Tensor
        # v: torch.Tensor
        # dropout_p: float
        # softmax_scale: float
        # causal: bool
        # window_size_left: int
        # window_size_right: int
        # bias: Optional[torch.Tensor]
        # alibi_slopes: Optional[torch.Tensor]
        # return_lse: bool
        # return_softmax: bool
        input_dims = event['args']['Input Dims']
        concrete_inputs = event['args']['Concrete Inputs']
        q_shape, k_shape, v_shape = input_dims[0], input_dims[1], input_dims[2]
        bhnd_idx = 0, 2, 1, 3
        sdpa_cfg = extract_sdpa_cfg(q_shape, k_shape, v_shape, bhnd_idx)
        B, N_Q, H_Q, N_KV, H_KV, d_h_qk, d_h_v = (sdpa_cfg[key] for key in ['B', 'N_Q', 'H_Q', 'N_KV', 'H_KV', 'd_h_qk', 'd_h_v'])
        dropout_p = 0.0
        if concrete_inputs[3] not in ('', 'None'):
            try:
                dropout_p = float(concrete_inputs[3])
            except (ValueError, TypeError):
                pass
        is_causal = concrete_inputs[5].lower() == 'true' if concrete_inputs[5] not in ('', 'None') else False

        return {"B": B, "N_Q": N_Q, "H_Q": H_Q, "N_KV": N_KV, "H_KV": H_KV, "d_h_qk": d_h_qk, "d_h_v": d_h_v,
                "dropout": dropout_p, "causal": is_causal, "flash_impl": True}

class aiter__flash_attn_backward(SDPA):
    
    @staticmethod
    def get_param_details(event):
        # the order of arguments for aiter::_flash_attn_backward is:
        # q: torch.Tensor
        # k: torch.Tensor
        # v: torch.Tensor
        # dropout_p: float
        # softmax_scale: float
        # causal: bool
        # window_size_left: int
        # window_size_right: int
        # bias: Optional[torch.Tensor]
        # alibi_slopes: Optional[torch.Tensor]
        # return_lse: bool
        # return_softmax: bool
        input_dims = event['args']['Input Dims']
        concrete_inputs = event['args']['Concrete Inputs']
        q_shape, k_shape, v_shape = input_dims[0], input_dims[1], input_dims[2]
        bhnd_idx = 0, 2, 1, 3
        sdpa_cfg = extract_sdpa_cfg(q_shape, k_shape, v_shape, bhnd_idx)
        B, N_Q, H_Q, N_KV, H_KV, d_h_qk, d_h_v = (sdpa_cfg[key] for key in ['B', 'N_Q', 'H_Q', 'N_KV', 'H_KV', 'd_h_qk', 'd_h_v'])
        dropout_p = 0.0
        if concrete_inputs[10] not in ('', 'None'):
            try:
                dropout_p = float(concrete_inputs[10])
            except (ValueError, TypeError):
                pass
        is_causal = concrete_inputs[12].lower() == 'true' if concrete_inputs[12] not in ('', 'None') else False

        return {"B": B, "N_Q": N_Q, "H_Q": H_Q, "N_KV": N_KV, "H_KV": H_KV, "d_h_qk": d_h_qk, "d_h_v": d_h_v,
                "dropout": dropout_p, "causal": is_causal, "flash_impl": True}

    def flops(self):
        return self.flops_bwd()
    
    def bytes(self, bytes_per_element=2):
        return self.bytes_bwd(bytes_per_element)


class UnaryElementwise:

    def __init__(self, event, arch=None, python_path=None):
        self.event = event
        self.param_details = self.get_param_details(event)
        self.nelems = prod(self.param_details['op_shape'])
        self.dtype_in_out = self.param_details['dtype_in_out']
        self.stride_input = self.param_details['stride_input']
        self.stride_output = self.param_details['stride_output']

        self.bpe_in = name2bpe(self.dtype_in_out[0])
        if self.dtype_in_out[1] is not None:
            self.bpe_out = name2bpe(self.dtype_in_out[1])
        else:
            # same as input
            self.bpe_out = self.bpe_in

    @staticmethod
    def flops_func(nelems):
        return nelems
    def flops(self):
        return self.flops_func(self.nelems)

    @staticmethod
    def bytes_func(nelems, bpe_in, bpe_out):
        if None in {bpe_in, bpe_out}:
            return None
        return nelems*bpe_in + nelems*bpe_out
    def bytes(self):
        return self.bytes_func(self.nelems, self.bpe_in, self.bpe_out)

class aten_unary_elementwise(UnaryElementwise):

    @staticmethod
    def get_param_details(event):
        args_input_dims = event['args']['Input Dims']
        op_shape = tuple(args_input_dims[0])
        dtype_in = event['args']['Input type'][0]
        stride_input = tuple(event['args']['Input Strides'][0])
        if len(args_input_dims) > 1 and args_input_dims[1]:
            dtype_out = event['args']['Input type'][1]
            stride_output = tuple(event['args']['Input Strides'][1])
        else:
            dtype_out = None
            stride_output = None
        return {"op_shape": op_shape, "dtype_in_out" : (dtype_in, dtype_out),
                "stride_input": stride_input, "stride_output": stride_output}
class BinaryElementwise:

    def __init__(self, event, arch=None, python_path=None):
        self.event = event
        self.param_details = self.get_param_details(event)
        broadcast_shape = self.get_broadcast_shape(self.param_details['shape_in1'], self.param_details['shape_in2'])
        self.nelems_in1 = prod(self.param_details['shape_in1'])
        self.nelems_in2 = prod(self.param_details['shape_in2'])
        self.nelems_out = prod(broadcast_shape)
        self.dtype_in1_in2_out = self.param_details['dtype_in1_in2_out']
        self.stride_input1 = self.param_details['stride_input1']
        self.stride_input2 = self.param_details['stride_input2']
        self.stride_output = self.param_details['stride_output']

        dtype_in1, dtype_in2, dtype_out = self.dtype_in1_in2_out
        self.bpe_in1 = name2bpe(dtype_in1)
        self.bpe_in2 = name2bpe(dtype_in2)
        if dtype_out is not None:
            self.bpe_out = name2bpe(dtype_out)
        elif self.bpe_in1 and self.bpe_in2:
            in1_is_tensor = self.param_details['shape_in1'] != ()
            in2_is_tensor = self.param_details['shape_in2'] != ()
            if in1_is_tensor and in2_is_tensor:
                # cast to higher precision if both are tensors
                self.bpe_out = max(self.bpe_in1, self.bpe_in2)
            else:
                self.bpe_out = self.bpe_in1
        else:
            self.bpe_out = None
    @staticmethod
    def flops_func(nelems_out):
        return nelems_out
    def flops(self):
        return self.flops_func(self.nelems_out)

    @staticmethod
    def bytes_func(nelems_in1, nelems_in2, nelems_out, bpe_in1, bpe_in2, bpe_out):
        if None in {bpe_in1, bpe_in2, bpe_out}:
            return None
        return nelems_in1*bpe_in1 + nelems_in2*bpe_in2 + nelems_out*bpe_out
    def bytes(self):
        return self.bytes_func(self.nelems_in1, self.nelems_in2, self.nelems_out, self.bpe_in1, self.bpe_in2, self.bpe_out)

    @staticmethod
    def get_broadcast_shape(shape1, shape2):
        # Align shapes to the right by pre-pending 1's
        ndim = max(len(shape1), len(shape2))
        shape1 = (1,) * (ndim - len(shape1)) + shape1
        shape2 = (1,) * (ndim - len(shape2)) + shape2
        result = []
        for d1, d2 in zip(shape1, shape2):
            if d1 != d2 and d1 != 1 and d2 != 1:
                raise ValueError("Shapes not broadcastable: {} and {}".format(shape1, shape2))
            result.append(max(d1, d2))
        return tuple(result)

class aten_binary_elementwise(BinaryElementwise):

    @staticmethod
    def get_param_details(event):
        args_input_dims = event['args']['Input Dims']
        shape_in1 = tuple(args_input_dims[0])
        shape_in2 = tuple(args_input_dims[1])
        dtype_in1 = event['args']['Input type'][0]
        dtype_in2 = event['args']['Input type'][1]
        stride_input1 = tuple(event['args']['Input Strides'][0])
        stride_input2 = tuple(event['args']['Input Strides'][1])

        if len(args_input_dims) > 2 and args_input_dims[2]:
            dtype_out = event['args']['Input type'][2]
            stride_output = tuple(event['args']['Input Strides'][2])
        else:
            dtype_out = None
            stride_output = None
        return {"shape_in1": shape_in1, "shape_in2": shape_in2,
                "dtype_in1_in2_out" : (dtype_in1, dtype_in2, dtype_out),
                "stride_input1": stride_input1, "stride_input2": stride_input2, "stride_output": stride_output}

class GroupedGemm:
    """
    Grouped General Matrix Multiplication (GEMM).

    This operation applies group-specific weight matrices to partitions of the
    input tensor `X`. The rows of `X` are already arranged in group order along
    the first dimension; the tensor `b_g` encodes how many rows belong to each
    group so that group boundaries can be identified.

    Inputs:
        X : tensor, shape (M, K)
            Input tensor, arranged in group order along the first dimension.
        W : tensor, shape (G, K, N)
            Weight tensors, one (K, N) tensor per group.
        b_g : tensor, shape (G,)
            Integer tensor specifying the number of rows from `X` assigned
            to each group. Must satisfy sum(b_g) = M.
    Outputs:
        Y : tensor, shape (M, N)
            The concatenated result of the groupwise multiplications.
            
    Computation is functionally equivalent to  (implementation detail will ofcourse be efficient):
        start = 0
        Y_parts = []
        for i in range(G):
            rows = b_g[i]
            X_g = X[start:start+rows]     # (rows, K)
            Y_g = X_g @ W[i]              # (rows, N)
            Y_parts.append(Y_g)
            start += rows
        Y = concat(Y_parts, dim=0)        # (M, N)

    Performance Model:
        Forward FLOPs:
            - Per group: 2 * b_g[i] * K * N
            - Total:     2 * M * K * N

        Backward FLOPs:
            - Per group: dX = dY @ W^T  (2 * b_g[i] * N * K)
                         dW = X^T @ dY  (2 * K * b_g[i] * N)
            - Total:     4 * M * K * N

        Forward bytes (assuming weights use same dtype as inputs):
            - Reads:  X (M*K), W (G*K*N)
            - Writes: Y (M*N)
            - Total bytes: (M*K + G*K*N) * bpe_in + (M*N) * bpe_out

        Backward bytes (streaming estimate, dY read twice):
            - Reads:  dY twice (2*M*N) * bpe_out
                      W once   (G*K*N) * bpe_in
                      X once   (M*K)   * bpe_in
            - Writes: dX       (M*K)   * bpe_in
                      dW       (G*K*N) * bpe_in
            - Total bytes: (2*M*N) * bpe_out + (2*M*K) * bpe_in + (2*G*K*N) * bpe_in
    """


    def __init__(self, event, arch=None, python_path=None):
        self.event = event
        self.param_details = self.get_param_details(event)
        self.arch = arch
        self.python_path = python_path
        self.M, self.K, self.G, self.N = (self.param_details[key] for key in ['M', 'K', 'G', 'N'])
        self.bpe_in = self.param_details['bpe_in']
        self.bpe_out = self.param_details['bpe_out']

    @staticmethod
    def flops_func(M, K, N):
        return 2 * M * K * N
    def flops(self):
        return self.flops_func(self.M, self.K, self.N)
    
    @staticmethod
    def bytes_func(M, K, N, G, bpe_in, bpe_out):
        return (M * K + G * K * N) * bpe_in + M * N * bpe_out
    def bytes(self):
        return self.bytes_func(self.M, self.K, self.N, self.G, 
                               self.bpe_in, self.bpe_out)

    @staticmethod
    def flops_bwd_func(M, K, N):
        return 4 * M * K * N
    def flops_bwd(self):
        return self.flops_bwd_func(self.M, self.K, self.N)
    
    @staticmethod
    def bytes_bwd_func(M, K, N, G, bpe_in, bpe_out):
        return (2 * M * N) * bpe_out + (2 * M * K) * bpe_in + (2 * G * K * N) * bpe_in
    def bytes_bwd(self):
        return self.bytes_bwd_func(self.M, self.K, self.N, self.G, 
                                 self.bpe_in, self.bpe_out)
