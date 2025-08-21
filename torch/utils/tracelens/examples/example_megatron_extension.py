import argparse
import json
import math
import re
import logging

def tree_postprocess_extension(trace_tree):
    """
    Context: In Transformer Engine v1, the blas GEMM calls are made by tex_ts::te_gemm_ts CPU ops.
    As a result we can parse the gemm shapes from these CPU ops.
    Hoewever, in Transformer Engine v2, the GEMM calls are made directly by the_Linear and _LayerNormLinear in fwd pass 
    and _LinearBackward and _LayerNormLinearBackward in bwd pass.
    Consequently, we cannot parse the gemm shapes from the _Linear and _LayerNormLinear CPU ops.
    Moreover, since the bwd pass of linear results in 2 gemm calls (xgrad and wgrad), 
    there is no way to directly infer which is xgrad and which is wgrad.

    Solution: To solve this, we create pseudo matmul ops for the fwd and bwd passes.
    We parse the input shape (B, L, D_in) and weight shape (D_out, D_in) from the fwd pass.
    From these, we can infer yfwd, xgrad and wgrad gemm shapes.
    Secondly, by correlating the order of the GPU events and the TEv2 code,
    we can infer which gpu event is xgrad and which is wgrad.

    In summary, we inject 3 pseudo ops as follows:
    fwd pass:
        Y = X.matmul(W^T) + B
    bwd pass gemm:
        X_grad = Y_grad.matmul(W)
        W_grad = Y_grad^T.matmul(X)
    """
    if '_Linear' in trace_tree.name2event_uids and 'tex_ts::te_gemm_ts' not in trace_tree.name2event_uids:
        fwd_op_events = [trace_tree.get_UID2event(uid) for uid in trace_tree.name2event_uids['_Linear']]
        for fwd_op_event in fwd_op_events:
            create_host_mm_ops_from_linear_op(trace_tree, fwd_op_event)
    if '_LayerNormLinear' in trace_tree.name2event_uids and 'tex_ts::te_gemm_ts' not in trace_tree.name2event_uids:
        fwd_op_events = [trace_tree.get_UID2event(uid) for uid in trace_tree.name2event_uids['_LayerNormLinear']]
        for fwd_op_event in fwd_op_events:
            create_host_mm_ops_from_layernormlinear_op(trace_tree, fwd_op_event)


def get_bwd_ops_for_fwd_op(trace_tree, fwd_op_event: dict) -> list[dict]:
    """
    Get backward operations for a given forward operation.
    """
    bwd_eventUIDs = fwd_op_event.get('bwd_events')
    if not bwd_eventUIDs:
        trace_tree.link_bwd_events(fwd_op_event['UID'])
        bwd_eventUIDs = fwd_op_event.get('bwd_events')
    bwd_events = [trace_tree.get_UID2event(uid) for uid in bwd_eventUIDs]
    return bwd_events

def set_bookkeeping_attr(tree, event: dict):
    UID = len(tree.events)
    event['UID'] = UID
    tree.events.append(event)
    tree.events_by_uid[UID] = event
    seq_num = event['args']['Sequence number']
    tree.seq_num2event_uids_map[seq_num].append(UID)

def is_gemm_kernel(kernel_event: dict) -> bool:
    assert kernel_event['cat'] == 'kernel'
    kernel_name = kernel_event['name']
    pattern = r'.*C.*_A.*_B.*'
    is_rocm_gemm = bool(re.match(pattern, kernel_name))
    is_cuda_gemm = kernel_name.startswith('nvjet')
    return is_rocm_gemm or is_cuda_gemm

def _create_host_mm_ops_common(trace_tree, fwd_op_event: dict, expected_name: str, 
                               w_idx: int, x_idx: int, fp8_bool_idx: int):
    """
    Create pseudo matmul ops for forward and backward passes.
    fwd pass:
        Y = X.matmul(W^T) + B
    bwd pass gemm:
        X_grad = Y_grad.matmul(W)
        W_grad = Y_grad^T.matmul(X)
        B_grad = Y_grad.sum(dim=0)
    """
    if fwd_op_event.get('name') != expected_name:
        logging.warning(f"[Warning] Expected op name {expected_name}, found {fwd_op_event['name']}")
        return
    prefix = expected_name

    fwd_gpu_event_ids = fwd_op_event.get('gpu_events', [])
    if not fwd_gpu_event_ids:
        logging.warning(f"[Warning] No GPU events found for fwd UID {fwd_op_event['UID']}")
        return

    fwd_gpu_events = [trace_tree.get_UID2event(uid) for uid in fwd_gpu_event_ids]
    fwd_gemm_kernels = [e for e in fwd_gpu_events if is_gemm_kernel(e)]

    if len(fwd_gemm_kernels) != 1:
        logging.warning(f"[Warning] Expected 1 GEMM kernel in fwd, found {len(fwd_gemm_kernels)}")
        return

    yfwd_kernel = fwd_gemm_kernels[0]

    # Link to backward
    bwd_ops = get_bwd_ops_for_fwd_op(trace_tree, fwd_op_event)
    if not bwd_ops:
        logging.warning(f"[Warning] No backward op found for fwd UID {fwd_op_event['UID']}")
        return

    bprop_gpu_event_ids = [uid for bwd_op in bwd_ops for uid in bwd_op.get('gpu_events', [])]
    bprop_gpu_events = [trace_tree.get_UID2event(uid) for uid in bprop_gpu_event_ids]
    bprop_gemm_kernels = [e for e in bprop_gpu_events if is_gemm_kernel(e)]
    def get_launcher_start(kernel_evt):
        launcher = trace_tree.get_parent_event(kernel_evt)
        return launcher.get('ts')
    bprop_gemm_kernels = sorted(bprop_gemm_kernels, key=lambda e: get_launcher_start(e)) #which 

    if len(bprop_gemm_kernels) != 2:
        logging.warning(f"[Warning] Expected 2 GEMM kernels in bwd, found {len(bprop_gemm_kernels)}")
        return

    # Transformer Engine first launches xgrad, then wgrad
    # ref: https://github.com/NVIDIA/TransformerEngine/blob/91405eb4a184b962edb7a211626b7e58a4a87cbc/transformer_engine/pytorch/module/linear.py#L405
    # ref: https://github.com/NVIDIA/TransformerEngine/blob/91405eb4/transformer_engine/pytorch/module/layernorm_linear.py#L472
    xgrad_kernel, wgrad_kernel = bprop_gemm_kernels

    try:
        input_dims = fwd_op_event['args']['Input Dims']
        input_types = fwd_op_event['args']['Input type']
        W_shape, inp_shape = input_dims[w_idx], input_dims[x_idx]
        W_dtype, inp_dtype = input_types[w_idx], input_types[x_idx]
    except Exception as e:
        logging.warning(f"[Warning] Missing shape info in fwd UID {fwd_op_event['UID']}: {e}")
        return

    assert inp_shape[-1] == W_shape[1]
    assert W_dtype == inp_dtype

    is_fp8 = fwd_op_event['args']['Concrete Inputs'][fp8_bool_idx] == 'True'
    if is_fp8:
        W_dtype = 'fp8'
        inp_dtype = 'fp8'

    X_shape = (math.prod(inp_shape[:-1]), inp_shape[-1])
    Y_grad_shape = (X_shape[0], W_shape[0])

    # Check if pseudo ops already exist
    seq_num = fwd_op_event['args']['Sequence number']
    seq_num_uids = trace_tree.seq_num2event_uids_map.get(seq_num, [])
    seq_num_evts = [trace_tree.get_UID2event(uid) for uid in seq_num_uids]

    existing = [e for e in seq_num_evts if e['name'] == f'{prefix}Backward_xgrad_mm']
    if existing:
        return

    # Create pseudo host ops
    inject_pseudo_op(trace_tree, yfwd_kernel, f'{prefix}_yfwd_mm', seq_num,
                     [X_shape, W_shape[::-1]], [inp_dtype, inp_dtype], [], [])
    inject_pseudo_op(trace_tree, xgrad_kernel, f'{prefix}Backward_xgrad_mm', seq_num,
                     [Y_grad_shape, W_shape], [inp_dtype, inp_dtype], [], [])
    inject_pseudo_op(trace_tree, wgrad_kernel, f'{prefix}Backward_wgrad_mm', seq_num,
                     [Y_grad_shape[::-1], X_shape], [inp_dtype, inp_dtype], [], [])

def create_host_mm_ops_from_linear_op(trace_tree, fwd_op_event: dict):
    # index 0 is Linear weight tensor, 1 is input tensor
    # ref: https://github.com/NVIDIA/TransformerEngine/blob/91405eb4a184b962edb7a211626b7e58a4a87cbc/transformer_engine/pytorch/module/linear.py#L405
    _create_host_mm_ops_common(
        trace_tree, fwd_op_event,
        expected_name='_Linear',
        w_idx=0,
        x_idx=1,
        fp8_bool_idx=4
    )

def create_host_mm_ops_from_layernormlinear_op(trace_tree, fwd_op_event: dict):
    # index 3 is Linear weight tensor, 0 is input tensor
    # ref: https://github.com/NVIDIA/TransformerEngine/blob/91405eb4a184b962edb7a211626b7e58a4a87cbc/transformer_engine/pytorch/module/layernorm_linear.py#L1434
    _create_host_mm_ops_common(
        trace_tree, fwd_op_event,
        expected_name='_LayerNormLinear',
        w_idx=3,
        x_idx=0,
        fp8_bool_idx=7
    )

def inject_pseudo_op(tree, kernel_evt, name, seq_num, dims=None, types=None, strides=None, concrete_inputs=None):
    """
    Create a pseudo host-side op for `kernel_evt` and make the *original*
    CPU op point to it instead of the kernel-launcher stub.

    Parameters
    ----------
    tree : TraceTree
    kernel_evt : dict          # the GPU kernel event
    name : str                 # pseudo-op name
    seq_num : int
    dims : list, optional  # 'Input Dims'
    types : list, optional  # 'Input type'
    strides : list, optional   # 'Input Strides'
    concrete_inputs : list, optional  # 'Concrete Inputs'
    """
    # ── climb two levels: kernel -> launcher -> original CPU op ───────────
    launcher_evt   = tree.get_parent_event(kernel_evt)          # cpu-launch stub
    orig_cpu_evt   = tree.get_parent_event(launcher_evt)        # high-level op

    pseudo_evt = {
        'ph' : 'X',
        'name' : name,
        'cat' : 'cpu_op',
        'pid' : orig_cpu_evt['pid'],
        'tid' : orig_cpu_evt['tid'],
        'args': {
            'Input Dims'     : orig_cpu_evt['args']['Input Dims'] if dims is None else dims,
            'Input type'     : orig_cpu_evt['args']['Input type'] if types is None else types,
            'Input Strides'  : orig_cpu_evt['args']['Input Strides'] if strides is None else strides,
            'Concrete Inputs': orig_cpu_evt['args']['Concrete Inputs'] if concrete_inputs is None else concrete_inputs,
            'Sequence number': seq_num,
            'External id'    : kernel_evt['args']['correlation'],
            'Pseudo op'      : True,
        },
        'children'  : [launcher_evt['UID']],    # we still nest the launcher
        'gpu_events': [kernel_evt['UID']],
    }
    set_bookkeeping_attr(tree, pseudo_evt)
    # ── re-wire the original CPU op ───────────────────────────────────────
    children = orig_cpu_evt['children']
    children.remove(launcher_evt['UID'])
    children.append(pseudo_evt['UID'])

# we also need to 
def categorize_extension(row, plugin):
    """
    Categorizer plugin to categorize the kernel launchers.
    """
    if row['name'] in ['_Linear_fwd_mm', '_LayerNormLinear_fwd_mm',
                          '_LinearBackward_xgrad_mm', '_LinearBackward_wgrad_mm',
                          '_LayerNormLinearBackward_xgrad_mm', '_LayerNormLinearBackward_wgrad_mm']:
          return 'GEMM'
    if row['name'] == 'FusedAttnFunc':
        return 'SDPA_fwd'
    if row['name'] == 'FusedAttnFuncBackward':
        return 'SDPA_bwd'
    return None


# extending the perf model to catch the pseudo ops
from TraceLens.PerfModel import GEMM, name2bpe

# Step 1: Define the new Perf Model class for the pseudo GEMM operations
# We already have a base class GEMM, so we can extend it
class tev2_pseudo_gemm(GEMM):
    # TODO: need to cleanup and reuse perf models better
    @staticmethod
    def get_param_details(event):
        input_dims = event['args']['Input Dims']
        A_shape, B_shape = input_dims[0], input_dims[1]
        M = A_shape[0]
        N = B_shape[1]
        K = A_shape[1]

        dtype_A_B = 'fp8', 'fp8'
        stride_A, stride_B = None, None
        return {"M": M, "N": N, "K": K, "bias": False,
                "stride_A": stride_A, "stride_B": stride_B,
                "dtype_A_B": dtype_A_B}

    def bytes(self):
        dtype_A_B = self.param_details['dtype_A_B']
        if dtype_A_B[0] != dtype_A_B[1]:
            raise ValueError(f"Data types of A and B are different: {dtype_A_B}")
        self.bpe_in = 1 #for fp8 gemm
        # irrespective of the input dtype, the output dtype is always fp16/bf16
        self.bpe_out = 2 
        return super().bytes(bpe_mat1=self.bpe_in, bpe_mat2=self.bpe_in,
                             bpe_bias=self.bpe_in, # does not matter
                             bpe_output=self.bpe_out)
    def flops_bwd(self):
        raise NotImplementedError("Backward pass for tev2_pseudo_gemm is not defined.")
    def bytes_bwd(self):
        raise NotImplementedError("Backward pass for tev2_pseudo_gemm is not defined.")

from TraceLens.PerfModel import SDPA, GroupedGemm, extract_sdpa_cfg
class transformer_engine_attention(SDPA):
    """
    Context: The FusedAttnFunc is a pytorch extention for the attention kernel.
    Unfortunately, the args does not have a bool flag for is_causal.
    Instead, it has a str arg which is not recorded in the trace.

    Solution: Based on the LLM use case we make the assumption that
    the attention is always causal.
    Since this might not be the case for other use cases, 
    we dont add this natively to the perf model and instead add it here 
    """
    @staticmethod
    def get_param_details(event):
    # ref TransformerEngine/transformer_engine/pytorch/cpp_extensions/fused_attn.py
    # https://github.com/NVIDIA/TransformerEngine/blob/51cd441501e8e6dee18c00056f008e1b53b89ebd/transformer_engine/pytorch/attention/dot_product_attention/backends.py#L881
        input_dims = event['args']['Input Dims']
        q_idx, k_idx, v_idx = 9, 10, 11 # this is the idx in the args list
        q_shape, k_shape, v_shape = input_dims[q_idx], input_dims[k_idx], input_dims[v_idx]
        bhnd_idx = (1, 2, 0, 3)
        sdpa_cfg = extract_sdpa_cfg(q_shape, k_shape, v_shape, bhnd_idx)
        B, N_Q, H_Q, N_KV, H_KV, d_h_qk, d_h_v = (sdpa_cfg[key] for key in ['B', 'N_Q', 'H_Q', 'N_KV', 'H_KV', 'd_h_qk', 'd_h_v'])

        strides = event['args']['Input Strides']
        q_strides, k_strides, v_strides = strides[q_idx], strides[k_idx], strides[v_idx]
        # convert stride to tuple
        q_strides, k_strides, v_strides = tuple(q_strides), tuple(k_strides), tuple(v_strides)

        is_causal = True
        dropout_p = 0.0
        flash_impl = True
        return {"B": B, "N_Q": N_Q, "H_Q": H_Q, "N_KV": N_KV, "H_KV": H_KV, "d_h_qk": d_h_qk, "d_h_v": d_h_v,
                "q_strides": q_strides, "k_strides": k_strides, "v_strides": v_strides,
                "dropout": dropout_p, "causal": is_causal, "flash_impl": flash_impl}

import ast
class custom_grouped_gemm(GroupedGemm):

    @staticmethod
    def get_param_details(event):
        input_dims = event['args']['Input Dims']
        X_dims = input_dims[0]
        W_dims = input_dims[1]
        transpose = ast.literal_eval(event['args']['Concrete Inputs'][3])
        M, K = X_dims
        if transpose:
            assert K == W_dims[-1], f"Expected K dimension to match W_dims[-1] for transposed GEMM, got {K} and {W_dims[-1]}"
            G, N, K = W_dims
        else:
            assert K == W_dims[1], f"Expected K dimension to match W_dims[1] for non-transposed GEMM, got {K} and {W_dims[1]}"
            G, K, N = W_dims
        input_types = event['args']['Input type']
        dtype_X = input_types[0]
        dtype_W = input_types[1]
        assert dtype_X == dtype_W, f"Expected input types to match, got {dtype_X} and {dtype_W}"
        bpe_in = name2bpe(dtype_X)
        if bpe_in == 1 or bpe_in == 2:
            bpe_out = 2
        else:
            raise ValueError(f"Expected bpe_in to be 1 or 2, got {bpe_in}")
        return {"M": M, "K": K, "N": N, "G": G,
                "bpe_in": bpe_in, "bpe_out": bpe_out}

# Step 2: Register the new Perf Model class in the mapping
perf_model_extension = {
    '_Linear_yfwd_mm': tev2_pseudo_gemm,
    '_LinearBackward_xgrad_mm': tev2_pseudo_gemm,
    '_LinearBackward_wgrad_mm': tev2_pseudo_gemm,
    '_LayerNormLinear_yfwd_mm': tev2_pseudo_gemm,
    '_LayerNormLinearBackward_xgrad_mm': tev2_pseudo_gemm,
    '_LayerNormLinearBackward_wgrad_mm': tev2_pseudo_gemm,

    'FusedAttnFunc': transformer_engine_attention,
    'GroupedGemm': custom_grouped_gemm,
}

dict_cat2names_extension = {
    'GEMM': ['_Linear_yfwd_mm', '_LinearBackward_xgrad_mm', '_LinearBackward_wgrad_mm',
             '_LayerNormLinear_yfwd_mm', '_LayerNormLinearBackward_xgrad_mm', '_LayerNormLinearBackward_wgrad_mm'],
    'SDPA': ['FusedAttnFunc'],
    'GroupedGEMM': ['GroupedGemm'],
}


