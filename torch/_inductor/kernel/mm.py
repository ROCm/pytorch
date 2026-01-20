# mypy: allow-untyped-defs
import functools
import json
import logging
import os
import subprocess
import tempfile
import time
from typing import Any, Optional
import math

import torch
from torch._dynamo.utils import counters, identity
from torch._inductor.autoheuristic.autoheuristic import AutoHeuristicSelectAlgorithm
from torch._inductor.autoheuristic.autoheuristic_utils import (
    AHContext,
    context_add_strides,
    context_add_using_tf32,
    mm_operations,
)
from torch._inductor.codegen.cpp_gemm_template import CppGemmTemplate
from torch._inductor.remote_gemm_autotune_cache import gen_best_config
from torch._inductor.virtualized import V
from torch.fx.experimental.proxy_tensor import make_fx
from torch.torch_version import TorchVersion

from .. import config as inductor_config
from ..codegen.cuda.gemm_template import CUTLASS2xGemmTemplate, CUTLASS3xGemmTemplate
from ..codegen.rocm.ck_tile_universal_gemm_template import CKTileGemmTemplate
from ..codegen.rocm.ck_universal_gemm_template import CKGemmTemplate
from ..codegen.subgraph import SubgraphChoiceCaller, SubgraphTemplate
from ..ir import Buffer, ChoiceCaller, FlexibleLayout, is_triton, Layout
from ..kernel_inputs import MMKernelInputs
from ..lowering import add_layout_constraint, constrain_to_fx_strides, register_lowering, empty_strided, zeros_strided
from ..select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
    realize_inputs,
    SymbolicGridFn,
    TritonTemplate,
)
from ..utils import (
    _use_cutlass_for_op,
    use_aten_gemm_kernels,
    use_ck_gemm_template,
    use_ck_tile_gemm_template,
    use_cpp_gemm_template,
    use_cutlass_template,
    use_decompose_k_choice,
    use_triton_template,
    use_triton_tma_template,
)
from .mm_common import _is_static_problem, mm_args, mm_grid, persistent_mm_grid


try:
    import triton

    triton_version = TorchVersion(triton.__version__)
    has_triton = True
except ImportError:
    triton_version = TorchVersion("0.0.0")
    has_triton = False

log = logging.getLogger(__name__)
aten = torch.ops.aten
prims = torch.ops.prims

# StreamK and Origami configuration
ENABLE_STREAMK = os.environ.get("TORCHINDUCTOR_ENABLE_STREAMK", "1") == "1"
# Origami is now enabled by default for StreamK (no env var needed)
STREAMK_MIN_PROBLEM_SIZE = int(os.environ.get("STREAMK_MIN_PROBLEM_SIZE", str(int(1e8))))

# Experimental force modes
FORCE_STREAMK = os.environ.get("FORCE_STREAMK", "0") == "1"
STREAMK_ONLY = os.environ.get("STREAMK_ONLY", "0") == "1"
DISABLE_OTHER_BACKENDS = os.environ.get("DISABLE_OTHER_BACKENDS", "0") == "1"

# StreamK Debug logging controls
STREAMK_DEBUG = os.environ.get("STREAMK_DEBUG", "0") == "1"
STREAMK_VERBOSE = os.environ.get("STREAMK_VERBOSE", "0") == "1"

# Startup verification - this should always print if the module is loaded
if STREAMK_DEBUG or STREAMK_VERBOSE or FORCE_STREAMK or STREAMK_ONLY:
    print(f"[StreamK-STARTUP] Debug flags: STREAMK_DEBUG={STREAMK_DEBUG}, STREAMK_VERBOSE={STREAMK_VERBOSE}")
    print(f"[StreamK-STARTUP] StreamK enabled: {ENABLE_STREAMK}, Origami enabled: True (default)")
    print(f"[StreamK-STARTUP] Min problem size: {STREAMK_MIN_PROBLEM_SIZE}")
    print(f"[StreamK-STARTUP] 🔥 FORCE modes: FORCE_STREAMK={FORCE_STREAMK}, STREAMK_ONLY={STREAMK_ONLY}")
    print(f"[StreamK-STARTUP] Disable others: {DISABLE_OTHER_BACKENDS}")

# Also log through the logging system
log.info(
    f"StreamK module loaded - DEBUG={STREAMK_DEBUG}, VERBOSE={STREAMK_VERBOSE}, "
    f"ENABLED={ENABLE_STREAMK}, FORCE={FORCE_STREAMK}"
)


def streamk_log_info(msg):
    """Always log StreamK info messages"""
    log.info(f"[StreamK] {msg}")


def streamk_log_debug(msg):
    """Log StreamK debug messages if enabled"""
    if STREAMK_DEBUG:
        log.info(f"[StreamK-DEBUG] {msg}")  # Use info level so it shows up
    else:
        log.debug(f"[StreamK] {msg}")


def streamk_log_verbose(msg):
    """Log StreamK verbose messages if enabled"""
    if STREAMK_VERBOSE:
        log.info(f"[StreamK-VERBOSE] {msg}")  # Use info level so it shows up
    elif STREAMK_DEBUG:
        log.debug(f"[StreamK] {msg}")


def log_choices_summary(choices, problem_desc):
    """Log summary of all choices for debugging"""
    if STREAMK_DEBUG or STREAMK_VERBOSE:
        streamk_log_debug(f"Choice summary for {problem_desc}:")
        choice_types = {}
        for choice in choices:
            choice_name = getattr(choice, 'name', str(type(choice).__name__))
            if 'streamk' in choice_name.lower():
                choice_name = f"🟢 {choice_name}"  # Highlight StreamK choices
            choice_types[choice_name] = choice_types.get(choice_name, 0) + 1

        for choice_type, count in sorted(choice_types.items()):
            streamk_log_debug(f"  - {choice_type}: {count} configs")
        streamk_log_debug(f"Total choices: {len(choices)}")


def test_streamk_logging():
    """Test function to verify logging is working"""
    print("[StreamK-TEST] Testing logging functions...")
    streamk_log_info("Test info message")
    streamk_log_debug("Test debug message")
    streamk_log_verbose("Test verbose message")
    print("[StreamK-TEST] If you see this, the module loaded successfully")


def _safe_even_k_check(k, block_k):
    """Safely check if K is evenly divisible by block_k, handling symbolic variables"""
    try:
        return k % block_k == 0
    except (TypeError, AttributeError):
        # Symbolic variable - assume it might not be even
        return False


# Run test if debug is enabled
if STREAMK_DEBUG:
    test_streamk_logging()


def _get_hardware_chiplet_count():
    """Get actual hardware chiplet count using origami detection"""
    try:
        import origami
        hardware = origami.get_hardware_for_device(0)
        num_xcds = getattr(hardware, 'NUM_XCD', 1)
        if STREAMK_DEBUG:
            print(f"[CHIPLET-DEBUG] Hardware detection: NUM_XCD={num_xcds}")
        return max(1, num_xcds)  # Ensure at least 1
    except (ImportError, AttributeError) as e:
        if STREAMK_DEBUG:
            print(f"[CHIPLET-DEBUG] Hardware detection failed: {e}, defaulting to 1 chiplet")
        return 1


class StreamKOrigamiSelector:
    """Origami-based selector for StreamK configuration following tritonBLAS pattern"""

    # Dtype to string mapping (from tritonBLAS)
    dtype_to_str = {
        torch.float32: "f32",
        torch.complex64: "c32",
        torch.complex128: "c64",
        torch.float64: "f64",
        torch.float16: "f16",
        torch.int32: "i32",
        torch.bfloat16: "bf16",
        torch.int8: "i8",
        torch.float8_e5m2: "f8",
        torch.float8_e4m3fn: "f8",
    }
    # Add FP8 FNUZ variants if available
    if hasattr(torch, "float8_e5m2fnuz"):
        dtype_to_str[torch.float8_e5m2fnuz] = "f8"
    if hasattr(torch, "float8_e4m3fnuz"):
        dtype_to_str[torch.float8_e4m3fnuz] = "f8"

    def __init__(self, M, N, K, a_dtype, b_dtype, c_dtype, device):
        init_start = time.perf_counter()
        if STREAMK_DEBUG:
            print(f"🔍 [ORIGAMI-SELECTOR-DEBUG] Initializing StreamKOrigamiSelector for {M}x{N}x{K}")
        self.M = M
        self.N = N
        self.K = K
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.c_dtype = c_dtype
        self.device = device

        # Get hardware information (tritonBLAS style)
        hw_start = time.perf_counter()
        try:
            # Try to use origami hardware detection
            import origami
            self.hardware = origami.get_hardware_for_device(0)
            self.num_sms = self.hardware.N_CU
            if STREAMK_DEBUG:
                print(f"🔍 [ORIGAMI-SELECTOR-DEBUG] Using origami hardware detection: N_CU={self.num_sms}")
        except (ImportError, AttributeError) as e:
            # Fallback to CUDA device properties
            print(f"🔍 [ORIGAMI-SELECTOR-DEBUG] Origami hardware detection failed: {e}, falling back to CUDA")
            try:
                if torch.cuda.is_available() and hasattr(device, 'index'):
                    props = torch.cuda.get_device_properties(device.index)
                    self.num_sms = props.multi_processor_count
                    # Create mock hardware object for compatibility
                    self.hardware = type('Hardware', (), {'N_CU': self.num_sms})()
                    print(f"🔍 [ORIGAMI-SELECTOR-DEBUG] Using CUDA device properties: multi_processor_count={self.num_sms}")
                else:
                    self.num_sms = 108  # Default fallback
                    self.hardware = type('Hardware', (), {'N_CU': 108})()
                    print(f"🔍 [ORIGAMI-SELECTOR-DEBUG] Using default fallback: num_sms=108")
            except Exception as e2:
                self.num_sms = 108
                self.hardware = type('Hardware', (), {'N_CU': 108})()
                print(f"🔍 [ORIGAMI-SELECTOR-DEBUG] CUDA detection failed: {e2}, using default: num_sms=108")

        # Initialize configuration ranges (from tritonBLAS)
        self.block_mn_range = [16, 32, 64, 128, 256]
        self.block_k_range = [16, 32, 64, 128, 256, 512]

        # Get element sizes and infer MI dimensions
        self.element_size_A = self._get_dtype_bits(a_dtype)
        self.element_size_B = self._get_dtype_bits(b_dtype)
        self.element_size_out = self._get_dtype_bits(c_dtype)

        # Set MI dtype - use input dtype for matrix instruction type
        input_dtype_for_mi = a_dtype if self._get_dtype_bits(a_dtype) <= self._get_dtype_bits(b_dtype) else b_dtype
        self.mi_dtype = self.dtype_to_str.get(input_dtype_for_mi, self.dtype_to_str.get(c_dtype))

        # Infer Matrix Instruction Dimensions (tritonBLAS style)
        self.MI_dim = self._infer_matrix_instruction_dimensions(self.element_size_A, self.element_size_B)

        # StreamK grid constants (from tritonBLAS)
        self.split_factors = [8, 6, 4, 3, 2, 1]
        self.tile_fractions = [0.0, 1.0/2.0, 1.0/8.0, 1.0/5.0, 1.0/4.0, 1.0/3.0]
        self.max_workspace = 128 * 1024 * 1024

        hw_end = time.perf_counter()
        if STREAMK_DEBUG:
            print(f"⏱️  [TIMING] Hardware detection took: {(hw_end - hw_start)*1000:.2f}ms")

        # Compute optimal configuration and grid
        config_start = time.perf_counter()
        self.config = self._compute_optimal_config()
        config_end = time.perf_counter()

        grid_start = time.perf_counter()
        self.grid = self._compute_streamk_grid()
        grid_end = time.perf_counter()

        init_end = time.perf_counter()

        if STREAMK_DEBUG:
            print(f"⏱️  [TIMING] Config computation took: {(config_end - config_start)*1000:.2f}ms")
            print(f"⏱️  [TIMING] Grid computation took: {(grid_end - grid_start)*1000:.2f}ms")
            print(f"⏱️  [TIMING] Total StreamKOrigamiSelector init took: {(init_end - init_start)*1000:.2f}ms")
            print(f"🔍 [ORIGAMI-SELECTOR-DEBUG] Final config: BLK_M={self.config[0]}, BLK_N={self.config[1]}, BLK_K={self.config[2]}, GROUP_M={self.config[3]}, GRID={self.grid}")

    def _get_dtype_bits(self, dtype):
        """Get bits for torch dtypes"""
        try:
            return torch.finfo(dtype).bits
        except TypeError:
            return torch.iinfo(dtype).bits

    def _infer_matrix_instruction_dimensions(self, element_size_A, element_size_B):
        """Infer MI dimensions based on hardware and data types (from tritonBLAS)"""
        MI_dim = None
        is_gfx942 = self.hardware.N_CU in [304, 80, 64]

        # gfx950
        if self.hardware.N_CU == 256:
            if max(element_size_A, element_size_B) == 32:  # FP32
                MI_dim = [16, 16, 4]
            elif max(element_size_A, element_size_B) == 16:  # FP16/BF16
                MI_dim = [16, 16, 32]
            elif max(element_size_A, element_size_B) <= 8:  # F4F6F8
                if hasattr(self.K, '__mod__') and self.K % 256 == 0:
                    self.block_k_range = [256]
                else:
                    self.block_k_range = [128]
                self.block_mn_range = [32, 64, 128, 256]
                MI_dim = [16, 16, 128]

        # gfx942 (304 CUs full, 80 CUs partitioned)
        elif is_gfx942:
            if max(element_size_A, element_size_B) == 32:  # FP32
                MI_dim = [16, 16, 4]
            elif max(element_size_A, element_size_B) == 16:  # FP16/BF16
                MI_dim = [16, 16, 16]
            elif max(element_size_A, element_size_B) == 8:  # F8
                MI_dim = [16, 16, 32]
                self.block_mn_range = self.block_mn_range + [512]
                self.block_k_range = self.block_k_range + [128, 256]
            elif max(element_size_A, element_size_B) < 8:  # F4F6
                raise ValueError("gfx942 doesn't support F4/F6")

        # gfx942 228 CUs
        elif self.hardware.N_CU == 228:
            if max(element_size_A, element_size_B) == 32:  # FP32
                MI_dim = [16, 16, 4]
            elif max(element_size_A, element_size_B) == 16:  # FP16/BF16
                MI_dim = [16, 16, 16]
            elif max(element_size_A, element_size_B) == 8:  # F8
                MI_dim = [16, 16, 32]
                self.block_mn_range = self.block_mn_range + [512]
                self.block_k_range = self.block_k_range + [128, 256]
            elif max(element_size_A, element_size_B) < 8:  # F4F6
                raise ValueError("gfx942 228CUs doesn't support F4/F6")

        # gfx90s 104 CUs
        elif self.hardware.N_CU == 104:
            if max(element_size_A, element_size_B) == 32:  # FP32
                MI_dim = [16, 16, 4]
            elif max(element_size_A, element_size_B) == 16:  # FP16/BF16
                MI_dim = [16, 16, 16]
            elif max(element_size_A, element_size_B) == 8:  # F8
                raise ValueError("gfx90s doesn't support F8")
            elif max(element_size_A, element_size_B) < 8:  # F4F6
                raise ValueError("gfx90s doesn't support F4/F6")

        # Default fallback for unknown architectures
        if MI_dim is None:
            if max(element_size_A, element_size_B) == 32:
                MI_dim = [16, 16, 4]
            elif max(element_size_A, element_size_B) == 16:
                MI_dim = [16, 16, 16]
            else:
                MI_dim = [16, 16, 32]

        return MI_dim

    def _compute_optimal_config(self):
        """Compute optimal tile configuration using tritonBLAS-style selection"""
        try:
            # Try to use origami for optimal tile selection
            import origami
            tiles_start = time.perf_counter()
            valid_tiles = self._get_valid_tiles()
            tiles_end = time.perf_counter()

            if STREAMK_DEBUG:
                print(f"⏱️  [TIMING] get_valid_tiles took: {(tiles_end - tiles_start)*1000:.2f}ms")

            macro_start = time.perf_counter()
            results = origami.select_best_macro_tile_size(
                self.M, self.N, self.K,
                1,  # Batch
                True,  # transA
                False,  # transB
                self.hardware,
                valid_tiles,
                self.element_size_A,
                self.element_size_B,
                self.element_size_out,
                origami.string_to_datatype(self.mi_dtype),
                0,  # MX Block Size
                0.8,  # H_L2
                False,  # debug
                False,  # Print
                6,  # WGM
            )
            macro_end = time.perf_counter()

            if STREAMK_DEBUG:
                print(f"⏱️  [TIMING] origami.select_best_macro_tile_size took: {(macro_end - macro_start)*1000:.2f}ms")

            best_result = results[0]

            # Heuristic weighting for gfx942
            if self.hardware.N_CU in [304, 80, 64]:
                if best_result[1] == 256 and best_result[2] == 256:
                    if results[0][0] * 1.00 > results[1][0]:
                        best_result = results[1]

            BLK_M, BLK_N, BLK_K = best_result[1], best_result[2], best_result[3]

            # Apply more accurate shared memory constraints matching tritonBLAS behavior
            # Real hardware limit is 65536 bytes, but tritonBLAS successfully uses larger tiles
            max_shared_memory = 65536

            # More accurate shared memory estimation:
            # - Only count A_shared (BLK_M * BLK_K) and B_shared (BLK_K * BLK_N) tiles
            # - Account for proper data types and padding
            element_size = 2 if self.a_dtype in (torch.float16, torch.bfloat16) else 4
            a_shared_size = BLK_M * BLK_K * element_size
            b_shared_size = BLK_K * BLK_N * element_size

            # Add minimal overhead (not the inflated 4KB I used before)
            padding_overhead = 1024  # 1KB for alignment and miscellaneous
            estimated_usage = a_shared_size + b_shared_size + padding_overhead

            # Since tritonBLAS succeeds with 256x256x64 (estimated ~67KB), be less conservative
            # Only reduce if we're significantly over the limit
            if estimated_usage > max_shared_memory * 1.1:  # 10% tolerance
                if STREAMK_DEBUG:
                    print(f"🔍 [ORIGAMI-SELECTOR-DEBUG] Shared memory constraint: {estimated_usage} > {max_shared_memory*1.1:.0f}, reducing block sizes")
                    print(f"🔍 [ORIGAMI-SELECTOR-DEBUG] A_shared={a_shared_size}, B_shared={b_shared_size}, overhead={padding_overhead}")

                # Try a middle ground first (192x192x48) before going to 128x128x32
                if BLK_M == 256 and BLK_N == 256:
                    BLK_M, BLK_N = 192, 192
                    BLK_K = min(BLK_K, 48)
                    # Re-estimate with new sizes
                    a_shared_size = BLK_M * BLK_K * element_size
                    b_shared_size = BLK_K * BLK_N * element_size
                    estimated_usage = a_shared_size + b_shared_size + padding_overhead

                # If still too big, then go conservative
                if estimated_usage > max_shared_memory * 1.05:  # 5% tolerance
                    if self.a_dtype in (torch.float16, torch.bfloat16):
                        BLK_M, BLK_N = 128, 128
                        BLK_K = 32
                    else:
                        BLK_M, BLK_N = 64, 64
                        BLK_K = 32

                if STREAMK_DEBUG:
                    print(f"🔍 [ORIGAMI-SELECTOR-DEBUG] Adjusted to: BLK_M={BLK_M}, BLK_N={BLK_N}, BLK_K={BLK_K}")
            else:
                if STREAMK_DEBUG:
                    print(f"🔍 [ORIGAMI-SELECTOR-DEBUG] Shared memory OK: {estimated_usage} <= {max_shared_memory*1.1:.0f}, keeping optimal sizes")

            # Get optimal group size
            try:
                wgm_start = time.perf_counter()
                group_m_results = origami.select_best_wgm(
                    self.M, self.N, self.K, 1, self.hardware,
                    BLK_M, BLK_N, BLK_K,
                    self.MI_dim[0], self.MI_dim[1], self.MI_dim[2],
                    [1, 2, 4, 6, 8],
                    self.element_size_A,
                    0.8, False, False
                )
                wgm_end = time.perf_counter()
                group_m = group_m_results[1]

                if STREAMK_DEBUG:
                    print(f"⏱️  [TIMING] origami.select_best_wgm took: {(wgm_end - wgm_start)*1000:.2f}ms")
            except:
                group_m = 8 if (BLK_M >= 128 and BLK_N >= 128) else 4

        except (ImportError, AttributeError, Exception) as e:
            # Fallback to heuristic-based selection
            log.debug(f"Origami optimization failed: {e}, using heuristics")

            # Select block sizes based on problem size and dtype - conservative for shared memory
            if self.a_dtype in (torch.float16, torch.bfloat16):
                if self.M >= 2048 and self.N >= 2048:
                    BLK_M, BLK_N = 128, 128  # Reduced from 256,128 for shared memory
                    BLK_K = 32  # Reduced from 64 for shared memory
                elif self.M >= 1024 and self.N >= 1024:
                    BLK_M, BLK_N = 128, 128
                    BLK_K = 32
                else:
                    BLK_M, BLK_N = 64, 64
                    BLK_K = 32
            elif self.a_dtype == torch.float32:
                if self.M >= 1024 and self.N >= 1024:
                    BLK_M, BLK_N = 64, 64  # Reduced for FP32
                    BLK_K = 16
                else:
                    BLK_M, BLK_N = 64, 64
                    BLK_K = 16
            else:
                BLK_M, BLK_N = 64, 64
                BLK_K = 32

            group_m = 8 if (BLK_M >= 128 and BLK_N >= 128) else 4

        return (BLK_M, BLK_N, BLK_K, group_m)

    def _get_valid_tiles(self):
        """Get valid tile configurations for origami"""
        import itertools
        return list(itertools.product(
            self.block_mn_range,
            self.block_mn_range,
            self.block_k_range,
            [self.MI_dim[0]],  # MI_M
            [self.MI_dim[1]],  # MI_N
            [self.MI_dim[2]],  # MI_K
            [1],  # kernel_occupancy
        ))

    def _compute_streamk_grid(self):
        """Compute StreamK grid size following tritonBLAS logic"""
        BLK_M, BLK_N, BLK_K, _ = self.config

        # Calculate total tiles
        tiles_m = math.ceil(self.M / BLK_M) if hasattr(self.M, '__truediv__') else (self.M + BLK_M - 1) // BLK_M
        tiles_n = math.ceil(self.N / BLK_N) if hasattr(self.N, '__truediv__') else (self.N + BLK_N - 1) // BLK_N
        total_tiles = tiles_m * tiles_n

        # StreamK grid computation (from tritonBLAS origami.py:301)
        sk_grid = total_tiles
        iters_per_tile = max(1, math.ceil(self.K / BLK_K) if hasattr(self.K, '__truediv__') else (self.K + BLK_K - 1) // BLK_K)

        # More tiles than CUs: try fractional splits to distribute work
        if total_tiles > self.num_sms:
            virt_cu_count = self.num_sms
            min_even_tiles = total_tiles / virt_cu_count

            for frac in self.tile_fractions:
                # Compute candidate grid with rounding
                frac_grid = int((total_tiles / (min_even_tiles + frac)) + 0.5)

                # Skip if this split leaves a remainder AND workspace is too large
                if (total_tiles % frac_grid != 0 and
                    self._partial_tile_size(frac_grid) > self.max_workspace):
                    continue

                # Accept the first grid no larger than the virtual CU count
                if frac_grid <= virt_cu_count:
                    sk_grid = frac_grid
                    break

        # Fewer tiles than CUs: split along k-dimension up to some factor
        elif total_tiles < self.num_sms:
            for factor in self.split_factors:
                split_grid = total_tiles * factor
                iters_per_cu = iters_per_tile // factor

                if split_grid <= self.num_sms and iters_per_cu >= 8:
                    sk_grid = split_grid
                    break

        # Final check: if the chosen grid leaves a remainder AND
        # workspace exceeds what the problem allows, fall back to no split
        if total_tiles % sk_grid != 0:
            sk_grid = total_tiles

        # Last wave optimization for gfx942
        if total_tiles >= self.hardware.N_CU:
            last_wave_remainder = total_tiles % self.hardware.N_CU

            if (last_wave_remainder < 128 and last_wave_remainder > 0 and
                self.hardware.N_CU in [304, 80, 64]):  # gfx942
                sk_grid = 256 if self.hardware.N_CU == 304 else 64

        return sk_grid

    def _partial_tile_size(self, sk_grid):
        """Compute partial tile size for workspace calculation"""
        BLK_M, BLK_N, _, _ = self.config
        bytes_per_elem = self.element_size_out // 8
        tile_size = BLK_M * BLK_N * bytes_per_elem
        return tile_size * sk_grid

    def get_config(self):
        """Return (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)"""
        return self.config

    def get_grid(self):
        """Return optimal StreamK grid size"""
        return self.grid


class MatmulHeuristicResult:
    """PyTorch equivalent of tritonBLAS MatmulHeuristicResult for StreamK"""

    def __init__(self, M, N, K, a_dtype, b_dtype, c_dtype, device, streamk=True):
        self.M = M
        self.N = N
        self.K = K
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.c_dtype = c_dtype
        self.device = device
        self.streamk = streamk

        # Get device properties
        try:
            if torch.cuda.is_available() and hasattr(device, 'index'):
                props = torch.cuda.get_device_properties(device.index)
                self.num_sms = props.multi_processor_count
            else:
                self.num_sms = 108  # Default fallback
        except:
            self.num_sms = 108

        # Compute configuration and grid
        self.config = self._prepare_config()
        if streamk:
            self.grid = self._compute_sk_grid()
        else:
            self.grid = self.num_sms

    def _prepare_config(self):
        """Prepare optimal block configuration following tritonBLAS patterns"""
        # Select block sizes based on problem size and dtype - conservative for shared memory limits
        # Always use tritonBLAS fixed settings: num_warps=8, num_stages=2
        if self.a_dtype in (torch.float16, torch.bfloat16):
            if self.M >= 4096 and self.N >= 4096:
                # Very large problems: use moderate sizes to fit in shared memory
                block_m, block_n = 128, 64
                block_k = 32
                num_warps = 8  # Fixed like tritonBLAS
                num_stages = 2  # Fixed like tritonBLAS
            elif self.M >= 2048 and self.N >= 2048:
                # Large problems: balanced sizes
                block_m, block_n = 128, 128
                block_k = 32
                num_warps = 8  # Fixed like tritonBLAS
                num_stages = 2  # Fixed like tritonBLAS
            elif self.M >= 1024 and self.N >= 1024:
                block_m, block_n = 128, 128
                block_k = 32
                num_warps = 8  # Fixed like tritonBLAS
                num_stages = 2  # Fixed like tritonBLAS
            else:
                block_m, block_n = 64, 64
                block_k = 32
                num_warps = 8  # Fixed like tritonBLAS
                num_stages = 2  # Fixed like tritonBLAS
        elif self.a_dtype == torch.float32:
            # Float32 uses more memory, be more conservative
            if self.M >= 2048 and self.N >= 2048:
                block_m, block_n = 64, 64
                block_k = 16
                num_warps = 8  # Fixed like tritonBLAS
                num_stages = 2  # Fixed like tritonBLAS
            elif self.M >= 1024 and self.N >= 1024:
                block_m, block_n = 64, 64
                block_k = 16
                num_warps = 8  # Fixed like tritonBLAS
                num_stages = 2  # Fixed like tritonBLAS
            else:
                block_m, block_n = 64, 64
                block_k = 16
                num_warps = 8  # Fixed like tritonBLAS
                num_stages = 2  # Fixed like tritonBLAS
        else:
            # Default configuration: conservative
            block_m, block_n = 64, 64
            block_k = 32
            num_warps = 8  # Fixed like tritonBLAS
            num_stages = 2  # Fixed like tritonBLAS

        # Group size heuristic
        group_m = 8 if (self.M >= 2048 and self.N >= 2048) else 4

        # Calculate StreamK tiles following TritonBLAS logic
        try:
            tiles_m = (self.M + block_m - 1) // block_m
            tiles_n = (self.N + block_n - 1) // block_n
            total_tiles = tiles_m * tiles_n
            # CRITICAL: actual_grid must match streamk_mm_grid logic to prevent hang
            actual_grid = min(self.num_sms, total_tiles)
            # TritonBLAS-aligned logic: STREAMK_TILES = total_tiles % grid
            # When total_tiles % grid == 0, no StreamK needed (classical tiling optimal)
            streamk_tiles = total_tiles % actual_grid if actual_grid > 0 else 0

            if STREAMK_DEBUG:
                print(f"🔍 [FALLBACK-STREAMK-DEBUG] TritonBLAS logic applied (fallback path):")
                print(f"   total_tiles={total_tiles}, actual_grid={actual_grid}")
                print(f"   STREAMK_TILES={streamk_tiles} (total_tiles % actual_grid)")
                print(f"   StreamK {'ENABLED' if streamk_tiles > 0 else 'DISABLED'} (classical tiling)")
        except (TypeError, AttributeError):
            # Handle symbolic dimensions
            streamk_tiles = min(32, self.num_sms)
            actual_grid = self.num_sms  # Conservative for symbolic

        # Detect quantization from input/output dtypes
        is_quantized = self._is_quantized_operation()

        # Set accumulator type based on quantization and output type
        if is_quantized and self.c_dtype == torch.int8:
            acc_type = "tl.int32"
        else:
            acc_type = "tl.float32"

        return {
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
            "GROUP_M": group_m,
            "STREAMK_TILES": streamk_tiles,
            "NUM_SMS": actual_grid,
            "NUM_XCDS": _get_hardware_chiplet_count(),
            "CHUNK_SIZE": min(4 * 4, actual_grid // _get_hardware_chiplet_count()),  # TritonBLAS-aligned
            "ACC_TYPE": acc_type,
            "QUANTIZED": is_quantized,
            "OUTPUT_DTYPE_IS_INT8": (self.c_dtype == torch.int8),
            "ALLOW_TF32": True,
            "EVEN_K": _safe_even_k_check(self.K, block_k),
            "BIAS": False,
            "USE_FAST_ACCUM": True,
            "CACHE_MODIFIER_A": None,  # TritonBLAS-aligned (None for typical matrices)
            "CACHE_MODIFIER_B": None,  # TritonBLAS-aligned (None for typical matrices)
            "INPUT_PRECISION": '"ieee"' if is_quantized else None,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }

    def _is_quantized_operation(self):
        """Check if this is a quantized operation based on dtypes"""
        quantized_dtypes = {torch.int8, torch.uint8}

        # Check for fp8 types if available
        if hasattr(torch, 'float8_e4m3fn'):
            quantized_dtypes.add(torch.float8_e4m3fn)
        if hasattr(torch, 'float8_e5m2'):
            quantized_dtypes.add(torch.float8_e5m2)

        return (
            self.a_dtype in quantized_dtypes or
            self.b_dtype in quantized_dtypes or
            self.c_dtype in quantized_dtypes
        )

    def _compute_sk_grid(self):
        """Compute StreamK grid size following tritonBLAS logic"""
        config = self.config
        block_m = config["BLOCK_M"]
        block_n = config["BLOCK_N"]
        block_k = config["BLOCK_K"]

        # Calculate total tiles
        try:
            tiles_m = math.ceil(self.M / block_m)
            tiles_n = math.ceil(self.N / block_n)
            total_tiles = tiles_m * tiles_n
        except (TypeError, AttributeError):
            # Handle symbolic dimensions
            total_tiles = 64  # Conservative estimate

        # StreamK grid computation similar to tritonBLAS
        sk_grid = total_tiles
        try:
            iters_per_tile = max(1, math.ceil(self.K / block_k))
        except (TypeError, AttributeError):
            iters_per_tile = 8  # Conservative estimate

        # More tiles than SMs: try to balance load
        if total_tiles > self.num_sms:
            # Try fractional splits
            tile_fractions = [0.0, 1.0/2.0, 1.0/8.0, 1.0/5.0, 1.0/4.0, 1.0/3.0]
            min_even_tiles = total_tiles / self.num_sms

            for frac in tile_fractions:
                frac_grid = int((total_tiles / (min_even_tiles + frac)) + 0.5)
                if frac_grid <= self.num_sms:
                    sk_grid = frac_grid
                    break

        # Fewer tiles than SMs: split along K dimension
        elif total_tiles < self.num_sms:
            split_factors = [8, 6, 4, 3, 2, 1]
            for factor in split_factors:
                split_grid = total_tiles * factor
                iters_per_sm = iters_per_tile // factor
                if split_grid <= self.num_sms and iters_per_sm >= 8:
                    sk_grid = split_grid
                    break

        return min(sk_grid, self.num_sms)

    def get_config(self):
        """Return configuration dict following tritonBLAS pattern"""
        return self.config

    def get_grid(self):
        """Return optimal grid size following tritonBLAS pattern"""
        return self.grid


# LRU cache for origami selector following tritonBLAS pattern
@functools.lru_cache(maxsize=1024)
def _make_streamk_selector(M, N, K, a_dtype, b_dtype, c_dtype, device_type):
    """Create cached origami selector following tritonBLAS pattern"""
    selector_start = time.perf_counter()
    if STREAMK_DEBUG:
        print(f"🔍 [CACHE-DEBUG] Creating StreamK selector for {M}x{N}x{K}")

    # Create a dummy device object for the selector
    device = torch.device(device_type)
    # Use the origami-based selector instead of the simple heuristic one
    origami_selector = StreamKOrigamiSelector(M, N, K, a_dtype, b_dtype, c_dtype, device)

    wrapper_start = time.perf_counter()

    # Create a wrapper that provides the same interface as MatmulHeuristicResult
    class OrigamiSelectorWrapper:
        def __init__(self, origami_selector):
            self.origami_selector = origami_selector

        def get_config(self):
            # Get config from origami selector and convert to expected format
            block_m, block_n, block_k, group_m = self.origami_selector.get_config()
            grid = self.origami_selector.get_grid()

            # Calculate total tiles for TritonBLAS-aligned STREAMK_TILES logic
            import math
            total_tiles_m = math.ceil(self.origami_selector.M / block_m)
            total_tiles_n = math.ceil(self.origami_selector.N / block_n)
            total_tiles = total_tiles_m * total_tiles_n

            # TritonBLAS-aligned logic: STREAMK_TILES = total_tiles % grid
            # When total_tiles % grid == 0, no StreamK needed (classical tiling optimal)
            streamk_tiles = total_tiles % grid if grid > 0 else 0

            if STREAMK_DEBUG:
                print(f"🔍 [STREAMK-SELECTOR-DEBUG] TritonBLAS logic applied:")
                print(f"   total_tiles={total_tiles}, grid={grid}")
                print(f"   STREAMK_TILES={streamk_tiles} (total_tiles % grid)")
                print(f"   StreamK {'ENABLED' if streamk_tiles > 0 else 'DISABLED'} (classical tiling)")

            config = {
                "BLOCK_M": block_m,
                "BLOCK_N": block_n,
                "BLOCK_K": block_k,
                "GROUP_M": group_m,
                "STREAMK_TILES": streamk_tiles,
                "NUM_SMS": grid,
                "EVEN_K": _safe_even_k_check(self.origami_selector.K, block_k),
                "ACC_TYPE": "tl.float32",
                "ALLOW_TF32": True,
                "CACHE_MODIFIER_A": None,  # TritonBLAS-aligned
                "CACHE_MODIFIER_B": None,  # TritonBLAS-aligned
                "CHUNK_SIZE": min(4 * 4, grid // _get_hardware_chiplet_count()),  # TritonBLAS-aligned
                "NUM_XCDS": _get_hardware_chiplet_count(),
                "BIAS": False,
                "INPUT_PRECISION": None,
                "OUTPUT_DTYPE_IS_INT8": False,
                "QUANTIZED": False,
                "USE_FAST_ACCUM": True,
                # Use tritonBLAS fixed settings to avoid shared memory issues
                "num_warps": 8,        # Fixed like tritonBLAS (not dynamic)
                "num_stages": 2,       # Fixed like tritonBLAS (always 2, never 3)
                "waves_per_eu": 0,     # Fixed like tritonBLAS
                "matrix_instr_nonkdim": 16,  # Fixed like tritonBLAS (mfmaInstrSize)
                "kpack": 1,            # Fixed like tritonBLAS
            }
            if STREAMK_DEBUG:
                print(f"🔍 [STREAMK-SELECTOR-DEBUG] Full origami config: {config}")
            return config

        def get_grid(self):
            return self.origami_selector.get_grid()

    wrapper_end = time.perf_counter()
    selector_end = time.perf_counter()

    if STREAMK_DEBUG:
        print(f"⏱️  [TIMING] StreamK selector creation took: {(wrapper_end - wrapper_start)*1000:.2f}ms")
        print(f"⏱️  [TIMING] _make_streamk_selector total took: {(selector_end - selector_start)*1000:.2f}ms")

    return OrigamiSelectorWrapper(origami_selector)

@SymbolicGridFn
def streamk_mm_grid(m, n, meta, *, cdiv, min):
    """True StreamK grid: one thread per SM for work distribution"""
    num_sms = meta.get("NUM_SMS", 108)
    total_tiles = cdiv(m, meta["BLOCK_M"]) * cdiv(n, meta["BLOCK_N"])
    return (min(num_sms, total_tiles), 1, 1)  # One thread per SM for StreamK algorithm


class StreamKTemplate(TritonTemplate):
    """
    Complete StreamK template implementing full tritonBLAS StreamK algorithm.

    This template stores directly to the output buffer using manual tl.store calls
    inside loops, which is required for the StreamK algorithm where each SM may
    process multiple tiles. The final {{store_output}} uses a False mask to avoid
    double-storing.
    """

    def __init__(self):
        super().__init__(
            name="mm_streamk",
            grid=streamk_mm_grid,
            source=r"""
{{def_kernel("A", "B", "A_SCALE_PTR", "B_SCALE_PTR", "BIAS_PTR", "WORKSPACE", "LOCKS")}}
    # Matrix dimensions (can vary between calls, so regular variables)
    M = {{size("A", 0)}}
    N = {{size("B", 1)}}
    K = {{size("A", 1)}}
    if M * N == 0:
        return

    # Get output pointer using template's ptr("C") macro
    C_OUT = {{ptr("C")}}

    # Memory strides - conditional constexpr based on K size to avoid issues with small transposed matrices
    stride_am: tl.constexpr = {{stride("A", 0)}}
    # For small K dimensions (<16), avoid constexpr to handle irregular memory patterns in transpose cases
    {% set k_size = size("A", 1)|int %}
    {% if k_size >= 16 %}
    stride_ak: tl.constexpr = {{stride("A", 1)}}
    stride_bk: tl.constexpr = {{stride("B", 0)}}
    {% else %}
    stride_ak = {{stride("A", 1)}}
    stride_bk = {{stride("B", 0)}}
    {% endif %}
    stride_bn: tl.constexpr = {{stride("B", 1)}}
    # Get output strides from template's layout using stride(None, dim)
    stride_cm = {{stride(None, 0)}}
    stride_cn = {{stride(None, 1)}}

    stride_bias: tl.constexpr = {{stride("BIAS_PTR", 0)}}
    stride_a_scale: tl.constexpr = {{stride("A_SCALE_PTR", 0)}}
    stride_b_scale: tl.constexpr = {{stride("B_SCALE_PTR", 0)}}

    # Note: BLOCK_M, BLOCK_N, BLOCK_K, NUM_XCDS, etc. are already passed as constexpr by template system

    # Memory stride assumptions for compiler optimization (tritonBLAS pattern)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # Complete StreamK algorithm implementation
    pid = tl.program_id(0)

    # Apply chiplet transformation for multi-die optimization (TritonBLAS pattern)
    if NUM_XCDS != 1:
        pid = triton_helpers.chiplet_transform_chunked(pid, NUM_SMS, NUM_XCDS, CHUNK_SIZE)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    total_tiles = num_pid_m * num_pid_n
    total_full_tiles = total_tiles - STREAMK_TILES

    acc_dtype = tl.float32

    # ========== Phase 1: Process Full Tiles ==========
    # Each SM processes multiple full tiles in a strided loop
    # Key fix: Store each tile INSIDE the loop (like tritonBLAS lines 152-158)
    for tile_id in range(pid, total_full_tiles, NUM_SMS):
        # Calculate tile coordinates with group reordering
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        # Calculate matrix indices - NOTE: Do NOT use modulo for correct boundary masking
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)

        # Set up base pointers
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        # Compute masks for boundary handling
        mask_m = rm[:, None] < M
        mask_n = rn[None, :] < N

        {% if BIAS %}
        bias_ = BIAS_PTR + rm * stride_bias
        bias = tl.load(bias_, mask=rm < M, other=0.0)
        {% endif %}

        # K dimension loop
        loop_k = tl.cdiv(K, BLOCK_K)
        {% if not EVEN_K %}
        loop_k -= 1
        {% endif %}
        # Fixed: Only assume when valid to prevent failure with K < BLOCK_K
        {% if EVEN_K %}
        tl.assume(loop_k >= 1)
        {% endif %}

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            if stride_ak == 1:
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)), mask=mask_m, other=0.0, cache_modifier=CACHE_MODIFIER_A)
            else:
                a = tl.load(tl.multiple_of(A_BASE, (16, 1)), mask=mask_m, other=0.0, cache_modifier=CACHE_MODIFIER_A)

            if stride_bk == 1:
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)), mask=mask_n, other=0.0, cache_modifier=CACHE_MODIFIER_B)
            else:
                b = tl.load(tl.multiple_of(B_BASE, (1, 16)), mask=mask_n, other=0.0, cache_modifier=CACHE_MODIFIER_B)

            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
            A_BASE += BLOCK_K * stride_ak
            B_BASE += BLOCK_K * stride_bk

        {% if not EVEN_K %}
        # Handle remaining K
        k = loop_k
        rk_rem = k * BLOCK_K + tl.arange(0, BLOCK_K)
        A_BASE = A + rm[:, None] * stride_am + rk_rem[None, :] * stride_ak
        B_BASE = B + rk_rem[:, None] * stride_bk + rn[None, :] * stride_bn
        if stride_ak == 1:
            A_BASE = tl.multiple_of(A_BASE, (1, 16))
        else:
            A_BASE = tl.multiple_of(A_BASE, (16, 1))
        if stride_bk == 1:
            B_BASE = tl.multiple_of(B_BASE, (16, 1))
        else:
            B_BASE = tl.multiple_of(B_BASE, (1, 16))
        a = tl.load(A_BASE, mask=mask_m & (rk_rem[None, :] < K), other=0.0, cache_modifier=CACHE_MODIFIER_A)
        b = tl.load(B_BASE, mask=mask_n & (rk_rem[:, None] < K), other=0.0, cache_modifier=CACHE_MODIFIER_B)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        {% endif %}

        {% if QUANTIZED %}
        rm_A_scale = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn_B_scale = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        A_scale = tl.load(A_SCALE_PTR + rm_A_scale * stride_a_scale, mask=rm_A_scale < M, other=0.0)
        B_scale = tl.load(B_SCALE_PTR + rn_B_scale * stride_b_scale, mask=rn_B_scale < N, other=0.0)
        acc *= A_scale[:, None] * B_scale[None, :]
        {% endif %}

        {% if BIAS %}
        c = acc.to({{dtype("C")}}) + bias[:, None]
        {% else %}
        c = acc.to({{dtype("C")}})
        {% endif %}

        # Store directly to output C using rm/rn without modulo for correct masking
        mask = (rm[:, None] < M) & (rn[None, :] < N)
        C_ = C_OUT + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        tl.store(C_, c, mask=mask)

    # ========== Phase 2: Process StreamK Tiles ==========
    if STREAMK_TILES == 0:
        return

    # Initialize workspace for this SM
    rm1 = tl.arange(0, BLOCK_M)
    rn1 = tl.arange(0, BLOCK_N)
    rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_M), BLOCK_M)
    rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_N), BLOCK_N)
    P_ = WORKSPACE + pid * BLOCK_M * BLOCK_N + rm1[:, None] * BLOCK_N + rn1[None, :]
    tl.store(P_, 0.0, cache_modifier=".wt")  # Efficient scalar store like TritonBLAS
    tl.store(LOCKS + pid, 0, cache_modifier=".wt")

    tl.assume(pid >= 0)
    iters_per_tile = tl.cdiv(K, BLOCK_K)
    total_streamk_iters = STREAMK_TILES * iters_per_tile
    streamk_iters_pcu = total_streamk_iters // NUM_SMS
    streamk_remainder_iters = total_streamk_iters % NUM_SMS
    start_iter = total_full_tiles * iters_per_tile + pid * streamk_iters_pcu + tl.minimum(pid, streamk_remainder_iters)
    last_iter = total_full_tiles * iters_per_tile + (pid + 1) * streamk_iters_pcu + tl.minimum(pid + 1, streamk_remainder_iters)

    # StreamK main loop
    while start_iter < last_iter:
        remainder = start_iter % iters_per_tile
        end_iter = tl.minimum(start_iter + (iters_per_tile - remainder), last_iter)
        tile_id = start_iter // iters_per_tile

        num_pid_in_group = GROUP_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        # NOTE: Do NOT use modulo for correct boundary masking
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak + BLOCK_K * stride_ak * remainder
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn + BLOCK_K * stride_bk * remainder
        if stride_ak == 1:
            A_BASE = tl.multiple_of(A_BASE, (1, 16))
        else:
            A_BASE = tl.multiple_of(A_BASE, (16, 1))
        if stride_bk == 1:
            B_BASE = tl.multiple_of(B_BASE, (16, 1))
        else:
            B_BASE = tl.multiple_of(B_BASE, (1, 16))

        # Compute masks for boundary handling
        mask_m = rm[:, None] < M
        mask_n = rn[None, :] < N

        {% if BIAS %}
        bias_ = BIAS_PTR + rm * stride_bias
        bias = tl.load(bias_, mask=rm < M, other=0.0)
        {% endif %}

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
        for current_iter in range(start_iter, end_iter):
            {% if EVEN_K %}
            # Add memory alignment hints like tritonBLAS full tiles loop
            if stride_ak == 1:
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)), mask=mask_m, other=0.0, cache_modifier=CACHE_MODIFIER_A)
            else:
                a = tl.load(tl.multiple_of(A_BASE, (16, 1)), mask=mask_m, other=0.0, cache_modifier=CACHE_MODIFIER_A)

            if stride_bk == 1:
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)), mask=mask_n, other=0.0, cache_modifier=CACHE_MODIFIER_B)
            else:
                b = tl.load(tl.multiple_of(B_BASE, (1, 16)), mask=mask_n, other=0.0, cache_modifier=CACHE_MODIFIER_B)
            {% else %}
            global_k_offset = (current_iter % iters_per_tile) * BLOCK_K
            k_mask = global_k_offset + rk < K
            # Apply memory alignment hints even with masking
            if stride_ak == 1:
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)), mask=mask_m & k_mask[None, :], other=0.0, cache_modifier=CACHE_MODIFIER_A)
            else:
                a = tl.load(tl.multiple_of(A_BASE, (16, 1)), mask=mask_m & k_mask[None, :], other=0.0, cache_modifier=CACHE_MODIFIER_A)

            if stride_bk == 1:
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)), mask=mask_n & k_mask[:, None], other=0.0, cache_modifier=CACHE_MODIFIER_B)
            else:
                b = tl.load(tl.multiple_of(B_BASE, (1, 16)), mask=mask_n & k_mask[:, None], other=0.0, cache_modifier=CACHE_MODIFIER_B)
            {% endif %}

            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
            A_BASE += BLOCK_K * stride_ak
            B_BASE += BLOCK_K * stride_bk

        {% if QUANTIZED %}
        rm_A_scale = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn_B_scale = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        A_scale = tl.load(A_SCALE_PTR + rm_A_scale * stride_a_scale, mask=rm_A_scale < M, other=0.0)
        B_scale = tl.load(B_SCALE_PTR + rn_B_scale * stride_b_scale, mask=rn_B_scale < N, other=0.0)
        acc *= A_scale[:, None] * B_scale[None, :]
        {% endif %}

        tile_iter = tile_id * iters_per_tile

        if start_iter != tile_iter:
            # Partial tile: store to workspace and signal
            tl.store(P_, acc, cache_modifier=".wt")
            tl.debug_barrier()
            tl.store(LOCKS + pid, 1, cache_modifier=".wt")
        else:
            # Complete tile: aggregate from other SMs using quadrant-based optimization
            next_pid = pid + 1
            tile_iter_end = tile_iter + iters_per_tile
            end = end_iter

            # ✅ OPTIMIZED: Split accumulator into 4 quadrants (exact tritonBLAS approach)
            # First split in M direction
            acc_m_reshaped = tl.reshape(acc, (2, BLOCK_M // 2, BLOCK_N))
            acc_m_permuted = tl.permute(acc_m_reshaped, (1, 2, 0))  # (M//2, N, 2)
            acc_top, acc_bottom = tl.split(acc_m_permuted)  # Split along last dimension

            # Remove singleton dimension - each is now (M//2, N)
            acc_top = tl.reshape(acc_top, (BLOCK_M // 2, BLOCK_N))
            acc_bottom = tl.reshape(acc_bottom, (BLOCK_M // 2, BLOCK_N))

            # Now split each half in N direction
            acc_top_reshaped = tl.reshape(acc_top, (BLOCK_M // 2, 2, BLOCK_N // 2))
            acc_top_permuted = tl.permute(acc_top_reshaped, (0, 2, 1))  # (M//2, N//2, 2)
            acc00, acc01 = tl.split(acc_top_permuted)  # Split along last dimension

            acc_bottom_reshaped = tl.reshape(acc_bottom, (BLOCK_M // 2, 2, BLOCK_N // 2))
            acc_bottom_permuted = tl.permute(acc_bottom_reshaped, (0, 2, 1))  # (M//2, N//2, 2)
            acc10, acc11 = tl.split(acc_bottom_permuted)  # Split along last dimension

            # Remove singleton dimensions - each is now (M//2, N//2)
            acc00 = tl.reshape(acc00, (BLOCK_M // 2, BLOCK_N // 2))
            acc01 = tl.reshape(acc01, (BLOCK_M // 2, BLOCK_N // 2))
            acc10 = tl.reshape(acc10, (BLOCK_M // 2, BLOCK_N // 2))
            acc11 = tl.reshape(acc11, (BLOCK_M // 2, BLOCK_N // 2))

            # Aggregate from other processing elements (exact tritonBLAS pattern)
            while (end < tile_iter_end and next_pid < NUM_SMS):
                while tl.load(LOCKS + next_pid, cache_modifier=".cv", volatile=True) != 1:
                    pass
                rm1 = tl.arange(0, BLOCK_M)
                rn1 = tl.arange(0, BLOCK_N)
                rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_M), BLOCK_M)
                rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_N), BLOCK_N)

                # Load P in 4 quadrants
                P_base = WORKSPACE + next_pid * BLOCK_M * BLOCK_N

                # Quadrant 00 (top-left)
                P_00 = P_base + tl.arange(0, BLOCK_M // 2)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N // 2)[None, :]
                acc00 += tl.load(P_00, cache_modifier=".cv")

                # Quadrant 01 (top-right)
                P_01 = P_base + tl.arange(0, BLOCK_M // 2)[:, None] * BLOCK_N + (tl.arange(0, BLOCK_N // 2)[None, :] + BLOCK_N // 2)
                acc01 += tl.load(P_01, cache_modifier=".cv")

                # Quadrant 10 (bottom-left)
                P_10 = P_base + (tl.arange(0, BLOCK_M // 2)[:, None] + BLOCK_M // 2) * BLOCK_N + tl.arange(0, BLOCK_N // 2)[None, :]
                acc10 += tl.load(P_10, cache_modifier=".cv")

                # Quadrant 11 (bottom-right)
                P_11 = P_base + (tl.arange(0, BLOCK_M // 2)[:, None] + BLOCK_M // 2) * BLOCK_N + (tl.arange(0, BLOCK_N // 2)[None, :] + BLOCK_N // 2)
                acc11 += tl.load(P_11, cache_modifier=".cv")

                end += streamk_iters_pcu + (next_pid < streamk_remainder_iters)
                next_pid += 1

            # Unified bias handling for Stream-K section (exact tritonBLAS pattern)
            {% if BIAS %}
            # Split bias for top and bottom halves
            bias_top = bias[:BLOCK_M // 2]
            bias_bottom = bias[BLOCK_M // 2:]

            bias_top_reshaped = tl.reshape(bias_top, (BLOCK_M // 2, 1))
            bias_bottom_reshaped = tl.reshape(bias_bottom, (BLOCK_M // 2, 1))

            acc00 += bias_top_reshaped
            acc01 += bias_top_reshaped
            acc10 += bias_bottom_reshaped
            acc11 += bias_bottom_reshaped
            {% endif %}

            # Convert each quadrant to output dtype
            c00 = acc00.to({{dtype("C")}})
            c01 = acc01.to({{dtype("C")}})
            c10 = acc10.to({{dtype("C")}})
            c11 = acc11.to({{dtype("C")}})

            # Store all 4 quadrants separately for optimal memory locality (tritonBLAS approach)
            # Calculate quadrant indices
            rm_top = pid_m * BLOCK_M + tl.arange(0, BLOCK_M // 2)
            rm_bottom = pid_m * BLOCK_M + tl.arange(BLOCK_M // 2, BLOCK_M)
            rn_left = pid_n * BLOCK_N + tl.arange(0, BLOCK_N // 2)
            rn_right = pid_n * BLOCK_N + tl.arange(BLOCK_N // 2, BLOCK_N)

            # Store quadrant 00 (top-left)
            mask00 = (rm_top < M)[:, None] & (rn_left < N)[None, :]
            tl.store(C_OUT + rm_top[:, None] * stride_cm + rn_left[None, :] * stride_cn, c00, mask=mask00)

            # Store quadrant 01 (top-right)
            mask01 = (rm_top < M)[:, None] & (rn_right < N)[None, :]
            tl.store(C_OUT + rm_top[:, None] * stride_cm + rn_right[None, :] * stride_cn, c01, mask=mask01)

            # Store quadrant 10 (bottom-left)
            mask10 = (rm_bottom < M)[:, None] & (rn_left < N)[None, :]
            tl.store(C_OUT + rm_bottom[:, None] * stride_cm + rn_left[None, :] * stride_cn, c10, mask=mask10)

            # Store quadrant 11 (bottom-right)
            mask11 = (rm_bottom < M)[:, None] & (rn_right < N)[None, :]
            tl.store(C_OUT + rm_bottom[:, None] * stride_cm + rn_right[None, :] * stride_cn, c11, mask=mask11)

        start_iter = end_iter

    # No store_output needed - all stores happen inside the loops above
    # This is a minimal dummy output point required by the template system
    # Use minimal scalar variables to avoid register pressure
    dummy_idx_m = 0
    dummy_idx_n = 0
    dummy_mask = False
    acc_dummy_scalar = tl.cast(0.0, dtype=acc_dtype)
    # Single-element store with False mask to prevent any actual output
    {{store_output(("dummy_idx_m", "dummy_idx_n"), "acc_dummy_scalar", "dummy_mask")}}
""",
            cache_codegen_enabled_for_template=True,
            prologue_loads_all_inputs=True,
        )

    def maybe_append_choice(
        self,
        choices,
        **kwargs,
    ):
        """StreamK choice generation with workspace and locks buffers.

        The template uses {{ptr("C")}} and {{stride(None, dim)}} to access the
        output buffer directly, enabling multiple stores inside loops.
        """
        if not ENABLE_STREAMK:
            return None  # Return None for success when StreamK is disabled

        # Skip StreamK for operations with epilogue fusion (e.g., _to_copy, activations).
        # StreamK templates handle stores directly inside the kernel using ptr("C"),
        # which doesn't work correctly with epilogue fusion since the epilogue
        # may register a different output buffer (e.g., for type conversion).
        epilogue_fn = kwargs.get('epilogue_fn', None)
        if epilogue_fn is not None and epilogue_fn is not identity:
            streamk_log_debug("Skipping StreamK: epilogue fusion detected (not compatible with ptr('C'))")
            return None

        try:
            # Extract the input nodes (A, B matrices, optional bias)
            input_nodes = kwargs.get('input_nodes', ())
            if len(input_nodes) < 2:
                raise ValueError(f"StreamK template requires at least 2 input nodes (A, B), got {len(input_nodes)}")

            A_node, B_node = input_nodes[0], input_nodes[1]
            bias_node = input_nodes[2] if len(input_nodes) > 2 else None
            layout = kwargs.get('layout')

            # Extract template configuration
            template_kwargs = dict(kwargs)
            template_kwargs.pop('input_nodes', None)
            template_kwargs.pop('layout', None)

            # Detect quantization from input dtypes
            is_quantized = self._detect_quantization(A_node, B_node, layout)
            template_kwargs.setdefault('QUANTIZED', is_quantized)

            # Set bias configuration based on whether bias_node is provided
            has_bias = bias_node is not None
            template_kwargs.setdefault('BIAS', has_bias)

            # Get StreamK parameters
            num_sms = template_kwargs.get('NUM_SMS', 108)
            block_m = template_kwargs.get('BLOCK_M', 128)
            block_n = template_kwargs.get('BLOCK_N', 128)

            # Get matrix dimensions
            M = A_node.get_size()[0]
            N = B_node.get_size()[1]

            # Create workspace buffer for partial results
            # Note: We still need to use IR buffer creation (empty_strided) for proper inductor integration
            # The global buffer optimization will be handled at the kernel execution level
            workspace_shape = [num_sms, block_m, block_n]
            workspace = empty_strided(
                workspace_shape,
                None,
                dtype=torch.float32,  # Always use float32 for accumulation
                device=layout.device,
            )

            # Create locks buffer for synchronization
            locks_shape = [num_sms]
            locks = empty_strided(
                locks_shape,
                None,
                dtype=torch.int32,  # Integer type for locks
                device=layout.device,
            )

            if STREAMK_DEBUG:
                print(f"🔍 [WORKSPACE-DEBUG] Created IR buffer nodes: workspace={workspace_shape}, locks={locks_shape}")

            # Create scale tensors for quantization support
            if is_quantized:
                a_scale = empty_strided(
                    [M],  # Row-wise scaling for A
                    None,
                    dtype=torch.float32,
                    device=layout.device,
                )
                b_scale = empty_strided(
                    [N],  # Column-wise scaling for B
                    None,
                    dtype=torch.float32,
                    device=layout.device,
                )
            else:
                # Create dummy scale tensors for non-quantized operations
                a_scale = empty_strided(
                    [1],  # Minimal size
                    None,
                    dtype=torch.float32,
                    device=layout.device,
                )
                b_scale = empty_strided(
                    [1],  # Minimal size
                    None,
                    dtype=torch.float32,
                    device=layout.device,
                )

            # Build input nodes in the order expected by the template:
            # def_kernel("A", "B", "A_SCALE_PTR", "B_SCALE_PTR", "BIAS_PTR", "WORKSPACE", "LOCKS")
            # Note: C_OUT is NOT passed as input - the template uses {{ptr("C")}} to get the output pointer


            if has_bias:
                streamk_input_nodes = (A_node, B_node, a_scale, b_scale, bias_node, workspace, locks)
            else:
                # Create a dummy bias buffer to maintain consistent signature
                dummy_bias = empty_strided(
                    [1],  # Minimal size
                    None,
                    dtype=layout.dtype,
                    device=layout.device,
                )
                streamk_input_nodes = (A_node, B_node, a_scale, b_scale, dummy_bias, workspace, locks)

            # Mark workspace and locks as mutated inputs
            mutated_inputs = [workspace, locks]

            # Force epilogue_fn=identity to disable epilogue fusion for StreamK.
            # StreamK templates handle stores directly inside the kernel using ptr("C"),
            # which doesn't work correctly with epilogue fusion since the epilogue
            # may register a different output buffer (e.g., for type conversion).
            # Remove any epilogue_fn that might have been passed in template_kwargs
            template_kwargs.pop('epilogue_fn', None)
            template_kwargs.pop('epilogue_fn_hash', None)

            # Call parent with full StreamK configuration
            # Disable epilogue fusion for StreamK because it stores directly in the kernel
            return super().maybe_append_choice(
                choices,
                input_nodes=streamk_input_nodes,
                layout=layout,
                mutated_inputs=mutated_inputs,
                epilogue_fn=identity,
                epilogue_fn_hash=None,
                allow_epilogue_fusion=False,
                **template_kwargs
            )

        except Exception as e:
            streamk_log_debug(f"Failed to add StreamK choice: {e}")
            import traceback
            if STREAMK_DEBUG:
                traceback.print_exc()
            return e  # Return the error

    def _detect_quantization(self, A_node, B_node, layout):
        """Detect if this is a quantized operation based on input types"""
        a_dtype = A_node.get_dtype()
        b_dtype = B_node.get_dtype()
        output_dtype = layout.dtype

        # Check for quantized input dtypes
        quantized_dtypes = {torch.int8, torch.uint8}

        # Check for fp8 types if available
        if hasattr(torch, 'float8_e4m3fn'):
            quantized_dtypes.add(torch.float8_e4m3fn)
        if hasattr(torch, 'float8_e5m2'):
            quantized_dtypes.add(torch.float8_e5m2)

        # Determine if quantized based on input or output types
        is_quantized = (
            a_dtype in quantized_dtypes or
            b_dtype in quantized_dtypes or
            output_dtype in quantized_dtypes
        )

        if is_quantized:
            streamk_log_debug(f"Detected quantized operation: A={a_dtype}, B={b_dtype}, output={output_dtype}")

        return is_quantized


# Global pre-allocated buffers (tritonBLAS pattern for optimal performance)
# These are reused across all StreamK calls to avoid allocation overhead
current_device_index = torch.cuda.current_device()
current_device = torch.cuda.get_device_properties(current_device_index)
MAX_SMS = current_device.multi_processor_count  # Hardware SMS count
MAX_BLOCK_SIZE = 256 * 256  # Max block size for typical configurations

# Pre-allocated global buffers (tritonBLAS approach)
_global_locks = None
_global_workspace = None

def _get_global_streamk_buffers():
    """Get or create global StreamK buffers following tritonBLAS pattern"""
    global _global_locks, _global_workspace

    if _global_locks is None or _global_workspace is None:
        # Initialize global buffers once per device
        _global_locks = torch.empty(MAX_SMS, device="cuda", dtype=torch.int32)
        _global_workspace = torch.empty(MAX_SMS, MAX_BLOCK_SIZE, device="cuda", dtype=torch.float32)

        if STREAMK_DEBUG:
            print(f"🔍 [WORKSPACE-DEBUG] Initialized global StreamK buffers: locks={_global_locks.shape}, workspace={_global_workspace.shape}")

    return _global_locks, _global_workspace

# Create the global StreamK template instance
mm_streamk_template = StreamKTemplate()


def should_use_streamk(m, n, k, dtype, device):
    """Heuristic to decide when StreamK is beneficial"""

    # Early symbolic variable detection - bypass StreamK completely for symbolic shapes
    def is_symbolic(val):
        """Check if a value is a symbolic variable"""
        try:
            # Try to convert to int - if it fails, it's likely symbolic
            int(val)
            return False
        except (TypeError, ValueError):
            # Check for common symbolic indicators
            val_str = str(val)
            return (
                hasattr(val, 'is_symbol') or
                val_str.startswith('s') or
                'Symbol' in str(type(val)) or
                'Expr' in str(type(val)) or
                any(c in val_str for c in ['s', 'Symbol', 'Expr', 'sympy'])
            )

    if is_symbolic(m) or is_symbolic(n) or is_symbolic(k):
        streamk_log_info(
            f"🚫 Symbolic variables detected ({m}x{n}x{k}). "
            f"Completely disabling StreamK to avoid Relational errors."
        )
        streamk_log_info(
            f"   StreamK will be skipped for this compilation to prevent symbolic math issues."
        )
        streamk_log_debug(f"   Variable types: m={type(m)}, n={type(n)}, k={type(k)}")
        return False

    # Force StreamK override - bypass all heuristics (but only for concrete shapes)
    if FORCE_STREAMK:
        streamk_log_info(f"🔥 FORCE_STREAMK=1: Using StreamK for {m}x{n}x{k} (bypassing heuristics)")
        # Still check basic compatibility
        if not torch.cuda.is_available():
            streamk_log_info(f"⚠️ CUDA not available, but FORCE_STREAMK=1. Proceeding anyway...")
        if str(device).startswith('mtia'):
            streamk_log_info(f"⚠️ MTIA device detected ({device}), StreamK may not be supported")
            return False
        return True

    if not ENABLE_STREAMK:
        streamk_log_debug(f"StreamK disabled via TORCHINDUCTOR_ENABLE_STREAMK for {m}x{n}x{k}")
        return False

    # Check device compatibility
    if str(device).startswith('mtia'):
        streamk_log_debug(f"MTIA device ({device}) detected, skipping StreamK for {m}x{n}x{k}")
        return False

    total_ops = m * n * k

    # StreamK typically helps with:
    # 1. Large problems where load balancing matters
    # 2. Irregular shapes that don't divide evenly across SMs
    # 3. High K dimension that benefits from splitting

    if not torch.cuda.is_available():
        log.debug(f"CUDA not available, skipping StreamK for {m}x{n}x{k}")
        return False

    # Get device properties
    try:
        props = torch.cuda.get_device_properties(device.index if hasattr(device, 'index') else 0)
        num_sms = props.multi_processor_count
    except:
        num_sms = 108  # Default fallback

    # Calculate regular tiling efficiency
    blocks_m = (m + 127) // 128  # Assume 128x128 tiles
    blocks_n = (n + 127) // 128
    total_blocks = blocks_m * blocks_n

    # Detailed decision criteria logging - handle symbolic variables safely
    try:
        # Try to evaluate expressions, fallback to safe defaults if symbolic
        large_problem = total_ops > STREAMK_MIN_PROBLEM_SIZE
    except (TypeError, AttributeError):
        # Symbolic expression, assume it could be large
        large_problem = True

    try:
        high_k = k > 2048
    except (TypeError, AttributeError):
        high_k = True  # Assume symbolic K could be high

    try:
        few_blocks = total_blocks < num_sms * 2
    except (TypeError, AttributeError):
        few_blocks = False  # Conservative assumption

    try:
        irregular_shape = (m % 128 != 0 and n % 128 != 0)
    except (TypeError, AttributeError):
        irregular_shape = True  # Assume symbolic shapes could be irregular

    use_streamk = large_problem or high_k or few_blocks or irregular_shape

    # Safe logging for symbolic variables
    try:
        total_ops_str = f"{total_ops:.2e}"
    except (TypeError, ValueError):
        total_ops_str = str(total_ops)

    try:
        total_blocks_str = str(total_blocks)
    except:
        total_blocks_str = "symbolic"

    streamk_log_verbose(f"StreamK decision for {m}x{n}x{k} (dtype={dtype}, total_ops={total_ops_str}):")
    streamk_log_verbose(f"  - Large problem (>{STREAMK_MIN_PROBLEM_SIZE:.0e}): {large_problem}")
    streamk_log_verbose(f"  - High K dimension (>2048): {high_k}")
    streamk_log_verbose(f"  - Few blocks ({total_blocks_str} < {num_sms * 2}): {few_blocks}")
    streamk_log_verbose(f"  - Irregular shape: {irregular_shape}")
    streamk_log_debug(f"Final decision: {'✅ USE StreamK' if use_streamk else '⏭️ SKIP StreamK'} for {m}x{n}x{k}")

    return use_streamk


def generate_streamk_configs(m, n, k, device, enable_origami=True):
    """Generate StreamK configs, optionally with Origami tuning"""
    log.debug(f"Generating StreamK configs for {m}x{n}x{k} (enable_origami={enable_origami})")

    configs = []

    # Get device properties
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(device.index if hasattr(device, 'index') else 0)
            num_sms = props.multi_processor_count
            log.debug(f"Device: {props.name}, SM count: {num_sms}")
        except:
            num_sms = 108  # Default fallback
            log.debug(f"Failed to get device properties, using default SM count: {num_sms}")
    else:
        num_sms = 108  # Default fallback
        log.debug(f"CUDA not available, using default SM count: {num_sms}")

    # Use Origami for advanced tuning (enabled by default for StreamK)
    if enable_origami:
        print(f"🔍 [PYTORCH-ORIGAMI-DEBUG] Attempting Origami tuning for StreamK {m}x{n}x{k}")
        log.info(f"Attempting Origami tuning for StreamK {m}x{n}x{k}")
        try:
            origami_configs = generate_origami_streamk_configs(m, n, k, device, num_sms)
            if origami_configs:
                print(f"🔍 [PYTORCH-ORIGAMI-DEBUG] ✓ Origami generated {len(origami_configs)} StreamK configs")
                print(f"🔍 [PYTORCH-ORIGAMI-DEBUG] First config: {origami_configs[0]}")
                log.info(f"✓ Origami generated {len(origami_configs)} StreamK configs")
                return origami_configs
            else:
                print(f"🔍 [PYTORCH-ORIGAMI-DEBUG] ✗ Origami returned no configs, falling back to heuristics")
                log.warning(f"✗ Origami returned no configs, falling back to heuristics")
        except Exception as e:
            print(f"🔍 [PYTORCH-ORIGAMI-DEBUG] ✗ Origami tuning failed: {e}, falling back to heuristic configs")
            log.warning(f"✗ Origami tuning failed: {e}, falling back to heuristic configs")

    # Fallback to heuristic-based config generation
    log.debug(f"Using heuristic-based StreamK config generation")
    heuristic_configs = generate_heuristic_streamk_configs(m, n, k, num_sms)
    log.debug(f"Generated {len(heuristic_configs)} heuristic StreamK configs")
    return heuristic_configs


def generate_heuristic_streamk_configs(m, n, k, num_sms):
    """Generate StreamK configs using heuristics"""
    streamk_log_debug(f"Generating heuristic StreamK configs for {m}x{n}x{k}, num_sms={num_sms}")

    configs = []

    # Base block sizes based on problem size - handle symbolic variables
    try:
        # Try to evaluate symbolic expressions
        m_large = m >= 256 and n >= 256
        m_medium = m >= 128 and n >= 128
    except (TypeError, AttributeError):
        # Symbolic dimensions - use conservative defaults
        streamk_log_debug("Symbolic dimensions detected, using conservative block size selection")
        m_large = True  # Assume could be large
        m_medium = True

    if m_large:
        block_ms = [128, 256]
        block_ns = [128, 256]
        streamk_log_debug("Large problem: using block sizes [128,256] x [128,256]")
    elif m_medium:
        block_ms = [64, 128]
        block_ns = [64, 128]
        streamk_log_debug("Medium problem: using block sizes [64,128] x [64,128]")
    else:
        block_ms = [64]
        block_ns = [64]
        streamk_log_debug("Small problem: using block sizes [64] x [64]")

    # Handle symbolic K dimension
    try:
        k_large = k >= 128
    except (TypeError, AttributeError):
        k_large = True  # Assume symbolic K could be large

    block_ks = [32, 64] if k_large else [32]
    streamk_log_debug(f"K dimension block sizes: {block_ks}")

    for block_m in block_ms:
        for block_n in block_ns:
            for block_k in block_ks:
                # Calculate number of tiles - handle symbolic dimensions
                try:
                    tiles_m = (m + block_m - 1) // block_m
                    tiles_n = (n + block_n - 1) // block_n
                    total_tiles = tiles_m * tiles_n

                    # Check if it's a large problem for XCDS optimization
                    large_problem_xcds = m * n > 1e6
                except (TypeError, AttributeError):
                    # Symbolic dimensions - use conservative defaults
                    tiles_m = 32  # Conservative estimate
                    tiles_n = 32
                    total_tiles = tiles_m * tiles_n
                    large_problem_xcds = True  # Assume could be large

                # Compute actual grid size - must match streamk_mm_grid logic
                # Grid is min(num_sms, total_tiles) to ensure all launched SMs have work
                actual_grid = min(num_sms, total_tiles)

                # TritonBLAS-aligned logic: STREAMK_TILES = total_tiles % grid
                # When total_tiles % grid == 0, no StreamK needed (classical tiling optimal)
                try:
                    streamk_tiles = total_tiles % actual_grid if actual_grid > 0 else 0
                except (TypeError, AttributeError):
                    # Handle symbolic dimensions
                    streamk_tiles = 0  # Conservative fallback

                if STREAMK_DEBUG:
                    print(f"🔍 [AUTO-STREAMK-DEBUG] TritonBLAS logic applied (autotune path):")
                    print(f"   total_tiles={total_tiles}, actual_grid={actual_grid}")
                    print(f"   STREAMK_TILES={streamk_tiles} (total_tiles % actual_grid)")
                    print(f"   StreamK {'ENABLED' if streamk_tiles > 0 else 'DISABLED'} (classical tiling)")

                # Only create StreamK configs when there are actual partial tiles to process
                # This aligns with TritonBLAS behavior: classical tiling when streamk_tiles == 0
                streamk_configs_to_try = [streamk_tiles] if streamk_tiles > 0 else [0]

                for streamk_tiles in streamk_configs_to_try:

                    for group_m in [1, 2, 4, 8]:
                        for num_warps in [4, 8]:
                            for num_stages in [2]:  # Fixed like tritonBLAS
                                # NUM_XCDS for cross-CU optimization - use actual hardware chiplet count
                                hw_chiplet_count = _get_hardware_chiplet_count()
                                if large_problem_xcds and hw_chiplet_count > 1:
                                    xcds_options = [1, hw_chiplet_count]  # Test single die vs full chiplet mode
                                else:
                                    xcds_options = [1]  # Small problems or single chiplet hardware
                                for num_xcds in xcds_options:
                                    # Apply TritonBLAS reference chunk size calculation
                                    # Set chunk size to same area as L2 tiles (TritonBLAS pattern)
                                    chunk_size = group_m * group_m
                                    chunk_size = min(chunk_size, actual_grid // num_xcds)

                                    # Create Triton config with proper format

                                    # Use standard template parameter names with quantization support
                                    # CRITICAL: NUM_SMS must equal actual_grid to prevent hang in spin-wait loop
                                    template_config = {
                                        "BLOCK_M": block_m,
                                        "BLOCK_N": block_n,
                                        "BLOCK_K": block_k,
                                        "GROUP_M": group_m,
                                        "STREAMK_TILES": streamk_tiles,
                                        "NUM_SMS": actual_grid,
                                        "NUM_XCDS": num_xcds,
                                        "CHUNK_SIZE": chunk_size,
                                        "ACC_TYPE": "tl.float32",  # Will be updated based on quantization
                                        "QUANTIZED": False,  # Default to non-quantized
                                        "OUTPUT_DTYPE_IS_INT8": False,
                                        "ALLOW_TF32": True,
                                        "EVEN_K": _safe_even_k_check(k, block_k),
                                        "BIAS": False,  # No bias for mm
                                        "USE_FAST_ACCUM": True,  # Enable fast accumulation for StreamK
                                        "CACHE_MODIFIER_A": None,  # TritonBLAS-aligned (None for typical matrices)
                                        "CACHE_MODIFIER_B": None,  # TritonBLAS-aligned (None for typical matrices)
                                        "INPUT_PRECISION": None,  # Set based on quantization
                                    }

                                    # Create triton Config object
                                    if has_triton:
                                        config = triton.Config(
                                            template_config,
                                            num_warps=num_warps,
                                            num_stages=num_stages
                                        )
                                    else:
                                        config = template_config
                                        config.update({"num_warps": num_warps, "num_stages": num_stages})
                                    configs.append(config)

    total_generated = len(configs)
    configs = configs[:20]  # Limit to prevent too many choices
    log.debug(f"Generated {total_generated} total configs, limited to {len(configs)} for performance")

    # Log a few sample configs for debugging
    if configs:
        log.debug(f"Sample StreamK config: {configs[0]}")
        if len(configs) > 1:
            log.debug(f"Sample StreamK config 2: {configs[1]}")

    return configs


def generate_origami_streamk_configs(m, n, k, device, num_sms):
    """Generate StreamK configs using tritonBLAS-style Origami integration"""
    try:
        print(f"🔍 [PYTORCH-ORIGAMI-DEBUG] Creating origami selector for {m}x{n}x{k}, device={device}, num_sms={num_sms}")
        log.debug(f"Creating origami selector for {m}x{n}x{k}")

        # Use tritonBLAS-style origami selector
        # Infer data types (assuming BF16 for this implementation)
        a_dtype = torch.bfloat16  # TODO: Pass actual dtypes from caller
        b_dtype = torch.bfloat16
        c_dtype = torch.bfloat16

        # Create origami selector following tritonBLAS pattern
        print(f"🔍 [PYTORCH-ORIGAMI-DEBUG] Creating StreamKOrigamiSelector with dtypes: {a_dtype}, {b_dtype}, {c_dtype}")
        origami_selector = StreamKOrigamiSelector(m, n, k, a_dtype, b_dtype, c_dtype, device)

        # Get optimal configuration
        BLK_M, BLK_N, BLK_K, GROUP_M = origami_selector.get_config()
        optimal_grid = origami_selector.get_grid()

        print(f"🔍 [PYTORCH-ORIGAMI-DEBUG] Origami selected: BLOCK_M={BLK_M}, BLOCK_N={BLK_N}, BLOCK_K={BLK_K}, GROUP_M={GROUP_M}, GRID={optimal_grid}")
        log.debug(f"Origami selected: BLOCK_M={BLK_M}, BLOCK_N={BLK_N}, BLOCK_K={BLK_K}, GROUP_M={GROUP_M}, GRID={optimal_grid}")

        # Create configuration in PyTorch Triton format
        config = {
            "BLOCK_SIZE_M": BLK_M,
            "BLOCK_SIZE_N": BLK_N,
            "BLOCK_SIZE_K": BLK_K,
            "GROUP_SIZE_M": GROUP_M,
            "NUM_SMS": optimal_grid,
            "CHUNK_SIZE": min(GROUP_M * GROUP_M, optimal_grid // _get_hardware_chiplet_count()),  # TritonBLAS-aligned
            "ACC_TYPE": "tl.float32",
            "ALLOW_TF32": True,
            "CACHE_MODIFIER_A": None,  # TritonBLAS-aligned
            "CACHE_MODIFIER_B": None,  # TritonBLAS-aligned
            "EVEN_K": _safe_even_k_check(k, BLK_K),
            # Use tritonBLAS fixed parameters
            "num_warps": 8,  # Fixed like tritonBLAS
            "num_stages": 2,  # Fixed like tritonBLAS
        }

        configs = [config]
        log.info(f"✓ Origami generated {len(configs)} StreamK configs for {m}x{n}x{k} using tritonBLAS integration")
        return configs

    except Exception as e:
        log.warning(f"Origami integration error: {e}")
        # Fallback to a basic configuration
        try:
            basic_config = {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
                "NUM_SMS": min(num_sms, 256),
                "CHUNK_SIZE": 16,
                "ACC_TYPE": "tl.float32",
                "ALLOW_TF32": True,
                "CACHE_MODIFIER_A": None,  # TritonBLAS-aligned
                "CACHE_MODIFIER_B": None,  # TritonBLAS-aligned
                "EVEN_K": _safe_even_k_check(k, 32),
                "num_warps": 8,  # Fixed like tritonBLAS
                "num_stages": 2,  # Fixed like tritonBLAS
            }
            log.debug(f"Using fallback configuration: {basic_config}")
            return [basic_config]
        except:
            return []


mm_template = TritonTemplate(
    name="mm",
    grid=mm_grid,
    source=(
        r"""
{{def_kernel("A", "B")}}
    M = {{size("A", 0)}}
    N = {{size("B", 1)}}
    K = {{size("A", 1)}}
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = {{stride("A", 0)}}
    stride_ak = {{stride("A", 1)}}
    stride_bk = {{stride("B", 0)}}
    stride_bn = {{stride("B", 1)}}

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        offs_a_m = rm % M
    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        offs_b_n = rn % N
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        {% if not EVEN_K %}
        a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
        b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)
        {% endif %}
        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        {{load_input("A", "a", ("idx_m", "idx_n"), mask=None if EVEN_K else "a_mask", indent_width=8)}}

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        {{load_input("B", "b", ("idx_m", "idx_n"), mask=None if EVEN_K else "b_mask", indent_width=8)}}

        {% if USE_FAST_ACCUM %}
        acc = tl.dot(a, b, acc, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
        {% else %}
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
        {% endif %}

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    {{store_output(("idx_m", "idx_n"), "acc", "mask")}}
"""
        if (torch.version.hip is None) or triton_version >= "3.3.0"
        # FIXME: To get around rocm failures like https://github.com/pytorch/pytorch/actions/runs/13123783322/job/36617154943
        # The only difference between the two templates is M >= BLOCK_M and N >= BLOCK_N checking.
        # See more details in https://github.com/pytorch/pytorch/pull/146293
        else r"""
{{def_kernel("A", "B")}}
    M = {{size("A", 0)}}
    N = {{size("B", 1)}}
    K = {{size("A", 1)}}
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = {{stride("A", 0)}}
    stride_ak = {{stride("A", 1)}}
    stride_bk = {{stride("B", 0)}}
    stride_bn = {{stride("B", 1)}}

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        offs_a_m = rm % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        offs_b_n = rn % N
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        {% if not EVEN_K %}
        a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
        b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)
        {% endif %}
        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        {{load_input("A", "a", ("idx_m", "idx_n"), mask=None if EVEN_K else "a_mask", indent_width=8)}}

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        {{load_input("B", "b", ("idx_m", "idx_n"), mask=None if EVEN_K else "b_mask", indent_width=8)}}
        {% if USE_FAST_ACCUM %}
        acc = tl.dot(a, b, acc, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
        {% else %}
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
        {% endif %}

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    {{store_output(("idx_m", "idx_n"), "acc", "mask")}}
"""
    ),
    cache_codegen_enabled_for_template=True,
    prologue_loads_all_inputs=True,
)

persistent_tma_mm_template = TritonTemplate(
    name="mm_persistent_tma",
    grid=persistent_mm_grid,
    source=r"""
{{def_kernel("A", "B")}}
    M = {{size("A", 0)}}
    N = {{size("B", 1)}}
    K = {{size("A", 1)}}
    if M * N == 0:
        # early exit due to zero-size input(s)
        return

    start_pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = grid_m * grid_n
    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1

    width = GROUP_M * grid_n
    rk_for_mask = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    {%- if TMA_EXPERIMENTAL_API %}
    workspace_base = ws_ptr + start_pid * 2 * TMA_SIZE
    a_desc_ptr = workspace_base
    b_desc_ptr = workspace_base + TMA_SIZE

    triton.language.extra.cuda.experimental_device_tensormap_create2d(
        desc_ptr=a_desc_ptr,
        global_address=A,
        load_size=[BLOCK_M, BLOCK_K] if A_ROW_MAJOR else [BLOCK_K, BLOCK_M],
        global_size=[M, K] if A_ROW_MAJOR else [K, M],
        element_ty=A.dtype.element_ty,
    )
    triton.language.extra.cuda.experimental_device_tensormap_create2d(
        desc_ptr=b_desc_ptr,
        global_address=B,
        load_size=[BLOCK_K, BLOCK_N] if B_ROW_MAJOR else [BLOCK_N, BLOCK_K],
        global_size=[K, N] if B_ROW_MAJOR else [N, K],
        element_ty=B.dtype.element_ty,
    )

    tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(a_desc_ptr)
    tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(b_desc_ptr)

    {%- else %}
    stride_am = {{stride("A", 0)}}
    stride_ak = {{stride("A", 1)}}
    stride_bk = {{stride("B", 0)}}
    stride_bn = {{stride("B", 1)}}
    a_desc = triton.language.make_tensor_descriptor(
        base=A,
        shape=[M, K] if A_ROW_MAJOR else [K, M],
        strides=[stride_am, 1] if A_ROW_MAJOR else [stride_ak, 1],
        block_shape=[BLOCK_M, BLOCK_K] if A_ROW_MAJOR else [BLOCK_K, BLOCK_M],
    )
    b_desc = triton.language.make_tensor_descriptor(
        base=B,
        shape=[K, N] if B_ROW_MAJOR else [N, K],
        strides=[stride_bk, 1] if B_ROW_MAJOR else [stride_bn, 1],
        block_shape=[BLOCK_K, BLOCK_N] if B_ROW_MAJOR else [BLOCK_N, BLOCK_K],
    )
    {%- endif %}

    pid_m = 0
    pid_n = 0
    rm = 0
    rn = 0

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            # re-order program ID for better L2 performance
            group_id = tile_id // width
            group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
            pid_m = group_id * GROUP_M + (tile_id % group_size)
            pid_n = (tile_id % width) // (group_size)

            rm = pid_m * BLOCK_M
            rn = pid_n * BLOCK_N

        rk = ki * BLOCK_K

        {%- if TMA_EXPERIMENTAL_API %}
        a = tl._experimental_descriptor_load(
            a_desc_ptr,
            [rm, rk] if A_ROW_MAJOR else [rk, rm],
            [BLOCK_M, BLOCK_K] if A_ROW_MAJOR else [BLOCK_K, BLOCK_M],
            A.dtype.element_ty,
        )
        b = tl._experimental_descriptor_load(
            b_desc_ptr,
            [rk, rn] if B_ROW_MAJOR else [rn, rk],
            [BLOCK_K, BLOCK_N] if B_ROW_MAJOR else [BLOCK_N, BLOCK_K],
            B.dtype.element_ty,
        )
        {%- else %}
        a = tl.load_tensor_descriptor(
            a_desc,
            [rm, rk] if A_ROW_MAJOR else [rk, rm],
        )
        b = tl.load_tensor_descriptor(
            b_desc,
            [rk, rn] if B_ROW_MAJOR else [rn, rk],
        )
        {%- endif %}
        acc += tl.dot(
            a if A_ROW_MAJOR else a.T,
            b if B_ROW_MAJOR else b.T,
            allow_tf32=ALLOW_TF32,
        )

        if ki == k_tiles - 1:
            # rematerialize rm and rn to save registers
            rcm = rm + tl.arange(0, BLOCK_M)
            rcn = rn + tl.arange(0, BLOCK_N)
            idx_m = rcm[:, None]
            idx_n = rcn[None, :]
            mask = (idx_m < M) & (idx_n < N)

            # inductor generates a suffix
            {{store_output(("idx_m", "idx_n"), "acc", "mask", indent_width=12)}}
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

""",
)

load_scales = r"""
@triton.jit
def load_scales(a_scale_ptr, b_scale_ptr, SCALING_ROWWISE: tl.constexpr):
    if SCALING_ROWWISE:
        # For row-wise scaling, we'll return the pointers
        return a_scale_ptr, b_scale_ptr
    else:
        # For per-tensor scaling, we'll load the scalar values
        a_scale = tl.load(a_scale_ptr)
        b_scale = tl.load(b_scale_ptr)
        return a_scale, b_scale
"""


apply_scaling = r"""
@triton.jit
def apply_scaling(
    accumulator,
    a_scale,
    b_scale,
    SCALING_ROWWISE: tl.constexpr,
    offs_cm,
    offs_cn,
    M,
    N,
    stride_a_scale_m,
    stride_b_scale_n,
):
    if SCALING_ROWWISE:
        # For row-wise scaling, we need to load the scales for each row/column
        a_scales = tl.load(
            a_scale + (offs_cm * stride_a_scale_m),
            mask=offs_cm < M,
            other=0.0,
        )
        b_scales = tl.load(
            b_scale + (offs_cn * stride_b_scale_n),
            mask=offs_cn < N,
            other=0.0,
        )
        acc_scale = a_scales[:, None] * b_scales[None, :]
    else:
        # For per-tensor scaling, we can directly use the loaded scalar values
        acc_scale = a_scale * b_scale

    return accumulator * acc_scale
"""


device_tma = r"""
{{def_kernel("A", "B", "A_inverse_scale", "B_inverse_scale")}}
    M = {{size("A", 0)}}
    N = {{size("B", 1)}}
    K = {{size("A", 1)}}
    if M * N == 0:
        # early exit due to zero-size input(s)
        return

    stride_am = {{stride("A", 0)}}
    stride_ak = {{stride("A", 1)}}
    stride_bk = {{stride("B", 0)}}
    stride_bn = {{stride("B", 1)}}

    if SCALING_ROWWISE:
        stride_a_scale_m = 1
        stride_b_scale_n = 1
    else:
        stride_a_scale_m = 0
        stride_b_scale_n = 0

    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n

    {%- if TMA_EXPERIMENTAL_API %}
    workspace_base = ws_ptr + start_pid * 2 * TMA_SIZE
    a_desc_ptr = workspace_base
    b_desc_ptr = workspace_base + TMA_SIZE

    triton.language.extra.cuda.experimental_device_tensormap_create2d(
        desc_ptr=a_desc_ptr,
        global_address=A,
        load_size=[BLOCK_M, BLOCK_K],
        global_size=[M, K],
        element_ty=A.dtype.element_ty,
    )
    triton.language.extra.cuda.experimental_device_tensormap_create2d(
        desc_ptr=b_desc_ptr,
        global_address=B,
        load_size=[BLOCK_N, BLOCK_K],
        global_size=[N, K],
        element_ty=B.dtype.element_ty,
    )

    tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(a_desc_ptr)
    tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(b_desc_ptr)

    {%- else %}
    stride_am = {{stride("A", 0)}}
    stride_bn = {{stride("B", 1)}}
    a_desc = triton.language.make_tensor_descriptor(
        base=A,
        shape=[M, K],
        strides=[stride_am, 1],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    b_desc = triton.language.make_tensor_descriptor(
        base=B,
        shape=[N, K],
        strides=[stride_bn, 1],
        block_shape=[BLOCK_N, BLOCK_K],
    )
    {%- endif %}

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1

    pid_m = 0
    pid_n = 0
    offs_am = 0
    offs_bn = 0

    num_pid_in_group = GROUP_M * num_pid_n
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    a_scale, b_scale = load_scales(A_inverse_scale, B_inverse_scale, SCALING_ROWWISE)

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            offs_am = pid_m * BLOCK_M
            offs_bn = pid_n * BLOCK_N

        offs_k = ki * BLOCK_K

        {%- if TMA_EXPERIMENTAL_API %}
        a = tl._experimental_descriptor_load(
            a_desc_ptr, [offs_am, offs_k], [BLOCK_M, BLOCK_K],  A.dtype.element_ty
        )
        b = tl._experimental_descriptor_load(
            b_desc_ptr, [offs_bn, offs_k], [BLOCK_N, BLOCK_K],  B.dtype.element_ty
        )
        {%- else %}
        a = tl.load_tensor_descriptor(a_desc, [offs_am, offs_k])
        b = tl.load_tensor_descriptor(b_desc, [offs_bn, offs_k])
        {%- endif %}
        if USE_FAST_ACCUM:
            accumulator = tl.dot(a, b.T, accumulator)
        else:
            accumulator += tl.dot(a, b.T)

        if ki == k_tiles - 1:
            # Apply inverse scaling
            offs_cm = offs_am + tl.arange(0, BLOCK_M)
            offs_cn = offs_bn + tl.arange(0, BLOCK_N)
            # Apply scaling
            accumulator = apply_scaling(
                accumulator,
                a_scale,
                b_scale,
                SCALING_ROWWISE,
                offs_cm,
                offs_cn,
                M,
                N,
                stride_a_scale_m,
                stride_b_scale_n,
            )

            idx_m = offs_cm[:, None]
            idx_n = offs_cn[None, :]
            mask = (idx_m < M) & (idx_n < N)
            # inductor generates a suffix
            {{store_output(("idx_m", "idx_n"), "accumulator", "mask", indent_width=12)}}
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
"""


scaled_mm_device_tma_template = TritonTemplate(
    name="scaled_mm_device_tma",
    grid=persistent_mm_grid,
    source=device_tma + load_scales + apply_scaling,
)


# prevent duplication registration of extern functions
@functools.cache
def lazy_register_extern_choice(fn):
    return ExternKernelChoice(fn)


aten_mm = ExternKernelChoice(torch.mm, "at::mm_out", op_overload=aten.mm.out)

aten_addmm = ExternKernelChoice(
    torch.addmm, "at::addmm_out", op_overload=aten.addmm.out
)

aten__int_mm = ExternKernelChoice(
    torch._int_mm, "at::_int_mm_out", op_overload=aten._int_mm.out
)

aten__sparse_semi_structured_mm = ExternKernelChoice(
    torch._sparse_semi_structured_mm,
    "at::_sparse_semi_structured_mm",
    has_out_variant=False,
    op_overload=aten._sparse_semi_structured_mm.default,
)

aten__fp8_mm = ExternKernelChoice(
    torch._scaled_mm, "at::_scaled_mm_out", op_overload=aten._scaled_mm.out
)


def _is_int8_mat(mat):
    return mat.get_dtype() in (torch.int8, torch.uint8)


def bias_addmm(inp, mat1, mat2, *, out=None, alpha=1, beta=1):
    """
    Giving torch.addmm a 1D tensor calls a different (faster) cublasLt
    kernel under the hood.  There are a few shapes where this is slower,
    but they are rare.
    """
    if (inp.stride(0) == 0 and inp.size(0) != 0) or inp.size(0) == 1:
        return torch.addmm(inp[0], mat1, mat2, out=out, alpha=alpha, beta=beta)
    return torch.addmm(inp, mat1, mat2, out=out, alpha=alpha, beta=beta)


def check_supported_striding(mat_a, mat_b) -> None:
    def is_row_major(stride) -> bool:
        return V.graph.sizevars.statically_known_equals(stride[1], 1)

    def is_col_major(stride) -> bool:
        return V.graph.sizevars.statically_known_equals(stride[0], 1)

    def has_zero_dim(size) -> bool:
        return bool(
            V.graph.sizevars.statically_known_equals(size[0], 0)
            or V.graph.sizevars.statically_known_equals(size[1], 0)
        )

    # Check mat_a (self) stride requirements
    torch._check(
        is_row_major(mat_a.get_stride()) or has_zero_dim(mat_a.get_size()),
        lambda: f"mat_a must be row_major, got stride {mat_a.get_stride()}",
    )

    # Check mat_b stride requirements
    torch._check(
        is_col_major(mat_b.get_stride()) or has_zero_dim(mat_b.get_size()),
        lambda: f"mat_b must be col_major, got stride {mat_b.get_stride()}",
    )


aten_bias_addmm = ExternKernelChoice(bias_addmm, None)


def decomposeK(a, b, k_splits):
    m = a.shape[0]
    n = b.shape[1]
    k = a.shape[1]

    k_parts = k // k_splits
    B = k_splits
    a_reshaped = torch.permute(a.reshape(m, B, k_parts), (1, 0, 2))
    b_reshaped = b.reshape(B, k_parts, n)
    result = torch.bmm(a_reshaped, b_reshaped, out_dtype=torch.float32)
    reduced_buf = torch.sum(result, 0)
    return reduced_buf.to(a.dtype)


class DecomposeKSugraphTemplate(SubgraphTemplate):
    def __init__(self):
        super().__init__(
            name="decompose_k",
        )

    def generate(  # type: ignore[override]
        self,
        input_nodes: list[Buffer],
        layout: Layout,
        k_split: int,
    ) -> SubgraphChoiceCaller:
        from torch._dispatch.python import enable_python_dispatcher

        from ..decomposition import select_decomp_table

        name = f"decompose_k_mm_{k_split}_split"
        description = f"{k_split=}"

        with enable_python_dispatcher():
            decompositions = select_decomp_table()
            fn = make_fx(
                functools.partial(decomposeK, k_splits=k_split),
                decompositions,
            )

            return super().generate(
                name=name,
                input_nodes=input_nodes,
                layout=layout,
                make_fx_graph=fn,
                description=description,
            )


decompose_k_subgraph_template = DecomposeKSugraphTemplate()


class ContiguousTemplate(SubgraphTemplate):
    def __init__(self, name: str, description: str, fn: Any):
        self.name = name
        self.description = description
        self.fn = fn
        super().__init__(
            name=name,
        )

    def generate(  # type: ignore[override]
        self,
        input_nodes: list[Buffer],
        layout: Layout,
    ) -> SubgraphChoiceCaller:
        from torch._dispatch.python import enable_python_dispatcher

        from ..decomposition import select_decomp_table

        with enable_python_dispatcher():
            decompositions = select_decomp_table()
            fn = make_fx(
                self.fn,
                decompositions,
            )

            return super().generate(
                name=self.name,
                input_nodes=input_nodes,
                layout=layout,
                make_fx_graph=fn,
                description=self.description,
            )


def contiguous_mm(a, b):
    return torch.mm(a, b.contiguous())


def contiguous_addmm(inp, a, b):
    return torch.addmm(inp, a, b.contiguous())


mm_contiguous_subgraph_template = ContiguousTemplate(
    "contiguous_mm", "contiguous mm", contiguous_mm
)
addmm_contiguous_subgraph_template = ContiguousTemplate(
    "contiguous_addmm", "contiguous addmm", contiguous_addmm
)


@register_lowering(aten.mm, type_promotion_kind=None)
def tuned_mm(mat1, mat2, *, layout=None):
    """
    Lowering for autotuning aten.mm with different backends (Aten, Triton, CUTLASS, etc.)
    """
    # TODO(coconutruben): integrate into MMKernelInputs when all callsites use that
    m, n, k, layout, mat1, mat2 = mm_args(mat1, mat2, layout=layout)

    # Early logging to confirm we're in the function
    if STREAMK_DEBUG:
        print(f"[StreamK-ENTRY] tuned_mm called for {m}x{n}x{k}")
        streamk_log_info(f"Entering tuned_mm for {m}x{n}x{k}")
    static_shape, is_nonzero = _is_static_problem(layout)
    name = "mm"

    # Create MMKernelInputs for standard MM at the top
    kernel_inputs = MMKernelInputs([mat1, mat2])

    # below is for getting an overview logging info of inductor mms
    counters["aten_mm_info"][f"aten.mm_{m}_{n}_{k}"] += 1
    log.info(
        "Tuned aten.mm: m=%s, n=%s, k=%s, mat1_dtype=%s, mat2_dtype=%s, output_layout=%s",
        m,
        n,
        k,
        mat1.get_dtype(),
        mat2.get_dtype(),
        layout,
    )

    aten_layout = layout
    if not (inductor_config.max_autotune or inductor_config.max_autotune_gemm):
        aten_layout = FlexibleLayout(
            device=layout.device, dtype=layout.dtype, size=layout.size
        )
    choices: list[ChoiceCaller] = []

    # Initialize StreamK usage flag
    streamk_should_be_used = False

    # Always generate autotuning choices for competition (unless explicitly disabled)
    if use_aten_gemm_kernels() and not DISABLE_OTHER_BACKENDS:
        choices.extend(
            V.choices.get_mm_configs(kernel_inputs, aten_layout, [aten_mm], "mm")
        )
    static_shape, is_nonzero = _is_static_problem(layout)

    if is_nonzero and use_triton_template(layout, check_max_autotune=False) and not DISABLE_OTHER_BACKENDS:
        # Get template choices using the new unified function
        choices.extend(
            V.choices.get_mm_configs(kernel_inputs, layout, [mm_template], "mm")
        )
        if use_triton_tma_template(mat1, mat2) and not DISABLE_OTHER_BACKENDS:
            # Get TMA template choices using the new unified function
            choices.extend(
                V.choices.get_mm_configs(
                    kernel_inputs, layout, [persistent_tma_mm_template], "mm"
                )
            )

        if use_decompose_k_choice(m, n, k) and not DISABLE_OTHER_BACKENDS:
            choices.extend(
                V.choices.get_mm_configs(
                    kernel_inputs, layout, [decompose_k_subgraph_template], "mm"
                )
            )
        if not DISABLE_OTHER_BACKENDS:
            choices.extend(
                V.choices.get_mm_configs(
                    kernel_inputs, layout, [mm_contiguous_subgraph_template], "mm"
                )
            )

    # Add StreamK+origami as competitor choice (following tritonBLAS pattern)
    if static_shape and is_nonzero and ENABLE_STREAMK:
        # Check if StreamK should be used (this handles symbolic detection)
        streamk_should_be_used = should_use_streamk(m, n, k, mat1.get_dtype(), layout.device)

        # Add StreamK as a competitor choice if recommended OR if forced (but respect symbolic detection)
        if streamk_should_be_used:
            if STREAMK_ONLY:
                streamk_log_info(f"🎯 STREAMK_ONLY=1: Will compete StreamK+origami vs autotuned mm for {m}x{n}x{k}")
                # Clear other choices to make it StreamK-only
                choices = []
            elif FORCE_STREAMK:
                streamk_log_info(f"🔥 FORCE_STREAMK=1: Adding StreamK+origami competitor for {m}x{n}x{k}")
            else:
                log.info(f"🏁 Adding StreamK+origami as competitor choice for {m}x{n}x{k}")
        elif STREAMK_ONLY or FORCE_STREAMK:
            # Force modes requested but StreamK was rejected (likely due to symbolic variables)
            streamk_log_info(f"⚠️ FORCE modes requested but StreamK was disabled (likely symbolic variables)")
            streamk_log_info(f"   STREAMK_ONLY={STREAMK_ONLY}, FORCE_STREAMK={FORCE_STREAMK}")
            streamk_log_info(f"   StreamK will be skipped to prevent errors.")

        # Only proceed with StreamK if it should be used
        if streamk_should_be_used:
            streamk_log_info(f"✅ StreamK+origami competing for {m}×{n}×{k} {mat1.get_dtype()}")

            try:
                # Use origami selector following tritonBLAS pattern
                # selector = _make_matmul_selector(M, N, K, a.dtype, b.dtype, c.dtype)
                selector = _make_streamk_selector(
                    m, n, k,
                    mat1.get_dtype(),
                    mat2.get_dtype(),
                    layout.dtype,  # c_dtype
                    str(layout.device)  # device_type as string
                )

                # Get optimal configuration from selector
                optimal_config = selector.get_config()
                optimal_grid = selector.get_grid()

                streamk_log_info(f"🎯 Origami selected competitor config: "
                                f"BLOCK_M={optimal_config['BLOCK_M']}, "
                                f"BLOCK_N={optimal_config['BLOCK_N']}, "
                                f"BLOCK_K={optimal_config['BLOCK_K']}, "
                                f"STREAMK_TILES={optimal_config['STREAMK_TILES']}")

                # Create single optimal StreamK competitor choice
                streamk_choices = []
                try:
                    streamk_log_debug("Adding StreamK+origami as competitor choice")

                    # Extract triton-specific parameters (use tritonBLAS defaults)
                    num_warps = optimal_config.pop("num_warps", 8)  # tritonBLAS default
                    num_stages = optimal_config.pop("num_stages", 2)  # tritonBLAS default

                    # Use the StreamK template with optimal configuration as competitor
                    error = mm_streamk_template.maybe_append_choice(
                        streamk_choices,
                        input_nodes=(kernel_inputs.nodes()[0], kernel_inputs.nodes()[1]),  # Just A and B
                        layout=layout,
                        num_warps=num_warps,
                        num_stages=num_stages,
                        **optimal_config
                    )

                    if error is not None:
                        streamk_log_debug(f"⚠️ StreamK competitor choice generation failed: {error}")
                        streamk_choices = []
                    else:
                        streamk_log_info(f"✅ StreamK+origami competitor choice created")

                except Exception as e:
                    streamk_log_debug(f"⚠️ StreamK+origami competitor creation failed: {e}")
                    if STREAMK_VERBOSE:
                        import traceback
                        streamk_log_verbose(f"   Full traceback: {traceback.format_exc()}")
                    streamk_choices = []

                # Add StreamK competitor to the choice pool for autotuning competition
                choices_before = len(choices)
                choices.extend(streamk_choices)

                if len(streamk_choices) > 0:
                    streamk_log_info(f"✅ StreamK+origami added as competitor choice "
                                    f"(total choices: {len(choices)}, mm+autotuned: {choices_before}, streamk: {len(streamk_choices)})")
                    streamk_log_info(f"🏁 Autotuning will now benchmark StreamK vs mm and select the winner")
                else:
                    streamk_log_info(f"❌ Failed to add StreamK competitor - autotuning will use mm choices only")

            except Exception as e:
                streamk_log_info(f"❌ StreamK+origami competitor setup failed: {e}")
                if "truth value of Relational" in str(e) or "cannot determine truth value" in str(e):
                    streamk_log_info(f"💀 Symbolic variable error detected in origami selector.")
                streamk_log_info(f"   Autotuning will proceed with mm choices only")

        streamk_log_info(f"🏁 StreamK+origami competition setup completed for {m}×{n}×{k}")
    else:
        if STREAMK_DEBUG:
            streamk_log_info(f"🚫 StreamK was NOT added as competitor for {m}×{n}×{k} (should_use={streamk_should_be_used})")
            streamk_log_info(f"   Autotuning will proceed with mm choices only")

    if (
        is_nonzero
        and use_cutlass_template(layout, m, n, k)
        and _use_cutlass_for_op("mm")
        and not STREAMK_ONLY
        and not DISABLE_OTHER_BACKENDS
    ):
        CUTLASS3xGemmTemplate.add_cutlass_gemm_choices(
            choices, layout, kernel_inputs.nodes()
        )

    if is_nonzero and use_ck_gemm_template(layout, m, n, k) and not STREAMK_ONLY and not DISABLE_OTHER_BACKENDS:
        CKGemmTemplate.add_ck_gemm_choices(choices, layout, kernel_inputs.nodes())
    if is_nonzero and use_ck_tile_gemm_template(layout, m, n, k) and not STREAMK_ONLY and not DISABLE_OTHER_BACKENDS:
        CKTileGemmTemplate.add_choices(choices, layout, kernel_inputs.nodes())

    if use_cpp_gemm_template(layout, mat1, mat2) and not STREAMK_ONLY and not DISABLE_OTHER_BACKENDS:
        CppGemmTemplate.add_choices(
            choices,
            layout,
            kernel_inputs.nodes(),
        )

    input_nodes = [mat1, mat2]
    if (
        is_nonzero
        and use_triton_template(layout)
        and torch._inductor.config.run_autoheuristic(name)
        and is_triton(mat1)
    ):
        always_included = []
        if use_aten_gemm_kernels():
            always_included.append("extern_mm")
        num_choices_before_extra_configs = len(choices)
        choices.extend(
            V.choices.get_mm_configs(
                # TODO(coconutruben): remove once we deprecate ah
                # mm-extra is a hack to keep the ah functionality alive
                # while we transition to the unified kwargs retrieval
                kernel_inputs,
                layout,
                [mm_template],
                "mm-ah",
            )
        )

        # using AutoHeuristic for ranking
        ah_choices = mm_autoheuristic(
            mat1,
            mat2,
            m,
            n,
            k,
            choices,
            name,
            input_nodes,
            mm_operations(),
            None,
            top_k=10,
            always_included=always_included,
        )
        if not torch._inductor.config.collect_autoheuristic(name):
            # if we are collecting data, we do not want to modify choices
            if ah_choices is not None and len(ah_choices) > 0:
                # the order in which autoheuristic returns choices is not the same as
                # as the order of choices, which affects things like epilogue fusion.
                # once epilogue fusion benchmarks choices in sorted order, I think we can
                # just use the order returned by autoheuristic
                choices = [choice for choice in choices if choice in ah_choices]
            else:
                choices = choices[:num_choices_before_extra_configs]

    for k in inductor_config.external_matmul:
        choices.append(
            lazy_register_extern_choice(k).bind(kernel_inputs.nodes(), layout)
        )

    best_config_future = None
    # Purposely not awaiting the future here - this kicks off the best config lookup at lowering time
    # The future will be awaited at scheduling time in select_algorithm.py
    if torch._inductor.config.remote_gemm_autotune_cache:
        best_config_future = gen_best_config(mat1, mat2)

    # Safety check for experimental modes
    if (STREAMK_ONLY or FORCE_STREAMK) and len(choices) == 0:
        streamk_log_info(f"⚠️  WARNING: No choices generated for {m}x{n}x{k} in force mode!")
        streamk_log_info(f"   This might indicate StreamK config generation failed or symbolic shapes detected.")
        streamk_log_info(f"   Adding a fallback choice to prevent crash...")

        # Detailed debugging for why no choices were generated
        streamk_log_info(f"🔍 DEBUGGING: Why no choices were generated:")
        streamk_log_info(f"   Problem size: {m}×{n}×{k}")
        streamk_log_info(f"   Mat1 dtype: {mat1.get_dtype()}, Mat2 dtype: {mat2.get_dtype()}")
        streamk_log_info(f"   Device: {layout.device}")
        streamk_log_info(f"   STREAMK_ONLY: {STREAMK_ONLY}, FORCE_STREAMK: {FORCE_STREAMK}")
        streamk_log_info(f"   streamk_should_be_used was: {streamk_should_be_used}")
        streamk_log_info(f"   Layout: {layout}")

        # Check if StreamK heuristic would approve this case
        try:
            would_use_streamk = should_use_streamk(m, n, k, mat1.get_dtype(), layout.device)
            streamk_log_info(f"   should_use_streamk check: {would_use_streamk}")
        except Exception as e:
            streamk_log_info(f"   should_use_streamk check failed: {e}")

        # Check config generation
        try:
            test_configs = generate_streamk_configs(m, n, k, layout.device, enable_origami=True)
            streamk_log_info(f"   Config generation test: {len(test_configs)} configs")
        except Exception as e:
            streamk_log_info(f"   Config generation test failed: {e}")

        # Add a basic fallback to prevent complete failure
        if use_aten_gemm_kernels():
            try:
                fallback_choices = list(V.choices.get_mm_configs(kernel_inputs, layout, [aten_mm], "mm"))
                choices.extend(fallback_choices)
                streamk_log_info(f"   Added {len(fallback_choices)} fallback choices")
            except Exception as e:
                streamk_log_info(f"   Fallback choice generation also failed: {e}")
                streamk_log_info(f"   This might be due to symbolic variables affecting all choice generation.")

    # Log final choice summary for StreamK debugging
    log_choices_summary(choices, f"{m}x{n}x{k} GEMM")

    streamk_log_debug(f"🏁 Final auto-tuning for {m}x{n}x{k} with {len(choices)} total choices")

    # Enhanced logging to track which template gets selected
    if STREAMK_DEBUG:
        streamk_log_info(f"📊 Choice Details for {m}x{n}x{k}:")
        for i, choice in enumerate(choices):
            choice_name = getattr(choice, 'name', str(type(choice).__name__))
            choice_template = getattr(choice, 'template', None)
            if choice_template:
                template_name = getattr(choice_template, 'name', 'unknown')
                if 'streamk' in template_name.lower():
                    streamk_log_info(f"  🟢 Choice {i}: {choice_name} (StreamK template: {template_name})")
                else:
                    streamk_log_info(f"  ⚪ Choice {i}: {choice_name} (template: {template_name})")
            else:
                streamk_log_info(f"  ⚪ Choice {i}: {choice_name} (no template info)")

    result = autotune_select_algorithm(
        name,
        choices,
        kernel_inputs.nodes(),
        layout,
        best_config_future=best_config_future,
    )

    # Log which choice was actually selected
    if STREAMK_DEBUG and hasattr(result, 'choice'):
        selected_choice = result.choice
        choice_name = getattr(selected_choice, 'name', str(type(selected_choice).__name__))
        choice_template = getattr(selected_choice, 'template', None)
        if choice_template:
            template_name = getattr(choice_template, 'name', 'unknown')
            if 'streamk' in template_name.lower():
                streamk_log_info(f"🎯 SELECTED StreamK: {choice_name} with template {template_name}")
            else:
                streamk_log_info(f"🎯 SELECTED non-StreamK: {choice_name} with template {template_name}")
        else:
            streamk_log_info(f"🎯 SELECTED: {choice_name} (no template info)")

    return result


@register_lowering(aten._int_mm, type_promotion_kind=None)
def tuned_int_mm(mat1, mat2, *, layout=None):
    # TODO(coconutruben): integrate into MMKernelInputs when all callsites use that
    m, n, k, layout, mat1, mat2 = mm_args(
        mat1, mat2, layout=layout, out_dtype=torch.int32
    )
    name = "int_mm"
    # below is for getting an overview logging info of inductor mms
    counters["aten_mm_info"][f"aten._int_mm_{m}_{n}_{k}"] += 1
    log.info(
        "Tuned aten._int_mm: m=%s, n=%s, k=%s, mat1_dtype=%s, mat2_dtype=%s, output_layout=%s",
        m,
        n,
        k,
        mat1.get_dtype(),
        mat2.get_dtype(),
        layout,
    )

    static_shape, is_nonzero = _is_static_problem(layout)
    use_cutlass = static_shape and is_nonzero and use_cutlass_template(layout, m, n, k)
    choices: list[ChoiceCaller] = []

    # Create MMKernelInputs for Int MM
    kernel_inputs = MMKernelInputs([mat1, mat2])
    if use_aten_gemm_kernels():
        choices.extend(
            V.choices.get_mm_configs(
                kernel_inputs,
                layout,
                [aten__int_mm],
                name,
            )
        )

    if use_cutlass and _use_cutlass_for_op(name):
        CUTLASS3xGemmTemplate.add_cutlass_gemm_choices(
            choices, layout, kernel_inputs.nodes(), fuseable=True, non_fuseable=True
        )

    if is_nonzero and use_triton_template(
        layout, enable_int32=True, check_max_autotune=False
    ):
        choices.extend(
            V.choices.get_mm_configs(kernel_inputs, layout, [mm_template], name)
        )

    return autotune_select_algorithm(name, choices, kernel_inputs.nodes(), layout)


@register_lowering(aten.addmm, type_promotion_kind=None)
def tuned_addmm(inp, mat1, mat2, *, alpha=1, beta=1, layout=None):
    """
    Lowering for autotuning aten.addmm with different backends (Aten, Triton, CUTLASS, etc.)
    """
    # TODO(coconutruben): integrate into MMKernelInputs when all callsites use that
    m, n, k, layout, mat1, mat2, inp_expanded = mm_args(mat1, mat2, inp, layout=layout)
    static_shape, is_nonzero = _is_static_problem(layout)
    name = "addmm"
    # Create MMKernelInputs for AddMM at the top
    kernel_inputs = MMKernelInputs(
        [inp_expanded, mat1, mat2], scalars=dict(alpha=alpha, beta=beta)
    )
    choices: list[ChoiceCaller] = []

    # below is for getting an overview logging info of inductor mms
    counters["aten_mm_info"][f"aten.addmm_{m}_{n}_{k}"] += 1
    log.info(
        "Tuned aten.addmm: m=%s, n=%s, k=%s, mat1_dtype=%s, mat2_dtype=%s, output_layout=%s",
        m,
        n,
        k,
        mat1.get_dtype(),
        mat2.get_dtype(),
        layout,
    )
    aten_layout = layout
    if (not is_nonzero) or (
        not (inductor_config.max_autotune or inductor_config.max_autotune_gemm)
    ):
        # Use a FlexibleLayout if we are not autotuning.
        # This allows padding strides for the output.
        from torch._inductor.ir import FixedLayout, FlexibleLayout

        if isinstance(layout, FixedLayout):
            aten_layout = FlexibleLayout(
                device=layout.device, dtype=layout.dtype, size=layout.size
            )
        # TODO(coconutruben): combine this with the main flow of addmm through
        # a subgraph or something as inp vs inp_expanded causes some slight numeric
        # differences
        kernel_inputs = MMKernelInputs(
            [inp, mat1, mat2], scalars=dict(alpha=alpha, beta=beta)
        )
        choices.extend(
            V.choices.get_mm_configs(
                kernel_inputs,
                aten_layout,
                [aten_addmm],
                name,
            )
        )
        return autotune_select_algorithm(name, choices, kernel_inputs.nodes(), layout)

    if use_aten_gemm_kernels():
        choices.extend(
            V.choices.get_mm_configs(
                kernel_inputs,
                aten_layout,
                [aten_bias_addmm],
                name,
            )
        )
        choices.extend(
            V.choices.get_mm_configs(
                kernel_inputs,
                aten_layout,
                [aten_addmm],
                name,
            )
        )

    if is_nonzero and use_triton_template(layout, check_max_autotune=False):
        # all the triton templates use the extra_kwargs
        # Get template choices using the new unified function
        choices.extend(
            V.choices.get_mm_configs(
                kernel_inputs,
                layout,
                [mm_template],
                name,
            )
        )

        if use_triton_tma_template(mat1, mat2):
            # Get TMA template choices using the new unified function
            choices.extend(
                V.choices.get_mm_configs(
                    kernel_inputs,
                    layout,
                    [persistent_tma_mm_template],
                    name,
                )
            )

        choices.extend(
            V.choices.get_mm_configs(
                kernel_inputs,
                layout,
                [addmm_contiguous_subgraph_template],
                "addmm",
            )
        )

    if (
        is_nonzero
        and use_cutlass_template(layout, m, n, k)
        and _use_cutlass_for_op(name)
    ):
        CUTLASS3xGemmTemplate.add_cutlass_gemm_choices(
            choices,
            layout,
            # reorder here because CUTLASS expects (x, w, bias) but torch
            # is bias, x, w
            kernel_inputs.nodes(reorder=[1, 2, 0]),
            alpha=alpha,
            beta=beta,
        )

    if is_nonzero and use_ck_gemm_template(layout, m, n, k):
        CKGemmTemplate.add_ck_gemm_choices(
            choices,
            layout,
            # reorder here because CK expects (x, w, bias) but torch
            # is bias, x, w
            kernel_inputs.nodes(reorder=[1, 2, 0]),
            alpha=alpha,
            beta=beta,
            input_reorder=[2, 0, 1],
        )

    if use_cpp_gemm_template(layout, mat1, mat2):
        CppGemmTemplate.add_choices(
            choices,
            layout,
            kernel_inputs.nodes(),
            alpha=alpha,
            beta=beta,
            has_bias=True,
        )

    return autotune_select_algorithm(name, choices, kernel_inputs.nodes(), layout)


@register_lowering(aten._sparse_semi_structured_mm, type_promotion_kind=None)
def tuned_sparse_semi_structured_mm(
    mat1, mat1_meta, mat2, *, out_dtype=None, layout=None
):
    from torch._inductor.select_algorithm import realize_inputs

    # TODO(coconturuben): support V.choices.get_mm_configs for sparse_semi_structured_mm
    mat1, mat1_meta, mat2 = realize_inputs(mat1, mat1_meta, mat2)
    m1, k1 = mat1.get_size()
    m2, _ = mat1_meta.get_size()
    k2, n = mat2.get_size()
    m = V.graph.sizevars.check_equals_and_simplify(m1, m2)
    k = V.graph.sizevars.check_equals_and_simplify(2 * k1, k2)
    if layout is None:
        from torch._inductor.ir import FixedLayout

        layout = FixedLayout(
            mat2.get_device(),
            out_dtype if out_dtype else mat2.get_dtype(),
            [m, n],
            [n, 1],
        )
    else:
        assert out_dtype is None, "out_dtype is ignored if layout is specified."

    choices = (
        [
            aten__sparse_semi_structured_mm.bind(
                (mat1, mat1_meta, mat2), layout, out_dtype=out_dtype
            )
        ]
        if use_aten_gemm_kernels()
        else []
    )

    if (
        m * n != 0
        and use_cutlass_template(layout, m, n, k)
        and _use_cutlass_for_op("sparse_semi_structured_mm")
    ):
        CUTLASS2xGemmTemplate.add_cutlass_gemm_choices(
            choices, layout, [mat1, mat2, mat1_meta], fuseable=True, non_fuseable=True
        )

    return autotune_select_algorithm(
        "sparse_semi_structured_mm", choices, (mat1, mat1_meta, mat2), layout
    )


add_layout_constraint(aten._scaled_mm.default, constrain_to_fx_strides)


@register_lowering(aten._scaled_mm.default, type_promotion_kind=None)  # type: ignore[misc]
def tuned_scaled_mm(
    mat_a,
    mat_b,
    scale_a,
    scale_b,
    bias=None,
    scale_result=None,
    out_dtype=None,
    use_fast_accum=False,
    layout=None,
):
    """
    Performs an optimized matrix multiplication where scaling factors are applied
    to the inputs and/or output.

    Args:
        mat1 (Tensor): First input matrix
        mat2 (Tensor): Second input matrix
        scale1 (Tensor): Scale factor applied to mat1 (supports broadcasting)
        scale2 (Tensor): Scale factor applied to mat2 (supports broadcasting)
        bias (Tensor, optional): Optional bias tensor to add to the result
        layout: Layout hint for optimization

    Returns:
        Tensor: The result of the scaled matrix multiplication
    """
    # TODO(coconutruben): integrate into MMKernelInputs when all callsites use that
    m, n, k, layout, mat_a, mat_b = mm_args(
        mat_a, mat_b, layout=layout, out_dtype=out_dtype
    )
    # below is for getting an overview logging info of inductor mms
    counters["aten_mm_info"][f"aten._scaled_mm.default_{m}_{n}_{k}"] += 1
    log.info(
        "Tuned aten._scaled_mm.default: m=%s, n=%s, k=%s, mat1_dtype=%s, mat2_dtype=%s, output_layout=%s",
        m,
        n,
        k,
        mat_a.get_dtype(),
        mat_b.get_dtype(),
        layout,
    )
    name = "scaled_mm"
    check_supported_striding(mat_a, mat_b)

    scale_a_real, scale_b_real = realize_inputs(scale_a, scale_b)

    input_nodes: list[Any]

    if not bias:
        input_nodes = [mat_a, mat_b, scale_a_real, scale_b_real]
    else:
        bias_real = realize_inputs(bias)
        input_nodes = [mat_a, mat_b, scale_a_real, scale_b_real, bias_real]

    # Create MMKernelInputs for Scaled MM (matrices are at indices 0, 1)
    kernel_inputs = MMKernelInputs(input_nodes, mat1_idx=0, mat2_idx=1)

    choices: list[ChoiceCaller] = []
    if use_aten_gemm_kernels():
        choices.extend(
            V.choices.get_mm_configs(
                kernel_inputs,
                layout,
                [aten__fp8_mm],
                name,
                kwarg_overrides={
                    aten__fp8_mm.uid: dict(
                        out_dtype=out_dtype, use_fast_accum=use_fast_accum
                    )
                },
            )
        )

    # We dont have triton lowerings for the MX variants yet
    if scale_a.dtype != torch.float32:
        return autotune_select_algorithm(name, choices, input_nodes, layout)

    _, is_nonzero = _is_static_problem(layout)

    if is_nonzero and use_triton_template(
        layout, enable_float8=True, check_max_autotune=False
    ):
        overriders = dict(USE_FAST_ACCUM=use_fast_accum)
        # TODO (paulzhan): There is no template that exists for bias and TMA
        # Don't run tma template currently if bias exists
        if use_triton_tma_template(mat_a, mat_b) and not bias:
            # Get TMA template choices using the new unified function
            choices.extend(
                V.choices.get_mm_configs(
                    kernel_inputs,
                    layout,
                    [scaled_mm_device_tma_template],
                    name,
                    kwarg_overrides={scaled_mm_device_tma_template.uid: overriders},
                )
            )

        # Get template choices using the new unified function
        choices.extend(
            V.choices.get_mm_configs(
                kernel_inputs,
                layout,
                [mm_template],
                name,
                kwarg_overrides={mm_template.uid: overriders},
            )
        )

    if (
        is_nonzero
        and use_cutlass_template(layout, m, n, k)
        and _use_cutlass_for_op(name)
    ):
        CUTLASS3xGemmTemplate.add_cutlass_gemm_choices(
            choices,
            layout,
            kernel_inputs.nodes(),  # type: ignore[arg-type]
            use_fast_accum=use_fast_accum,  # type: ignore[arg-type]
        )

    if is_nonzero and use_ck_gemm_template(layout, m, n, k):
        CKGemmTemplate.add_ck_gemm_choices(choices, layout, kernel_inputs.nodes())

    return autotune_select_algorithm(name, choices, kernel_inputs.nodes(), layout)


@functools.cache
def _is_sm7x_or_older_gpu(index: Optional[int]) -> bool:
    props = torch.cuda.get_device_properties(index or 0)
    return props.major <= 7


def dims_are_int(dims):
    return all(isinstance(dim, int) for dim in dims)


def mm_autoheuristic(
    mat1,
    mat2,
    m,
    n,
    k,
    choices,
    name,
    input_nodes,
    ops,
    precondition,
    top_k: Optional[int] = None,
    always_included=None,
):
    m, n, k = get_size_hints(mat1, mat2, m, n, k)
    if not dims_are_int([m, n, k]):
        return None
    mat1_stride, mat2_stride = get_size_hints_strides(mat1, mat2)

    def get_context(m, k, n, mat1, mat2, mat1_stride, mat2_stride):
        context = AHContext()
        context.add_feature("m", m)
        context.add_feature("k", k)
        context.add_feature("n", n)
        context.add_feature("mat1_dtype", mat1.layout.dtype, is_categorical=True)
        context.add_feature("mat2_dtype", mat2.layout.dtype, is_categorical=True)
        context_add_strides(context, "mat1", mat1_stride)
        context_add_strides(context, "mat2", mat2_stride)
        context.add_feature(
            "mat1_iscontig", mat1.layout.is_contiguous(), is_categorical=True
        )
        context.add_feature(
            "mat2_iscontig", mat2.layout.is_contiguous(), is_categorical=True
        )
        if name == "mm":
            context_add_using_tf32(context, mat1.layout.dtype)
        return context

    def fallback():
        return None

    context = get_context(m, k, n, mat1, mat2, mat1_stride, mat2_stride)
    autoheuristic = AutoHeuristicSelectAlgorithm(
        fallback=fallback,
        choices=choices,
        input_nodes=input_nodes,
        context=context,
        name=name,
        augment_context=ops,
        precondition=precondition,
    )

    if top_k is not None:
        # TODO: is there a cleaner way to ensure aten.mm is always included?
        return autoheuristic.get_top_k_choices_caller(
            top_k, always_included=always_included
        )

    return autoheuristic.get_choice_caller()


def get_size_hints(mat1, mat2, m, n, k):
    if not isinstance(m, int) or not isinstance(k, int):
        (m, k) = V.graph.sizevars.size_hints(
            mat1.get_size(),
            fallback=torch._inductor.config.unbacked_symint_fallback,
        )

    if not isinstance(n, int) or not isinstance(k, int):
        (k, n) = V.graph.sizevars.size_hints(
            mat2.get_size(),
            fallback=torch._inductor.config.unbacked_symint_fallback,
        )
    return m, n, k


def get_size_hints_strides(mat1, mat2):
    mat1_stride = mat1.layout.stride
    mat2_stride = mat2.layout.stride
    strides = [mat1_stride, mat2_stride]
    strides_hints = []
    for stride in strides:
        if not isinstance(stride, int):
            stride = V.graph.sizevars.size_hints(
                stride,
                fallback=torch._inductor.config.unbacked_symint_fallback,
            )
        strides_hints.append(stride)
    return strides_hints[0], strides_hints[1]
