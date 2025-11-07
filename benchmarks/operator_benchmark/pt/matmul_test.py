import operator_benchmark as op_bench

import torch


"""Microbenchmarks for MatMul operator"""

# Configs for PT Matmul operator
mm_short_configs = op_bench.config_list(
    attr_names=["M", "N", "K", "trans_a", "trans_b"],
    attrs=[
        [1, 1, 1, True, False],
        [128, 128, 128, True, False],
        [256, 256, 256, False, True],
    ],
<<<<<<< HEAD
    cross_product_configs={"device": ["cpu", "cuda"]},
=======
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    tags=["short"],
)


mm_long_configs = op_bench.cross_product_configs(
<<<<<<< HEAD
    M=[256, 1024, 3000],
    N=[512, 4096],
    K=[512, 4096],
    trans_a=[False, True],
    trans_b=[True, False],
    device=["cuda"],
    dtype=[torch.float16, torch.bfloat16, torch.float32],
=======
    M=[32],
    N=[512, 128],
    K=[64],
    trans_a=[False, True],
    trans_b=[True, False],
    device=["cpu", "cuda"],
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    tags=["long"],
)


class MatMulBenchmark(op_bench.TorchBenchmarkBase):
<<<<<<< HEAD
    def init(self, M, N, K, trans_a, trans_b, device, dtype=torch.float):
        # Create tensors without requires_grad first, then set it separately
        # This avoids creating graph leaves that cannot be deep copied
        if trans_a:
            input_one = torch.rand(M, N, device=device, dtype=dtype)
        else:
            input_one = torch.rand(N, M, device=device, dtype=dtype).t()

        if trans_b:
            input_two = torch.rand(N, K, device=device, dtype=dtype)
        else:
            input_two = torch.rand(K, N, device=device, dtype=dtype).t()

        # Set requires_grad after tensor creation to avoid graph leaf issues
        if self.auto_set():
            input_one.requires_grad_(True)
        if self.auto_set():
            input_two.requires_grad_(True)

        self.inputs = {
            "input_one": input_one,
            "input_two": input_two,
=======
    def init(self, M, N, K, trans_a, trans_b, device):
        self.inputs = {
            "input_one": torch.rand(M, N, device=device)
            if trans_a
            else torch.rand(N, M, device=device).t(),
            "input_two": torch.rand(N, K, device=device)
            if trans_b
            else torch.rand(K, N, device=device).t(),
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        }
        self.set_module_name("matmul")

    def forward(self, input_one, input_two):
        return torch.matmul(input_one, input_two)


op_bench.generate_pt_test(mm_long_configs + mm_short_configs, MatMulBenchmark)
<<<<<<< HEAD
op_bench.generate_pt_gradient_test(mm_long_configs, MatMulBenchmark)
=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
