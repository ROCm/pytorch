# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# from cffi import FFI
# ffi = FFI()

# def read_file(file_path):
#     with open(file_path, 'r') as file:
#         all_text = file.read()
#         return all_text

# # cpp_code = read_file('test_eigen_log.cpp')
# cpp_code = read_file('test_eigen.cpp')

# ffi.cdef(cpp_code)


# C = ffi.dlopen('/opt/rocm/lib/librocsolver.so')
# # C.rocsolver_log_begin()
# C.run_eigen()
# exit()

from itertools import product
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import triton  # @manual=//triton:triton

torch.manual_seed(42)


def benchmark(n: int, dtype=torch.float32) -> None:
    r = torch.rand(n, n, device="cuda", dtype=dtype)
    a = (r + r.T) / 2
    res = triton.testing.do_bench(lambda: torch.linalg.eigh(a))
    print(f"{n}: {res:.3f} ms")


"""
1) Instead of samplind a random matrix and using eigenvalues computed with a cpu
eigensolver as ground truth, let us start by sampling a set `l` of numbers and
building a matrix that realizes `l` as its spectrum. This way we test true
accuracy, otherwise, we would just test which method is closest to the cpu
eigensolver.

2) The accuracy of the Jacobi eigensolver is measured using a backward error
estimate, thus we need to have a look at this result to ascertain whether
Jacobi is working properly. Although an accurate implementation of Jacobi will
lead to typically accurate matrix entries, this is not a given (for example,
in the presence of catastrophic cancellation during matrix reconstruction) and
it is important for us to pinpoint whether the possible loss of accuracy is in
Jacobi or elsewhere.
"""


def compute_cond_num(l: torch.Tensor) -> float:
    absl = torch.abs(l)
    return (torch.max(absl) / torch.min(absl)).cpu().item()


def prepare_input(
    n: int, batch_size: Optional[int] = None, dtype=torch.float32
) -> (torch.Tensor, torch.Tensor):
    if batch_size is None:
        r = torch.rand(n, n, device="cuda", dtype=dtype)
        Q, _ = torch.linalg.qr(r)
        eps_ = torch.finfo(dtype).eps

        # Sample eigenvalues deterministically on interval [eps, sqrt(n)/2]
        # flake8: noqa
        l = torch.linspace(eps_, np.sqrt(n) / 2, n, device="cuda", dtype=dtype)

        # Create test input matrix
        a = Q.T @ torch.diag(l) @ Q
        return l, a

    l_list = []
    a_list = []
    for _ in range(batch_size):
        r = torch.rand(n, n, device="cuda", dtype=dtype)
        Q, _ = torch.linalg.qr(r)
        eps_ = torch.finfo(dtype).eps

        # Sample eigenvalues deterministically on interval [eps, sqrt(n)/2]
        # flake8: noqa
        l = torch.linspace(eps_, np.sqrt(n) / 2, n, device="cuda", dtype=dtype)

        # Create test input matrix
        a = Q.T @ torch.diag(l) @ Q
        l_list.append(l)
        a_list.append(a)
    return torch.stack(l_list), torch.stack(a_list)


def run(n: int, dtype=torch.float32) -> None:
    ## Prepare input:
    # Sample random orthogonal matrix
    r = torch.rand(n, n, device="cuda", dtype=dtype)
    Q, _ = torch.linalg.qr(r)
    eps_ = torch.finfo(dtype).eps

    # Sample eigenvalues deterministically on interval [eps, sqrt(n)/2]
    # flake8: noqa
    l = torch.linspace(eps_, np.sqrt(n) / 2, n, device="cuda", dtype=dtype)

    # Create test input matrix
    a = Q.T @ torch.diag(l) @ Q

    ## Run test:
    print("\n<<<")
    print("::: Testing with n = ", n)
    a_cpu = a.clone().detach().cpu().to(dtype)
    L_cpu = torch.diag(l).clone().detach().cpu().to(dtype)

    L, Q = torch.linalg.eigh(a)
    L = torch.diag(L.real)
    a2 = Q @ L @ Q.T
    a_norm_fro = torch.linalg.matrix_norm(a, "fro")

    # Compute eigendecomposition error (Frobenius norm):
    el = torch.linalg.matrix_norm(L.cpu() - L_cpu.to(dtype), "fro")
    print("Error (Frobenius): ", el / (n * eps_ * a_norm_fro))

    # Compute reconstruction error (Frobenius norm):
    rl = torch.linalg.matrix_norm(a - a2, "fro")
    print("Reconstruction error (Frobenius): ", rl / (n * eps_ * a_norm_fro))

    try:
        torch.testing.assert_close(a2.cpu(), a_cpu.to(dtype), rtol=1e-4, atol=1e-4)
    except Exception as e:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(e)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(">>>\n")


def condition_number(n: int, dtype=torch.float32) -> None:
    condition_numbers = []
    ntests = 1000

    for _ in range(ntests):
        l = np.sqrt(n) * (torch.rand(n, device="cuda", dtype=dtype) - 1 / 2)
        cond_num = compute_cond_num(l)
        condition_numbers.append(cond_num)

    print("max(condition number): ", max(condition_numbers))
    plt.hist(condition_numbers, bins=20, color="skyblue", edgecolor="black")
    plt.yscale("log")
    plt.title("Distribution of Condition Number")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig("cond_number.png")


def main() -> None:
    batches = [1, 2, 4, 8, 10, 16, 32, 64, 128, 256]
    sizes = [2, 4, 8, 16, 32, 48, 64, 80, 96, 128, 256, 512]
    modes = ["SYEVD", "SYEVJ", "SYEVD_BATCHED", "SYEVJ_BATCHED", "CUDA", "ROCM"]
    dtypes = [torch.float32, torch.float64]
    # dtypes = [torch.float64]
    print(f"dtype         | batch | shape | {' | '.join(f'{m}' for m in modes)}")
    for dtype in dtypes:
        for n, x in product(batches, sizes):
            l_list, a_list = prepare_input(x, n, dtype=dtype)
            print(f"{dtype} | {n:>5} | {x:>5}", end="")
            for mode in modes:
                os.environ["PYTORCH_ROCM_EIGEN_MODE"] = mode
                l_list_res, _ = torch.linalg.eigh(a_list)
                err = None
                try:
                    torch.testing.assert_close(l_list, l_list_res, rtol=1e-4, atol=1e-4)
                except Exception as e:
                    err = (l_list-l_list_res).abs().max()
        
                res1 = triton.testing.do_bench(lambda: torch.linalg.eigh(a_list))

                # def run_seq(a_list):
                #     for a in a_list:
                #         torch.linalg.eigh(a)

                # res2 = triton.testing.do_bench(lambda: run_seq(a_list))

                print(f" | {(1.0 if not err else -1.0)*res1:.3f}", end="")
            print()
    # for size in [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
    #    run(size, torch.float32)
    # condition_number(1024, torch.float32)
    # benchmark(1024, torch.float32)


if __name__ == "__main__":
    main()
    # C.rocsolver_log_end()
