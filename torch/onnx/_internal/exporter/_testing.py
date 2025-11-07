"""Test utilities for ONNX export."""

from __future__ import annotations


__all__ = ["assert_onnx_program"]

<<<<<<< HEAD
from typing import Any, Literal, TYPE_CHECKING
=======
from typing import Any, TYPE_CHECKING
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

import torch
from torch.utils import _pytree


if TYPE_CHECKING:
    from torch.onnx._internal.exporter import _onnx_program


def assert_onnx_program(
    program: _onnx_program.ONNXProgram,
    *,
    rtol: float | None = None,
    atol: float | None = None,
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
    strategy: str | None = "TorchExportNonStrictStrategy",
<<<<<<< HEAD
    backend: Literal["onnxruntime", "reference"] = "onnxruntime",
=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
) -> None:
    """Assert that the ONNX model produces the same output as the PyTorch ExportedProgram.

    Args:
        program: The ``ONNXProgram`` to verify.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        args: The positional arguments to pass to the program.
            If None, the default example inputs in the ExportedProgram will be used.
        kwargs: The keyword arguments to pass to the program.
            If None, the default example inputs in the ExportedProgram will be used.
        strategy: Assert the capture strategy used to export the program. Values can be
            class names like "TorchExportNonStrictStrategy".
            If None, the strategy is not asserted.
<<<<<<< HEAD
        backend: The backend to use for evaluating the ONNX program.
            Supported values are "onnxruntime" and "reference".
=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    """
    if strategy is not None:
        if program._capture_strategy != strategy:
            raise ValueError(
                f"Expected strategy '{strategy}' is used to capture the exported program, "
                f"but got '{program._capture_strategy}'."
            )
    exported_program = program.exported_program
    if exported_program is None:
        raise ValueError(
            "The ONNXProgram does not contain an ExportedProgram. "
            "To verify the ONNX program, initialize ONNXProgram with an ExportedProgram, "
            "or assign the ExportedProgram to the ONNXProgram.exported_program attribute."
        )
    if args is None and kwargs is None:
        # User did not provide example inputs, use the default example inputs
        if exported_program.example_inputs is None:
            raise ValueError(
                "No example inputs provided and the exported_program does not contain example inputs. "
                "Please provide arguments to verify the ONNX program."
            )
        args, kwargs = exported_program.example_inputs
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    torch_module = exported_program.module()
    torch_outputs, _ = _pytree.tree_flatten(torch_module(*args, **kwargs))
    # ONNX outputs are always real, so we need to convert torch complex outputs to real representations
    torch_outputs_adapted = []
    for output in torch_outputs:
<<<<<<< HEAD
        # ONNX graph does not support None outputs, so we skip them
        if output is None:
            continue
=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        if not isinstance(output, torch.Tensor):
            torch_outputs_adapted.append(torch.tensor(output))
        elif torch.is_complex(output):
            torch_outputs_adapted.append(torch.view_as_real(output))
        else:
            torch_outputs_adapted.append(output)
<<<<<<< HEAD

    # Obtain the ONNX outputs using the specified backend
    if backend == "onnxruntime":
        onnx_outputs = program(*args, **kwargs)
    elif backend == "reference":
        onnx_outputs = program.call_reference(*args, **kwargs)
    else:
        raise ValueError(
            f"Unsupported backend '{backend}'. Supported backends are 'onnxruntime' and 'reference'."
        )

=======
    onnx_outputs = program(*args, **kwargs)
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    # TODO(justinchuby): Include output names in the error message
    torch.testing.assert_close(
        tuple(onnx_outputs),
        tuple(torch_outputs_adapted),
        rtol=rtol,
        atol=atol,
        equal_nan=True,
        check_device=False,
    )
