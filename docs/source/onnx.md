# torch.onnx


## Overview

[Open Neural Network eXchange (ONNX)](https://onnx.ai/) is an open standard
format for representing machine learning models. The `torch.onnx` module captures the computation graph from a
native PyTorch {class}`torch.nn.Module` model and converts it into an
[ONNX graph](https://github.com/onnx/onnx/blob/main/docs/IR.md).

The exported model can be consumed by any of the many
[runtimes that support ONNX](https://onnx.ai/supported-tools.html#deployModel), including
Microsoft's [ONNX Runtime](https://www.onnxruntime.ai).

<<<<<<< HEAD
=======
**There are two flavors of ONNX exporter API that you can use, as listed below.**
Both can be called through function {func}`torch.onnx.export`.
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
Next example shows how to export a simple model.

```python
import torch

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 128, 5)

    def forward(self, x):
        return torch.relu(self.conv1(x))

input_tensor = torch.rand((1, 1, 128, 128), dtype=torch.float32)

model = MyModel()

torch.onnx.export(
    model,                  # model to export
    (input_tensor,),        # inputs of the model,
    "my_model.onnx",        # filename of the ONNX model
    input_names=["input"],  # Rename inputs for the ONNX model
    dynamo=True             # True or False to select the exporter to use
)
```

<<<<<<< HEAD

## torch.export-based ONNX Exporter

*The torch.export-based ONNX exporter is the newest exporter for PyTorch 2.6 and newer*

{ref}`torch.export <torch.export>` engine is leveraged to produce a traced graph representing only the Tensor computation of the function in an
Ahead-of-Time (AOT) fashion. The resulting traced graph (1) produces normalized operators in the functional
ATen operator set (as well as any user-specified custom operators), (2) has eliminated all Python control
flow and data structures (with certain exceptions), and (3) records the set of shape constraints needed to
show that this normalization and control-flow elimination is sound for future inputs, before it is finally
translated into an ONNX graph.

{doc}`Learn more about the torch.export-based ONNX Exporter <onnx_export>`

## Frequently Asked Questions

Q: I have exported my LLM model, but its input size seems to be fixed?

  The tracer records the shapes of the example inputs. If the model should accept
  inputs of dynamic shapes, set ``dynamic_shapes`` when calling {func}`torch.onnx.export`.

Q: How to export models containing loops?

  See {ref}`torch.cond <cond>`.

=======
Next sections introduce the two versions of the exporter.

## TorchDynamo-based ONNX Exporter

*The TorchDynamo-based ONNX exporter is the newest (and Beta) exporter for PyTorch 2.1 and newer*

TorchDynamo engine is leveraged to hook into Python's frame evaluation API and dynamically rewrite its
bytecode into an FX Graph. The resulting FX Graph is then polished before it is finally translated into an
ONNX graph.

The main advantage of this approach is that the [FX graph](https://pytorch.org/docs/stable/fx.html) is captured using
bytecode analysis that preserves the dynamic nature of the model instead of using traditional static tracing techniques.

{doc}`Learn more about the TorchDynamo-based ONNX Exporter <onnx_dynamo>`

## TorchScript-based ONNX Exporter

*The TorchScript-based ONNX exporter is available since PyTorch 1.2.0*

[TorchScript](https://pytorch.org/docs/stable/jit.html) is leveraged to trace (through {func}`torch.jit.trace`)
the model and capture a static computation graph.

As a consequence, the resulting graph has a couple limitations:

* It does not record any control-flow, like if-statements or loops;
* Does not handle nuances between `training` and `eval` mode;
* Does not truly handle dynamic inputs

As an attempt to support the static tracing limitations, the exporter also supports TorchScript scripting
(through {func}`torch.jit.script`), which adds support for data-dependent control-flow, for example. However, TorchScript
itself is a subset of the Python language, so not all features in Python are supported, such as in-place operations.

{doc}`Learn more about the TorchScript-based ONNX Exporter <onnx_torchscript>`
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

## Contributing / Developing

The ONNX exporter is a community project and we welcome contributions. We follow the
[PyTorch guidelines for contributions](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md), but you might
also be interested in reading our [development wiki](https://github.com/pytorch/pytorch/wiki/PyTorch-ONNX-exporter).

<<<<<<< HEAD

## torch.onnx APIs

```{eval-rst}
.. automodule:: torch.onnx
```

### Functions

```{eval-rst}
.. autofunction:: export
    :noindex:
.. autofunction:: is_in_onnx_export
    :noindex:
```

### Classes

```{eval-rst}
.. autoclass:: ONNXProgram
    :noindex:
.. autoclass:: OnnxExporterError
    :noindex:
```

=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
```{eval-rst}
.. toctree::
    :hidden:

<<<<<<< HEAD
    onnx_export
    onnx_ops
    onnx_verification
    onnx_testing
```

### Deprecated APIs

```{eval-rst}
.. deprecated:: 2.6
    These functions are deprecated and will be removed in a future version.

.. autofunction:: register_custom_op_symbolic
.. autofunction:: unregister_custom_op_symbolic
.. autofunction:: select_model_mode_for_export
```

```{eval-rst}
.. py:module:: torch.onnx.errors
.. py:module:: torch.onnx.operators
=======
    onnx_dynamo
    onnx_ops
    onnx_verification
    onnx_dynamo_onnxruntime_backend
    onnx_torchscript
```

<!-- This module needs to be documented. Adding here in the meantime
for tracking purposes -->
```{eval-rst}
.. py:module:: torch.onnx.errors
.. py:module:: torch.onnx.operators
.. py:module:: torch.onnx.symbolic_caffe2
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
.. py:module:: torch.onnx.symbolic_helper
.. py:module:: torch.onnx.symbolic_opset10
.. py:module:: torch.onnx.symbolic_opset11
.. py:module:: torch.onnx.symbolic_opset12
.. py:module:: torch.onnx.symbolic_opset13
.. py:module:: torch.onnx.symbolic_opset14
.. py:module:: torch.onnx.symbolic_opset15
.. py:module:: torch.onnx.symbolic_opset16
.. py:module:: torch.onnx.symbolic_opset17
.. py:module:: torch.onnx.symbolic_opset18
.. py:module:: torch.onnx.symbolic_opset19
.. py:module:: torch.onnx.symbolic_opset20
.. py:module:: torch.onnx.symbolic_opset7
.. py:module:: torch.onnx.symbolic_opset8
.. py:module:: torch.onnx.symbolic_opset9
.. py:module:: torch.onnx.utils
```
