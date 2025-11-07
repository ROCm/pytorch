<<<<<<< HEAD
---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  execution_timeout: 30
  execution_show_tb: True
  merge_streams: True
---

=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
(torch.export)=

# torch.export

<<<<<<< HEAD
=======
:::{warning}
This feature is a prototype under active development and there WILL BE
BREAKING CHANGES in the future.
:::

>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
## Overview

{func}`torch.export.export` takes a {class}`torch.nn.Module` and produces a traced graph
representing only the Tensor computation of the function in an Ahead-of-Time
(AOT) fashion, which can subsequently be executed with different outputs or
serialized.

<<<<<<< HEAD
```{code-cell}
import torch
from torch.export import export, ExportedProgram
=======
```python
import torch
from torch.export import export
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

class Mod(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        a = torch.sin(x)
        b = torch.cos(y)
        return a + b

example_args = (torch.randn(10, 10), torch.randn(10, 10))

<<<<<<< HEAD
exported_program: ExportedProgram = export(Mod(), args=example_args)
print(exported_program)
```

=======
exported_program: torch.export.ExportedProgram = export(
    Mod(), args=example_args
)
print(exported_program)
```

```python
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, x: "f32[10, 10]", y: "f32[10, 10]"):
            # code: a = torch.sin(x)
            sin: "f32[10, 10]" = torch.ops.aten.sin.default(x)

            # code: b = torch.cos(y)
            cos: "f32[10, 10]" = torch.ops.aten.cos.default(y)

            # code: return a + b
            add: f32[10, 10] = torch.ops.aten.add.Tensor(sin, cos)
            return (add,)

    Graph signature:
        ExportGraphSignature(
            input_specs=[
                InputSpec(
                    kind=<InputKind.USER_INPUT: 1>,
                    arg=TensorArgument(name='x'),
                    target=None,
                    persistent=None
                ),
                InputSpec(
                    kind=<InputKind.USER_INPUT: 1>,
                    arg=TensorArgument(name='y'),
                    target=None,
                    persistent=None
                )
            ],
            output_specs=[
                OutputSpec(
                    kind=<OutputKind.USER_OUTPUT: 1>,
                    arg=TensorArgument(name='add'),
                    target=None
                )
            ]
        )
    Range constraints: {}
```

>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
`torch.export` produces a clean intermediate representation (IR) with the
following invariants. More specifications about the IR can be found
{ref}`here <export.ir_spec>`.

- **Soundness**: It is guaranteed to be a sound representation of the original
  program, and maintains the same calling conventions of the original program.
- **Normalized**: There are no Python semantics within the graph. Submodules
  from the original programs are inlined to form one fully flattened
  computational graph.
- **Graph properties**: The graph is purely functional, meaning it does not
  contain operations with side effects such as mutations or aliasing. It does
  not mutate any intermediate values, parameters, or buffers.
- **Metadata**: The graph contains metadata captured during tracing, such as a
  stacktrace from user's code.

Under the hood, `torch.export` leverages the following latest technologies:

- **TorchDynamo (torch._dynamo)** is an internal API that uses a CPython feature
  called the Frame Evaluation API to safely trace PyTorch graphs. This
  provides a massively improved graph capturing experience, with much fewer
  rewrites needed in order to fully trace the PyTorch code.
- **AOT Autograd** provides a functionalized PyTorch graph and ensures the graph
  is decomposed/lowered to the ATen operator set.
- **Torch FX (torch.fx)** is the underlying representation of the graph,
  allowing flexible Python-based transformations.

### Existing frameworks

{func}`torch.compile` also utilizes the same PT2 stack as `torch.export`, but
is slightly different:

- **JIT vs. AOT**: {func}`torch.compile` is a JIT compiler whereas
  which is not intended to be used to produce compiled artifacts outside of
  deployment.
- **Partial vs. Full Graph Capture**: When {func}`torch.compile` runs into an
  untraceable part of a model, it will "graph break" and fall back to running
  the program in the eager Python runtime. In comparison, `torch.export` aims
  to get a full graph representation of a PyTorch model, so it will error out
  when something untraceable is reached. Since `torch.export` produces a full
  graph disjoint from any Python features or runtime, this graph can then be
  saved, loaded, and run in different environments and languages.
- **Usability tradeoff**: Since {func}`torch.compile` is able to fallback to the
  Python runtime whenever it reaches something untraceable, it is a lot more
  flexible. `torch.export` will instead require users to provide more
  information or rewrite their code to make it traceable.

Compared to {func}`torch.fx.symbolic_trace`, `torch.export` traces using
TorchDynamo which operates at the Python bytecode level, giving it the ability
to trace arbitrary Python constructs not limited by what Python operator
overloading supports. Additionally, `torch.export` keeps fine-grained track of
tensor metadata, so that conditionals on things like tensor shapes do not
fail tracing. In general, `torch.export` is expected to work on more user
programs, and produce lower-level graphs (at the `torch.ops.aten` operator
level). Note that users can still use {func}`torch.fx.symbolic_trace` as a
preprocessing step before `torch.export`.

Compared to {func}`torch.jit.script`, `torch.export` does not capture Python
<<<<<<< HEAD
control flow or data structures, unless using explicit {ref}`control flow operators <cond>`,
but it supports more Python language features due to its comprehensive coverage
over Python bytecodes. The resulting graphs are simpler and only have straight
line control flow, except for explicit control flow operators.

Compared to {func}`torch.jit.trace`, `torch.export` is sound:
it can trace code that performs integer computation on sizes and records
all of the side-conditions necessary to ensure that a particular
trace is valid for other inputs.

## Exporting a PyTorch Model

The main entrypoint is through {func}`torch.export.export`, which takes a
{class}`torch.nn.Module` and sample inputs, and
captures the computation graph into an {class}`torch.export.ExportedProgram`. An
example:

```{code-cell}
import torch
from torch.export import export, ExportedProgram
=======
control flow or data structures, but it supports more Python language features
than TorchScript (as it is easier to have comprehensive coverage over Python
bytecodes). The resulting graphs are simpler and only have straight line control
flow (except for explicit control flow operators).

Compared to {func}`torch.jit.trace`, `torch.export` is sound: it is able to
trace code that performs integer computation on sizes and records all of the
side-conditions necessary to show that a particular trace is valid for other
inputs.

## Exporting a PyTorch Model

### An Example

The main entrypoint is through {func}`torch.export.export`, which takes a
callable ({class}`torch.nn.Module`, function, or method) and sample inputs, and
captures the computation graph into an {class}`torch.export.ExportedProgram`. An
example:

```python
import torch
from torch.export import export
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

# Simple module for demonstration
class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, padding=1
        )
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3)

    def forward(self, x: torch.Tensor, *, constant=None) -> torch.Tensor:
        a = self.conv(x)
        a.add_(constant)
        return self.maxpool(self.relu(a))

example_args = (torch.randn(1, 3, 256, 256),)
example_kwargs = {"constant": torch.ones(1, 16, 256, 256)}

<<<<<<< HEAD
exported_program: ExportedProgram = export(
    M(), args=example_args, kwargs=example_kwargs
)
print(exported_program)

# To run the exported program, we can use the `module()` method
print(exported_program.module()(torch.randn(1, 3, 256, 256), constant=torch.ones(1, 16, 256, 256)))
=======
exported_program: torch.export.ExportedProgram = export(
    M(), args=example_args, kwargs=example_kwargs
)
print(exported_program)
```

```python
ExportedProgram:
    class GraphModule(torch.nn.Module):
    def forward(self, p_conv_weight: "f32[16, 3, 3, 3]", p_conv_bias: "f32[16]", x: "f32[1, 3, 256, 256]", constant: "f32[1, 16, 256, 256]"):
            # code: a = self.conv(x)
            conv2d: "f32[1, 16, 256, 256]" = torch.ops.aten.conv2d.default(x, p_conv_weight, p_conv_bias, [1, 1], [1, 1])

            # code: a.add_(constant)
            add_: "f32[1, 16, 256, 256]" = torch.ops.aten.add_.Tensor(conv2d, constant)

            # code: return self.maxpool(self.relu(a))
            relu: "f32[1, 16, 256, 256]" = torch.ops.aten.relu.default(add_)
            max_pool2d: "f32[1, 16, 85, 85]" = torch.ops.aten.max_pool2d.default(relu, [3, 3], [3, 3])
            return (max_pool2d,)

Graph signature:
    ExportGraphSignature(
        input_specs=[
            InputSpec(
                kind=<InputKind.PARAMETER: 2>,
                arg=TensorArgument(name='p_conv_weight'),
                target='conv.weight',
                persistent=None
            ),
            InputSpec(
                kind=<InputKind.PARAMETER: 2>,
                arg=TensorArgument(name='p_conv_bias'),
                target='conv.bias',
                persistent=None
            ),
            InputSpec(
                kind=<InputKind.USER_INPUT: 1>,
                arg=TensorArgument(name='x'),
                target=None,
                persistent=None
            ),
            InputSpec(
                kind=<InputKind.USER_INPUT: 1>,
                arg=TensorArgument(name='constant'),
                target=None,
                persistent=None
            )
        ],
        output_specs=[
            OutputSpec(
                kind=<OutputKind.USER_OUTPUT: 1>,
                arg=TensorArgument(name='max_pool2d'),
                target=None
            )
        ]
    )
Range constraints: {}
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
```

Inspecting the `ExportedProgram`, we can note the following:

- The {class}`torch.fx.Graph` contains the computation graph of the original
  program, along with records of the original code for easy debugging.
- The graph contains only `torch.ops.aten` operators found [here](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml)
<<<<<<< HEAD
  and custom operators.
=======
  and custom operators, and is fully functional, without any inplace operators
  such as `torch.add_`.
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
- The parameters (weight and bias to conv) are lifted as inputs to the graph,
  resulting in no `get_attr` nodes in the graph, which previously existed in
  the result of {func}`torch.fx.symbolic_trace`.
- The {class}`torch.export.ExportGraphSignature` models the input and output
  signature, along with specifying which inputs are parameters.
- The resulting shape and dtype of tensors produced by each node in the graph is
<<<<<<< HEAD
  noted. For example, the `conv2d` node will result in a tensor of dtype
  `torch.float32` and shape (1, 16, 256, 256).

## Expressing Dynamism

By default `torch.export` will trace the program assuming all input shapes are
**static**, and specializing the exported program to those dimensions. One
consequence of this is that at runtime, the program won’t work on inputs with
different shapes, even if they’re valid in eager mode.

An example:

```{code-cell}
import torch
import traceback as tb

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.branch1 = torch.nn.Sequential(
            torch.nn.Linear(64, 32), torch.nn.ReLU()
        )
        self.branch2 = torch.nn.Sequential(
            torch.nn.Linear(128, 64), torch.nn.ReLU()
        )
        self.buffer = torch.ones(32)

    def forward(self, x1, x2):
        out1 = self.branch1(x1)
        out2 = self.branch2(x2)
        return (out1 + self.buffer, out2)

example_args = (torch.randn(32, 64), torch.randn(32, 128))

ep = torch.export.export(M(), example_args)
print(ep)

example_args2 = (torch.randn(64, 64), torch.randn(64, 128))
try:
    ep.module()(*example_args2)  # fails
except Exception:
    tb.print_exc()
```


However, some dimensions, such as a batch dimension, can be dynamic and vary
from run to run. Such dimensions must be specified by using the
{func}`torch.export.Dim()` API to create them and by passing them into
{func}`torch.export.export()` through the `dynamic_shapes` argument.

```{code-cell}
import torch
=======
  noted. For example, the `convolution` node will result in a tensor of dtype
  `torch.float32` and shape (1, 16, 256, 256).

(non-strict-export)=

### Non-Strict Export

In PyTorch 2.3, we introduced a new mode of tracing called **non-strict mode**.
It's still going through hardening, so if you run into any issues, please file
them to Github with the "oncall: export" tag.

In *non-strict mode*, we trace through the program using the Python interpreter.
Your code will execute exactly as it would in eager mode; the only difference is
that all Tensor objects will be replaced by ProxyTensors, which will record all
their operations into a graph.

In *strict* mode, which is currently the default, we first trace through the
program using TorchDynamo, a bytecode analysis engine. TorchDynamo does not
actually execute your Python code. Instead, it symbolically analyzes it and
builds a graph based on the results. This analysis allows torch.export to
provide stronger guarantees about safety, but not all Python code is supported.

An example of a case where one might want to use non-strict mode is if you run
into a unsupported TorchDynamo feature that might not be easily solved, and you
know the python code is not exactly needed for computation. For example:

```python
import contextlib
import torch

class ContextManager():
    def __init__(self):
        self.count = 0
    def __enter__(self):
        self.count += 1
    def __exit__(self, exc_type, exc_value, traceback):
        self.count -= 1

class M(torch.nn.Module):
    def forward(self, x):
        with ContextManager():
            return x.sin() + x.cos()

export(M(), (torch.ones(3, 3),), strict=False)  # Non-strict traces successfully
export(M(), (torch.ones(3, 3),))  # Strict mode fails with torch._dynamo.exc.Unsupported: ContextManager
```

In this example, the first call using non-strict mode (through the
`strict=False` flag) traces successfully whereas the second call using strict
mode (default) results with a failure, where TorchDynamo is unable to support
context managers. One option is to rewrite the code (see {ref}`Limitations of torch.export <limitations-of-torch-export>`),
but seeing as the context manager does not affect the tensor
computations in the model, we can go with the non-strict mode's result.

(training-export)=

### Export for Training and Inference

In PyTorch 2.5, we introduced a new API called {func}`export_for_training`.
It's still going through hardening, so if you run into any issues, please file
them to Github with the "oncall: export" tag.

In this API, we produce the most generic IR that contains all ATen operators
(including both functional and non-functional) which can be used to train in
eager PyTorch Autograd. This API is intended for eager training use cases such as PT2 Quantization
and will soon be the default IR of torch.export.export. To read further about
the motivation behind this change, please refer to
<https://dev-discuss.pytorch.org/t/why-pytorch-does-not-need-a-new-standardized-operator-set/2206>

When this API is combined with {func}`run_decompositions()`, you should be able to get inference IR with
any desired decomposition behavior.

To show some examples:

```python
class ConvBatchnorm(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 1, 1)
        self.bn = torch.nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return (x,)

mod = ConvBatchnorm()
inp = torch.randn(1, 1, 3, 3)

ep_for_training = torch.export.export_for_training(mod, (inp,))
print(ep_for_training)
```

```python
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, p_conv_weight: "f32[3, 1, 1, 1]", p_conv_bias: "f32[3]", p_bn_weight: "f32[3]", p_bn_bias: "f32[3]", b_bn_running_mean: "f32[3]", b_bn_running_var: "f32[3]", b_bn_num_batches_tracked: "i64[]", x: "f32[1, 1, 3, 3]"):
            conv2d: "f32[1, 3, 3, 3]" = torch.ops.aten.conv2d.default(x, p_conv_weight, p_conv_bias)
            add_: "i64[]" = torch.ops.aten.add_.Tensor(b_bn_num_batches_tracked, 1)
            batch_norm: "f32[1, 3, 3, 3]" = torch.ops.aten.batch_norm.default(conv2d, p_bn_weight, p_bn_bias, b_bn_running_mean, b_bn_running_var, True, 0.1, 1e-05, True)
            return (batch_norm,)
```

From the above output, you can see that {func}`export_for_training` produces pretty much the same ExportedProgram
as {func}`export` except for the operators in the graph. You can see that we captured batch_norm in the most general
form. This op is non-functional and will be lowered to different ops when running inference.

You can also go from this IR to an inference IR via {func}`run_decompositions` with arbitrary customizations.

```python
# Lower to core aten inference IR, but keep conv2d
decomp_table = torch.export.default_decompositions()
del decomp_table[torch.ops.aten.conv2d.default]
ep_for_inference = ep_for_training.run_decompositions(decomp_table)

print(ep_for_inference)
```

```python
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, p_conv_weight: "f32[3, 1, 1, 1]", p_conv_bias: "f32[3]", p_bn_weight: "f32[3]", p_bn_bias: "f32[3]", b_bn_running_mean: "f32[3]", b_bn_running_var: "f32[3]", b_bn_num_batches_tracked: "i64[]", x: "f32[1, 1, 3, 3]"):
            conv2d: "f32[1, 3, 3, 3]" = torch.ops.aten.conv2d.default(x, p_conv_weight, p_conv_bias)
            add: "i64[]" = torch.ops.aten.add.Tensor(b_bn_num_batches_tracked, 1)
            _native_batch_norm_legit_functional = torch.ops.aten._native_batch_norm_legit_functional.default(conv2d, p_bn_weight, p_bn_bias, b_bn_running_mean, b_bn_running_var, True, 0.1, 1e-05)
            getitem: "f32[1, 3, 3, 3]" = _native_batch_norm_legit_functional[0]
            getitem_3: "f32[3]" = _native_batch_norm_legit_functional[3]
            getitem_4: "f32[3]" = _native_batch_norm_legit_functional[4]
            return (getitem_3, getitem_4, add, getitem)
```

Here you can see that we kept `conv2d` op in the IR while decomposing the rest. Now the IR is a functional IR
containing core aten operators except for `conv2d`.

You can do even more customization by directly registering your chosen decomposition behaviors.

You can do even more customizations by directly registering custom decomp behaviour

```python
# Lower to core aten inference IR, but customize conv2d
decomp_table = torch.export.default_decompositions()

def my_awesome_custom_conv2d_function(x, weight, bias, stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1):
    return 2 * torch.ops.aten.convolution(x, weight, bias, stride, padding, dilation, False, [0, 0], groups)

decomp_table[torch.ops.aten.conv2d.default] = my_awesome_conv2d_function
ep_for_inference = ep_for_training.run_decompositions(decomp_table)

print(ep_for_inference)
```

```python
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, p_conv_weight: "f32[3, 1, 1, 1]", p_conv_bias: "f32[3]", p_bn_weight: "f32[3]", p_bn_bias: "f32[3]", b_bn_running_mean: "f32[3]", b_bn_running_var: "f32[3]", b_bn_num_batches_tracked: "i64[]", x: "f32[1, 1, 3, 3]"):
            convolution: "f32[1, 3, 3, 3]" = torch.ops.aten.convolution.default(x, p_conv_weight, p_conv_bias, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
            mul: "f32[1, 3, 3, 3]" = torch.ops.aten.mul.Tensor(convolution, 2)
            add: "i64[]" = torch.ops.aten.add.Tensor(b_bn_num_batches_tracked, 1)
            _native_batch_norm_legit_functional = torch.ops.aten._native_batch_norm_legit_functional.default(mul, p_bn_weight, p_bn_bias, b_bn_running_mean, b_bn_running_var, True, 0.1, 1e-05)
            getitem: "f32[1, 3, 3, 3]" = _native_batch_norm_legit_functional[0]
            getitem_3: "f32[3]" = _native_batch_norm_legit_functional[3]
            getitem_4: "f32[3]" = _native_batch_norm_legit_functional[4];
            return (getitem_3, getitem_4, add, getitem)
```

### Expressing Dynamism

By default `torch.export` will trace the program assuming all input shapes are
**static**, and specializing the exported program to those dimensions. However,
some dimensions, such as a batch dimension, can be dynamic and vary from run to
run. Such dimensions must be specified by using the
{func}`torch.export.Dim` API to create them and by passing them into
{func}`torch.export.export` through the `dynamic_shapes` argument. An example:

```python
import torch
from torch.export import Dim, export
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.branch1 = torch.nn.Sequential(
            torch.nn.Linear(64, 32), torch.nn.ReLU()
        )
        self.branch2 = torch.nn.Sequential(
            torch.nn.Linear(128, 64), torch.nn.ReLU()
        )
        self.buffer = torch.ones(32)

    def forward(self, x1, x2):
        out1 = self.branch1(x1)
        out2 = self.branch2(x2)
        return (out1 + self.buffer, out2)

example_args = (torch.randn(32, 64), torch.randn(32, 128))

# Create a dynamic batch size
<<<<<<< HEAD
batch = torch.export.Dim("batch")
# Specify that the first dimension of each input is that batch size
dynamic_shapes = {"x1": {0: batch}, "x2": {0: batch}}

ep = torch.export.export(
    M(), args=example_args, dynamic_shapes=dynamic_shapes
)
print(ep)

example_args2 = (torch.randn(64, 64), torch.randn(64, 128))
ep.module()(*example_args2)  # success
=======
batch = Dim("batch")
# Specify that the first dimension of each input is that batch size
dynamic_shapes = {"x1": {0: batch}, "x2": {0: batch}}

exported_program: torch.export.ExportedProgram = export(
    M(), args=example_args, dynamic_shapes=dynamic_shapes
)
print(exported_program)
```

```python
ExportedProgram:
class GraphModule(torch.nn.Module):
    def forward(self, p_branch1_0_weight: "f32[32, 64]", p_branch1_0_bias: "f32[32]", p_branch2_0_weight: "f32[64, 128]", p_branch2_0_bias: "f32[64]", c_buffer: "f32[32]", x1: "f32[s0, 64]", x2: "f32[s0, 128]"):

         # code: out1 = self.branch1(x1)
        linear: "f32[s0, 32]" = torch.ops.aten.linear.default(x1, p_branch1_0_weight, p_branch1_0_bias)
        relu: "f32[s0, 32]" = torch.ops.aten.relu.default(linear)

         # code: out2 = self.branch2(x2)
        linear_1: "f32[s0, 64]" = torch.ops.aten.linear.default(x2, p_branch2_0_weight, p_branch2_0_bias)
        relu_1: "f32[s0, 64]" = torch.ops.aten.relu.default(linear_1)

         # code: return (out1 + self.buffer, out2)
        add: "f32[s0, 32]" = torch.ops.aten.add.Tensor(relu, c_buffer)
        return (add, relu_1)

Range constraints: {s0: VR[0, int_oo]}
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
```

Some additional things to note:

- Through the {func}`torch.export.Dim` API and the `dynamic_shapes` argument, we specified the first
  dimension of each input to be dynamic. Looking at the inputs `x1` and
<<<<<<< HEAD
  `x2`, they have a symbolic shape of `(s0, 64)` and `(s0, 128)`, instead of
  the `(32, 64)` and `(32, 128)` shaped tensors that we passed in as example inputs.
=======
  `x2`, they have a symbolic shape of (s0, 64) and (s0, 128), instead of
  the (32, 64) and (32, 128) shaped tensors that we passed in as example inputs.
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  `s0` is a symbol representing that this dimension can be a range
  of values.
- `exported_program.range_constraints` describes the ranges of each symbol
  appearing in the graph. In this case, we see that `s0` has the range
  [0, int_oo]. For technical reasons that are difficult to explain here, they are
  assumed to be not 0 or 1. This is not a bug, and does not necessarily mean
  that the exported program will not work for dimensions 0 or 1. See
  [The 0/1 Specialization Problem](https://docs.google.com/document/d/16VPOa3d-Liikf48teAOmxLc92rgvJdfosIy-yoT38Io/edit?fbclid=IwAR3HNwmmexcitV0pbZm_x1a4ykdXZ9th_eJWK-3hBtVgKnrkmemz6Pm5jRQ#heading=h.ez923tomjvyk)
  for an in-depth discussion of this topic.

<<<<<<< HEAD

In the example, we used `Dim("batch")` to create a dynamic dimension. This is
the most explicit way to specify dynamism. We can also use `Dim.DYNAMIC` and
`Dim.AUTO` to specify dynamism. We will go over both methods in the next section.

### Named Dims

For every dimension specified with `Dim("name")`, we will allocate a symbolic
shape. Specifying a `Dim` with the same name will result in the same symbol
to be generated. This allows users to specify what symbols are allocated for
each input dimension.

```python
batch = Dim("batch")
dynamic_shapes = {"x1": {0: dim}, "x2": {0: batch}}
```

For each `Dim`, we can specify minimum and maximum values. We also allow
specifying relations between `Dim`s in univariate linear expressions: `A * dim + B`.
This allows users to specify more complex constraints like integer divisibility
for dynamic dimensions. These features allow for users to place explicit
restrictions on the dynamic behavior of the `ExportedProgram` produced.

```python
dx = Dim("dx", min=4, max=256)
dh = Dim("dh", max=512)
dynamic_shapes = {
    "x": (dx, None),
    "y": (2 * dx, dh),
}
```

However, `ConstraintViolationErrors` will be raised if the while tracing, we emit guards
that conflict with the relations or static/dynamic specifications given. For
example, in the above specification, the following is asserted:

* `x.shape[0]` is to have range `[4, 256]`, and related to `y.shape[0]` by `y.shape[0] == 2 * x.shape[0]`.
* `x.shape[1]` is static.
* `y.shape[1]` has range `[0, 512]`, and is unrelated to any other dimension.

If any of these assertions are found to be incorrect while tracing (ex.
`x.shape[0]` is static, or `y.shape[1]` has a smaller range, or
`y.shape[0] != 2 * x.shape[0]`), then a `ConstraintViolationError` will be
raised, and the user will need to change their `dynamic_shapes` specification.

### Dim Hints

Instead of explicitly specifying dynamism using `Dim("name")`, we can let
`torch.export` infer the ranges and relationships of the dynamic values using
`Dim.DYNAMIC`. This is also a more convenient way to specify dynamism when you
don't know specifically *how* dynamic your dynamic values are.

```python
dynamic_shapes = {
    "x": (Dim.DYNAMIC, None),
    "y": (Dim.DYNAMIC, Dim.DYNAMIC),
}
```

We can also specify min/max values for `Dim.DYNAMIC`, which will serve as hints
to export. But if while tracing export found the range to be different, it will
automatically update the range without raising an error. We also cannot specify
relationships between dynamic values. Instead, this will be inferred by export,
and exposed to users through an inspection of assertions within the graph.  In
this method of specifying dynamism, `ConstraintViolationErrors` will **only** be
raised if the specified value is inferred to be **static**.

An even more convenient way to specify dynamism is to use `Dim.AUTO`, which will
behave like `Dim.DYNAMIC`, but will **not** raise an error if the dimension is
inferred to be static. This is useful for when you have no idea what the dynamic
values are, and want to export the program with a "best effort" dynamic approach.

### ShapesCollection

When specifying which inputs are dynamic via `dynamic_shapes`, we must specify
the dynamism of every input. For example, given the following inputs:

```python
args = {"x": tensor_x, "others": [tensor_y, tensor_z]}
```

we would need to specify the dynamism of `tensor_x`, `tensor_y`, and `tensor_z`
along with the dynamic shapes:

```python
# With named-Dims
dim = torch.export.Dim(...)
dynamic_shapes = {"x": {0: dim, 1: dim + 1}, "others": [{0: dim * 2}, None]}

torch.export(..., args, dynamic_shapes=dynamic_shapes)
```

However, this is particularly complicated as we need to specify the
`dynamic_shapes` specification in the same nested input structure as the input
arguments. Instead, an easier way to specify dynamic shapes is with the helper
utility {class}`torch.export.ShapesCollection`, where instead of specifying the
dynamism of every single input, we can just assign directly which input
dimensions are dynamic.

```{code-cell}
import torch

class M(torch.nn.Module):
    def forward(self, inp):
        x = inp["x"] * 1
        y = inp["others"][0] * 2
        z = inp["others"][1] * 3
        return x, y, z

tensor_x = torch.randn(3, 4, 8)
tensor_y = torch.randn(6)
tensor_z = torch.randn(6)
args = {"x": tensor_x, "others": [tensor_y, tensor_z]}

dim = torch.export.Dim("dim")
sc = torch.export.ShapesCollection()
sc[tensor_x] = (dim, dim + 1, 8)
sc[tensor_y] = {0: dim * 2}

print(sc.dynamic_shapes(M(), (args,)))
ep = torch.export.export(M(), (args,), dynamic_shapes=sc)
print(ep)
```

### AdditionalInputs

In the case where you don't know how dynamic your inputs are, but you have an
ample set of testing or profiling data that can provide a fair sense of
representative inputs for a model, you can use
{class}`torch.export.AdditionalInputs` in place of `dynamic_shapes`. You can
specify all the possible inputs used to trace the program, and
`AdditionalInputs` will infer which inputs are dynamic based on which input
shapes are changing.

Example:

```{code-cell}
import dataclasses
import torch
import torch.utils._pytree as pytree

@dataclasses.dataclass
class D:
    b: bool
    i: int
    f: float
    t: torch.Tensor

pytree.register_dataclass(D)

class M(torch.nn.Module):
    def forward(self, d: D):
        return d.i + d.f + d.t

input1 = (D(True, 3, 3.0, torch.ones(3)),)
input2 = (D(True, 4, 3.0, torch.ones(4)),)
ai = torch.export.AdditionalInputs()
ai.add(input1)
ai.add(input2)

print(ai.dynamic_shapes(M(), input1))
ep = torch.export.export(M(), input1, dynamic_shapes=ai)
print(ep)
```

## Serialization

To save the `ExportedProgram`, users can use the {func}`torch.export.save` and
{func}`torch.export.load` APIs. The resulting file is a zipfile with a specific
structure. The details of the structure are defined in the
{ref}`PT2 Archive Spec <export.pt2_archive>`.
=======
We can also specify more expressive relationships between input shapes, such as
where a pair of shapes might differ by one, a shape might be double of
another, or a shape is even. An example:

```python
class M(torch.nn.Module):
    def forward(self, x, y):
        return x + y[1:]

x, y = torch.randn(5), torch.randn(6)
dimx = torch.export.Dim("dimx", min=3, max=6)
dimy = dimx + 1

exported_program = torch.export.export(
    M(), (x, y), dynamic_shapes=({0: dimx}, {0: dimy}),
)
print(exported_program)
```

```python
ExportedProgram:
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[s0]", y: "f32[s0 + 1]"):
        # code: return x + y[1:]
        slice_1: "f32[s0]" = torch.ops.aten.slice.Tensor(y, 0, 1, 9223372036854775807)
        add: "f32[s0]" = torch.ops.aten.add.Tensor(x, slice_1)
        return (add,)

Range constraints: {s0: VR[3, 6], s0 + 1: VR[4, 7]}
```

Some things to note:

- By specifying `{0: dimx}` for the first input, we see that the resulting
  shape of the first input is now dynamic, being `[s0]`. And now by specifying
  `{0: dimy}` for the second input, we see that the resulting shape of the
  second input is also dynamic. However, because we expressed `dimy = dimx + 1`,
  instead of `y`'s shape containing a new symbol, we see that it is
  now being represented with the same symbol used in `x`, `s0`. We can
  see that relationship of `dimy = dimx + 1` is being shown through `s0 + 1`.
- Looking at the range constraints, we see that `s0` has the range [3, 6],
  which is specified initially, and we can see that `s0 + 1` has the solved
  range of [4, 7].

### Serialization

To save the `ExportedProgram`, users can use the {func}`torch.export.save` and
{func}`torch.export.load` APIs. A convention is to save the `ExportedProgram`
using a `.pt2` file extension.
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

An example:

```python
import torch
<<<<<<< HEAD
=======
import io
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

class MyModule(torch.nn.Module):
    def forward(self, x):
        return x + 10

<<<<<<< HEAD
exported_program = torch.export.export(MyModule(), (torch.randn(5),))
=======
exported_program = torch.export.export(MyModule(), torch.randn(5))
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

torch.export.save(exported_program, 'exported_program.pt2')
saved_exported_program = torch.export.load('exported_program.pt2')
```

<<<<<<< HEAD
(training-export)=

## Export IR, Decompositions

The graph produced by `torch.export` returns a graph containing only
[ATen operators](https://pytorch.org/cppdocs/#aten), which are the basic unit of
computation in PyTorch. As there are over
3000 ATen operators, export provides a way to narrow down the operator set used
in the graph based on certain characteristics, creating different IRs.

By default, export produces the most generic IR which contains all ATen
operators, including both functional and non-functional operators. A functional
operator is one that does not contain any mutations or aliasing of the inputs.
You can find a list of all ATen operators
[here](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml)
and you can inspect if an operator is functional by checking
`op._schema.is_mutable`.

This generic IR can be used to train in eager PyTorch Autograd.

```{code-cell}
import torch

class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 1, 1)
        self.bn = torch.nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return (x,)

ep_for_training = torch.export.export(M(), (torch.randn(1, 1, 3, 3),))
print(ep_for_training.graph_module.print_readable(print_output=False))
```

However, if you want to use the IR for inference, or decrease the amount of
operators being used, you can lower the graph through the
{func}`ExportedProgram.run_decompositions` API. This method decomposes the
ATen operators into the ones specified in the decomposition table, and
functionalizes the graph.

By specifying an empty set, we're only performing functionalization, and does
not do any additional decompositions. This results in an IR which contains ~2000
operators (instead of the 3000 operators above), and is ideal for inference cases.

```{code-cell}
import torch

class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 1, 1)
        self.bn = torch.nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return (x,)

ep_for_training = torch.export.export(M(), (torch.randn(1, 1, 3, 3),))
with torch.no_grad():
    ep_for_inference = ep_for_training.run_decompositions(decomp_table={})
print(ep_for_inference.graph_module.print_readable(print_output=False))
```

As we can see, the previously in-place operator,
`torch.ops.aten.add_.default` has now been replaced with
`torch.ops.aten.add.default`, a functional operator.

We can also further lower this exported program to an operator set which only
contains the
`Core ATen Operator Set <https://pytorch.org/docs/main/torch.compiler_ir.html#core-aten-ir>`__,
which is a collection of only ~180 operators. This IR is optimal for backends
who do not want to reimplement all ATen operators.

```{code-cell}
import torch

class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 1, 1)
        self.bn = torch.nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return (x,)

ep_for_training = torch.export.export(M(), (torch.randn(1, 1, 3, 3),))
with torch.no_grad():
    core_aten_ir = ep_for_training.run_decompositions(decomp_table=None)
print(core_aten_ir.graph_module.print_readable(print_output=False))
```

We now see that `torch.ops.aten.conv2d.default` has been decomposed
into `torch.ops.aten.convolution.default`. This is because `convolution`
is a more "core" operator, as operations like `conv1d` and `conv2d` can be
implemented using the same op.

We can also specify our own decomposition behaviors:

```{code-cell}
class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 1, 1)
        self.bn = torch.nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return (x,)

ep_for_training = torch.export.export(M(), (torch.randn(1, 1, 3, 3),))

my_decomp_table = torch.export.default_decompositions()

def my_awesome_custom_conv2d_function(x, weight, bias, stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1):
    return 2 * torch.ops.aten.convolution(x, weight, bias, stride, padding, dilation, False, [0, 0], groups)

my_decomp_table[torch.ops.aten.conv2d.default] = my_awesome_custom_conv2d_function
my_ep = ep_for_training.run_decompositions(my_decomp_table)
print(my_ep.graph_module.print_readable(print_output=False))
```

Notice that instead of `torch.ops.aten.conv2d.default` being decomposed
into `torch.ops.aten.convolution.default`, it is now decomposed into
`torch.ops.aten.convolution.default` and `torch.ops.aten.mul.Tensor`,
which matches our custom decomposition rule.
=======
### Specializations

A key concept in understanding the behavior of `torch.export` is the
difference between *static* and *dynamic* values.

A *dynamic* value is one that can change from run to run. These behave like
normal arguments to a Python function—you can pass different values for an
argument and expect your function to do the right thing. Tensor *data* is
treated as dynamic.

A *static* value is a value that is fixed at export time and cannot change
between executions of the exported program. When the value is encountered during
tracing, the exporter will treat it as a constant and hard-code it into the
graph.

When an operation is performed (e.g. `x + y`) and all inputs are static, then
the output of the operation will be directly hard-coded into the graph, and the
operation won’t show up (i.e. it will get constant-folded).

When a value has been hard-coded into the graph, we say that the graph has been
*specialized* to that value.

The following values are static:

#### Input Tensor Shapes

By default, `torch.export` will trace the program specializing on the input
tensors' shapes, unless a dimension is specified as dynamic via the
`dynamic_shapes` argument to `torch.export`. This means that if there exists
shape-dependent control flow, `torch.export` will specialize on the branch
that is being taken with the given sample inputs. For example:

```python
import torch
from torch.export import export

class Mod(torch.nn.Module):
    def forward(self, x):
        if x.shape[0] > 5:
            return x + 1
        else:
            return x - 1

example_inputs = (torch.rand(10, 2),)
exported_program = export(Mod(), example_inputs)
print(exported_program)
```

```python
ExportedProgram:
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[10, 2]"):
        # code: return x + 1
        add: "f32[10, 2]" = torch.ops.aten.add.Tensor(x, 1)
        return (add,)
```

The conditional of (`x.shape[0] > 5`) does not appear in the
`ExportedProgram` because the example inputs have the static
shape of (10, 2). Since `torch.export` specializes on the inputs' static
shapes, the else branch (`x - 1`) will never be reached. To preserve the dynamic
branching behavior based on the shape of a tensor in the traced graph,
{func}`torch.export.Dim` will need to be used to specify the dimension
of the input tensor (`x.shape[0]`) to be dynamic, and the source code will
need to be {ref}`rewritten <data-shape-dependent-control-flow>`.

Note that tensors that are part of the module state (e.g. parameters and
buffers) always have static shapes.

#### Python Primitives

`torch.export` also specializes on Python primitives,
such as `int`, `float`, `bool`, and `str`. However they do have dynamic
variants such as `SymInt`, `SymFloat`, and `SymBool`.

For example:

```python
import torch
from torch.export import export

class Mod(torch.nn.Module):
    def forward(self, x: torch.Tensor, const: int, times: int):
        for i in range(times):
            x = x + const
        return x

example_inputs = (torch.rand(2, 2), 1, 3)
exported_program = export(Mod(), example_inputs)
print(exported_program)
```

```python
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, x: "f32[2, 2]", const, times):
            # code: x = x + const
            add: "f32[2, 2]" = torch.ops.aten.add.Tensor(x, 1)
            add_1: "f32[2, 2]" = torch.ops.aten.add.Tensor(add, 1)
            add_2: "f32[2, 2]" = torch.ops.aten.add.Tensor(add_1, 1)
            return (add_2,)
```

Because integers are specialized, the `torch.ops.aten.add.Tensor` operations
are all computed with the hard-coded constant `1`, rather than `const`. If
a user passes a different value for `const` at runtime, like 2, than the one used
during export time, 1, this will result in an error.
Additionally, the `times` iterator used in the `for` loop is also "inlined"
in the graph through the 3 repeated `torch.ops.aten.add.Tensor` calls, and the
input `times` is never used.

#### Python Containers

Python containers (`List`, `Dict`, `NamedTuple`, etc.) are considered to
have static structure.
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

(limitations-of-torch-export)=

## Limitations of torch.export

<<<<<<< HEAD
=======
### Graph Breaks

>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
As `torch.export` is a one-shot process for capturing a computation graph from
a PyTorch program, it might ultimately run into untraceable parts of programs as
it is nearly impossible to support tracing all PyTorch and Python features. In
the case of `torch.compile`, an unsupported operation will cause a "graph
break" and the unsupported operation will be run with default Python evaluation.
In contrast, `torch.export` will require users to provide additional
<<<<<<< HEAD
information or rewrite parts of their code to make it traceable.

{ref}`Draft-export <export.draft_export>` is a great resource for listing out
graphs breaks that will be encountered when tracing the program, along with
additional debug information to solve those errors.

{ref}`ExportDB <torch.export_db>` is also great resource for learning about the
kinds of programs that are supported and unsupported, along with ways to rewrite
programs to make them traceable.

### TorchDynamo unsupported

When using `torch.export` with `strict=True`, this will use TorchDynamo to
evaluate the program at the Python bytecode level to trace the program into a
graph. Compared to previous tracing frameworks, there will be significantly
fewer rewrites required to make a program traceable, but there will still be
some Python features that are unsupported. An option to get past dealing with
this graph breaks is by using
{ref}`non-strict export <non-strict-export>` through changing the `strict` flag
to `strict=False`.
=======
information or rewrite parts of their code to make it traceable. As the
tracing is based on TorchDynamo, which evaluates at the Python
bytecode level, there will be significantly fewer rewrites required compared to
previous tracing frameworks.

When a graph break is encountered, {ref}`ExportDB <torch.export_db>` is a great
resource for learning about the kinds of programs that are supported and
unsupported, along with ways to rewrite programs to make them traceable.

An option to get past dealing with this graph breaks is by using
{ref}`non-strict export <non-strict-export>`
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

(data-shape-dependent-control-flow)=

### Data/Shape-Dependent Control Flow

Graph breaks can also be encountered on data-dependent control flow (`if
x.shape[0] > 2`) when shapes are not being specialized, as a tracing compiler cannot
possibly deal with without generating code for a combinatorially exploding
number of paths. In such cases, users will need to rewrite their code using
special control flow operators. Currently, we support {ref}`torch.cond <cond>`
to express if-else like control flow (more coming soon!).

<<<<<<< HEAD
You can also refer to this
[tutorial](https://docs.pytorch.org/tutorials/intermediate/torch_export_tutorial.html#data-dependent-errors)
for more ways of addressing data-dependent errors.

### Missing Fake/Meta Kernels for Operators

When tracing, a FakeTensor kernel (aka meta kernel) is required for all
operators. This is used to reason about the input/output shapes for this
operator.

Please see this [tutorial](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html)
for more details.
=======
### Missing Fake/Meta/Abstract Kernels for Operators

When tracing, a FakeTensor kernel (aka meta kernel, abstract impl) is
required for all operators. This is used to reason about the input/output shapes
for this operator.

Please see {func}`torch.library.register_fake` for more details.
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

In the unfortunate case where your model uses an ATen operator that is does not
have a FakeTensor kernel implementation yet, please file an issue.

## Read More

```{toctree}
:caption: Additional Links for Export Users
:maxdepth: 1

<<<<<<< HEAD
export/api_reference
export/programming_model
export/ir_spec
export/pt2_archive
export/draft_export
export/joint_with_descriptors
cond
generated/exportdb/index
torch.compiler_aot_inductor
torch.compiler_ir
=======
export.programming_model
export.ir_spec
draft_export
torch.compiler_transformations
torch.compiler_ir
generated/exportdb/index
cond
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
```

```{toctree}
:caption: Deep Dive for PyTorch Developers
:maxdepth: 1

<<<<<<< HEAD
torch.compiler_dynamic_shapes
torch.compiler_fake_tensor
torch.compiler_transformations
=======
torch.compiler_dynamo_overview
torch.compiler_dynamo_deepdive
torch.compiler_dynamic_shapes
torch.compiler_fake_tensor
```

## API Reference

```{eval-rst}
.. automodule:: torch.export
```

```{eval-rst}
.. autofunction:: export
```

```{eval-rst}
.. autofunction:: save
```

```{eval-rst}
.. autofunction:: load
```

```{eval-rst}
.. autofunction:: draft_export
```

```{eval-rst}
.. autofunction:: register_dataclass
```

```{eval-rst}
.. autoclass:: torch.export.dynamic_shapes.Dim
```

```{eval-rst}
.. autoclass:: torch.export.dynamic_shapes.ShapesCollection

    .. automethod:: dynamic_shapes
```

```{eval-rst}
.. autoclass:: torch.export.dynamic_shapes.AdditionalInputs

    .. automethod:: add
    .. automethod:: dynamic_shapes
    .. automethod:: verify
```

```{eval-rst}
.. autofunction:: torch.export.dynamic_shapes.refine_dynamic_shapes_from_suggested_fixes
```

```{eval-rst}
.. autoclass:: ExportedProgram

    .. attribute:: graph
    .. attribute:: graph_signature
    .. attribute:: state_dict
    .. attribute:: constants
    .. attribute:: range_constraints
    .. attribute:: module_call_graph
    .. attribute:: example_inputs
    .. automethod:: module
    .. automethod:: run_decompositions
```

```{eval-rst}
.. autoclass:: ExportGraphSignature
```

```{eval-rst}
.. autoclass:: ModuleCallSignature
```

```{eval-rst}
.. autoclass:: ModuleCallEntry
```

```{eval-rst}
.. automodule:: torch.export.decomp_utils
```

```{eval-rst}
.. autoclass:: CustomDecompTable

    .. automethod:: copy
    .. automethod:: items
    .. automethod:: keys
    .. automethod:: materialize
    .. automethod:: pop
    .. automethod:: update
```

```{eval-rst}
.. autofunction:: torch.export.exported_program.default_decompositions
```

```{eval-rst}
.. automodule:: torch.export.exported_program
```

```{eval-rst}
.. automodule:: torch.export.graph_signature
```

```{eval-rst}
.. autoclass:: ExportGraphSignature

    .. automethod:: replace_all_uses
    .. automethod:: get_replace_hook
```

```{eval-rst}
.. autoclass:: ExportBackwardSignature
```

```{eval-rst}
.. autoclass:: InputKind
```

```{eval-rst}
.. autoclass:: InputSpec
```

```{eval-rst}
.. autoclass:: OutputKind
```

```{eval-rst}
.. autoclass:: OutputSpec
```

```{eval-rst}
.. autoclass:: SymIntArgument
```

```{eval-rst}
.. autoclass:: SymBoolArgument
```

```{eval-rst}
.. autoclass:: SymFloatArgument
```

```{eval-rst}
.. autoclass:: CustomObjArgument
```

```{eval-rst}
.. py:module:: torch.export.dynamic_shapes
```

```{eval-rst}
.. py:module:: torch.export.custom_ops
```

```{eval-rst}
.. automodule:: torch.export.unflatten
    :members:
```

```{eval-rst}
.. automodule:: torch.export.custom_obj
```

```{eval-rst}
.. automodule:: torch.export.experimental
```

```{eval-rst}
.. automodule:: torch.export.passes
```

```{eval-rst}
.. autofunction:: torch.export.passes.move_to_device_pass
```

```{eval-rst}
.. automodule:: torch.export.pt2_archive
```

```{eval-rst}
.. automodule:: torch.export.pt2_archive.constants
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
```
