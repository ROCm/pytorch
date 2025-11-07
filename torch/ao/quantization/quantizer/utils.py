<<<<<<< HEAD
from collections.abc import Callable
from typing import Optional

from torch.ao.quantization.pt2e.utils import _is_sym_size_node
from torch.ao.quantization.quantizer.quantizer import (
    QuantizationAnnotation,
    QuantizationSpecBase,
)
from torch.fx import Node


__all__: list[str] = []


def _annotate_input_qspec_map(
    node: Node, input_node: Node, qspec: Optional[QuantizationSpecBase]
) -> None:
=======
# mypy: allow-untyped-defs

from torch.ao.quantization.pt2e.utils import _is_sym_size_node
from torch.ao.quantization.quantizer.quantizer import QuantizationAnnotation
from torch.fx import Node


def _annotate_input_qspec_map(node: Node, input_node: Node, qspec):
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    quantization_annotation = node.meta.get(
        "quantization_annotation", QuantizationAnnotation()
    )
    if quantization_annotation.input_qspec_map is None:
        quantization_annotation.input_qspec_map = {}
    quantization_annotation.input_qspec_map[input_node] = qspec
    node.meta["quantization_annotation"] = quantization_annotation


<<<<<<< HEAD
def _annotate_output_qspec(node: Node, qspec: Optional[QuantizationSpecBase]) -> None:
=======
def _annotate_output_qspec(node: Node, qspec):
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    quantization_annotation = node.meta.get(
        "quantization_annotation", QuantizationAnnotation()
    )
    quantization_annotation.output_qspec = qspec
    node.meta["quantization_annotation"] = quantization_annotation


<<<<<<< HEAD
def _node_only_used_for_sym_size(node: Node, partition_nodes: list[Node]) -> bool:
=======
def _node_only_used_for_sym_size(node: Node, partition_nodes: list[Node]):
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    """
    This utility is used to handle cases when dynami_shape=True tracing leads
    to symint nodes in the pattern of linear module. In those cases, we need to
    distinguish between the nodes that are in input for just extracting value of
<<<<<<< HEAD
    some dimensions (and symint nodes) vs. the one that is activation.
=======
    some dimentions (and symint nodes) vs. the one that is activation.
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    For example:
    graph(x, y, weight):
       size_0 = torch.ops.aten.sym_size([x], [0])
       size_1 = torch.ops.aten.sym_size([y], [1])
       view_size = size_0 * size_1
       size_3 = torch.ops.aten.sym_size([x], [2])
       vie_out = torch.ops.aten.view(x, [view_size, size_3])
       return mm(view_out, weight)
    In the example above y node is not actual input. It exist only to extract size_1
    """
    if _is_sym_size_node(node):
        return True

    return all(
        ((user not in partition_nodes) or _is_sym_size_node(user))
        for user in node.users
    )


<<<<<<< HEAD
def _get_module_name_filter(module_name: str) -> Callable[[Node], bool]:
=======
def _get_module_name_filter(module_name: str):
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    """Get the module_name_filter function for a given module name, the filter accepts
    a node and checks if the node comes from a module that has certain module name

    For example:
        node: linear_op = call_function[...](...)  # comes from a module with name blocks.sub.linear1


    >> module_name_filter = _get_module_name_filter("blocks.sub")
    >> print(module_name_filter(node))
    True  # the node is from "blocks.sub" based on the fully qualified name "blocks.sub.linear1"
    """

    def module_name_filter(n: Node) -> bool:
        # example: {
        #    'L__self___sub': ("L['self'].sub", <class '....Sub'>),
        #    'L__self___sub_linear': ("L['self'].sub.linear", <class 'torch.nn.modules.linear.Linear'>)
        # }
        # get_attr nodes doesn't have nn_module_stack?
        nn_module_stack = n.meta.get("nn_module_stack", {})

<<<<<<< HEAD
        def _normalize_path(n: str) -> str:
=======
        def _normalize_path(n):
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
            prefix = 0
            # TODO This is non standard behavior and should be removed when we migrate off capture_pre_autograd_graph.
            if n.startswith("L['self']."):
                prefix = len("L['self'].")
            return n[prefix:]

        names = [_normalize_path(n) for n, _ in nn_module_stack.values()]
        return module_name in names

    return module_name_filter
