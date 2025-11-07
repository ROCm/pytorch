from __future__ import annotations

from torchgen.api import dispatcher
from torchgen.api.types import (
    BaseCppType,
    BaseCType,
    Binding,
    boolT,
    ConstRefCType,
    CType,
    longT,
    NamedCType,
    tensorT,
)
from torchgen.model import (
    Argument,
    BaseTy,
    BaseType,
    FunctionSchema,
    NativeFunction,
    NativeFunctionsViewGroup,
)


# This file describes the translation of JIT schema to API's used
<<<<<<< HEAD
# when creating `ViewMeta` specializations that are used by the functionalization pass.
# These API's mostly follow the dispatcher API, with one difference:
# - While the forward function just directly calls into the at::_ops API
#   (following the dispatcher convention), the logic here for the reverse function
#   is responsible for generating both the call-site, and the declarations
#   (which are implemented manually in the at::functionalization::impl namespace).

=======
# when creating view lambdas that are used by the functionalization pass.
# There are two types of lambdas: forward lambdas and reverse lambdas.
# These API's mostly follow the dispatcher API, with a few quirks:
# - The lambda capture has to convert reference types to value types
# - While the forward lambda just directly calls into the at::_ops API
#   (following the dispatcher convention), the logic here for the reverse lambda
#   is responsible for generating both the call-site, and the declarations
#   (which are implemented manually in the at::functionalization::impl namespace).

# The lambdas generated for each view op in the functionalization pass are of the form
# [capture_arguments](outer_arguments) -> returns_type {
#     return name(inner_arguments);
# }

>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
# Define some specific lambda input arguments.
base_binding = Binding(
    name="base",
    nctype=NamedCType(name="base", type=ConstRefCType(BaseCType(tensorT))),
    argument=Argument(
        name="base", type=BaseType(BaseTy.Tensor), default=None, annotation=None
    ),
    default=None,
)
<<<<<<< HEAD

has_symbolic_inputs_binding = Binding(
    name="has_symbolic_inputs",
    nctype=NamedCType(name="has_symbolic_inputs", type=BaseCType(boolT)),
    argument=Argument(
        name="has_symbolic_inputs",
        type=BaseType(BaseTy.bool),
        default=None,
        annotation=None,
    ),
    default=None,
)
=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
mutated_view_binding = Binding(
    name="mutated_view",
    nctype=NamedCType(name="mutated_view", type=ConstRefCType(BaseCType(tensorT))),
    argument=Argument(
        name="base", type=BaseType(BaseTy.Tensor), default=None, annotation=None
    ),
    default=None,
)
<<<<<<< HEAD
out_index_binding = Binding(
    name="out_index",
    nctype=NamedCType(name="out_index", type=BaseCType(longT)),
    argument=Argument(
        name="out_index", type=BaseType(BaseTy.int), default=None, annotation=None
=======
mutated_view_idx_binding = Binding(
    name="mutated_view_idx",
    nctype=NamedCType(name="mutated_view_idx", type=BaseCType(longT)),
    argument=Argument(
        name="base", type=BaseType(BaseTy.Tensor), default=None, annotation=None
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    ),
    default=None,
)
reapply_views_binding = Binding(
    name="reapply_views",
    nctype=NamedCType(name="reapply_views", type=BaseCType(boolT)),
    argument=Argument(
        name="reapply_views", type=BaseType(BaseTy.bool), default=None, annotation=None
    ),
    default=None,
)

InverseReturnModeT = BaseCppType("at::functionalization", "InverseReturnMode")
inverse_return_mode_binding = Binding(
    name="inverse_return_mode",
    nctype=NamedCType(name="inverse_return_mode", type=BaseCType(InverseReturnModeT)),
    argument=Argument(
        name="inverse_return_mode",
        # NB: not actually a bool but it doesn't matter because this isn't used
        type=BaseType(BaseTy.bool),
        default=None,
        annotation=None,
    ),
    default=None,
)


<<<<<<< HEAD
# Name of the `ViewMeta` specialization class created.
def classname(func: FunctionSchema, with_namespace: bool = False) -> str:
    namespace = "at::functionalization::" if with_namespace else ""
    return f"{namespace}{func.name.unambiguous_name()}_ViewMeta"


# Name of the operation called inside the `forward`/`reverse` implementations.
=======
# The lambda capture itself doesn't have a name.
# The name returned here corresponds to the name of the inner function called by the lambda.
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
def name(
    g: NativeFunctionsViewGroup,
    *,
    is_reverse: bool,
    include_namespace: bool,
    reapply_views: bool | None = None,
) -> str:
    if reapply_views is None:
        # reapply_views is only important for the fwd lambda,
        # since we always plumb the runtime "reapply_views" argument into the reverse function.
        assert is_reverse
    if is_reverse:
        return reverse_name(g.view, include_namespace)
    # in the forward case, we just directly call into the at::_ops API (so we always need the namespace)
    assert include_namespace
    assert g.view_copy is not None
    api_name = (
        g.view.func.name.unambiguous_name()
        if reapply_views
        else g.view_copy.func.name.unambiguous_name()
    )
    return f"at::_ops::{api_name}::call"


def reverse_name(f: NativeFunction, include_namespace: bool) -> str:
    # for the reverse: we plumb the "reapply_views" flag into that function and support
    # both copy and non-copy variants. (We could avoid doing that, but that would require
    # writing out twice as many view inverse functions).
    api_name = f.func.name.unambiguous_name()
    # in the reverse case, we codegen both the call-sites (which need the full namespace) and the declarations (which don't)
    if include_namespace:
        return f"at::functionalization::FunctionalInverses::{api_name}_inverse"
    else:
        return f"{api_name}_inverse"


<<<<<<< HEAD
=======
def capture_arguments(func: FunctionSchema, *, is_reverse: bool) -> list[Binding]:
    # capture arguments include all arguments except `self`.
    # Importantly, they don't include any C++ reference types (or else we'll get a dangling reference in the capture),
    # So any reference types (IntArrayRef) need to be converted to value types (vector<int64_t>)
    args = func.arguments.flat_all
    assert args[0].type == BaseType(BaseTy.Tensor)
    non_self_args = args[1:]
    non_self_value_bindings = [
        dispatcher.argument(a, remove_non_owning_ref_types=True) for a in non_self_args
    ]

    all_bindings = [
        inverse_return_mode_binding if is_reverse else reapply_views_binding
    ]
    all_bindings.extend(non_self_value_bindings)
    return all_bindings


>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
def returns_type(func: FunctionSchema) -> CType:
    # Assertion: all view ops return tensor-like outputs
    assert len(func.returns) >= 1
    for ret in func.returns:
        assert ret.type.is_tensor_like()
    # However, the return type of the lambda is always an individual tensor.
    # For multi-tensor outputs, each tensor needs to be tracked individually.
    return BaseCType(tensorT)


<<<<<<< HEAD
# Checks whether `func` might return more than one value.
def is_multi_output(func: FunctionSchema) -> bool:
    return len(func.returns) > 1 or (
        len(func.returns) == 1 and func.returns[0].type.is_list_like() is not None
    )


# `ViewMeta` specialization constructor parameters.
def base_ctor_arguments(func: FunctionSchema) -> list[Binding]:
    # All specializations are parematerized by `has_symbolic_inputs` flag.
    arguments = [has_symbolic_inputs_binding]

    # If `func` might return more than 1 value, we also parameterize this specialization
    # with the output index.
    if is_multi_output(func):
        arguments.append(out_index_binding)

    return arguments


# `ViewMeta` specialized class' constructor arguments.
#
# Values needed specifically by this specialization, that the base class does not need.
# Same as the class' attributes, but non-owning.
def extra_ctor_arguments(func: FunctionSchema) -> list[Binding]:
    return attributes(func, owning=False)


# `ViewMeta` specialized class' non-static member data.
#
# Essential data for calling the instance's `forward` and `reverse functions. You can
# think of them as values that should be captured from the functionalization kernel.
def attributes(func: FunctionSchema, owning: bool = True) -> list[Binding]:
    args = func.arguments.flat_all
    assert args[0].type == BaseType(BaseTy.Tensor)
    return [
        reapply_views_binding,
        inverse_return_mode_binding,
        *[dispatcher.argument(a, remove_non_owning_ref_types=owning) for a in args[1:]],
    ]


def op_arguments(func: FunctionSchema, is_reverse: bool) -> list[Binding]:
=======
def outer_arguments(*, is_reverse: bool) -> list[Binding]:
    if is_reverse:
        return [base_binding, mutated_view_binding, mutated_view_idx_binding]
    else:
        return [base_binding, mutated_view_idx_binding]


def inner_call_index(func: FunctionSchema) -> Binding | None:
    # For view ops that return multiple tensors (like `split`), we generate a separate lambda for each output.
    # When we replay a view op that returns multiple tensors, we need to index into the output appropriately
    if len(func.returns) > 1 or (
        len(func.returns) == 1 and func.returns[0].type.is_list_like()
    ):
        return mutated_view_idx_binding
    return None


def inner_arguments(func: FunctionSchema, is_reverse: bool) -> list[Binding]:
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    args = func.arguments.flat_all
    assert args[0].type == BaseType(BaseTy.Tensor)
    non_self_args = args[1:]
    # The forward lambda calls the at::_ops API, while the reverse lambda calls the view inverse API.
    # Both of these follow the dispatcher API.
    non_self_bindings = [dispatcher.argument(a) for a in non_self_args]
    if not is_reverse:
        # the forward lambda swaps out the original tensor argument with the lambd arg "base"
        return [base_binding] + non_self_bindings
    else:
        # the reverse lambda does the same, but with an additional "mutated_view" arg
        # additionally, we have a calling convention: for view ops that return multiple tensor outputs
        # their corresponding view_inverse function takes in an additional index argument.
<<<<<<< HEAD
        if is_multi_output(func):
=======
        index_binding = inner_call_index(func)
        if index_binding is not None:
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
            return [
                base_binding,
                mutated_view_binding,
                inverse_return_mode_binding,
<<<<<<< HEAD
                out_index_binding,
=======
                index_binding,
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
            ] + non_self_bindings
        else:
            return [
                base_binding,
                mutated_view_binding,
                inverse_return_mode_binding,
            ] + non_self_bindings
