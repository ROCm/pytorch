<<<<<<< HEAD
=======
# mypy: allow-untyped-defs
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
from torch.fx.experimental.migrate_gradual_types.constraint import (
    BinConstraintD,
    BVar,
    DVar,
    TVar,
)
from torch.fx.experimental.migrate_gradual_types.operation import op_leq


<<<<<<< HEAD
def gen_tvar(curr: int) -> tuple[TVar, int]:
=======
def gen_tvar(curr):
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    """
    Generate a tensor variable
    :param curr: The current counter
    :return: a tensor variable and the updated counter
    """
    curr += 1
    return TVar(curr), curr


<<<<<<< HEAD
def gen_dvar(curr: int) -> tuple[DVar, int]:
=======
def gen_dvar(curr):
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    """
    Generate a dimension variable
    :param curr: the current counter
    :return: a dimension variable and an updated counter
    """
    curr += 1
    return DVar(curr), curr


<<<<<<< HEAD
def gen_bvar(curr: int) -> tuple[BVar, int]:
=======
def gen_bvar(curr):
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    """
    Generate a boolean variable
    :param curr: the current counter
    :return: a boolean variable and an updated counter
    """
    curr += 1
    return BVar(curr), curr


<<<<<<< HEAD
def gen_tensor_dims(n: int, curr: int) -> tuple[list[DVar], int]:
=======
def gen_tensor_dims(n, curr):
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    """
    Generate a list of tensor dimensions
    :param n:  the number of dimensions
    :param curr: the current counter
    :return: a list of dimension variables and an updated counter
    """
    dims = []
    for _ in range(n):
        dvar, curr = gen_dvar(curr)
        dims.append(dvar)
    return dims, curr


<<<<<<< HEAD
def gen_nat_constraints(list_of_dims: list[DVar]) -> list[BinConstraintD]:
=======
def gen_nat_constraints(list_of_dims):
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    """
    Generate natural number constraints for dimensions
    """
    return [BinConstraintD(0, d, op_leq) for d in list_of_dims]
