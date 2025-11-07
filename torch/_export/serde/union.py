# mypy: allow-untyped-defs
import functools
from collections.abc import Hashable
<<<<<<< HEAD
from dataclasses import dataclass, fields
from typing import TypeVar
from typing_extensions import dataclass_transform


T = TypeVar("T", bound="_Union")
=======
from dataclasses import fields
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))


class _UnionTag(str):
    __slots__ = ("_cls",)
    _cls: Hashable

    @staticmethod
    def create(t, cls):
        tag = _UnionTag(t)
        assert not hasattr(tag, "_cls")
        tag._cls = cls
        return tag

    def __eq__(self, cmp) -> bool:
        assert isinstance(cmp, str)
        other = str(cmp)
<<<<<<< HEAD
        assert other in _get_field_names(self._cls), (
            f"{other} is not a valid tag for {self._cls}. Available tags: {_get_field_names(self._cls)}"
        )
=======
        assert other in _get_field_names(
            self._cls
        ), f"{other} is not a valid tag for {self._cls}. Available tags: {_get_field_names(self._cls)}"
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        return str(self) == other

    def __hash__(self):
        return hash(str(self))


@functools.cache
def _get_field_names(cls) -> set[str]:
    return {f.name for f in fields(cls)}


<<<<<<< HEAD
# If you turn a schema class that inherits from union into a dataclass, please use
# this decorator to configure it. It's safe, faster and allows code sharing.
#
# For example, _union_dataclass customizes the __eq__ method to only check the type
# and value property instead of default implementation of dataclass which goes
# through every field in the dataclass.
@dataclass_transform(eq_default=False)
def _union_dataclass(cls: type[T]) -> type[T]:
    assert issubclass(cls, _Union), f"{cls} must inheirt from {_Union}."
    return dataclass(repr=False, eq=False)(cls)


=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
class _Union:
    _type: _UnionTag

    @classmethod
    def create(cls, **kwargs):
        assert len(kwargs) == 1
        obj = cls(**{**{f.name: None for f in fields(cls)}, **kwargs})  # type: ignore[arg-type]
        obj._type = _UnionTag.create(next(iter(kwargs.keys())), cls)
        return obj

    def __post_init__(self):
<<<<<<< HEAD
        assert not any(
            f.name in ("type", "_type", "create", "value")
            for f in fields(self)  # type: ignore[arg-type, misc]
        )
=======
        assert not any(f.name in ("type", "_type", "create", "value") for f in fields(self))  # type: ignore[arg-type, misc]
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

    @property
    def type(self) -> str:
        try:
            return self._type
        except AttributeError as e:
            raise RuntimeError(
                f"Please use {type(self).__name__}.create to instantiate the union type."
            ) from e

    @property
    def value(self):
        return getattr(self, self.type)

    def __getattribute__(self, name):
        attr = super().__getattribute__(name)
        if attr is None and name in _get_field_names(type(self)) and name != self.type:  # type: ignore[arg-type]
            raise AttributeError(f"Field {name} is not set.")
        return attr

<<<<<<< HEAD
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Union):
            return False
        return self.type == other.type and self.value == other.value

=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"{type(self).__name__}({self.type}={getattr(self, self.type)})"
