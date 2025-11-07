"""
This package implements variable tracking and symbolic execution capabilities for Dynamo,
which are essential for converting Python code into FX graphs. It provides a comprehensive
set of variable types that handle different Python constructs during tracing.

Each variable type (like BuiltinVariable, TensorVariable, NNModuleVariable, etc.) is responsible
for tracking and symbolically executing operations on specific Python objects. This enables
Dynamo to:
- Track the flow of values through Python code
- Maintain correct semantics during graph conversion
- Handle complex Python features like context managers, iterators, and custom objects
- Support both eager and symbolic execution modes

The VariableTracker base class provides the foundation for all variable types, with each
subclass implementing specific behavior for different Python constructs. This modular design
allows Dynamo to accurately trace and optimize Python code while preserving its semantics.
"""

from .base import VariableTracker
from .builtin import BuiltinVariable
from .constant import ConstantVariable, EnumVariable
from .ctx_manager import (
    CatchWarningsCtxManagerVariable,
    ContextWrappingVariable,
    CUDADeviceVariable,
    DeterministicAlgorithmsVariable,
    DisabledSavedTensorsHooksVariable,
    DualLevelContextManager,
    DynamoConfigPatchVariable,
<<<<<<< HEAD
    ErrorOnGraphBreakVariable,
    FSDPParamGroupUseTrainingStateVariable,
    FxTracebackAnnotateVariable,
=======
    FSDPParamGroupUseTrainingStateVariable,
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    GradIncrementNestingCtxManagerVariable,
    GradInplaceRequiresGradCtxManagerVariable,
    GradModeVariable,
    InferenceModeVariable,
    JvpIncrementNestingCtxManagerVariable,
    SDPAKernelVariable,
    SetFwdGradEnabledContextManager,
<<<<<<< HEAD
    TemporarilyPopInterpreterStackCtxManagerVariable,
    VmapIncrementNestingCtxManagerVariable,
    WithEnterFunctionVariable,
=======
    StreamContextVariable,
    StreamVariable,
    TemporarilyPopInterpreterStackCtxManagerVariable,
    VmapIncrementNestingCtxManagerVariable,
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    WithExitFunctionVariable,
)
from .dicts import (
    ConstDictVariable,
    DefaultDictVariable,
    DictKeySetVariable,
    FrozensetVariable,
    MappingProxyVariable,
    NNModuleHooksDictVariable,
    SetVariable,
)
from .distributed import BackwardHookVariable, DistributedVariable, PlacementVariable
from .functions import (
    BuiltinMethodVariable,
    CollectionsNamedTupleFunction,
    CreateTMADescriptorExperimentalVariable,
    CreateTMADescriptorStableVariable,
    FunctionDecoratedByContextlibContextManagerVariable,
    FunctoolsPartialVariable,
    FunctoolsWrapsVariable,
    LocalGeneratorFunctionVariable,
    LocalGeneratorObjectVariable,
    NestedUserFunctionVariable,
    PolyfilledFunctionVariable,
    SkipFunctionVariable,
    TMADescriptorExperimentalVariable,
    TMADescriptorStableVariable,
    UserFunctionVariable,
    UserMethodVariable,
    WrapperUserFunctionVariable,
    WrapperUserMethodVariable,
)
from .higher_order_ops import (
    FunctionalCallVariable,
    FunctorchHigherOrderVariable,
<<<<<<< HEAD
    ReparametrizeModuleCallVariable,
=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    TorchHigherOrderOperatorVariable,
)
from .iter import (
    CountIteratorVariable,
<<<<<<< HEAD
=======
    CycleIteratorVariable,
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    FilterVariable,
    IteratorVariable,
    ItertoolsVariable,
    MapVariable,
<<<<<<< HEAD
    ObjectIteratorVariable,
=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    RepeatIteratorVariable,
    ZipVariable,
)
from .lazy import LazyVariableTracker
from .lists import (
    BaseListVariable,
    ListIteratorVariable,
    ListVariable,
    NamedTupleVariable,
    RangeVariable,
    SliceVariable,
    TupleIteratorVariable,
    TupleVariable,
)
from .misc import (
    AutogradFunctionContextVariable,
    AutogradFunctionVariable,
    CellVariable,
    DeletedVariable,
    ExceptionVariable,
    GetAttrVariable,
    LambdaVariable,
    MethodWrapperVariable,
    NewGlobalVariable,
    NumpyVariable,
    PythonModuleVariable,
    RandomClassVariable,
    RandomVariable,
    RegexPatternVariable,
    StringFormatVariable,
    SuperVariable,
    TorchVersionVariable,
    TypingVariable,
    UnknownVariable,
    WeakRefVariable,
)
from .nn_module import (
    FSDPManagedNNModuleVariable,
    NNModuleVariable,
    UnspecializedBuiltinNNModuleVariable,
    UnspecializedNNModuleVariable,
)
from .optimizer import OptimizerVariable
from .sdpa import SDPAParamsVariable
<<<<<<< HEAD
from .streams import EventVariable, StreamContextVariable, StreamVariable
=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
from .tensor import (
    DataPtrVariable,
    FakeItemVariable,
    NumpyNdarrayVariable,
    SymNodeVariable,
    TensorVariable,
    UnspecializedPythonVariable,
    UntypedStorageVariable,
)
from .torch import TorchCtxManagerClassVariable, TorchInGraphFunctionVariable
from .user_defined import (
<<<<<<< HEAD
    FrozenDataClassVariable,
=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    MutableMappingVariable,
    RemovableHandleVariable,
    UserDefinedClassVariable,
    UserDefinedDictVariable,
    UserDefinedExceptionClassVariable,
    UserDefinedExceptionObjectVariable,
    UserDefinedListVariable,
    UserDefinedObjectVariable,
<<<<<<< HEAD
    UserDefinedSetVariable,
=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    UserDefinedTupleVariable,
)


__all__ = [
    "AutogradFunctionContextVariable",
    "AutogradFunctionVariable",
    "BackwardHookVariable",
    "BaseListVariable",
    "BuiltinVariable",
    "CatchWarningsCtxManagerVariable",
    "ConstantVariable",
    "ConstDictVariable",
    "ContextWrappingVariable",
    "CountIteratorVariable",
    "CreateTMADescriptorExperimentalVariable",
    "CreateTMADescriptorStableVariable",
    "CUDADeviceVariable",
<<<<<<< HEAD
=======
    "CycleIteratorVariable",
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    "DataPtrVariable",
    "DefaultDictVariable",
    "DeletedVariable",
    "DeterministicAlgorithmsVariable",
    "DictKeySetVariable",
    "DynamoConfigPatchVariable",
    "EnumVariable",
    "FakeItemVariable",
    "GetAttrVariable",
    "GradModeVariable",
    "IteratorVariable",
    "ItertoolsVariable",
    "LambdaVariable",
    "LazyVariableTracker",
    "ListIteratorVariable",
    "ListVariable",
    "NamedTupleVariable",
    "NestedUserFunctionVariable",
    "CellVariable",
    "NewGlobalVariable",
    "NNModuleVariable",
    "NumpyNdarrayVariable",
    "NumpyVariable",
    "OptimizerVariable",
    "PlacementVariable",
    "PolyfilledFunctionVariable",
    "PythonModuleVariable",
    "RangeVariable",
    "RegexPatternVariable",
    "RemovableHandleVariable",
    "RepeatIteratorVariable",
    "SDPAParamsVariable",
<<<<<<< HEAD
    "ErrorOnGraphBreakVariable",
=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    "SkipFunctionVariable",
    "SliceVariable",
    "StringFormatVariable",
    "SuperVariable",
    "TemporarilyPopInterpreterStackCtxManagerVariable",
    "TensorVariable",
    "TMADescriptorExperimentalVariable",
    "TMADescriptorStableVariable",
    "TorchCtxManagerClassVariable",
    "TorchInGraphFunctionVariable",
    "TorchVersionVariable",
    "TupleVariable",
    "UnknownVariable",
    "UnspecializedNNModuleVariable",
    "UnspecializedPythonVariable",
    "UntypedStorageVariable",
    "UserDefinedClassVariable",
    "UserDefinedTupleVariable",
    "UserDefinedObjectVariable",
    "UserFunctionVariable",
    "UserMethodVariable",
    "VariableTracker",
<<<<<<< HEAD
    "WithEnterFunctionVariable",
=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    "WithExitFunctionVariable",
    "MappingProxyVariable",
]
