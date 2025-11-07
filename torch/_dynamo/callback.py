"""
This module provides callback management functionality for TorchDynamo's compilation process.

It implements a thread-safe system for registering, managing and executing callbacks that run
at the start and end of TorchDynamo compilations. Key features include:

- Registration and deregistration of compilation callbacks
- Thread-safe callback handling with proper locking mechanisms
- Prevention of duplicate callback execution when configured
- Decorator utilities for easy callback registration
- Context manager for controlled callback lifecycle

The module centers around the CompilationCallbackHandler class which maintains separate
lists for start and end callbacks, manages their execution order, and ensures thread-safety.
Utility decorators @on_compile_start and @on_compile_end provide a convenient way to
register compilation hooks.

Example usage:
    @on_compile_start
    def my_start_callback():
        print("Starting compilation")

    @on_compile_end
    def my_end_callback():
        print("Compilation complete")
"""

<<<<<<< HEAD
=======
import enum
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
import threading
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field  # noqa: F811
<<<<<<< HEAD
from typing import Any, Callable, Optional

from torch._utils_internal import justknobs_check
=======
from typing import Any, Callable


class CallbackTrigger(enum.Enum):
    # most common case, dynamo attempts to trace a new frame
    DYNAMO = 1
    # backward compilation can be deferred to runtime
    LAZY_BACKWARD = 2
    # some backends autotune at runtime
    TRITON_AUTOTUNING = 3
    # cudagraphs record at runtime
    CUDAGRAPH_RECORDING = 4


@dataclass
class CallbackArgs:
    callback_trigger: CallbackTrigger
    compile_id: str
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))


@dataclass
class CompilationCallbackHandler:
<<<<<<< HEAD
    start_callbacks: list[Callable[[], None]] = field(default_factory=list)
    end_callbacks: list[Callable[[], None]] = field(default_factory=list)

    __prevent_duplicate_callbacks: Optional[bool] = field(
        default=None, init=False, repr=False
    )
=======
    start_callbacks: list[Callable[[CallbackArgs], None]] = field(default_factory=list)
    end_callbacks: list[Callable[[CallbackArgs], None]] = field(default_factory=list)

>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    __pending_callbacks_counter: int = field(default=0, init=False, repr=False)
    __pending_callbacks_counter_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def register_start_callback(
<<<<<<< HEAD
        self, callback: Callable[[], None]
    ) -> Callable[[], None]:
=======
        self, callback: Callable[[CallbackArgs], None]
    ) -> Callable[[CallbackArgs], None]:
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        """
        Register a callback function to be called when the compilation starts.

        Args:
        - callback (Callable): The callback function to register.
        """
        self.start_callbacks.append(callback)
        return callback

<<<<<<< HEAD
    def register_end_callback(self, callback: Callable[[], None]) -> Callable[[], None]:
=======
    def register_end_callback(
        self, callback: Callable[[CallbackArgs], None]
    ) -> Callable[[CallbackArgs], None]:
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        """
        Register a callback function to be called when the compilation ends.

        Args:
        - callback (Callable): The callback function to register.
        """
        self.end_callbacks.append(callback)
        return callback

<<<<<<< HEAD
    def remove_start_callback(self, callback: Callable[[], None]) -> None:
=======
    def remove_start_callback(self, callback: Callable[[CallbackArgs], None]) -> None:
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        """
        Remove a registered start callback function.

        Args:
        - callback (Callable): The callback function to remove.
        """
        self.start_callbacks.remove(callback)

<<<<<<< HEAD
    def remove_end_callback(self, callback: Callable[[], None]) -> None:
=======
    def remove_end_callback(self, callback: Callable[[CallbackArgs], None]) -> None:
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        """
        Remove a registered end callback function.

        Args:
        - callback (Callable): The callback function to remove.
        """
        self.end_callbacks.remove(callback)

<<<<<<< HEAD
    def run_start_callbacks(self) -> None:
=======
    def run_start_callbacks(self, args: CallbackArgs) -> None:
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        """
        Execute all registered start callbacks.
        """
        for callback in self.start_callbacks:
<<<<<<< HEAD
            callback()

    def run_end_callbacks(self) -> None:
=======
            callback(args)

    def run_end_callbacks(self, args: CallbackArgs) -> None:
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        """
        Execute all registered end callbacks.
        """
        for callback in self.end_callbacks:
<<<<<<< HEAD
            callback()

    @property
    def prevent_duplicate_callbacks(self) -> bool:
        if self.__prevent_duplicate_callbacks is None:
            self.__prevent_duplicate_callbacks = justknobs_check(
                "pytorch/dynamo:prevent_duplicate_callbacks"
            )
        return self.__prevent_duplicate_callbacks

    @contextmanager
    def install_callbacks(self) -> Generator[None, Any, Any]:
        """
        Context manager to install the callbacks and run them when the context is exited.
        """
        if self.prevent_duplicate_callbacks:
            try:
                with self.__pending_callbacks_counter_lock:
                    if self.__pending_callbacks_counter == 0:
                        self.run_start_callbacks()
                    self.__pending_callbacks_counter += 1
                yield
            finally:
                with self.__pending_callbacks_counter_lock:
                    assert self.__pending_callbacks_counter > 0, (
                        "Pending callbacks counter cannot become negative."
                    )
                    if self.__pending_callbacks_counter == 1:
                        self.run_end_callbacks()
                    self.__pending_callbacks_counter -= 1
        else:
            try:
                self.run_start_callbacks()
                yield
            finally:
                self.run_end_callbacks()
=======
            callback(args)

    @contextmanager
    def install_callbacks(
        self, trigger: CallbackTrigger, compile_id: str
    ) -> Generator[None, Any, Any]:
        """
        Context manager to install the callbacks and run them when the context is exited.
        """
        args = CallbackArgs(trigger, compile_id)
        try:
            with self.__pending_callbacks_counter_lock:
                if self.__pending_callbacks_counter == 0:
                    self.run_start_callbacks(args)
                self.__pending_callbacks_counter += 1
            yield
        finally:
            with self.__pending_callbacks_counter_lock:
                assert self.__pending_callbacks_counter > 0, (
                    "Pending callbacks counter cannot become negative."
                )
                if self.__pending_callbacks_counter == 1:
                    self.run_end_callbacks(args)
                self.__pending_callbacks_counter -= 1
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

    def clear(self) -> None:
        """
        Clear all registered callbacks.
        """
        self.start_callbacks.clear()
        self.end_callbacks.clear()
<<<<<<< HEAD
=======
        assert self.__pending_callbacks_counter == 0
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))


callback_handler = CompilationCallbackHandler()


<<<<<<< HEAD
def on_compile_start(callback: Callable[[], None]) -> Callable[[], None]:
=======
def on_compile_start(
    callback: Callable[[CallbackArgs], None],
) -> Callable[[CallbackArgs], None]:
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    """
    Decorator to register a callback function for the start of the compilation.
    """
    callback_handler.register_start_callback(callback)
    return callback


<<<<<<< HEAD
def on_compile_end(callback: Callable[[], None]) -> Callable[[], None]:
=======
def on_compile_end(
    callback: Callable[[CallbackArgs], None],
) -> Callable[[CallbackArgs], None]:
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    """
    Decorator to register a callback function for the end of the compilation.
    """
    callback_handler.register_end_callback(callback)
    return callback
