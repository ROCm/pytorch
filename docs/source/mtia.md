# torch.mtia

<<<<<<< HEAD
The MTIA backend is implemented out of the tree, only interfaces are defined here.
=======
The MTIA backend is implemented out of the tree, only interfaces are be defined here.
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

```{eval-rst}
.. automodule:: torch.mtia
```

```{eval-rst}
.. currentmodule:: torch.mtia
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    StreamContext
    current_device
    current_stream
    default_stream
    device_count
    init
    is_available
<<<<<<< HEAD
    is_bf16_supported
=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    is_initialized
    memory_stats
    get_device_capability
    empty_cache
    record_memory_history
    snapshot
    attach_out_of_memory_observer
    set_device
    set_stream
    stream
    synchronize
    device
    set_rng_state
    get_rng_state
    DeferredMtiaCallError
```

## Streams and events

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    Event
    Stream
```
