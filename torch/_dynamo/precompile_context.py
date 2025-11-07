<<<<<<< HEAD
import copy
import json
import logging
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, Optional, TypeVar

import torch
from torch._dynamo.package import (
    _BackendId,
    _DynamoCacheEntry,
    DynamoCache,
    PrecompileCacheEntry,
)
=======
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Generic, Optional, TypeVar
from typing_extensions import override

from torch.compiler._cache import (
    _serialize_single_cache,
    CacheArtifact,
    CacheArtifactFactory,
    CacheArtifactManager,
    CacheArtifactsResult,
    CacheInfo,
)
from torch.utils._appending_byte_serializer import AppendingByteSerializer
from torch.utils._ordered_set import OrderedSet
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))


"""
Classes and implementations related to precompile
"""

T = TypeVar("T")
<<<<<<< HEAD
logger = logging.getLogger(__name__)


@dataclass
class BackendCacheArtifact(Generic[T]):
    """
    Represents a single serializable backend artifact from a dynamo backend.
    Each BackendCacheArtifact has a key associated with it along with some
    serializable content.
=======


class PrecompileCacheArtifact(CacheArtifact, Generic[T]):
    """
    Data for each cache artifact that will be serialized and deserialized by
    PrecompileContext, rather than CacheArtifactManager.
    T represents the deserialized type of the artifact, i.e. the return type of after_deserialization

    PrecompileCacheArtifact is a frozen dataclass - you can add new serializable fields and metadata specific to your own artifacts
    as needed, and use them in after_deserialization.
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

    Example implementation:

    class MyPrecompileCacheArtifact(PrecompileCacheArtifact[MySerializableType]):
        my_field: int

        def after_deserialization(self) -> MySerializableType:
            result = pickle.loads(self.content)
            # Do some extra work post deserialization
            result.my_post_deserialization_function(self.my_field)
            return result
    """

<<<<<<< HEAD
    key: str
    content: Any
=======
    @override
    def populate_cache(self) -> None:
        raise RuntimeError("Precompile cache artifacts do not populate caches")

    @override
    def precompile_compatible(self) -> bool:
        return True
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

    @abstractmethod
    def after_deserialization(self) -> T:
        """
        Code to be run after reading raw byte contents from disk.
        Generally converts self.content from raw bytes back into its original form.
        """
        ...

<<<<<<< HEAD
    def edit_contents(self, edit_fn: Callable[..., Any]) -> None:
        """
        Edit the contents of the artifact.
        """
        self.content = edit_fn(self.content)


class EagerCacheArtifact(BackendCacheArtifact[Any]):
    def after_deserialization(self) -> Any:
        return self.content


class BypassDynamoCacheEntry(Exception):
    pass


class PrecompileContext:
=======

class PrecompileContext(CacheArtifactManager):
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    """
    PrecompileContext is a special CacheArtifactManager for handling precompilation
    It uses the same interface as CacheArtifactManager, but handles deserialization differently: instead
    of placing each artifact into respective caches, it will stitch all the cache artifacts for a single key
    together and place it into a global Precompile Cache.

<<<<<<< HEAD
    PrecompileContext has two main portions: dynamo_cache_entries and backend_cache_artifacts.
    When saving, PrecompileContext.serialize() will serialize all dynamo cache entries along with any PrecompileCacheArtifacts that
    are needed to save those dynamo cache entries.

    The following artifact types are supported by PrecompileContext:
     - BundledAOTAutogradCacheArtifact

    """

    # Protected by the compile_lock
    # _backend_artifacts_by_key organizes results by the key of each artifact.
    # Each object here must be serializable
    _backend_artifacts_by_key: dict[_BackendId, BackendCacheArtifact[Any]] = {}

    # On call to `serialize()`, all cache artifacts in _dynamo_cache_entries are converted
    # into DynamoCacheArtifacts and added to _new_cache_artifacts for serialization
    _dynamo_cache_entries: dict[str, _DynamoCacheEntry] = {}

    @classmethod
    def clear(cls) -> None:
        cls._backend_artifacts_by_key.clear()
        cls._dynamo_cache_entries.clear()

    @classmethod
    def record_artifact(
        cls,
        artifact: BackendCacheArtifact[Any],
    ) -> None:
        """
        Records a backend artifact to be used with dynamo cache entries
        """
        cls._backend_artifacts_by_key[_BackendId(artifact.key)] = copy.deepcopy(
            artifact
        )

    @classmethod
    def record_dynamo_cache_entry(
        cls, cache_entry: _DynamoCacheEntry, key: str
    ) -> None:
        cls._dynamo_cache_entries[key] = cache_entry

    @classmethod
    def edit_artifact(cls, key: str, edit_fn: Callable[..., Any]) -> None:
        """
        Edit the content of an existing artifact
        """
        assert key in cls._backend_artifacts_by_key, f"Key {key} not found in artifacts"
        artifact = cls._backend_artifacts_by_key[_BackendId(key)]
        artifact.edit_contents(edit_fn)

    @classmethod
    def serialize_artifact_by_key(cls, key: str) -> Optional[BackendCacheArtifact[Any]]:
        """
        Return the backend cache artifact with the associated key
        """
        return cls._backend_artifacts_by_key.get(_BackendId(key), None)

    @staticmethod
    def dump_debug_info(
        dynamo_entries: dict[str, _DynamoCacheEntry],
        backend_artifacts: dict[_BackendId, BackendCacheArtifact[Any]],
    ) -> dict[str, Any]:
        """
        Return a JSON serializable debug dump of all entries in the precompile context
        Called in serialize before serialization, and in populate_caches after deserialization
        """
        # Print debug information
        debug_info: defaultdict[str, list[Any]] = defaultdict(list)
        for key, cache_entry in dynamo_entries.items():
            info = cache_entry.debug_info()
            info["key"] = key
            debug_info["dynamo"].append(info)

        for artifact in backend_artifacts.values():
            debug_info["backends"].append(artifact.key)

        return debug_info

    @classmethod
    def save_to_dynamo_cache(cls) -> dict[str, Any]:
        precompile_cache_entries, debug_info = cls.create_cache_entries()
        for key, entry in precompile_cache_entries.items():
            DynamoCache.write(entry, key)
        return debug_info

    @classmethod
    def create_cache_entries(
        cls,
    ) -> tuple[dict[str, PrecompileCacheEntry], dict[str, Any]]:
        """
        Grabs all the cache entries in the precompile context and
        stitches them together into full PrecompileCacheEntries.
        """
        dynamo_entries = cls._dynamo_cache_entries
        backend_artifacts = cls._backend_artifacts_by_key

        num_artifacts = len(dynamo_entries)

        debug_info = PrecompileContext.dump_debug_info(
            dynamo_entries, backend_artifacts
        )
        debug_str = json.dumps(
            {
                "num_entries": num_artifacts,
                "artifacts": debug_info,
            },
        )
        torch._logging.trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "dynamo_cache_entries",
                "encoding": "json",
            },
            payload_fn=lambda: debug_str,
            expect_trace_id=False,
        )

        precompile_cache_entries = {}

        for key, cache_entry in dynamo_entries.items():
            try:
                result = PrecompileCacheEntry.from_cache_entry(
                    cache_entry, backend_artifacts
                )
                if result is not None:
                    precompile_cache_entries[key] = result
            except Exception as e:
                logger.warning("Failed to create cache entry %s", key, exc_info=True)

                error = e
                data = json.dumps(
                    {
                        "key": key,
                        "error": str(error),
                    }
                )
                torch._logging.trace_structured(
                    "artifact",
                    metadata_fn=lambda: {
                        "name": "dynamo_cache_exception",
                        "encoding": "json",
                    },
                    payload_fn=lambda: data,
                )
                continue
        return precompile_cache_entries, debug_info
=======
    The following artifact types are supported by PrecompileContext:
     - BundledAOTAutogradCacheArtifact
     - CodeStateArtifact (from torch._dynamo.package once available)
    """

    # Protected by the compile_lock
    # _new_cache_artifacts_by_key organizes results by the key of each artifact.
    # This allows us to implement serialize_by_key easily.
    # On call to `serialize()`, all cache artifacts in _new_cache_artifacts_by_key
    # are transferred to _new_cache_artifacts before serialization.
    _new_cache_artifacts_by_key: dict[str, CacheArtifact] = {}
    _new_cache_artifacts: CacheArtifactsResult = defaultdict(list)
    # Keep a separate seen artifacts list to make avoid unnecessary duplicates
    # This list will not be cleared between serialize() calls
    _seen_artifacts: OrderedSet[CacheArtifact] = OrderedSet()
    # When serialize() is called, artifacts are transferred from _cache_artifacts to
    # internal data structure of the _serializer
    # This allows us to only pay the cost of serialization if serialize() is called
    _serializer: AppendingByteSerializer[tuple[str, list[CacheArtifact]]] = (
        AppendingByteSerializer(serialize_fn=_serialize_single_cache)
    )
    _cache_info: CacheInfo = CacheInfo()

    @classmethod
    def clear(cls) -> None:
        cls._new_cache_artifacts_by_key.clear()
        super().clear()

    @override
    @classmethod
    def record_artifact(
        cls,
        artifact_type: str,
        key: str,
        content: Any,
    ) -> None:
        """
        Called from each caching operation to record the artifact in this
        "mega" list
        """
        artifact = CacheArtifactFactory.encode_create(artifact_type, key, content)
        # TODO: although this covers completely same artifacts, it's possible
        # with AOTAutogradCacheEntries to have multiple artifacts whose keys
        # (i.e. backend_ids) are different, but whose contents are equal.
        # In those cases, it would be much better if we only serialize once instead
        # of N times.
        if artifact in cls._seen_artifacts:
            return

        cls._new_cache_artifacts_by_key[key] = artifact
        cls._seen_artifacts.add(artifact)

    @classmethod
    def _save_artifacts_by_type(cls) -> None:
        """
        We normally record artifacts by key, but serialization expects them to be organized
        by artifact type. This function transfers artifacts from _new_cache_artifacts_by_key to _new_cache_artifacts
        """
        for artifact in cls._new_cache_artifacts_by_key.values():
            cls._new_cache_artifacts[artifact.__class__.type()].append(artifact)
        cls._new_cache_artifacts_by_key.clear()

    @classmethod
    def serialize_artifact_by_key(cls, key: str) -> Optional[CacheArtifact]:
        """
        Serialize all artifacts with the given key returned in a list.
        """
        return cls._new_cache_artifacts_by_key.get(key, None)

    @classmethod
    def serialize(cls) -> Optional[tuple[bytes, CacheInfo]]:
        cls._save_artifacts_by_type()
        return super().serialize()

    @staticmethod
    def populate_caches(artifacts: CacheArtifactsResult) -> CacheInfo:
        raise NotImplementedError("TODO")

    @classmethod
    def _ensure_cache_artifacts_registered(cls) -> None:
        from torch._functorch._aot_autograd.autograd_cache import (  # noqa: F401
            BundledAOTAutogradCacheArtifact,
        )
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
