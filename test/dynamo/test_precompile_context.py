# Owner(s): ["module: dynamo"]
<<<<<<< HEAD
=======

>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._functorch
<<<<<<< HEAD
from torch._dynamo.precompile_context import BackendCacheArtifact, PrecompileContext
=======
from torch._dynamo.precompile_context import PrecompileContext
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
from torch._functorch import config as functorch_config
from torch._functorch._aot_autograd.autograd_cache import (
    BundledAOTAutogradCacheArtifact,
)
from torch._inductor.test_case import TestCase as InductorTestCase
from torch.testing._internal.inductor_utils import GPU_TYPE, requires_triton


@functorch_config.patch({"enable_autograd_cache": True})
<<<<<<< HEAD
@torch._dynamo.config.patch(
    {"caching_precompile": True}
=======
@functorch_config.patch(
    {"bundled_autograd_cache": True}
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
)  # Requires bundledaotautograd cache for now
class PrecompileContextTests(InductorTestCase):
    def setUp(self):
        """
        Reset all counters and caches before each unit test
        """
        super().setUp()
        # Clear PrecompileContext cache artifacts
        PrecompileContext.clear()

    @requires_triton()
    def test_basic(self):
        """
        Test that after torch.compile, PrecompileContext._new_cache_artifacts length is 1
        """

        def simple_function(x):
            return x.sin() + x.cos()

        compiled_fn = torch.compile(simple_function)

        # Run the compiled function
        x = torch.randn(10, device=GPU_TYPE, requires_grad=True)
        result = compiled_fn(x)
        result.sum().backward()
<<<<<<< HEAD
        self.assertEqual(len(PrecompileContext._dynamo_cache_entries), 1)
        self.assertEqual(len(PrecompileContext._backend_artifacts_by_key), 1)
        cache_entries, _ = PrecompileContext.create_cache_entries()
        self.assertEqual(len(cache_entries), 1)

    @requires_triton()
    def test_serialize_by_key(self):
        def simple_function(x):
            return x.sin() + x.cos()

        compiled_fn = torch.compile(simple_function)

        # Run the compiled function
        x = torch.randn(10, device=GPU_TYPE, requires_grad=True)
        result = compiled_fn(x)
        result.sum().backward()
        self.assertEqual(len(PrecompileContext._dynamo_cache_entries), 1)
        self.assertEqual(len(PrecompileContext._backend_artifacts_by_key), 1)
        for key in PrecompileContext._backend_artifacts_by_key.keys():
            result = PrecompileContext.serialize_artifact_by_key(key)
            assert isinstance(result, BackendCacheArtifact)
            self.assertEqual(result.key, key)

        # This should still work
        result, _ = PrecompileContext.create_cache_entries()
        assert len(result) == 1

    @requires_triton()
    def test_editable(self):
=======
        # Check that PrecompileContext._new_cache_artifacts_by_key has length 1
        self.assertEqual(len(PrecompileContext._new_cache_artifacts_by_key), 1)

        self.assertEqual(len(PrecompileContext._new_cache_artifacts), 0)
        result = PrecompileContext.serialize()
        assert result is not None
        serialized, cache_info = result
        self.assertEqual(len(cache_info.precompile_aot_autograd_artifacts), 1)

        artifacts = PrecompileContext.deserialize(serialized)
        assert artifacts is not None
        deserialized = artifacts["precompile_aot_autograd"]
        assert len(deserialized) == 1
        entry = deserialized[0]
        assert isinstance(entry, BundledAOTAutogradCacheArtifact)
        entry = entry.after_deserialization()
        # Now that we've serialized, there should be no new cache artifacts
        self.assertEqual(
            len(PrecompileContext._new_cache_artifacts["precompile_aot_autograd"]), 0
        )

    @requires_triton()
    def test_serialize_by_key(self):
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        """
        Test that after torch.compile, PrecompileContext._new_cache_artifacts length is 1
        """

        def simple_function(x):
            return x.sin() + x.cos()

        compiled_fn = torch.compile(simple_function)

        # Run the compiled function
        x = torch.randn(10, device=GPU_TYPE, requires_grad=True)
        result = compiled_fn(x)
        result.sum().backward()
<<<<<<< HEAD
        self.assertEqual(len(PrecompileContext._dynamo_cache_entries), 1)
        self.assertEqual(len(PrecompileContext._backend_artifacts_by_key), 1)
        # Find the key for the artifact of type "precompile_aot_autograd"
        key = next(iter(PrecompileContext._backend_artifacts_by_key))

        def edit_fn(x):
            x._my_private_field = 42
            return x

        PrecompileContext.edit_artifact(key, edit_fn)

=======
        # Check that PrecompileContext._new_cache_artifacts_by_key has length 1
        # TODO: the key right now is the AOTAutogradCacheKey, but will be backend_id once
        # we have torch._dynamo.package implemented
        self.assertEqual(len(PrecompileContext._new_cache_artifacts_by_key), 1)
        key = next(iter(PrecompileContext._new_cache_artifacts_by_key.keys()))
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        result = PrecompileContext.serialize_artifact_by_key(key)
        assert isinstance(result, BundledAOTAutogradCacheArtifact)
        self.assertEqual(result.key, key)

<<<<<<< HEAD
        result, _ = PrecompileContext.create_cache_entries()
        assert len(result) == 1
        aot_autograd_artifacts = next(iter(result.values())).backends
        assert len(aot_autograd_artifacts) == 1
        entry = next(iter(aot_autograd_artifacts.values())).content
        self.assertEqual(entry._my_private_field, 42)
=======
        self.assertEqual(len(PrecompileContext._new_cache_artifacts), 0)
        result = PrecompileContext.serialize()
        assert result is not None
        _, cache_info = result
        self.assertEqual(len(cache_info.precompile_aot_autograd_artifacts), 1)
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
