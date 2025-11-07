# mypy: allow-untyped-defs

import torch.distributed.rpc as rpc
<<<<<<< HEAD
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
    RpcAgentTestFixture,
)
from torch.testing._internal.common_distributed import (
    tp_transports,
)
=======
from torch.testing._internal.common_distributed import tp_transports
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
    RpcAgentTestFixture,
)
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))


class TensorPipeRpcAgentTestFixture(RpcAgentTestFixture):
    @property
    def rpc_backend(self):
<<<<<<< HEAD
        return rpc.backend_registry.BackendType[
            "TENSORPIPE"
        ]
=======
        return rpc.backend_registry.BackendType["TENSORPIPE"]
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

    @property
    def rpc_backend_options(self):
        return rpc.backend_registry.construct_rpc_backend_options(
<<<<<<< HEAD
            self.rpc_backend,
            init_method=self.init_method,
            _transports=tp_transports()
=======
            self.rpc_backend, init_method=self.init_method, _transports=tp_transports()
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        )

    def get_shutdown_error_regex(self):
        # FIXME Once we consolidate the error messages returned by the
        # TensorPipe agent put some more specific regex here.
        error_regexes = [".*"]
        return "|".join([f"({error_str})" for error_str in error_regexes])

    def get_timeout_error_regex(self):
        return "RPC ran for more than"
