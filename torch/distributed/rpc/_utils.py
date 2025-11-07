# mypy: allow-untyped-defs
import logging
from contextlib import contextmanager
from typing import cast

<<<<<<< HEAD
from . import api, TensorPipeAgent

=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

logger = logging.getLogger(__name__)


@contextmanager
def _group_membership_management(store, name, is_join):
    token_key = "RpcGroupManagementToken"
    join_or_leave = "join" if is_join else "leave"
    my_token = f"Token_for_{name}_{join_or_leave}"
    while True:
        # Retrieve token from store to signal start of rank join/leave critical section
        returned = store.compare_set(token_key, "", my_token).decode()
        if returned == my_token:
            # Yield to the function this context manager wraps
            yield
            # Finished, now exit and release token
            # Update from store to signal end of rank join/leave critical section
            store.set(token_key, "")
            # Other will wait for this token to be set before they execute
            store.set(my_token, "Done")
            break
        else:
            # Store will wait for the token to be released
            try:
                store.wait([returned])
            except RuntimeError:
                logger.error(
                    "Group membership token %s timed out waiting for %s to be released.",
                    my_token,
                    returned,
                )
                raise


def _update_group_membership(worker_info, my_devices, reverse_device_map, is_join):
<<<<<<< HEAD
=======
    from . import api, TensorPipeAgent

>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    agent = cast(TensorPipeAgent, api._get_current_rpc_agent())
    ret = agent._update_group_membership(
        worker_info, my_devices, reverse_device_map, is_join
    )
    return ret
