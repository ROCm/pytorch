# mypy: allow-untyped-defs
<<<<<<< HEAD
import torchvision

=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
import torch
from torch.distributed._tools import MemoryTracker


<<<<<<< HEAD
def run_one_model(net: torch.nn.Module, input: torch.Tensor):
    net.cuda()
    input = input.cuda()
=======
def run_one_model(net: torch.nn.Module, input: torch.Tensor, device: str = "cuda"):
    net.to(device)
    input = input.to(device)
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

    # Create the memory Tracker
    mem_tracker = MemoryTracker()
    # start_monitor before the training iteration starts
    mem_tracker.start_monitor(net)

    # run one training iteration
    net.zero_grad(True)
    loss = net(input)
    if isinstance(loss, dict):
        loss = loss["out"]
    loss.sum().backward()
    net.zero_grad(set_to_none=True)

    # stop monitoring after the training iteration ends
    mem_tracker.stop()
    # print the memory stats summary
    mem_tracker.summary()
    # plot the memory traces at operator level
    mem_tracker.show_traces()


<<<<<<< HEAD
run_one_model(torchvision.models.resnet34(), torch.rand(32, 3, 224, 224, device="cuda"))
=======
if __name__ == "__main__":
    import torchvision

    dev = "cuda"
    run_one_model(
        torchvision.models.resnet34(),
        torch.rand(32, 3, 224, 224, device=dev),
        device=dev,
    )
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
