## Understanding PyTorch Trace: A Deep Dive into Execution and Analysis

You followed the [PyTorch profiling guide](../conceptual/torch_profiling_guide.ipynb), collected a trace, and even opened it in [Perfetto](https://ui.perfetto.dev/). But what does that trace actually mean, and how do you read insights from it?  
That’s exactly what we’ll unpack here.  

When you run a PyTorch model on a GPU, there’s a hidden dance between the CPU (host) and the GPU (accelerator). PyTorch trace lets you peek into this choreography—revealing how your code executes, where time is spent, and what might be slowing you down.  

This blog is the first in a series—here we’ll focus on **understanding single-GPU traces**. In a follow-up, we’ll move on to **multi-GPU profiling**.


### The Execution Model: Host, GPU, and the Async Magic  
![Async Execution Example](./imgs/profiling_analysis_fig1.png)
To understand profiling, you need a mental model of how PyTorch (or any application) runs code across the CPU and GPU.  
**Figure 1**: Async Execution Example

The **host (CPU)** is the conductor. It manages processes and threads, moving through the application call stack until it issues GPU runtime or driver API calls such as `cudaLaunchKernel` or `hipLaunchKernel`. The **GPU** is the performer. It executes tasks with massive parallelism, often involving thousands of threads working on matrix multiplications, convolutions, or reductions.  

Most of these commands are asynchronous. The host queues the "commands" (kernels) and immediately continues, while the GPU independently drains its queue. This overlap is what enables high performance. Sometimes, however, the host introduces synchronization points such as `cudaDeviceSynchronize` or `hipDeviceSynchronize`. In those cases the CPU waits until the GPU finishes all queued work. Synchronization is necessary for correctness, but too many syncs can quickly turn into bottlenecks.  

One more twist: host events come with detailed call stacks (Python → C++ → runtime), so you can see exactly how an operation was triggered. GPU events, on the other hand, are flat. A GPU kernel cannot call another GPU kernel on its own—the launch must always come from the host. That’s why GPU traces appear as a simple queue of kernels, memcopies, and memsets with no stack information.  

A **kernel** itself is just a GPU function that runs across thousands of lightweight threads in parallel. In the trace, each kernel shows up as a single event: you can see when it started, how long it ran, and on which stream—but not what happened inside. To peer into that level of detail (thread execution, memory transactions, warp occupancy), you need dedicated kernel profilers such as [rocprof-compute](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/) for AMD GPUs or [Nsight Compute](https://developer.nvidia.com/nsight-compute) for NVIDIA GPUs. These tools complement the trace view by showing what happens inside the GPU during a kernel event.  




---

### A Simple Example

Let’s dig deeper into what a trace actually shows. Below is a snippet from our trace.

![Perfetto trace showing idle time between kernels](./imgs/profiling_analysis_fig2.png)  
**Figure 2**: Perfetto trace highlighting idle time between convolution and batch normalization kernels.

The red dashed region marks idle time: the GPU is waiting for the host to issue the next command. This happens when the CPU frontend cannot keep up with the GPU’s execution speed. Reducing such idle gaps is an important optimization goal. To see why these gaps occur, let’s follow the path from a high-level PyTorch call down to the GPU kernels that actually run.

---

Take a simple example:

    y = nn.Conv2d(3, 64, kernel_size=3)(x)

At the Python level, this looks like a single operation. In the trace, however, it expands into several layers:  
- **Python frontend**: `nn.Conv2d` in `torch.nn`  
- **ATen**: the call lowers into PyTorch’s tensor library, [ATen](https://pytorch.org/cppdocs/notes/aten.html), and shows up as `aten::conv2d` in the trace  
- **Backend wrapper**: ATen provides wrappers that call into vendor libraries. On AMD GPUs, you’ll see `aten::miopen_convolution`, which wraps [MIOpen](https://rocm.docs.amd.com/projects/MIOpen/en/latest/) commands. On NVIDIA GPUs, the equivalent is `aten::cudnn_convolution`, which wraps [cuDNN](https://developer.nvidia.com/cudnn) calls  
- **GPU kernels**: the backend library enqueues device kernels such as `igemm_fwd_gtcx2_nhwc` that perform the actual convolution on the GPU  

This gives us a clear understanding of how high-level code is translated into GPU execution.  


---
Another important detail is **tensor metadata**. Python-level operations in the trace do not carry shape information. But if you enable `record_shapes=True` in the profiler, the "cpu_op" category (like `aten::convolution`) include **input shapes, strides, and dtypes**.  

For example, in the figure below the first tensor is the activation (`[5, 64, 56, 56]`) and the second tensor is the convolution filter (`[64, 64, 3, 3]`).  
![Perfetto trace showing recorded input shapes](./imgs/profiling_analysis_fig3.png)
*Figure 3: Shapes recorded on the backend op when `record_shapes=True`.*  

This metadata becomes very useful for deeper analysis and debugging—for instance, when matching trace events to model architecture, analyzing kernel efficiency, or spotting unusual stride/dtype patterns.

---

That’s all you need for a clean first pass: know what the host and GPU are doing, read the timeline, and use recorded shapes to ground what you see. If you want UI shortcuts, how memcpys show up, or the structure of the raw trace file, jump to the appendix below.

**Your Profiling Journey Begins**: Perfetto view give you the raw signals; turning them into insight is a skill you’ll build with practice. And TraceLens is here to turbocharge your journey.

---
### Appendix

#### Perfetto UI Tips

Perfetto is a powerful trace viewer, but it takes some practice to navigate effectively. A few basics:  

- **Zoom and pan** with `Ctrl + scroll` to zoom in/out; click and drag to pan.  
- **Event details**: click any event to open the *Current Selection* panel below. This shows start time, duration, and arguments. With `record_shapes=True`, backend ops (in the `cpu_op` category) also show tensor shapes, dtypes, and strides.  

Perfetto also links host launches and GPU execution with arrows called *flows*. These are crucial for connecting what you see on the CPU timeline with what actually runs on the GPU.  

When you select a runtime launch event such as `hipExtModuleLaunchKernel`, the **Following Flow** jumps you forward to the GPU kernel it triggered:  

![Following flow from host launch to GPU kernel](./imgs/profiling_analysis_fig4.png)  
*Figure 4: Following flow from a host launch (`hipExtModuleLaunchKernel`) to the corresponding GPU kernel event.*  

Conversely, when you select a GPU kernel event, the **Preceding Flow** takes you back to the runtime call on the host that launched it:  

![Preceding flow from GPU kernel back to host launch](./imgs/profiling_analysis_fig5.png)  
*Figure 5: Preceding flow from a GPU kernel (`SubTensorOpWithScalar1d`) back to its launch on the host.*  

These flows are the bridge between the Python-level trace and the GPU execution timeline. They let you answer both:  
- *Which Python/ATen op caused this kernel to run?*  
- *What GPU work did this runtime call produce?*  



---

#### Memory Copy Events
Not all GPU activity is compute. Profiling traces also show **memory transfers**:  

- **H2D (Host to Device)**: copies data from CPU to GPU, usually synchronous and PCIe bandwidth-limited.  
- **D2H (Device to Host)**: copies results back to CPU, also synchronous and PCIe bandwidth-limited.  
- **D2D (Device to Device)**: moves data between GPU buffers, asynchronous and limited by HBM bandwidth.  

Recent versions even record measured bandwidth for these events in the trace arguments.  
Importantly, memcpy events use the GPU’s DMA engines, not compute cores—so they don’t directly compete with kernel execution.

---

#### JSON Format
Behind the Perfetto UI, the PyTorch profiler saves traces as JSON. Each entry is an event dictionary with:  

- **Timestamps**: `ts` (start), `dur` (duration).  
- **Process/thread IDs**: `pid`, `tid`. CPU events use real PIDs/TIDs; GPU events use pseudo-PIDs for devices and TIDs for streams.  
- **Category**: e.g. `python_function`, `cpu_op`, `cuda_runtime`, `kernel`, `gpu_memcpy`.  
- **Args**: extra info such as shapes, dtypes, strides, or bandwidth.  

The JSON format makes traces scriptable: you can parse them to build custom reports or run automated analysis outside Perfetto. This is what exactly TraceLens does.


#### Autograd in the Trace

Autograd introduces another dimension to the trace: the forward and backward passes are captured on **different threads within the same process**. In the timeline, you’ll typically see:  
- `python3.x [main thread]` for the forward pass,  
- `pt_autograd_0` (or similar) for the backward pass.  

These are linked at the `aten::convolution` layer of the call stack. Perfetto uses flows to connect the forward convolution op to its corresponding backward node.  

![Forward and backward convolution linked in the trace](./imgs/profiling_analysis_fig6.png)  
*Figure 6: Example of `aten::convolution` (forward) linked to `ConvolutionBackward0` (backward) via flows.*  

Key points:  
- The linkage happens at **`aten::convolution`**, not `aten::conv2d`. The higher-level `aten::conv2d` call funnels into `aten::convolution` before reaching the backend.  
- Forward and backward both eventually call into the same GPU backend (MIOpen/cuDNN), so you’ll see similar kernel launches in both directions.  
- The backward pass typically runs more kernels, since it must compute gradients for both activations and weights, making it more expensive than forward.  
- Because autograd runs on its own thread, you can easily separate model execution (forward path) from gradient computation (backward path) when analyzing traces.  

Together with the flows we discussed earlier, this lets you connect *a forward convolution op* to its *exact backward counterpart* and then follow the chain down to GPU kernels.
