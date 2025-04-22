Jax analysis, particularly reading the protobuf files, has been tested with tensorboard 2.19.0 and tensorboard-plugin-profile 2.19.0 and protobuf 5.29.2.
Other versions may not work

Analyze Jax computations including GEMM analysis
Run this with the xplane.pb or json.gz and jit_train_step.gfx942_gpu_after_optimizations.txt
```
from TraceLens.TraceLens import JaxAnalyses
import sys
import pandas as pd
filename_path = sys.argv[1]
averages, categorized, additional_events = JaxAnalyses.summarize_gpu_events(filename_path)
pd.set_option('display.max_rows', None)
print("Average utilization by type of kernel")
print(averages)
print("XLA computations (% for all GPUs)")
print(categorized)
print("Uncategorized XLA computations (% for all GPUs)")
print(additional_events)
if len(sys.argv)>2:
    print("GEMMs")
    print(JaxAnalyses.summarize_gpu_gemm_events(sys.argv[2]))
```

Standalone Jax GEMM analysis from protobuf (from profiler) or xla dump (jit_train_step.gfx942_gpu_after_optimizations.txt):
```
from TraceLens.TraceLens import JaxAnalyses
import sys
import pandas as pd
pd.set_option('display.max_rows', None)
print("GEMMs")
filename = sys.argv[1]
if filename.endswith("pb"):
    gemms = JaxAnalyses.summarize_gpu_gemm_events_from_pb(filename)
else:
    gemms = JaxAnalyses.summarize_gpu_gemm_events_from_xla(filename)
print(gemms)
```

Anylyze Jax communications
Run this with the xplane.pb or json.gz and  jit_train_step.gfx942_gpu_after_optimizations-buffer-assignment.txt
```
from TraceLens.TraceLens import JaxAnalyses
import sys
import pandas as pd
profile_path = sys.argv[1]
xla_path = sys.argv[2]
summarized_events = JaxAnalyses.summarize_gpu_communication_events(profile_path, xla_path)
for (df, bw_data, count_data, time_by_size, range_data) in filter(lambda x: len(x[0]) > 0, summarized_events.values()):
    print(f"Stats for {df['base_collective'][0]}")
    print("Bandwidth")
    print(bw_data)
    print("counts")
    print(count_data)
    print("buffer sizes")
    print(time_by_size)
    print("time_in_ranges")
    print(range_data)
```