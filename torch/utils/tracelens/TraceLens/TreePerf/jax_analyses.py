import glob
import math
import os
import pandas as pd
import re
import string
from typing import Callable
try:
    from enum import StrEnum
except ImportError:
    try:
        from backports.strenum import StrEnum
    # fallback for Python 3.10
    except ImportError:
        from strenum import StrEnum

from .gpu_event_analyser import GPUEventAnalyser, JaxGPUEventAnalyser
from ..PerfModel import perf_model
from ..util import TraceEventUtils, DataLoader, JaxProfileProcessor

class JaxAnalyses:
    # keywords for splitting jax events
    GemmKeys = ["Cijk", "gemm", "nvjet", "cublasLt"]
    FABwdKeys = ["FmhaBwd"]
    FAFwdKeys = ["FmhaFwd"]
    FAV3Keys = ["kernel_func"] # find a more precise way to do this
    ConvKeys = ["FillBuffer"]
    TEKeys = ["transformer_engine"]
    ClassCategories = {
        "GEMM": GemmKeys,
        "FA BWD": FABwdKeys,
        "FA FWD": FAFwdKeys,
        "FA V3": FAV3Keys,
        "Conv": ConvKeys,
        "TE": TEKeys,
    }
    UncategorizedEventKey = "Uncategorized Events"

    @staticmethod
    def breakdown_compute_events(event_list, group_by_gpu: bool = True, group_by_name = False):
        def add_event(cur_event_list, name, duration):
            current = cur_event_list.get(name, [0, 0])
            current[0] += 1
            current[1] += duration
            if current[0] == 1:
                cur_event_list[name] = current

        categorized_events = {}
        uncategorized_events = {}
        for compute_event in event_list:
            if group_by_gpu:
                gpu = int(compute_event['pid'])
                if gpu in categorized_events:
                    cur_categorized_list = categorized_events[gpu]
                    cur_uncategorized_list = uncategorized_events[gpu]
                else:
                    cur_categorized_list = {}
                    categorized_events[gpu] = cur_categorized_list
                    cur_uncategorized_list = {}
                    uncategorized_events[gpu] = cur_uncategorized_list
            else:
                cur_categorized_list = categorized_events
                cur_uncategorized_list = uncategorized_events

            name=compute_event[TraceEventUtils.TraceKeys.Name]
            duration=compute_event[TraceEventUtils.TraceKeys.Duration]
            found = False
            for category, filters in JaxAnalyses.ClassCategories.items():
                if any(f in name for f in filters):
                    add_event(cur_categorized_list, category, duration)
                    found = True
                    break
            if not found:
                if group_by_name:
                    name = name.rstrip(string.digits)
                add_event(cur_categorized_list, JaxAnalyses.UncategorizedEventKey, duration)
                add_event(cur_uncategorized_list, name, duration)

        return categorized_events, uncategorized_events

    @staticmethod
    def create_breakdown_df(events: dict, total_time, num_gpus: int = 8):
        df = pd.DataFrame.from_dict(events, orient='index', columns=['count', 'time'])
        # convert time to ms by div by 1e3
        df['time ms'] = df['time'] / 1e3
        df['time ms per gpu'] = df['time ms'] / num_gpus
        df['percent'] = df['time'] / total_time * 100
        df = df.drop(columns=['time'])
        df = df.sort_values("percent", ascending=False)
        return df

    @staticmethod
    def default_gpu_event_filter(event: dict):
        return event.get("tid", 200) < 100 # ignore of supplemental events

    @staticmethod
    def get_just_gpu_events(events):
        return dict(filter(lambda v: len(v[1].get(GPUEventAnalyser.computation_key, {})) > 0, events.items()))


    def create_gpu_summary(analyzer: JaxGPUEventAnalyser, group_kernels_by_name: bool = False):
        all_events = analyzer.get_gpu_event_lists(event_filter = JaxAnalyses.default_gpu_event_filter)

        # create an average across GPUs
        average_gpu_metrics = None
        num_gpus = 0
        for pid, cur_events in all_events.items():
            if pid <= 100:
                num_gpus += 1
                analyzer.verify_dict_gpu_event_lists(cur_events)
                current_metrics = analyzer.compute_metrics_dict(cur_events)
                if average_gpu_metrics is None:
                    average_gpu_metrics = current_metrics
                else:
                    for k, v in current_metrics.items():
                        average_gpu_metrics[k] += v
        for k in average_gpu_metrics.keys():
            average_gpu_metrics[k] /= num_gpus

        # find compute times
        just_gpu_events = JaxAnalyses.get_just_gpu_events(all_events)
        all_gpu_compute_events = [e for ge in just_gpu_events.values() for e in ge[GPUEventAnalyser.computation_key]]
        categorized_times, uncategorized_times = JaxAnalyses.breakdown_compute_events(all_gpu_compute_events,
                                                                           group_by_gpu = False,
                                                                           group_by_name = group_kernels_by_name)

        categorized_df = JaxAnalyses.create_breakdown_df(categorized_times, average_gpu_metrics["computation_time"] * num_gpus, num_gpus)
        uncategorized_df = JaxAnalyses.create_breakdown_df(uncategorized_times, categorized_times[JaxAnalyses.UncategorizedEventKey][1], num_gpus)
        return analyzer.get_breakdown_df_from_dict(average_gpu_metrics), categorized_df, uncategorized_df

    @staticmethod
    def summarize_gpu_events(filename, save_preprocessed = False):
        data = DataLoader.load_data(filename_path=filename, save_preprocessed=save_preprocessed)
        events = data['traceEvents']
        my_gpu_event_analyser = JaxGPUEventAnalyser(events)
        return JaxAnalyses.create_gpu_summary(my_gpu_event_analyser)

    communication_events_map={"all-gather-start":"all-gather", "all-reduce-start":"all-reduce", "reduce-scatter":"reduce-scatter", "collective-permute-start": "collective-permute"}

    # filename here is the "after-buffer-assignment" xla file
    @staticmethod
    def process_communication_events_from_xla_dump(xla_file_name: str) -> dict:
        communication_events={key:[] for key in JaxAnalyses.communication_events_map.keys()}

        event_key=str.join('|', JaxAnalyses.communication_events_map.keys())
        pattern = re.compile(rf"^.*value:.*({event_key})\.?([\d]+)?.*size=(\d+).*: ([a-zA-Z\d].*)\[.*$")
        with open(xla_file_name, "r") as f:
            for line in f:
                m=pattern.search(line)
                if m:
                    communication_events[m.group(1)].append([m.group(2), m.group(3), m.group(4)])
        return communication_events

    # filename here is the regular XLA file, not the "after-buffer-assignment" file
    @staticmethod
    def process_gemm_events_from_xla_dump(xla_file_name: str) -> dict:
        return JaxProfileProcessor.process_gemm_ops(JaxProfileProcessor.process_xla_file(xla_file_name))

    @staticmethod
    def process_gemm_events_from_pb(pb_file_name: str, module_name: str = "jit_train_step") -> dict:
        return JaxProfileProcessor.process_gemm_ops(
            JaxProfileProcessor.process_protobuf_file(pb_file_name, module_name))


    # this function only takes the minimum of each instance of the communication across all steps
    # ideally it would be nice to aggregate for each step instead, if we can find the step from the messsage
    @staticmethod
    def process_communication_events_from_profile(analyzer: JaxGPUEventAnalyser, messages: dict) -> dict:
        all_events = analyzer.get_gpu_event_lists(event_filter = JaxAnalyses.default_gpu_event_filter)
        just_gpu_events = JaxAnalyses.get_just_gpu_events(all_events)
        all_comm_events = [e for ge in just_gpu_events.values() for e in ge[GPUEventAnalyser.communication_key]]
        num_gpus = len(just_gpu_events)

        rccl_stats={}

        for i in all_comm_events:
            pid=i[TraceEventUtils.TraceKeys.PID]
            dur=i[TraceEventUtils.TraceKeys.Duration]
            op = i["args"]["hlo_op"]
            if op.startswith('reduce-scatter'):
                op = '.'.join(op.split('.')[:2]) # need to remove sub-communications from reduce-scatter only
            current = rccl_stats.get(op, [math.inf] * num_gpus)
            current[pid-1] = min(dur, current[pid-1])
            rccl_stats[op] = current


        #each dict is indexed by the hlo_op, and the value is a list [duration, total message size, number of tuple arguments,algbw]
        output = {}
        for msg_type, msg_values in messages.items():
            coll_dict={}
            output[JaxAnalyses.communication_events_map[msg_type]] = coll_dict
            for msg in msg_values:
                collname=f"{msg_type}.{msg[0]}" if msg[0] is not None else msg_type
                collsize=int(msg[1])
                collval = rccl_stats.get(collname, None)
                if (collval is not None):
                    current = coll_dict.get(collname, [min(collval),0,0,0])
                    current[1] += collsize
                    current[2] += 1
                    coll_dict[collname] = current
                else:
                    print(collname," not found")
            scale = num_gpus if "reduce-scatter" in msg_type else 1
            for collname, current in coll_dict.items():
                current[3]=current[1]*scale*0.001/current[0]

        return output

    @staticmethod
    def summarize_communication_data(comm_event_data):
        summary_data = {}
        for collective, collective_stats in comm_event_data.items():
            current_data = [[collective, xfer_name, data[0], data[1] / 1024, data[3]]
                            for xfer_name, data in collective_stats.items()]
            df = pd.DataFrame(data=current_data,
                            columns = [
                                "base_collective",
                                "collective_name",
                                "latency_us",
                                "buffer_size_kb",
                                "effective_bw" ])

            bandwidth_stats = (
                df.groupby(["base_collective", "buffer_size_kb"])["effective_bw"]
                .agg(["mean", "std"])
                .reset_index()
            )

            # Group by collective type and buffer size for call counts
            call_counts = (
                df.groupby(["base_collective", "buffer_size_kb"])
                .size()
                .reset_index(name="count")
            )

            bw_data = bandwidth_stats.sort_values("buffer_size_kb")
            count_data = call_counts[
                call_counts["base_collective"] == collective
            ].sort_values("buffer_size_kb")

            time_by_size = (
                df.groupby("buffer_size_kb")["latency_us"].sum().reset_index()
            )
            total_time_us = time_by_size["latency_us"].sum()
            time_by_size["percentage"] = (time_by_size["latency_us"] / total_time_us) * 100 if total_time_us > 0 else 0


            # Calculate time spent in each bandwidth range
            bw_thresholds = [0, 50, 100, 200, 300, 400]
            total_time = (df["latency_us"].sum()) / 1e6  # Convert to seconds
            time_in_ranges = []
            labels = []

            for i in range(len(bw_thresholds) - 1):
                mask = (df["effective_bw"] >= bw_thresholds[i]) & (
                    df["effective_bw"] < bw_thresholds[i + 1]
                )
                time_in_range = (df[mask]["latency_us"].sum()) / 1e6
                percentage = (time_in_range / total_time) * 100 if total_time > 0 else 0
                time_in_ranges.append(percentage)
                labels.append(f"{bw_thresholds[i]}-{bw_thresholds[i+1]} GB/s")

            # Add the highest range
            mask = df["effective_bw"] >= bw_thresholds[-1]
            time_in_range = (df[mask]["latency_us"].sum()) / 1e6
            percentage = (time_in_range / total_time) * 100 if total_time > 0 else 0
            time_in_ranges.append(percentage)
            range_data=pd.DataFrame(zip(labels, time_in_ranges), columns=("Bandwidth range", "Percentage of time"))

            summary_data[collective]=(df, bw_data, count_data, time_by_size, range_data)
        return summary_data


    @staticmethod
    def summarize_gpu_communication_events(profile_filename, xla_filename):
        # summarizes communication events from a single step
        data = DataLoader.load_data(profile_filename)
        events = data['traceEvents']
        my_gpu_event_analyser = JaxGPUEventAnalyser(events)
        comm_xla_events = JaxAnalyses.process_communication_events_from_xla_dump(xla_filename)
        processed = JaxAnalyses.process_communication_events_from_profile(my_gpu_event_analyser, comm_xla_events)
        return JaxAnalyses.summarize_communication_data(processed)

    @staticmethod
    def summarize_gpu_gemm_events_from_xla(xla_filename):
        gemms = JaxAnalyses.process_gemm_events_from_xla_dump(xla_filename)
        return pd.DataFrame.from_dict(gemms, orient='index',  columns = JaxProfileProcessor.gemm_columns)

    @staticmethod
    def summarize_gpu_gemm_events_from_pb(pb_filename, module_name: str = "jit_train_step"):
        gemms = JaxAnalyses.process_gemm_events_from_pb(pb_filename, module_name)
        return pd.DataFrame.from_dict(gemms, orient='index',  columns = JaxProfileProcessor.gemm_columns)

    @staticmethod
    def gemm_performance_from_pb(pb_file_name, module_name: str = "jit_train_step", arch: dict = None):
        all_profile_events = DataLoader.load_data(filename_path=pb_file_name)["traceEvents"]
        metadata = TraceEventUtils.get_metadata(all_profile_events)
        events = TraceEventUtils.split_events_by_pid_tid(TraceEventUtils.non_metadata_events(all_profile_events))
        if (module_name is None):
            # extract the first module name from the "XLA Modules:" thread
            xla_module_thread = TraceEventUtils.find_thread_by_item_in_metadata(
                metadata[1],
                lambda x: x[0] is not None and x[1][TraceEventUtils.MetadataFields.ThreadName] == TraceEventUtils.JaxSpecialThreads.XlaModules)
            module_name = events[1][xla_module_thread][0][TraceEventUtils.TraceKeys.Name].split("(")[0]
        hlo_ops = JaxProfileProcessor.process_protobuf_file(pb_file_name, module_name)
        gemm_ops = JaxProfileProcessor.process_gemm_ops(hlo_ops)
        # the keys in gemm ops start with % while the keys in the events don't - strip out the %
        gemm_ops = dict((x[0][1:], x[1]) for x in gemm_ops.items())
        main_thread_id = TraceEventUtils.find_thread_by_item_in_metadata(
            metadata[1],
            lambda x: x[0] is not None and x[1][TraceEventUtils.MetadataFields.ThreadName].startswith(TraceEventUtils.JaxSpecialThreads.StreamPrefix))
        main_thread_events = events[1][main_thread_id]
        main_thread_gemms = filter(lambda x: TraceEventUtils.TraceKeys.Args in x and x[TraceEventUtils.TraceKeys.Args][JaxAnalyses.JaxKernelEventArgs.hlo_op] in gemm_ops, main_thread_events)
        metrics = [JaxAnalyses.gemm_perf_metrics(event, gemm_ops[event[TraceEventUtils.TraceKeys.Args][JaxAnalyses.JaxKernelEventArgs.hlo_op]], False, arch) for event in main_thread_gemms]
        return pd.DataFrame(metrics)

    class JaxGemm(perf_model.GEMM):
        @staticmethod
        def get_param_details(event):
            hlo_args = event[JaxAnalyses.JaxKernelEventArgs.hlo_op]
            dict_dtype2gemmologist = {
                'f32': 'fp32',
                'f16': 'fp16',
                'bf16': 'bf16',
                'f8': 'fp8',
                'fp8': 'fp8',
            }
            return {
                "M": hlo_args["M"],
                "N": hlo_args["N"],
                "K": hlo_args["K"],
                "bias": hlo_args["Beta"] != 0,
                "stride_A": None,
                "stride_B": None,
                "dtype_A_B": (hlo_args["Type"], hlo_args["Type"]),
                "Op B": hlo_args["Batch"],
                "gemmologist_dtype": dict_dtype2gemmologist.get(hlo_args["Type"])
            }

        def flops(self):
            """Total FLOPs for the entire batch."""
            return self.param_details["Op B"] * super().flops()

        def bytes(self):
            size_map = {
                "f32": 4,
                "f16": 2,
                "bf16": 2,
                "f8": 1,
                "fp8": 1,
            }
            dtype_A_B = self.param_details['dtype_A_B']
            bpe = size_map[dtype_A_B[0]]
            per_batch = super().bytes(bpe_mat1=bpe, bpe_mat2=bpe,
                                   bpe_bias=bpe,   # not used, but keeps call signature
                                   bpe_output=bpe)
            return None if per_batch is None else self.param_details['Op B'] * per_batch
        def flops_bwd(self):
            raise NotImplementedError("Backward pass for JaxGemm is not defined.")
        def bytes_bwd(self, _):
            raise NotImplementedError("Backward pass for JaxGemm is not defined.")


    @staticmethod
    def get_perf_model(event: dict):
        name = event[TraceEventUtils.TraceKeys.Name]
        if any(f in name for f in JaxAnalyses.GemmKeys):
            return JaxAnalyses.JaxGemm
        return None

    @staticmethod
    def gemm_perf_metrics(event, op_params, bwd: bool = False, arch = None):
        perf_model_class = JaxAnalyses.get_perf_model(event)
        # the class structure of the perf_model class doesn't make it easy to add additional parameters to the event,
        # so make a copy of the event with the hlo op info inside it
        event_copy = dict(event)
        event_copy[JaxAnalyses.JaxKernelEventArgs.hlo_op] = op_params
        # the perf model needs a kernel names field
        event_copy["kernel_names"] = [event[TraceEventUtils.TraceKeys.Name]]
        perf_model = perf_model_class(event_copy, arch=arch)

        gflops = (perf_model.flops() if not bwd else perf_model.flops_bwd())/ 1e9
        time = event[TraceEventUtils.TraceKeys.Duration]

        tflops_per_s = (gflops / 1e3) / (time / 1e6) if time > 0 else float('nan')

        bytes_moved = perf_model.bytes() if not bwd else perf_model.bytes_bwd()

        # Return metrics
        dict_metrics = {
            'GFLOPS': gflops,
            'Kernel Time (µs)': time,
            'TFLOPS/s': tflops_per_s,
        }
        if bytes_moved is not None:
            dict_metrics['Data Moved (MB)'] = bytes_moved / (1024 * 1024)
            dict_metrics['FLOPS/Byte'] = (gflops * 1e9) / bytes_moved if bytes_moved > 0 else float('nan')
            dict_metrics['TB/s'] = (bytes_moved / 1e12) / (time / 1e6) if time > 0 else float('nan')
        else:
            dict_metrics['Data Moved (MB)'] = float('nan')
            dict_metrics['FLOPS/Byte'] = float('nan')
            dict_metrics['TB/s'] = float('nan')

        if hasattr(perf_model, "gemmologist_time"):
            dict_metrics['Gemmologist Time (µs)'] = perf_model.gemmologist_time
            dict_metrics['Gemmologist TFLOPS/s'] = (gflops / 1e3) / (perf_model.gemmologist_time / 1e6) if perf_model.gemmologist_time > 0 else float('nan')
            # dict_metrics['Gemmologist cmd'] = perf_model.gemmologist_cmd

        for key, value in perf_model.param_details.items():
            dict_metrics[f"param: {key}"] = value

        return dict_metrics



