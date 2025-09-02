import itertools
import json
import os
import re
import glob

try:
    from enum import StrEnum
except ImportError:
    try:
        from backports.strenum import StrEnum
    # fallback for Python 3.10
    except ImportError:
        from strenum import StrEnum
from typing import List, Dict, Callable, Tuple

# generic data loader class for json, json.gz, or tensorboard pb files
# tensorboard pb files are useful for Jax in particular because the json.gz traces produced by jax can have incorrect timestamps and missing information
class DataLoader:
    @staticmethod
    def load_data(filename_path:str, save_preprocessed: bool = False) -> dict:
        if filename_path.endswith('pb'):
            from tensorboard_plugin_profile.convert import raw_to_tool_data as convert
            data, _ = convert.xspace_to_tool_data([filename_path], "trace_viewer@^", {})
            data = data.decode("utf-8") # we get bytes back from the call above
        elif filename_path.endswith('json.gz'):
            import gzip
            with gzip.open(filename_path, 'r') as fin:
                data = fin.read().decode('utf-8')
        elif filename_path.endswith('json'):
            with open(filename_path, 'r') as fin:
                data = fin.read()
        else:
            raise ValueError("Unknown file type",filename_path)
        if (save_preprocessed):
            with open(filename_path.replace("pb", "processed.json"), 'w') as writefile:
                writefile.write(data)
        return json.loads(data)
    
class JaxProfileProcessor:
    gemm_columns = ["Batch", "M", "N", "K", "Beta", "Type"]

    @staticmethod
    def process_xla_file(xla_file_name):
        hlo_ops={}
        with open(xla_file_name, "r") as f:
            for line in f:
                JaxProfileProcessor.process_line(hlo_ops, line)
        return hlo_ops

    @staticmethod
    def process_protobuf_file(protobuf_file_name, module_name):
        from tensorboard_plugin_profile.convert import raw_to_tool_data as convert
        # look to see if the protobuf file has already been extracted
        dir_name = os.path.dirname(protobuf_file_name) + "/"
        hlo_filename = glob.glob(dir_name + os.path.sep + module_name + "*hlo_proto.pb")
        if len(hlo_filename) != 1:
            convert.xspace_to_tool_names([protobuf_file_name])
        hlo_filename = glob.glob(dir_name + os.path.sep + module_name + "*hlo_proto.pb")
        #assert len(hlo_filename) == 0
        if len(hlo_filename) > 1:
            print('Multiple matching hlo_filenames:')
            print(hlo_filename)
        elif len(hlo_filename) == 0:
            print('No matching hlo_filenames:')
            print(hlo_filename)

        # need to make sure that the pb exists and get the numerical suffix into the module name
        # and remove '.hlo_proto.pb'
        module_name = os.path.splitext(os.path.splitext(os.path.basename(hlo_filename[0]))[0])[0]

        hlo_ops={}
        graph_viewer_options= {
            'node_name': "",
            'module_name': module_name,
            'graph_width': 2,
            'show_metadata': True,
            'merge_fusion': True,
            'type': "long_txt"
        }
        params = {'graph_viewer_options': graph_viewer_options }
        data, _ = convert.xspace_to_tool_data(
                [dir_name], "graph_viewer^", params)
        data = data.decode("utf-8").split('\n')
        for line in data:
            JaxProfileProcessor.process_line(hlo_ops, line)
        return hlo_ops

    @staticmethod
    def process_line(hlo_ops: dict, line: str):
        line_processed=line.strip()
        if (("metadata" in line_processed and not(re.search(r"\)$",line_processed)) and not(line_processed.startswith("ROOT")))
            or any(t in line_processed for t in ["get-tuple-element", "bf16", "f8", "f16", "f32", "f64"])
            and not(line_processed.startswith("HloModule "))):
            k,v=JaxProfileProcessor.get_dict(hlo_ops, line_processed)
            hlo_ops[k]=v
            return True
        return False

    @staticmethod
    def get_operands(operands):
        operands=re.sub(r'^.*?\(', '', operands)
        operands=re.sub(r'\).*?$', '', operands)
        operands_m=re.findall(r"[bfs][0-9\[\]\{,a-z]*}",operands)
        if operands_m:
            return operands_m
        return operands.split(",")

    @staticmethod
    def get_dict(hlo_ops: dict, line):
        dict_line={}
        line=re.sub(r"\),",")",line)
        line=re.sub(r", ",",",line)
        line=re.sub(r" %","%",line)
        backend_config=re.search(r"backend_config=\{[a-zA-Z_=\"\(\)\/0-9\ @.-:,\[\]\{\}]*",line)
        metadata=re.search(r"metadata=\{[a-zA-Z_=\"\(\)\/0-9\ @.-]*",line)
        custom_call_target=re.search(r"custom_call_target=\"[a-zA-Z_=\"\(\)\/0-9\ @.\-\$]*",line)
        line=line.split(" ")
        key=line[0]
        dict_line["output"]=line[2]
        dict_line["operands"] = operands = JaxProfileProcessor.get_operands(line[3])
        dict_line["computation"]="rest"
        if metadata is not None:
            dict_line["metadata"]=metadata[0]
            if backend_config is not None:
                dict_line["backend_config"]=backend_config[0]
            if custom_call_target is not None:
                gemm_keys = ["matmul", "cublas"]
                dict_line["custom_call_target"]=custom_call_target[0]
                if any(k in dict_line["custom_call_target"] for k in gemm_keys):
                    if "f8" in str(custom_call_target[0]):
                        dict_line["type"]="fp8"
                        dict_line["computation"]="gemm"
                    else:
                        # use the input type to determine the GEMM type
                        gemm_type = JaxProfileProcessor.get_operand_type(hlo_ops, operands[0])
                        if not all(JaxProfileProcessor.get_operand_type(hlo_ops, o) == gemm_type for o in operands[1:]):
                            raise Exception("Input operand type mismatch", line)
                        dict_line["type"]=gemm_type
                        dict_line["computation"]="gemm"
        return (key,dict_line)
    @staticmethod
    def get_operand_type(hlo_ops: dict, operand : str) -> str:
        dtypes = ["bf16", "f16", "f32", "f8", "fp8"]
        # if the operand is a slice of something else, then the type might be at the beginning of the operand name
        for t in dtypes:
            if operand.startswith(t):
                return t
        # otherwise look it up
        output = hlo_ops[operand]["output"]
        for t in dtypes:
            if output.startswith(t):
                return t
        return None

    @staticmethod
    def process_gemm_ops(hlo_ops: dict):
        def get_sizes(str_size):
            match=(re.search(r".*\[(.*)\]",str_size))
            if match is not None:
                m=match.group(1)
                s=m.split(",")
                if len(s)>3:
                    raise ValueError("tensor size is more than 3?",str_size)
                return s

            else:
                raise ValueError(str_size)
        dtypes=["bf16", "f16", "f32", "f8", "fp8"]
        gemm_dict={}
        for opname,op in hlo_ops.items():
            if "gemm" in op["computation"].lower():
                if "backend_config" not in op:
                    raise ValueError("Gemm backend config information mnissing!", op)
                backend_config=op["backend_config"]
                beta=re.search(r"\"beta\":[01],",backend_config)[0].split(":")[1].split(",")[0]
                lhs_dim=re.search(r"\"lhs_contracting_dimensions\":\[[\"012]*\]",backend_config)[0].split(":")[1].split("\"")[1]
                rhs_dim=re.search(r"\"rhs_contracting_dimensions\":\[[\"012]*\]",backend_config)[0].split(":")[1].split("\"")[1]
                outputs = op["output"]
                if outputs.startswith("("):
                    if not outputs.endswith(")"):
                        raise ValueError("Mistmatched parens in outputs in ",outputs)
                    output_list = outputs[1:-2].split("},")
                    # this code assumes that the first output is the one we care about
                    # we should be able to make this an RE
                    sizes_string=[[i, d] for i in output_list for d in dtypes if i.startswith(d)]
                    if len(sizes_string) != 1:
                        raise ValueError("Did not find wide output ",op)
                    sizes_string = sizes_string[0]
                    sizes_string[0] = sizes_string[0] + "}" # restore the } that was removed
                else:
                    sizes_string = outputs

                operand_list=[]
                for opid in op["operands"]:
                    if ("[" in opid and "]" in opid):
                        # pb format, shapes in operand list
                        operand_list.append(opid)
                    else:
                        output = hlo_ops[opid]["output"]
                        if any(output.startswith(d) for d in dtypes + ["f8"]) and not output.endswith("[]"):
                            operand_list.append(hlo_ops[opid]["output"])
                if int(beta)==1 and len(operand_list)<3:
                    print("Bias is set, however on;y two operands found!",op)
                if len(operand_list)>3 or len(operand_list) == 0:
                    raise ValueError("Invalid operand list",op,operand_list)
                c_order=re.search(r"\{[012,]*",sizes_string[0])[0].split("{")[1]
                c=get_sizes(sizes_string[0])
                a=get_sizes(operand_list[0])
                b=get_sizes(operand_list[1])
                batch=1
                if a[int(lhs_dim)]!=b[int(rhs_dim)]:
                    raise ValueError("contracting dimension not matching",backend_config)
                k=a[int(lhs_dim)]
                a.remove(k)
                b.remove(k)
                if len(c)>2:
                    batch=c[0]
                    a.remove(batch)
                    b.remove(batch)
                if "0,1" in c_order:
                    n=b[0] if len(b) > 0 else 1
                    m=a[0] if len(a) > 0 else 1
                else:
                    n=a[0] if len(a) > 0 else 1
                    m=b[0] if len(b) > 0 else 1
                gemm_dict[opname]={
                    "Batch": int(batch),
                    "M": int(m),
                    "N": int(n),
                    "K": int(k),
                    "Beta": int(beta),
                    "Type": op["type"],
                    "Computation": "gemm",
                }
        return gemm_dict
    
# Trace event utilities to help with traces in the Google Trace Event format
# https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview?tab=t.0
# This trace event format includes both Pytorch and Jax traces (and anything that can be viewed in Perfetto)
class TraceEventUtils:
    class TraceKeys(StrEnum):
        PID       = 'pid'
        TID       = 'tid'
        Phase     = 'ph'
        Args      = 'args'
        Name      = 'name'
        TimeStamp = 'ts'
        Duration  = 'dur'
        Category  = 'cat'
        TimeEnd   = 't_end'
        UID       = 'UID'

    class TracePhases(StrEnum):
        DurationBegin = 'B'
        DurationEnd   = 'E'
        Complete      = 'X'
        Counter       = 'C'
        Sample        = 'P'
        Metadata      = 'M'

    class MetadataFields(StrEnum):
        ProcessName   = 'process_name'
        ProcessLabels = 'process_labels'
        ProcessSort   = 'process_sort_index'
        ThreadName    = 'thread_name'
        ThreadSort    = 'thread_sort_index'

    class ArgNames(StrEnum):
        Name        = 'name'
        SortIndex   = 'sort_index'
        StreamIndex = 'stream_index'
        Labels      = 'labels'

    class GpuEventCategories(StrEnum):
        Kernel = 'kernel'
        MemSet = 'gpu_memset'
        MemCpy = 'gpu_memcpy'

    class CpuEventCategories(StrEnum):
        Kernel  = 'cpu_op'
        Runtime = 'cuda_runtime'
        Driver  = 'cuda_driver'

    class JaxSpecialThreads(StrEnum):
        FrameworkCallStack = "Framework Name Scope"
        FrameworkOps       = "Framework Ops"
        XlaModules         = "XLA Modules"
        XlaOps             = "XLA Ops"
        pyXla             = 'py_xla'
        SourceCode         = "Source Code"
        Steps              = "Steps"
        StreamPrefix       = "Stream #"

    class JaxKernelEventArgs(StrEnum):
        hlo_module     = "hlo module"
        hlo_op         = "hlo_op"
        name           = "name" # name hierarchy, not always the same as the stack we see in framework ops
        correlation_id = "correlation_id" # can link to CPU threads
        group_id       = "group_id"


    @staticmethod
    def split_by_field(events: List[dict], field: str, defaultKey: str = None) -> Dict[str, List]:
        return dict(itertools.groupby(events, lambda event: event.get(field, defaultKey)))

    # Merges metadata events into a dictionary hierarchy per process
    # Process
    # None: {process_name, process_sort_index}
    # Thread_id: {thread_name, thread_sort_index} for each Thread_id
    @staticmethod
    def get_metadata(events: List[dict]) -> Dict[str, Dict[str, str]]:
        def get_metadata_val(x: dict) -> str:
            arg_labels = {
                TraceEventUtils.MetadataFields.ProcessName: TraceEventUtils.ArgNames.Name,
                TraceEventUtils.MetadataFields.ProcessLabels: TraceEventUtils.ArgNames.Labels,
                TraceEventUtils.MetadataFields.ProcessSort: TraceEventUtils.ArgNames.SortIndex,
                TraceEventUtils.MetadataFields.ThreadName: TraceEventUtils.ArgNames.Name,
                TraceEventUtils.MetadataFields.ThreadSort: TraceEventUtils.ArgNames.SortIndex,
            }
            key = x[TraceEventUtils.TraceKeys.Name]
            return (key, x[TraceEventUtils.TraceKeys.Args][arg_labels[key]])
        metadata_fields = itertools.takewhile(lambda x: x[TraceEventUtils.TraceKeys.Phase] == TraceEventUtils.TracePhases.Metadata, events)
        by_process = itertools.groupby(metadata_fields, lambda event: event[TraceEventUtils.TraceKeys.PID])
        # TID is not required for process-specific tags, so use null thread id for them
        fully_processed = map(lambda kv: (kv[0], itertools.groupby(kv[1], lambda event: event.get(TraceEventUtils.TraceKeys.TID))), by_process)
        return dict(map(lambda kv: (kv[0], dict(map(lambda kv1: (kv1[0], dict(map(lambda event: (get_metadata_val(event)), kv1[1]))), kv[1]))), fully_processed))

    @staticmethod
    def non_metadata_events(events:List[dict]) -> List[dict]:
        return list(itertools.dropwhile(lambda e: e[TraceEventUtils.TraceKeys.Phase] == TraceEventUtils.TracePhases.Metadata, events))

    @staticmethod
    def default_categorizer(event: dict) -> str:
        return event.get(TraceEventUtils.TraceKeys.Category)
    
    # TODO separate util class for Jax 
    # returns a curried function to categorizes events based on the
    # metadata extracted from the events list
    @staticmethod
    def prepare_event_categorizer(events: list[dict]) -> Callable[[dict], str]:
        metadata = TraceEventUtils.get_metadata(events)
        return lambda event: TraceEventUtils.get_event_category(metadata, event)
    
    # TODO separate util class for Jax 
    @staticmethod
    def get_event_category(metadata: dict, event: dict):
        if event.get(TraceEventUtils.TraceKeys.Phase == TraceEventUtils.TracePhases.Metadata):
            return "metadata"
        elif (TraceEventUtils.TraceKeys.PID in event and TraceEventUtils.TraceKeys.TID in event):
            pid = event[TraceEventUtils.TraceKeys.PID]
            tid = event[TraceEventUtils.TraceKeys.TID]
            ThreadName = metadata[pid][tid][TraceEventUtils.MetadataFields.ThreadName]
            if ThreadName == TraceEventUtils.JaxSpecialThreads.FrameworkCallStack:
                return "cpu_op"
            elif TraceEventUtils.JaxSpecialThreads.pyXla in ThreadName:
                return "cpu_op"
            elif ThreadName == TraceEventUtils.JaxSpecialThreads.XlaOps:
                return "python function"
            elif ThreadName.startswith("Stream"):
                name = event[TraceEventUtils.TraceKeys.Name]
                if any(name.lower().startswith(x) for x in ['copy', 'memcpy']):
                    return "memcpy"
                if any(name.lower().startswith(x) for x in ['memset']):
                    return "memset"
                return "kernel"
        return "Unknown"

    @staticmethod
    def split_events_by_pid_tid(events: List[dict]) -> Dict[str, Dict[str, List[Dict]]]:
        event_dict={}
        for event in TraceEventUtils.non_metadata_events(events):
            pid=event.get(TraceEventUtils.TraceKeys.PID)
            tid=event.get(TraceEventUtils.TraceKeys.TID)
            if pid in event_dict:
                pid_events = event_dict[pid]
            else:
                pid_events = event_dict[pid] = {}
            if tid in pid_events:
                pid_events[tid].append(event)
            else:
                pid_events[tid] = [event]
        return event_dict

    @staticmethod
    def sort_events_by_timestamp_duration(events: List[dict]) -> None:
        events.sort(key = lambda x: (x.get(TraceEventUtils.TraceKeys.TimeStamp), x.get(TraceEventUtils.TraceKeys.Duration)))

    @staticmethod
    def find_thread_by_item_in_metadata(metadata: dict[int, dict], select_item: Callable[[int], bool]) -> int:
        return next(filter(select_item, metadata.items()))[0]

    @staticmethod
    def compute_event_end_times(events: List[dict]) -> None:
        for event in events:
            TraceEventUtils.compute_single_event_end_time(event)

    @staticmethod
    def compute_single_event_end_time(event: dict) -> None:
        if TraceEventUtils.TraceKeys.TimeStamp in event and TraceEventUtils.TraceKeys.Duration in event and TraceEventUtils.TraceKeys.TimeEnd not in event:
            event[TraceEventUtils.TraceKeys.TimeEnd] = event[TraceEventUtils.TraceKeys.TimeStamp] + event[TraceEventUtils.TraceKeys.Duration]














