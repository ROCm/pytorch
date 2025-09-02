# MIT License

# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from collections import defaultdict
from typing import Dict, Any, Callable
import TraceLens.util
from .util import tev2_create_pseudo_host_mm_ops

from ..util import TraceEventUtils, JaxProfileProcessor


from abc import ABC, abstractmethod

class BaseTraceToTree(ABC):
    def __init__(self, events_data,
                 prune_nongpu_paths=True,
                 compute_end_times=True,
                 linking_key: str = None,
                 event_to_category: Callable[[dict], str] = None):

        self.events = [{**data, TraceLens.util.TraceEventUtils.TraceKeys.UID: i} for i, data in enumerate(events_data)]
        self.events_by_uid = {event[TraceLens.util.TraceEventUtils.TraceKeys.UID]: event for event in self.events}
        
        if compute_end_times:
            self._compute_event_end_times()
        if linking_key is not None:
            self.linking_key = linking_key
        else:
            self._set_linking_key()

        if event_to_category is not None:
            self.event_to_category = event_to_category
        else:
            self.event_to_category = self.default_categorizer()

        self.cpu_root_nodes = []
        self.prune_nongpu_paths = prune_nongpu_paths
        self.name2event_uids = defaultdict(list)
        

    @abstractmethod
    def default_categorizer(self) -> None:
        """
        Sets the default categorizer for the class.

        This abstract method should be implemented by subclasses to define
        how the default categorizer is set.
        """
        pass

    @abstractmethod
    def build_tree(self) -> None:
        """
        Constructs a tree structure from the provided trace data.

        This abstract method should be implemented by subclasses to define
        the logic for building a tree representation based on trace information.
        """
        pass

    @abstractmethod
    def _set_linking_key(self):
        """
        Set the linking key for the trace events.
       
        """
        pass

    @abstractmethod
    def add_gpu_ops_to_tree(self):
        """
        Add gpu operation to the tree
       
        """
        pass

    def build_host_call_stack_tree(self, add_python_func=False):
    # 1. Filter and sort events based on their start timestamps.
    #    - Include only CPU, CUDA runtime, and optionally Python function events.
    # 2. Iterate through the sorted events and maintain a stack to track the current call hierarchy.
    #    - Pop events from the stack if they end before the current event starts to find the parent.
    #    - Set the parent of the current event as the top of the stack if the stack is not empty.
    #    - Push the current event onto the stack.
    #    - For CPU operations:
    #      - Mark as a root node if it is the first CPU operation in the stack.
    #      - Increment the count of CPU operations in the stack.
        def event_filter(event):
            cat = self.event_to_category(event)
            event['cat'] = cat
            is_cpu_or_cuda_event = cat in {'cpu_op', 'cuda_runtime', 'cuda_driver'}
            is_python_event = self.event_to_category(event) == 'python_function'
            return is_cpu_or_cuda_event or (add_python_func and is_python_event)
        print(f"Building CPU op tree with add_python_func={add_python_func}")

        self.add_python_func = add_python_func
        list_events = filter(event_filter, self.events)

        events_sorted = sorted(list_events, key=lambda e: e[TraceLens.util.TraceEventUtils.TraceKeys.TimeStamp])
        dict_pidtid2stack = defaultdict(list)
        dict_pidtid2num_cpu_ops = defaultdict(int)

        for event in events_sorted:
            event['tree'] = True
            self.name2event_uids[event[TraceLens.util.TraceEventUtils.TraceKeys.Name]].append(event[TraceLens.util.TraceEventUtils.TraceKeys.UID])

            pid = event.get(TraceLens.util.TraceEventUtils.TraceKeys.PID)
            tid = event.get(TraceLens.util.TraceEventUtils.TraceKeys.TID)
            stack_key = (pid, tid)
            stack = dict_pidtid2stack[stack_key]

            while stack and event[TraceLens.util.TraceEventUtils.TraceKeys.TimeStamp] >= stack[-1][TraceLens.util.TraceEventUtils.TraceKeys.TimeEnd]:
                popped_event = stack.pop()
                if self.event_to_category(popped_event) == 'cpu_op':
                    dict_pidtid2num_cpu_ops[stack_key] -= 1

            if stack and event[TraceLens.util.TraceEventUtils.TraceKeys.TimeEnd] > stack[-1][TraceLens.util.TraceEventUtils.TraceKeys.TimeEnd]:
                #TODO add following to logging when logging level is debug
                # print(f"Invalid event ordering: {event[TraceLens.util.TraceEventUtils.TraceKeys.Name]} ends after the stack top event.")
                continue

            if stack:
                parent = stack[-1]
                parent.setdefault('children', []).append(event[TraceLens.util.TraceEventUtils.TraceKeys.UID])
                event['parent'] = parent[TraceLens.util.TraceEventUtils.TraceKeys.UID]

            stack.append(event)
            if self.event_to_category(event) == 'cpu_op':
                if dict_pidtid2num_cpu_ops[stack_key] == 0:
                    event['cpu_op_root'] = True
                    self.cpu_root_nodes.append(event[TraceLens.util.TraceEventUtils.TraceKeys.UID])
                dict_pidtid2num_cpu_ops[stack_key] += 1

    def label_non_gpu_paths(self):
        # 1. Iterate through non GPU nodes and chck the gpu_events list
        # 2. If the gpu_events list is empty, mark the node as non_gpu_path
        for event in self.events:
            # Skip GPU events
            if self.event_to_category(event) in {'kernel', 'gpu_memset', 'gpu_memcpy'}:
                continue
            # Now, we are dealing with non-GPU events
            if 'gpu_events' not in event:
                event['non_gpu_path'] = True

    def get_UID2event(self, UID):
        return self.events_by_uid[UID]

    def get_parent_event(self, event):
        if event.get('parent') is None:
            return None
        return self.get_UID2event(event['parent'])

    def get_children_events(self, event):
        if 'children' not in event:
            return []
        return [self.get_UID2event(child_UID) for child_UID in event['children']]

    def _compute_event_end_times(self) -> None:
        TraceLens.util.TraceEventUtils.compute_event_end_times(self.events)

    def _annotate_gpu_events_with_stream_index(self):
        """
        This function preprocesses the GPU events in the perf_analyzer object.
        """
        # 1. we create a dict stream -> events
        dict_stream2events = {}
        for event in self.events:
            stream =  event.get('args', {}).get('stream', None)
            if stream is not None:
                if stream not in dict_stream2events:
                    dict_stream2events[stream] = []
                dict_stream2events[stream].append(event)

        # 2. we sort the events in each stream by their timestamp
        for stream, events in dict_stream2events.items():
            dict_stream2events[stream] = sorted(events, key=lambda x: x[TraceLens.util.TraceEventUtils.TraceKeys.TimeStamp])

        # 3. we create a dict stream, index -> event
        #    and we set the stream index in the event
        dict_stream_index2event = {}
        for stream, events in dict_stream2events.items():
            for i, event in enumerate(events):
                dict_stream_index2event[(stream, i)] = event
                event[TraceLens.util.TraceEventUtils.TraceKeys.Args][TraceLens.util.TraceEventUtils.ArgNames.StreamIndex] = i
        # now we set this dict in the perf_analyzer
        self.dict_stream_index2event = dict_stream_index2event

    def _preprocess_and_index_events(self) -> None:
        # 1. Create a dictionary to map the linking id to the start and end ac2g events
        # 2. Create a dictionary to map the event key (by default (pid, tid)), and linking id to the actual event
        # 3. Create a dictionary to map the sequence number to the list of event uids
        # 4. Create a dictionary to map the python id to the event uid
        # This is done to quickly link events based on various keys

        self.ac2g_event_map = {'start': {}, 'end': {}}
        self.pid_tid_event_map = {}
        self.seq_num2event_uids_map = {} #from seq id to list uids
        self.dict_pythonID2UID = {}

        for event in self.events:
            # Process ac2g events
            if self.event_to_category(event) == 'ac2g':
                if event['ph'] == 's':
                    self.ac2g_event_map['start'][event['id']] = event
                elif event['ph'] == 'f':
                    self.ac2g_event_map['end'][event['id']] = event
                continue

            # Process PID-TID-linking key events
            pid = event.get(TraceLens.util.TraceEventUtils.TraceKeys.PID)
            tid = event.get(TraceLens.util.TraceEventUtils.TraceKeys.TID)
            link_id = event.get(TraceLens.util.TraceEventUtils.TraceKeys.Args, {}).get(self.linking_key)
            if None not in [pid, tid, link_id]:
                self.pid_tid_event_map[(pid, tid, link_id)] = event

            # Process sequence number events
            seq_num = event.get(TraceLens.util.TraceEventUtils.TraceKeys.Args, {}).get('Sequence number')
            if seq_num is not None:
                self.seq_num2event_uids_map.setdefault(seq_num, []).append(event[TraceLens.util.TraceEventUtils.TraceKeys.UID])

            # Process python_function events
            if self.event_to_category(event) == 'python_function':
                self.dict_pythonID2UID[event[TraceLens.util.TraceEventUtils.TraceKeys.Args]['Python id']] = event[TraceLens.util.TraceEventUtils.TraceKeys.UID]


class JaxTraceToTree(BaseTraceToTree):
    def __init__(self, events_data,
                 prune_nongpu_paths=True,
                 compute_end_times=True,
                 linking_key: str = None,
                 event_to_category: Callable[[dict], str] = TraceEventUtils.prepare_event_categorizer):

        super().__init__(events_data,
                         prune_nongpu_paths=prune_nongpu_paths,
                         compute_end_times=compute_end_times,
                         linking_key=linking_key,
                         event_to_category=event_to_category)
        self._preprocess_and_index_events()
        self._annotate_gpu_events_with_stream_index()

        self.linking_key_to_uid_map = defaultdict(list)
        self.hlo_ops = defaultdict(list)
        self.metadata = dict
    
    @staticmethod
    def default_categorizer(event: dict) -> str:
        """
        Returns the category of the given event dictionary using the TraceEventUtils.prepare_event_categorizer method.

        Args:
            event (dict): The event data to categorize.

        Returns:
            str: The category assigned to the event.
        """
        return TraceEventUtils.prepare_event_categorizer(event)
    
    def _set_linking_key(self) -> None:
        """
        Sets the linking key attribute to 'correlation_id'.

        This method assigns the string 'correlation_id' to the instance variable
        `linking_key`, which is used to link or correlate traces within the tree structure.
        """
        self.linking_key = 'correlation_id'
    
    def add_gpu_ops_to_tree(self) -> None:
        """
        Associates GPU operation events with their corresponding parent events in the event tree.

        Iterates through all events and, for those with a process ID (pid) less than or equal to 100,
        checks if the event has a parent. If so, it sets the 'gpu_events' field of the event and its
        ancestors to include the unique identifier (UID) of the GPU event. This allows for tracking
        GPU operations across the event hierarchy.

        Assumes that each event is a dictionary containing at least 'pid', 'parent', and UID fields,
        and that TraceLens.util.TraceEventUtils.TraceKeys.UID provides the key for the UID.

        Returns:
            None
        """
        for event in self.events:
            # GPU
            if event.get('pid')<= 100:
                # set the parents['gpu_events'] to the corresponding gpu event
                if 'parent' in event.keys():
                    corresponding_gpu_event = event
                    event['gpu_events'] = [corresponding_gpu_event[TraceLens.util.TraceEventUtils.TraceKeys.UID]]
                    while self.get_parent_event(event):
                        parent = self.get_parent_event(event)
                        parent.setdefault('gpu_events', []).append(corresponding_gpu_event[TraceLens.util.TraceEventUtils.TraceKeys.UID])
                        event = parent

    def build_tree(self, metadata: Dict, pb_file_name: str, add_python_func=False) -> None:
        """
        Builds a hierarchical tree structure from trace metadata and protobuf file.

        This method sets up the necessary metadata, processes HLO operations from the protobuf file,
        creates internal mappings, links CPU and GPU operations, and constructs the host call stack tree.
        Optionally, it can include Python function calls in the tree. GPU operations are added to the tree,
        and if pruning of non-GPU paths is enabled, those paths are labeled accordingly.

        Args:
            metadata (Dict): Metadata information required for building the tree.
            pb_file_name (str): Path to the protobuf file containing HLO operations.
            add_python_func (bool, optional): If True, includes Python function calls in the tree. Defaults to False.

        Returns:
            None
        """
        self._set_metadata(metadata)
        self._set_hlo_ops(pb_file_name)
        self._create_linking_key_to_uid_map()
        self._link_cpu_gpu()
        print(f"Building tree with add_python_func={add_python_func}")
        self.build_host_call_stack_tree(add_python_func)
        self.add_gpu_ops_to_tree()
        if self.prune_nongpu_paths:
            self.label_non_gpu_paths()


        
    def _set_metadata(self, metadata: Dict) -> None:
        """
        Sets the metadata for the current object and updates each event in `self.events`
        with corresponding process and thread information.

        Args:
            metadata (Dict): A dictionary containing process and thread metadata,
                indexed by process ID (`pid`) and thread ID (`tid`).

        Side Effects:
            - Assigns the provided metadata to `self.metadata`.
            - For each event in `self.events`, adds 'process' and 'thread' keys
              with values from the metadata dictionary.
        """
        self.metadata = metadata
        for event in self.events:
            pid = event.get('pid')
            tid = event.get('tid')
            event['process'] = self.metadata[pid][None]
            event['thread'] = self.metadata[pid][tid]


    def _set_hlo_ops(self, pb_file_name: str) -> None:
        """
        Processes events to extract unique HLO modules and populates the `hlo_ops` dictionary
        with operations parsed from the specified protobuf file.

        Args:
            pb_file_name (str): The path to the protobuf file containing HLO operation definitions.

        Side Effects:
            Updates the `self.hlo_ops` dictionary with keys as unique HLO module names found in events,
            and values as the result of processing the protobuf file for each module.
        """
        hlo_module_list = []
        for event in self.events:
            if 'args' in event.keys():
                if 'hlo_module' in event.get('args').keys():
                    hlo_module = event.get('args').get('hlo_module')
                    if hlo_module not in hlo_module_list:
                        hlo_module_list.append(hlo_module)
                        self.hlo_ops[hlo_module] = JaxProfileProcessor.process_protobuf_file(pb_file_name, hlo_module)

    def _create_linking_key_to_uid_map(self) -> None:
        """
        Populates the linking_key_to_uid_map with mappings from a specified linking key
        found in the 'args' field of each event to the corresponding event UID.

        Iterates over all events, and for each event that contains the linking key in its
        'args' dictionary, appends the event's UID to the list associated with that key
        in linking_key_to_uid_map.

        Returns:
            None
        """
        for uid, event in enumerate(self.events):
            if 'args' in event.keys():
                if self.linking_key in event.get('args').keys():
                    key = event.get('args').get(self.linking_key)
                    self.linking_key_to_uid_map[key].append(uid)

    def _link_cpu_gpu(self) -> None:
        """
        Links CPU and GPU events based on a shared linking key, establishing parent-child relationships
        between corresponding events. For each CPU event (identified by PID 701) that contains the linking key,
        finds associated GPU events and updates their hierarchical structure in the event tree.

        - Adds GPU event UIDs as children to the CPU event.
        - Marks both CPU and GPU events as part of the tree.
        - Sets the CPU event as the parent of the GPU event.
        - Appends the CPU event UID to the list of root nodes.
        - If GPU event contains HLO operation metadata, attaches it to the event; otherwise, logs missing metadata.

        Assumes the existence of:
        - self.events: List of event dictionaries.
        - self.linking_key: Key used to link CPU and GPU events.
        - self.linking_key_to_uid_map: Mapping from linking key to event UIDs.
        - self.events_by_uid: Mapping from UID to event dictionary.
        - self.cpu_root_nodes: List to collect root CPU event UIDs.
        - self.hlo_ops: Dictionary containing HLO operation metadata.
        """
        for event in self.events:
            if 'args' in event.keys():
                if self.linking_key in event.get('args').keys():
                    if event[TraceEventUtils.TraceKeys.PID]==701:
                        key = event.get('args').get(self.linking_key)
                        linking_uids = self.linking_key_to_uid_map[key]
                        uid = event[TraceEventUtils.TraceKeys.UID]
                        for linking_uid in linking_uids:
                            if linking_uid!=uid:
                                GPU_event = self.events_by_uid[linking_uid]
                                event.setdefault('children', []).append(linking_uid)
                                event['tree'] = True
                                GPU_event['parent'] = uid
                                GPU_event['tree'] = True
                                self.cpu_root_nodes.append(uid)
                                if 'hlo_op' in GPU_event.get('args').keys():
                                    hlo_op = '%' + GPU_event.get('args').get('hlo_op')
                                    hlo_module = GPU_event.get('args').get('hlo_module')
                                    if (hlo_module in self.hlo_ops.keys()) and (hlo_op in self.hlo_ops.get(hlo_module).keys()):
                                        GPU_event['metadata'] = self.hlo_ops.get(hlo_module).get(hlo_op)
                                    else:
                                        print('Missing hlo_op: ', hlo_op)
                                        print('in hlo_module: ', GPU_event['args']['hlo_module'])

class TraceToTree:
    def __init__(self, events_data,
                 prune_nongpu_paths=True,
                 compute_end_times=True,
                 linking_key: str = None,
                 event_to_category: Callable[[dict], str] = TraceLens.util.TraceEventUtils.default_categorizer):
        self.events = [{**data, TraceLens.util.TraceEventUtils.TraceKeys.UID: i} for i, data in enumerate(events_data)]
        self.events_by_uid = {event[TraceLens.util.TraceEventUtils.TraceKeys.UID]: event for event in self.events}
        self.event_to_category = event_to_category
        if compute_end_times:
            self._compute_event_end_times()
        if linking_key is not None:
            self.linking_key = linking_key
        else:
            self._set_linking_key()
        self._preprocess_and_index_events()
        self._annotate_gpu_events_with_stream_index()
        self.cpu_root_nodes = []
        self.prune_nongpu_paths = prune_nongpu_paths
        self.name2event_uids = defaultdict(list)

    @staticmethod
    def default_categorizer(event: dict) -> str:
        return event.get(TraceLens.util.TraceEventUtils.TraceKeys.Category)

    def _compute_event_end_times(self) -> None:
        TraceLens.util.TraceEventUtils.compute_event_end_times(self.events)

    def _set_linking_key(self):
        launch_event = next(
            ( event for event in self.events if self.event_to_category(event) in ['cuda_runtime', 'cuda_driver'] and 'launch' in event.get(TraceLens.util.TraceEventUtils.TraceKeys.Name, '').lower() )
            , None)
        self.linking_key = 'correlation' if launch_event is not None and 'correlation' in launch_event[TraceLens.util.TraceEventUtils.TraceKeys.Args] else 'External id'

    # TODO base class includes this, remove
    def _preprocess_and_index_events(self) -> None:
        # 1. Create a dictionary to map the linking id to the start and end ac2g events
        # 2. Create a dictionary to map the event key (by default (pid, tid)), and linking id to the actual event
        # 3. Create a dictionary to map the sequence number to the list of event uids
        # 4. Create a dictionary to map the python id to the event uid
        # This is done to quickly link events based on various keys

        self.ac2g_event_map = {'start': {}, 'end': {}}
        self.pid_tid_event_map = {}
        self.seq_num2event_uids_map = {} #from seq id to list uids
        self.dict_pythonID2UID = {}

        for event in self.events:
            # Process ac2g events
            if self.event_to_category(event) == 'ac2g':
                if event['ph'] == 's':
                    self.ac2g_event_map['start'][event['id']] = event
                elif event['ph'] == 'f':
                    self.ac2g_event_map['end'][event['id']] = event
                continue

            # Process PID-TID-linking key events
            pid = event.get(TraceLens.util.TraceEventUtils.TraceKeys.PID)
            tid = event.get(TraceLens.util.TraceEventUtils.TraceKeys.TID)
            link_id = event.get(TraceLens.util.TraceEventUtils.TraceKeys.Args, {}).get(self.linking_key)
            if None not in [pid, tid, link_id]:
                self.pid_tid_event_map[(pid, tid, link_id)] = event

            # Process sequence number events
            seq_num = event.get(TraceLens.util.TraceEventUtils.TraceKeys.Args, {}).get('Sequence number')
            if seq_num is not None:
                self.seq_num2event_uids_map.setdefault(seq_num, []).append(event[TraceLens.util.TraceEventUtils.TraceKeys.UID])

            # Process python_function events
            if self.event_to_category(event) == 'python_function':
                self.dict_pythonID2UID[event[TraceLens.util.TraceEventUtils.TraceKeys.Args]['Python id']] = event[TraceLens.util.TraceEventUtils.TraceKeys.UID]

    # TODO base class includes this, remove
    def build_host_call_stack_tree(self, add_python_func=False):
    # 1. Filter and sort events based on their start timestamps.
    #    - Include only CPU, CUDA runtime, and optionally Python function events.
    # 2. Iterate through the sorted events and maintain a stack to track the current call hierarchy.
    #    - Pop events from the stack if they end before the current event starts to find the parent.
    #    - Set the parent of the current event as the top of the stack if the stack is not empty.
    #    - Push the current event onto the stack.
    #    - For CPU operations:
    #      - Mark as a root node if it is the first CPU operation in the stack.
    #      - Increment the count of CPU operations in the stack.
        def event_filter(event):
            is_cpu_or_cuda_event = self.event_to_category(event) in {'cpu_op', 'cuda_runtime', 'cuda_driver'}
            is_python_event = self.event_to_category(event) == 'python_function'
            return is_cpu_or_cuda_event or (add_python_func and is_python_event)
        print(f"Building CPU op tree with add_python_func={add_python_func}")

        self.add_python_func = add_python_func
        list_events = filter(event_filter, self.events)

        events_sorted = sorted(list_events, key=lambda e: e[TraceLens.util.TraceEventUtils.TraceKeys.TimeStamp])
        dict_pidtid2stack = defaultdict(list)
        dict_pidtid2num_cpu_ops = defaultdict(int)

        for event in events_sorted:
            event['tree'] = True
            self.name2event_uids[event[TraceLens.util.TraceEventUtils.TraceKeys.Name]].append(event[TraceLens.util.TraceEventUtils.TraceKeys.UID])

            pid = event.get(TraceLens.util.TraceEventUtils.TraceKeys.PID)
            tid = event.get(TraceLens.util.TraceEventUtils.TraceKeys.TID)
            stack_key = (pid, tid)
            stack = dict_pidtid2stack[stack_key]

            while stack and event[TraceLens.util.TraceEventUtils.TraceKeys.TimeStamp] >= stack[-1][TraceLens.util.TraceEventUtils.TraceKeys.TimeEnd]:
                popped_event = stack.pop()
                if self.event_to_category(popped_event) == 'cpu_op':
                    dict_pidtid2num_cpu_ops[stack_key] -= 1

            if stack and event[TraceLens.util.TraceEventUtils.TraceKeys.TimeEnd] > stack[-1][TraceLens.util.TraceEventUtils.TraceKeys.TimeEnd]:
                #TODO add following to logging when logging level is debug
                # print(f"Invalid event ordering: {event[TraceLens.util.TraceEventUtils.TraceKeys.Name]} ends after the stack top event.")
                continue

            if stack:
                parent = stack[-1]
                parent.setdefault('children', []).append(event[TraceLens.util.TraceEventUtils.TraceKeys.UID])
                event['parent'] = parent[TraceLens.util.TraceEventUtils.TraceKeys.UID]

            stack.append(event)
            if self.event_to_category(event) == 'cpu_op':
                if dict_pidtid2num_cpu_ops[stack_key] == 0:
                    event['cpu_op_root'] = True
                    self.cpu_root_nodes.append(event[TraceLens.util.TraceEventUtils.TraceKeys.UID])
                dict_pidtid2num_cpu_ops[stack_key] += 1

    def add_gpu_ops_to_tree(self):
        for event in self.events:
            if self.event_to_category(event) not in {'cuda_runtime', 'cuda_driver'}:
                continue
            corresponding_gpu_event = self._find_corresponding_output_event(event)
            if not corresponding_gpu_event:
                continue
            event.setdefault('children', []).append(corresponding_gpu_event[TraceLens.util.TraceEventUtils.TraceKeys.UID])
            corresponding_gpu_event['parent'] = event[TraceLens.util.TraceEventUtils.TraceKeys.UID]
            corresponding_gpu_event['tree'] = True
            self.name2event_uids[corresponding_gpu_event[TraceLens.util.TraceEventUtils.TraceKeys.Name]].append(corresponding_gpu_event[TraceLens.util.TraceEventUtils.TraceKeys.UID])

            # set the parents['gpu_events'] to the corresponding gpu event
            event['gpu_events'] = [corresponding_gpu_event[TraceLens.util.TraceEventUtils.TraceKeys.UID]] # runtime event will have only one corresponding gpu event
            while self.get_parent_event(event):
                parent = self.get_parent_event(event)
                parent.setdefault('gpu_events', []).append(corresponding_gpu_event[TraceLens.util.TraceEventUtils.TraceKeys.UID])
                event = parent

    # TODO base class includes this, remove
    def label_non_gpu_paths(self):
        # 1. Iterate through non GPU nodes and chck the gpu_events list
        # 2. If the gpu_events list is empty, mark the node as non_gpu_path

        for event in self.events:
            # Skip GPU events
            if self.event_to_category(event) in {'kernel', 'gpu_memset', 'gpu_memcpy'}:
                continue
            # Now, we are dealing with non-GPU events
            if 'gpu_events' not in event:
                event['non_gpu_path'] = True

    def build_tree(self, add_python_func=False) -> None:
        print(f"Building tree with add_python_func={add_python_func}")
        self.build_host_call_stack_tree(add_python_func)
        self.add_gpu_ops_to_tree()

        tev2_create_pseudo_host_mm_ops(self)

        if self.prune_nongpu_paths:
            self.label_non_gpu_paths()

    # TODO base class includes this, remove
    def get_UID2event(self, UID):
        return self.events_by_uid[UID]

    # TODO base class includes this, remove
    def get_parent_event(self, event):
        if event.get('parent') is None:
            return None
        return self.get_UID2event(event['parent'])

    # TODO base class includes this, remove
    def get_children_events(self, event):
        if 'children' not in event:
            return []
        return [self.get_UID2event(child_UID) for child_UID in event['children']]

    def get_node_by_ext_id_pid_tid(self, ext_id, pid, tid):
        for event in self.events:
            if (event.get(TraceLens.util.TraceEventUtils.TraceKeys.Args, {}).get('External id') == ext_id
                and event.get(TraceLens.util.TraceEventUtils.TraceKeys.PID) == pid
                and event.get(TraceLens.util.TraceEventUtils.TraceKeys.TID) == tid):
                return event
        return None

    def traverse_subtree_and_print(self, node: Dict[str, Any], prune_non_gpu: bool = True, cpu_op_fields: tuple[str, ...] = ()) -> None:
        """
        Initiates traversal of a subtree of profiling events and prints them in a hierarchical call stack format.

        Args:
            node (Dict[str, Any]): The root node of the subtree.
            prune_non_gpu (bool): If True, prunes events that do not lead to GPU events.
            cpu_op_fields (tuple[str, ...]): Optional tuple to specify printing additional details for CPU operations. 
                It will be some subset of ['Input Dims', 'Input type', 'Input Strides', 'Concrete Inputs'].

        Prints:
            A structured representation of the subtree with details about each event.
        """
        self._traverse_subtree_recursive(node, prune_non_gpu, cpu_op_fields=cpu_op_fields,
                                        _prefix="", is_last=True)

    def _traverse_subtree_recursive(self, node: Dict[str, Any], prune_non_gpu: bool, 
                                    cpu_op_fields: tuple[str],
                                _prefix: str, is_last: bool) -> None:
        connector = "└── " if is_last else "├── "
        name = node.get(TraceLens.util.TraceEventUtils.TraceKeys.Name, 'Unknown')
        max_len = 64
        if len(name) > max_len:
            name = name[:max_len] + '..'

        cat =self.event_to_category(node)
        print_str = f"{_prefix}{connector}UID: {node[TraceLens.util.TraceEventUtils.TraceKeys.UID]}, Category: {cat}, Name: {name}"

        if cat in {'kernel', 'gpu_memset', 'gpu_memcpy'}:
            print_str += f", Duration: {node.get(TraceLens.util.TraceEventUtils.TraceKeys.Duration)}"

        print(print_str)

        if cat == 'cpu_op':
            args = node.get(TraceLens.util.TraceEventUtils.TraceKeys.Args, {})
            cpu_detail_prefix = _prefix + ("    " if is_last else "│   ") + "|   "
            details_emitted = False
            for detail in cpu_op_fields:
                if detail in args:
                    detail_value = args[detail]
                    print_str = f"{cpu_detail_prefix}{detail}: {detail_value}"
                    print(print_str)
                    details_emitted = True
            if details_emitted:
                print(cpu_detail_prefix)

        children = self.get_children_events(node)
        if prune_non_gpu:
            children = [child for child in children if 'non_gpu_path' not in child]

        child_count = len(children)
        new_prefix = _prefix + ("    " if is_last else "│   ")

        for i, child in enumerate(children):
            self._traverse_subtree_recursive(child, prune_non_gpu,
                                            cpu_op_fields=cpu_op_fields,
                                            _prefix=new_prefix, is_last=(i == child_count - 1))

    def traverse_parents_and_print(self, node: Dict[str, Any], cpu_op_fields: tuple[str, ...] = ()) -> None:
        """
        Traverses the parent nodes of a given event node and prints their details
        in a hierarchical format, starting from the node itself and going up to the root.

        Args:
            node (Dict[str, Any]): The event node from which to start traversing upwards.
            cpu_op_fields (tuple[str, ...]): Optional tuple to specify printing additional details for CPU operations.
                It will be some subset of ['Input Dims', 'Input type', 'Input Strides', 'Concrete Inputs'].
        """

        depth = 0
        while True:
            if depth == 0:
                print("Node:")
            else:
                print(f"{depth}-up:")

            # Print category and name
            # print(f"  cat: {self.event_to_category(node)}")
            name = node.get(TraceLens.util.TraceEventUtils.TraceKeys.Name, 'Unknown')
            max_len = 64
            if len(name) > max_len:
                name = name[:max_len] + '..'
            print_str = f"  UID: {node[TraceLens.util.TraceEventUtils.TraceKeys.UID]}, Category: {self.event_to_category(node)}, Name: {name}"
            # Print duration if category is kernel, gpu_memset, or gpu_memcpy
            if self.event_to_category(node) in {'kernel', 'gpu_memset', 'gpu_memcpy'}:
                duration = node.get(TraceLens.util.TraceEventUtils.TraceKeys.Duration)
                if duration is not None:
                    print_str += f", Duration: {duration}"
            print(print_str)
            # Print additional CPU operation details if applicable
            if self.event_to_category(node) == 'cpu_op':
                args = node.get(TraceLens.util.TraceEventUtils.TraceKeys.Args, {})
                cpu_detail_prefix = " "*4
                for detail in cpu_op_fields:
                    if detail in args:
                        detail_value = args[detail]
                        print_str = f"{cpu_detail_prefix}{detail}: {detail_value}"
                        print(print_str)

            # Move to the parent node
            parent_node = self.get_parent_event(node)
            if parent_node is None:
                return node
            node = parent_node
            depth += 1

    def get_seq_nums_for_node_subtree(self, node_UID):
        seq_nums = set()
        event = self.events_by_uid[node_UID]
        if event.get(TraceLens.util.TraceEventUtils.TraceKeys.Args, {}).get('Sequence number') is not None:
            seq_nums.add(event[TraceLens.util.TraceEventUtils.TraceKeys.Args]['Sequence number'])
        if 'children' in event:
            for child_UID in event['children']:
                seq_nums.update(self.get_seq_nums_for_node_subtree(child_UID))
        return seq_nums

    def link_bwd_events(self, event_UID):
        fwd_event = self.events_by_uid[event_UID]
        seq_nums = self.get_seq_nums_for_node_subtree(event_UID)
        bwd_event_UIDs = []
        for seq_num in seq_nums:
            for seq_num_match_UID in self.seq_num2event_uids_map.get(seq_num, []):
                if not self.events_by_uid[seq_num_match_UID].get(TraceLens.util.TraceEventUtils.TraceKeys.Name).startswith('autograd::engine::evaluate_function:'):
                    continue
                bwd_event_UIDs.append(seq_num_match_UID)
                bwd_event = self.events_by_uid[seq_num_match_UID]
                bwd_event['fwd_event'] = event_UID
                break
        fwd_event['bwd_events'] = bwd_event_UIDs

    def _find_corresponding_output_event(self, input_event):
        # 1. Get the linking id from the input event
        # 2. Find the corresponding start and end ac2g events for the linking id
        # 3. Find the output event using the pid, tid, and linking id of the end ac2g event
        link_id = input_event.get(TraceLens.util.TraceEventUtils.TraceKeys.Args, {}).get(self.linking_key)
        ac2g_start_event = self.ac2g_event_map['start'].get(link_id)
        ac2g_end_event = self.ac2g_event_map['end'].get(link_id)

        if not ac2g_start_event:
            return None

        if not ac2g_end_event:
            # print(f"Warning: start ac2g event found for {self.linking_key}={link_id} but no corresponding end ac2g event found.")
            # print(f"Input event name: {input_event[TraceLens.util.TraceEventUtils.TraceKeys.Name]}")
            # print(('-'*64))
            return None

        pid = ac2g_end_event.get(TraceLens.util.TraceEventUtils.TraceKeys.PID)
        tid = ac2g_end_event.get(TraceLens.util.TraceEventUtils.TraceKeys.TID)
        link_id = ac2g_end_event.get('id')

        output_event = self.pid_tid_event_map.get((pid, tid, link_id))
        return output_event

    def get_nn_module_children(self, nn_module_event: Dict[str, Any]):
        """
        Get the UIDs of the nn.Module children of the provided nn.Module event.
        """
        if not self.add_python_func:
            raise ValueError("This method requires the add_python_func flag to be set to True when building the tree.")
        # if the nn.Module children are already cached, return them
        if 'nn_module_children' in nn_module_event:
            return nn_module_event['nn_module_children']
        nn_module_children = []
        for child_UID in nn_module_event.get('children', []):
            child = self.get_UID2event(child_UID)
            if self._is_nn_module_event(child):
                nn_module_children.append(child_UID)
            else:
                nn_module_children.extend(self.get_nn_module_children(self.get_UID2event(child_UID)))
        # cache the nn.Module children for later use
        nn_module_event['nn_module_children'] = nn_module_children
        # set parent for each child
        for child_UID in nn_module_children:
            child = self.get_UID2event(child_UID)
            child['nn_module_parent'] = nn_module_event[TraceLens.util.TraceEventUtils.TraceKeys.UID]
        return nn_module_children

    def get_nn_module_parent(self, nn_module_event: Dict[str, Any]):
        """
        Get the UID of the nn.Module parent of the provided nn.Module event.
        """
        if not self.add_python_func:
            raise ValueError("This method requires the add_python_func flag to be set to True when building the tree.")
        # if the nn.Module parent is already cached, return it
        if 'nn_module_parent' in nn_module_event:
            return nn_module_event['nn_module_parent']
        # find the parent, traverse up the tree until we find a nn.Module event or parent is None
        parent_UID = nn_module_event.get('parent')
        while parent_UID is not None:
            parent = self.get_UID2event(parent_UID)
            if self._is_nn_module_event(parent):
                nn_module_event['nn_module_parent'] = parent_UID
                return parent_UID
            parent_UID = parent.get('parent')
        # if no parent is found, return None
        return None

    def _is_nn_module_event(self, event: Dict[str, Any]) -> bool:
        return self.event_to_category(event) == 'python_function' and event.get(TraceLens.util.TraceEventUtils.TraceKeys.Name, '').startswith('nn.Module:')

    def _annotate_gpu_events_with_stream_index(self):
        """
        This function preprocesses the GPU events in the perf_analyzer object.
        """
        # 1. we create a dict stream -> events
        dict_stream2events = {}
        for event in self.events:
            stream =  event.get('args', {}).get('stream', None)
            if stream is not None:
                if stream not in dict_stream2events:
                    dict_stream2events[stream] = []
                dict_stream2events[stream].append(event)

        # 2. we sort the events in each stream by their timestamp
        for stream, events in dict_stream2events.items():
            dict_stream2events[stream] = sorted(events, key=lambda x: x[TraceLens.util.TraceEventUtils.TraceKeys.TimeStamp])

        # 3. we create a dict stream, index -> event
        #    and we set the stream index in the event
        dict_stream_index2event = {}
        for stream, events in dict_stream2events.items():
            for i, event in enumerate(events):
                dict_stream_index2event[(stream, i)] = event
                event[TraceLens.util.TraceEventUtils.TraceKeys.Args][TraceLens.util.TraceEventUtils.ArgNames.StreamIndex] = i
        # now we set this dict in the perf_analyzer
        self.dict_stream_index2event = dict_stream_index2event
