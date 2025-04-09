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
from typing import Dict, Any

class TraceToTree:
    def __init__(self, events_data, prune_nongpu_paths=True):
        self.events = [{**data, 'UID': i} for i, data in enumerate(events_data)]
        self.events_by_uid = {event['UID']: event for event in self.events}
        self._compute_event_end_times()
        self._set_linking_key()
        self._preprocess_and_index_events()
        self.cpu_root_nodes = []
        self.prune_nongpu_paths = prune_nongpu_paths

    def _compute_event_end_times(self) -> None:
        for event in self.events:
            if 'ts' in event and 'dur' in event:
                event['t_end'] = event['ts'] + event['dur']

    def _set_linking_key(self):
        launch_event = next(
            ( event for event in self.events if event.get('cat') in ['cuda_runtime', 'cuda_driver'] and 'launch' in event.get('name', '').lower() )
            , None)
        self.linking_key = 'correlation' if 'correlation' in launch_event['args'] else 'External id'

    def _preprocess_and_index_events(self) -> None:
        # 1. Create a dictionary to map the linking id to the start and end ac2g events
        # 2. Create a dictionary to map the pid, tid, and linking id to the actual event
        # 3. Create a dictionary to map the sequence number to the list of event uids
        # 4. Create a dictionary to map the python id to the event uid
        # This is done to quickly link events based on various keys

        self.ac2g_event_map = {'start': {}, 'end': {}}
        self.pid_tid_event_map = {}
        self.seq_num2event_uids_map = {} #from seq id to list uids
        self.dict_pythonID2UID = {}

        for event in self.events:
            # Process ac2g events
            if event.get('cat') == 'ac2g':
                if event['ph'] == 's':
                    self.ac2g_event_map['start'][event['id']] = event
                elif event['ph'] == 'f':
                    self.ac2g_event_map['end'][event['id']] = event
                continue

            # Process PID-TID-linking key events
            pid = event.get('pid')
            tid = event.get('tid')
            link_id = event.get('args', {}).get(self.linking_key)
            if None not in [pid, tid, link_id]:
                self.pid_tid_event_map[(pid, tid, link_id)] = event

            # Process sequence number events
            seq_num = event.get('args', {}).get('Sequence number')
            if seq_num is not None:
                self.seq_num2event_uids_map.setdefault(seq_num, []).append(event['UID'])

            # Process python_function events
            if event.get('cat') == 'python_function':
                self.dict_pythonID2UID[event['args']['Python id']] = event['UID']

    def build_host_call_stack_tree(self, add_python_func=False) -> None:
    # 1. Filter and sort events based on their start timestamps.
    #    - Include only CPU, CUDA runtime, and optionally Python function events.
    # 2. Iterate through the sorted events and maintain a stack to track the current call hierarchy.
    #    - Pop events from the stack if they end before the current event starts to find the parent.
    #    - Set the parent of the current event as the top of the stack if the stack is not empty.
    #    - Push the current event onto the stack.
    #    - For CPU operations:
    #      - Mark as a root node if it is the first CPU operation in the stack.
    #      - Increment the count of CPU operations in the stack.

        print(f"Building CPU op tree with add_python_func={add_python_func}")
        self.add_python_func = add_python_func
        list_events = []
        for event in self.events:
            is_cpu_or_cuda_event = event.get('cat') in {'cpu_op', 'cuda_runtime', 'cuda_driver'}
            is_python_event = event.get('cat') == 'python_function'
            if is_cpu_or_cuda_event or (add_python_func and is_python_event):
                list_events.append(event)

        events_sorted = sorted(list_events, key=lambda e: e['ts'])
        dict_pidtid2stack = defaultdict(list)
        dict_pidtid2num_cpu_ops = defaultdict(int)

        for event in events_sorted:
            event['tree'] = True

            pid = event.get('pid')
            tid = event.get('tid')
            stack_key = (pid, tid)
            stack = dict_pidtid2stack[stack_key]

            while stack and event['ts'] >= stack[-1]['t_end']:
                popped_event = stack.pop()
                if popped_event.get('cat') == 'cpu_op':
                    dict_pidtid2num_cpu_ops[stack_key] -= 1

            if stack and event['t_end'] > stack[-1]['t_end']:
                #TODO add following to logging when logging level is debug
                # print(f"Invalid event ordering: {event['name']} ends after the stack top event.")
                continue

            if stack:
                parent = stack[-1]
                parent.setdefault('children', []).append(event['UID'])
                event['parent'] = parent['UID']

            stack.append(event)
            if event.get('cat') == 'cpu_op':
                if dict_pidtid2num_cpu_ops[stack_key] == 0:
                    event['cpu_op_root'] = True
                    self.cpu_root_nodes.append(event['UID'])
                dict_pidtid2num_cpu_ops[stack_key] += 1

    def add_gpu_ops_to_tree(self):
        for event in self.events:
            if event.get('cat') not in {'cuda_runtime', 'cuda_driver'}:
                continue
            corresponding_gpu_event = self._find_corresponding_output_event(event)
            if not corresponding_gpu_event:
                continue
            event.setdefault('children', []).append(corresponding_gpu_event['UID'])
            corresponding_gpu_event['parent'] = event['UID']
            corresponding_gpu_event['tree'] = True

            # set the parents['gpu_events'] to the corresponding gpu event
            event['gpu_events'] = [corresponding_gpu_event['UID']] # runtime event will have only one corresponding gpu event
            while self.get_parent_event(event):
                parent = self.get_parent_event(event)
                parent.setdefault('gpu_events', []).append(corresponding_gpu_event['UID'])
                event = parent

    def label_non_gpu_paths(self):
        # 1. Iterate through non GPU nodes and chck the gpu_events list
        # 2. If the gpu_events list is empty, mark the node as non_gpu_path

        for event in self.events:
            # Skip GPU events
            if event.get('cat') in {'kernel', 'gpu_memset', 'gpu_memcpy'}:
                continue
            # Now, we are dealing with non-GPU events
            if 'gpu_events' not in event:
                event['non_gpu_path'] = True

    def build_tree(self, add_python_func=False) -> None:
        print(f"Building tree with add_python_func={add_python_func}")
        self.build_host_call_stack_tree(add_python_func)
        self.add_gpu_ops_to_tree()
        if self.prune_nongpu_paths:
            self.label_non_gpu_paths()

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

    def get_node_by_ext_id_pid_tid(self, ext_id, pid, tid):
        for event in self.events:
            if event.get('args', {}).get('External id') == ext_id and event.get('pid') == pid and event.get('tid') == tid:
                return event
        return None


    def traverse_subtree_and_print(self, node: Dict[str, Any], prune_non_gpu: bool = True) -> None:
        """
        Initiates traversal of a subtree of profiling events and prints them in a hierarchical call stack format.

        Args:
            node (Dict[str, Any]): The root node of the subtree.
            prune_non_gpu (bool): If True, prunes events that do not lead to GPU events.

        Prints:
            A structured representation of the subtree with details about each event.
        """
        self._traverse_subtree_recursive(node, prune_non_gpu, _prefix="", is_last=True)

    def _traverse_subtree_recursive(self, node: Dict[str, Any], prune_non_gpu: bool,
                                _prefix: str, is_last: bool) -> None:
        connector = "└── " if is_last else "├── "
        name = node.get('name', 'Unknown')
        max_len = 64
        if len(name) > max_len:
            name = name[:max_len] + '...'

        cat = node.get('cat')
        print_str = f"{_prefix}{connector}UID: {node['UID']}, Category: {cat}, Name: {name}"

        if cat in {'kernel', 'gpu_memset', 'gpu_memcpy'}:
            print_str += f", Duration: {node.get('dur')}"

        print(print_str)

        children = self.get_children_events(node)
        if prune_non_gpu:
            children = [child for child in children if 'non_gpu_path' not in child]

        child_count = len(children)
        new_prefix = _prefix + ("    " if is_last else "│   ")

        for i, child in enumerate(children):
            self._traverse_subtree_recursive(child, prune_non_gpu,
                                            new_prefix, is_last=(i == child_count - 1))

    def traverse_parents_and_print(self, node):
        depth = 0
        while True:
            if depth == 0:
                print("Node:")
            else:
                print(f"{depth}-up:")

            # Print category and name
            print(f"  cat: {node['cat']}")
            name = node.get('name', 'Unknown')
            max_len = 64
            if len(name) > max_len:
                name = name[:max_len] + '...'
            print(f"  name: {name}")

            # Move to the parent node
            node = self.get_parent_event(node)
            if node is None:
                break
            depth += 1

    def get_seq_nums_for_node_subtree(self, node_UID):
        seq_nums = set()
        event = self.events_by_uid[node_UID]
        if event.get('args', {}).get('Sequence number') is not None:
            seq_nums.add(event['args']['Sequence number'])
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
                if not self.events_by_uid[seq_num_match_UID].get('name').startswith('autograd::engine::evaluate_function:'):
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
        link_id = input_event.get('args', {}).get(self.linking_key)
        ac2g_start_event = self.ac2g_event_map['start'].get(link_id)
        ac2g_end_event = self.ac2g_event_map['end'].get(link_id)

        if not ac2g_start_event:
            return None

        if not ac2g_end_event:
            # print(f"Warning: start ac2g event found for {self.linking_key}={link_id} but no corresponding end ac2g event found.")
            # print(f"Input event name: {input_event['name']}")
            # print(('-'*64))
            return None

        pid = ac2g_end_event.get('pid')
        tid = ac2g_end_event.get('tid')
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
        return nn_module_children

    def _is_nn_module_event(self, event: Dict[str, Any]) -> bool:
        return event.get('cat') == 'python_function' and event.get('name', '').startswith('nn.Module:')