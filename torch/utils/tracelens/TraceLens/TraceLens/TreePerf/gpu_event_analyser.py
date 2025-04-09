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

import pandas as pd
import itertools

class GPUEventAnalyser:
    def __init__(self, events):
        """
        Initialize with a list of event dictionaries.
        """
        self.events = events


    @staticmethod
    def merge_intervals(intervals):
        """
        Merge a list of intervals (each as a (start, end) tuple) into a union of non-overlapping intervals.
        """
        if not intervals:
            return []
        intervals = sorted(intervals, key=lambda x: x[0])
        merged = [intervals[0]]
        for start, end in intervals[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        return merged


    @staticmethod
    def subtract_intervalsA_from_B(intervals_to_subtract, intervals):
        """Subtract set of intervals from another set of intervals.

        Intervals in sets are expected to be non-overlapping and sorted.

        Returns a list of intervals as (start, end) tuples.
        """
        result = []
        a_idx = 0
        a_len = len(intervals_to_subtract)

        for b_start, b_end in intervals:
            current = b_start
            while a_idx < a_len and intervals_to_subtract[a_idx][1] <= b_start:
                a_idx += 1

            temp_idx = a_idx
            while temp_idx < a_len and intervals_to_subtract[temp_idx][0] < b_end:
                a_start, a_end = intervals_to_subtract[temp_idx]
                if a_start > current:
                    result.append((current, min(b_end, a_start)))
                current = max(current, a_end)
                if current >= b_end:
                    break
                temp_idx += 1
            if current < b_end:
                result.append((current, b_end))

        return result


    all_gpu_key = 'all_gpu'
    computation_key = 'computation'
    communication_key = 'communication'
    memcpy_key = 'memcpy'
    all_cpu_key = 'all_cpu'
    gpu_event_keys = [all_gpu_key, computation_key, communication_key, memcpy_key]
    cpu_event_keys = [all_cpu_key]
    @property
    @staticmethod
    def all_event_keys():
        return itertools.chain(GPUEventAnalyser.gpu_event_keys, GPUEventAnalyser.cpu_event_keys)

    def get_gpu_event_lists(self):
        """
        Return a dictionary of lists of events, categorized by event types
        Event types are all gpu events, computation, communication, and memcpy.
        Be sure that the returned events have 'ts' and 't_end' fields.
        The default implementation is for PyTorch json trace format.
        Inherit the class and reimplement this method for your profile format.
        """

        # note all events are not gpu events
        # the events list contains gpu events as well as host side events

        gpu_events = []
        comp_events = []
        comm_events = []
        memcpy_events = []

        for event in self.events:

            #TODO: ideally we want to get gpu events based on process id
            # That will be done shortly
            category = event.get('cat')
            if category in {'kernel', 'gpu_memcpy', 'gpu_memset'}:
                if 't_end' not in event:
                    event['t_end'] = event['ts'] + event['dur']
                gpu_events.append(event)

                if category == 'gpu_memcpy':
                    memcpy_events.append(event)
                elif category in {'kernel', 'gpu_memset'}:
                    if 'nccl' in event.get('name'):
                        comm_events.append(event)
                    else:
                        comp_events.append(event)
                else:
                    raise ValueError(f"Unknown event category: {category}")
        return {
            GPUEventAnalyser.all_gpu_key: gpu_events,
            GPUEventAnalyser.computation_key: comp_events,
            GPUEventAnalyser.communication_key: comm_events,
            GPUEventAnalyser.memcpy_key: memcpy_events,
        }


    @staticmethod
    def verify_dict_gpu_event_lists(dict_gpu_event_lists):
        # first check if the keys are correct
        # note the check before is a linear lookup, but there are only 4 elements in the list
        if not all (key in GPUEventAnalyser.gpu_event_keys for key in dict_gpu_event_lists):
            raise ValueError(f"Expected keys: {GPUEventAnalyser.gpu_event_keys}, " +
                             f"got: {dict_gpu_event_lists.keys()}")
        # next check if the events have 'ts' and 't_end' fields
        for _, events in dict_gpu_event_lists.items():
            for event in events:
                if 'ts' not in event or 't_end' not in event:
                    raise ValueError(f"Event {event} does not have 'ts' or 't_end' fields")
        if len(dict_gpu_event_lists['all_gpu']) == 0:
            raise ValueError("No GPU events found in the trace")

    @staticmethod
    def compute_metrics_dict(dict: dict):
        dict_intervals = {}
        for key, events in dict.items():
            dict_intervals[key] = [(event['ts'], event['t_end']) for event in events]

        # Merge intervals within each category.
        comp_union = GPUEventAnalyser.merge_intervals(dict_intervals['computation'])
        comm_union = GPUEventAnalyser.merge_intervals(dict_intervals['communication'])
        memcpy_union = GPUEventAnalyser.merge_intervals(dict_intervals['memcpy'])
        all_intervals = GPUEventAnalyser.merge_intervals(dict_intervals['all_gpu'])

        # end of the last event - start of the first event
        total_time = all_intervals[-1][1] - all_intervals[0][0]


        comp_time = sum(end - start for start, end in comp_union)

        total_comm_time = sum(end - start for start, end in comm_union)
        exposed_comm_intervals = GPUEventAnalyser.subtract_intervalsA_from_B(comp_union, comm_union)
        exposed_comm_time = sum(end - start for start, end in exposed_comm_intervals)

        total_memcpy_time = sum(end - start for start, end in memcpy_union)
        memcpy_minus_compute = GPUEventAnalyser.subtract_intervalsA_from_B(comp_union, memcpy_union)
        exposed_memcpy_intervals = GPUEventAnalyser.subtract_intervalsA_from_B(comm_union, memcpy_minus_compute)
        exposed_memcpy_time = sum(end - start for start, end in exposed_memcpy_intervals)

        busy_time = sum(end - start for start, end in all_intervals)
        idle_time = total_time - busy_time

        # assert that compute + exposed comm + exposed memcpy + idle = total time
        assert abs(comp_time + exposed_comm_time + exposed_memcpy_time + idle_time - total_time) < 1e-6

        return {
            "computation_time": comp_time,
            "exposed_comm_time": exposed_comm_time,
            "exposed_memcpy_time": exposed_memcpy_time,
            "busy_time": busy_time,
            "idle_time": idle_time,
            "total_time": total_time,
            "total_comm_time": total_comm_time,
            "total_memcpy_time": total_memcpy_time,
        }


    def compute_metrics(self):
        """
        Compute various metrics from the GPU event data.
        Computation is defined as the time spent in computation kernels.
        Communication is defined as the time spent in communication kernels.
        Memcpy is defined as the time spent in memcpy kernels.
        Exposed communication time is the time spent in communication kernels that is not overlapped by computation.
        Exposed memcpy time is the time spent in memcpy kernels that is not overlapped by computation or communication.
        """

        # Categorize events.
        dict_gpu_event_lists = self.get_gpu_event_lists()
        GPUEventAnalyser.verify_dict_gpu_event_lists(dict_gpu_event_lists)

        return GPUEventAnalyser.compute_metrics_dict(dict_gpu_event_lists)

    @staticmethod
    def get_breakdown_df_from_dict(dict_metrics: dict):
        df = pd.DataFrame(dict_metrics.items(), columns=['type', 'time'])
        # convert time to ms by div by 1e3
        df['time ms'] = df['time'] / 1e3
        df['percent'] = df['time'] / df.loc[df['type'] == 'total_time', 'time'].values[0] * 100
        df = df.drop(columns=['time'])

        return df

    def get_breakdown_df(self):
        dict_metrics = self.compute_metrics()
        return GPUEventAnalyser.get_breakdown_df_from_dict(dict_metrics)

# Pytorch GPU event analyser inherits everything from the base class
class PytorchGPUEventAnalyser(GPUEventAnalyser):
    pass

# Jax GPU event analyser supports multiple GPUs
class JaxGPUEventAnalyser(GPUEventAnalyser):
    def get_gpu_event_lists(self, gpu_pid = None):
        """
        Return a dictionory of GPU to dictionaries of lists of events,
        categorized by event types
        Event types are all gpu events, computation, communication, and memcpy.
        Be sure that the returned events have 'ts' and 't_end' fields.
        The default implementation is for PyTorch json trace format.
        Inherit the class and reimplement this method for your profile format.

        If pid is passed in, returns just a dictionary of that pid's events
        """

        # note all events are not gpu events
        # the events list contains gpu events as well as host side events
        return_dict = {}
        for event in self.events:
            pid = event.get('pid')
            # jax uses pid > 100 for CPU evens
            # skip some dictionary setup events that do not have ts
            if 'ts' in event:
                if pid < 100:
                    cur_dict = return_dict.get(pid)
                    if cur_dict is None:
                        cur_dict = {key: [] for key in GPUEventAnalyser.gpu_event_keys}
                        return_dict[pid] = cur_dict
                    if 't_end' not in event:
                        event['t_end'] = event['ts'] + event['dur']
                    cur_dict[GPUEventAnalyser.all_gpu_key].append(event)
                    name = event.get('name')
                    if name.startswith('Copy') or name.startswith('Memcpy'):
                        cur_dict[GPUEventAnalyser.memcpy_key].append(event)
                    elif name.startswith('nccl'):
                        cur_dict[GPUEventAnalyser.communication_key].append(event)
                    else:
                        cur_dict[GPUEventAnalyser.computation_key].append(event)
                else:
                    cur_dict = return_dict.get(pid)
                    if cur_dict is None:
                        cur_dict = {key: [] for key in GPUEventAnalyser.cpu_event_keys}
                        return_dict[pid] = cur_dict
                    cur_dict[GPUEventAnalyser.all_cpu_key].append(event)
        if gpu_pid is None:
            return return_dict
        else:
            return return_dict.get(gpu_pid, {})

    def compute_metrics(self):
        """
        Compute various metrics from the GPU event data.
        Computation is defined as the time spent in computation kernels.
        Communication is defined as the time spent in communication kernels.
        Memcpy is defined as the time spent in memcpy kernels.
        Exposed communication time is the time spent in communication kernels that is not overlapped by computation.
        Exposed memcpy time is the time spent in memcpy kernels that is not overlapped by computation or communication.
        """

        # Categorize events.
        # get GPU 0 (PID 1) for Jax
        dict_gpu_event_lists = self.get_gpu_event_lists(1)
        GPUEventAnalyser.verify_dict_gpu_event_lists(dict_gpu_event_lists)

        return GPUEventAnalyser.compute_metrics_dict(dict_gpu_event_lists)

    def get_breakdown_df_multigpu(self):
        events = self.get_gpu_event_lists()
        gpu_frames = {}
        for gpu_id, cur_events in events.items():
            if gpu_id <= 100:
                self.verify_dict_gpu_event_lists(cur_events)
                cur_metrics = GPUEventAnalyser.compute_metrics_dict(cur_events)
                gpu_frames[gpu_id - 1] = GPUEventAnalyser.get_breakdown_df_from_dict(cur_metrics)
        return gpu_frames
