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

import json
import gzip
import os
from functools import partial
import copy
from collections import defaultdict
from typing import Dict, Any, Callable

# TODO: warning should show the stack as well
import warnings
import pprint
import pandas as pd
import numpy as np

from ..PerfModel.torch_op_mapping import op_to_perf_model_class_map, categorize_torch_op, dict_cat2names
from .gpu_event_analyser import GPUEventAnalyser, JaxGPUEventAnalyser
from .jax_analyses import JaxAnalyses
from ..Trace2Tree.trace_to_tree import TraceToTree
from ..util import DataLoader, TraceEventUtils

class TreePerfAnalyzer:
    @staticmethod
    def from_file(profile_filepath, jax: bool = False, *args, **kwargs) -> "TreePerfAnalyzer":
        # Creates a TreePerfAnalyzer from the trace in the provided filepath.
        # *args, **kwargs are passed to the TreePerfAnalyzer constructor.

        data = DataLoader.load_data(profile_filepath)
        data = data['traceEvents']

        categorizer =  TraceToTree.default_categorizer if not jax else TraceEventUtils.prepare_event_categorizer(data)
        data = data if not jax else TraceEventUtils.non_metadata_events(data)
        tree = TraceToTree(data, event_to_category=categorizer)
        return TreePerfAnalyzer(tree, jax=jax, event_to_category=categorizer, *args, **kwargs)

    def __init__(self, tree: TraceToTree, add_python_func=False, arch=None, jax = False, python_path=None, event_to_category: Callable[[dict], str] = TraceEventUtils.default_categorizer):
        self.jax = jax
        self.GPUEventAnalyser = GPUEventAnalyser if not jax else JaxGPUEventAnalyser
        self.tree = tree
        self.add_python_func = add_python_func
        self.arch = arch
        self.python_path = python_path
        self.event_to_category = event_to_category
        # we check if profile contains python func events
        self.with_python_stack = next((True for event in self.tree.events if self.event_to_category(event) == 'python_func'), False)
        self.tree.build_tree(add_python_func=add_python_func)
        self.op_to_perf_model_class_map = op_to_perf_model_class_map
        self.op_categorizer = categorize_torch_op
        self.dict_cat2names = dict_cat2names

    def agg_kernels_in_subtree(self, event, filter_func=None, verbose=False):
        if filter_func is None:
            filter_func = lambda x: True
        if self.event_to_category(event) in {'kernel', 'gpu_memcpy', 'gpu_memset'}:
            if not filter_func(event):
                return 0, []
            if verbose:
                print(f"Found kernel event, duration: {event['dur']}, name: {event['name']}")
            return event['dur'], [event['UID']]
        total_dur = 0
        list_kernels = []
        for child_UID in event.get('children', []):
            child = self.tree.get_UID2event(child_UID)
            child_total_dur, child_list_kernels = self.agg_kernels_in_subtree(child, filter_func, verbose)
            total_dur += child_total_dur
            list_kernels.extend(child_list_kernels)
        return total_dur, list_kernels

    def loop_and_aggregate_kernels(self, events, filter_func=None, verbose=False):
        total_kernel_time = 0
        list_kernels = []
        for event in events:
            this_total_kernel_time, this_list_kernels = self.agg_kernels_in_subtree(event, filter_func, verbose=False)
            total_kernel_time += this_total_kernel_time
            list_kernels.extend(this_list_kernels)
        return total_kernel_time, list_kernels

    @staticmethod
    def non_data_mov_filter(event):
        DATA_MOVEMENT_PATTERNS = ['at::native::direct_copy_kernel_cuda', 'transpose_']
        return not any(pattern in event['name'] for pattern in DATA_MOVEMENT_PATTERNS)

    def compute_perf_metrics(self, event, bwd=False,
                             non_data_mov=False, perf_model_class=None):

        # Handle kernel aggregation
        if bwd:
            if not event.get('bwd_events'):
                self.tree.link_bwd_events(event['UID'])
            cpu_op_uids = event['bwd_events']
        else:
            cpu_op_uids = [event['UID']]
        cpu_op_list = [self.tree.get_UID2event(uid) for uid in cpu_op_uids]
        _, list_kernelUIDS = self.loop_and_aggregate_kernels(cpu_op_list)
        list_kernels = [self.tree.events_by_uid[uid] for uid in list_kernelUIDS]
        busy_kernel_time = 0
        if len(list_kernels) > 0:
            busy_kernel_time = GPUEventAnalyser(list_kernels).compute_metrics()['busy_time']
        _, list_non_data_mov_kernelUIDs = self.loop_and_aggregate_kernels(cpu_op_list, filter_func=self.non_data_mov_filter)
        list_non_data_mov_kernels = [self.tree.events_by_uid[uid] for uid in list_non_data_mov_kernelUIDs]
        busy_non_data_mov_time = 0
        if len(list_non_data_mov_kernels) > 0:
            busy_non_data_mov_time = GPUEventAnalyser(list_non_data_mov_kernels).compute_metrics()['busy_time']
        event['kernel_details'] = [ {'name': kernel['name'],
                                     'dur': kernel['dur'],
                                     'stream': kernel.get('args', {}).get('stream', None)}
                                    for kernel in list_kernels]

        # Select the appropriate dictionary for FLOPS and memory functions
        if perf_model_class is None:
            perf_model_class = self.op_to_perf_model_class_map.get(event['name'])
        perf_model = perf_model_class(event, arch=self.arch, python_path=self.python_path)

        gflops = (perf_model.flops() if not bwd else perf_model.flops_bwd())/ 1e9

        tflops_per_s = (gflops / 1e3) / (busy_kernel_time / 1e6) if busy_kernel_time > 0 else float('nan')

        non_data_mov_tflops_per_s = (gflops / 1e3) / (busy_non_data_mov_time / 1e6) if busy_non_data_mov_time > 0 else float('nan')
        bytes_moved = perf_model.bytes() if not bwd else perf_model.bytes_bwd()

        dict_metrics = {
            'GFLOPS': gflops,
            'Kernel Time (µs)': busy_kernel_time,
            'TFLOPS/s': tflops_per_s,
        }         
        if non_data_mov:
            dict_metrics['Non-Data-Mov Kernel Time (µs)'] = busy_non_data_mov_time
            dict_metrics['Non-Data-Mov TFLOPS/s'] = non_data_mov_tflops_per_s
        if bytes_moved is not None:
            dict_metrics['Data Moved (MB)'] = bytes_moved / (1024 * 1024)
            dict_metrics['FLOPS/Byte'] = (gflops * 1e9) / bytes_moved if bytes_moved > 0 else float('nan')
            dict_metrics['TB/s'] = (bytes_moved / 1e12) / (busy_kernel_time / 1e6) if busy_kernel_time > 0 else float('nan')
        else:
            dict_metrics['Data Moved (MB)'] = float('nan')
            dict_metrics['FLOPS/Byte'] = float('nan')
            dict_metrics['TB/s'] = float('nan')

        if hasattr(perf_model, "get_simulation_time") and not bwd:
            # This is for the case where we have a simulated time
            # for the forward pass, but not for the backward pass
            simulated_time = perf_model.get_simulation_time()
            if simulated_time:
                dict_metrics['Simulated Time (µs)']= simulated_time
                dict_metrics['Simulated TFLOPS/s'] = (gflops / 1e3) / (
                            simulated_time / 1e6) if simulated_time > 0 else float('nan')

        if hasattr(perf_model, "get_simulation_time_bwd") and bwd:
            # This is for the case where we have a simulated time
            # for the forward pass, but not for the backward pass
            simulated_time = perf_model.get_simulation_time_bwd()
            if simulated_time:
                dict_metrics['Simulated Time (µs)'] = simulated_time
                dict_metrics['Simulated TFLOPS/s'] = (gflops / 1e3) / (
                            simulated_time / 1e6) if simulated_time > 0 else float('nan')

        for key, value in perf_model.param_details.items():
            dict_metrics[f"param: {key}"] = value

        return dict_metrics

    def compute_fwd_perf_metrics(self, event, non_data_mov=False):
        return self.compute_perf_metrics(event, bwd=False, non_data_mov=non_data_mov)
    def compute_bwd_perf_metrics(self, event, non_data_mov=False):
        return self.compute_perf_metrics(event, bwd=True, non_data_mov=non_data_mov)

    def build_df_perf_metrics(self, events, bwd=False,
                              non_data_mov=False, include_kernel_details=False, include_args=False,
                              dict_name_to_perf_model=None):
        if len(events) == 0:
            warnings.warn("Input list of events is empty. Returning an empty DataFrame.")
            return pd.DataFrame()
        rows = []
        list_warn_non_zero_flops_and_zero_time = []
        list_warn_perf_metrics_failed = []
        list_no_bwd_events = []
        for event in events:
            metrics_event = {'cat': self.event_to_category(event), 'name': event['name'],
                             'UID': event['UID'],
                        'pid': event['pid'], 'tid': event['tid'],
                        'external_id': event['args'].get('External id')}
            if include_args:
                args_cols = ['Input Dims', 'Input type', 'Input Strides', 'Concrete Inputs']
                metrics_event.update((arg, event['args'].get(arg)) for arg in args_cols)
            if dict_name_to_perf_model and event['name'] in dict_name_to_perf_model:
                perf_model_class = dict_name_to_perf_model[event['name']]
            else:
                perf_model_class = None
            try:
                dict_perf_metrics = self.compute_perf_metrics(event, bwd=bwd,
                                                            non_data_mov=non_data_mov, perf_model_class=perf_model_class)
            except Exception as e:
                list_warn_perf_metrics_failed.append(event)
                continue
            # handle warnings
            if bwd and not event.get('bwd_events'):
                list_no_bwd_events.append(event)
                continue
            if dict_perf_metrics['GFLOPS'] > 0 and dict_perf_metrics['Kernel Time (µs)'] == 0:
                list_warn_non_zero_flops_and_zero_time.append(event)

            if dict_perf_metrics is not None:
                metrics_event.update(dict_perf_metrics)
            if include_kernel_details:
                if 'kernel_details' in event:
                    metrics_event['kernel_details'] = event['kernel_details']
            rows.append(metrics_event)

        self._show_warnings(list_warn_non_zero_flops_and_zero_time,
                            list_no_bwd_events,
                            list_warn_perf_metrics_failed,
                            len(events))
        df_perf_metrics = pd.DataFrame(rows)
        return df_perf_metrics

    @staticmethod
    def _show_warnings(list_warn_non_zero_flops_and_zero_time,
                        list_no_bwd_events,
                        list_warn_perf_metrics_failed,
                        total_events):
        # we need to say a/b  events had this issue and one example is following
        # where b is total events
        if len(list_warn_non_zero_flops_and_zero_time) > 0:
            warnings.warn(f"Found {len(list_warn_non_zero_flops_and_zero_time)}/{total_events} events with non-zero GFLOPS and zero Kernel Time (µs).")
            warnings.warn(f"Example event: {pprint.pformat(list_warn_non_zero_flops_and_zero_time[0])}")
        if len(list_no_bwd_events) > 0:
            warnings.warn(f"Found {len(list_no_bwd_events)}/{total_events} events without backward events.")
            warnings.warn(f"Example event: {pprint.pformat(list_no_bwd_events[0])}")
        if len(list_warn_perf_metrics_failed) > 0:
            warnings.warn(f"Found {len(list_warn_perf_metrics_failed)}/{total_events} events with failed performance metric computation.")
            warnings.warn(f"Example event: {pprint.pformat(list_warn_perf_metrics_failed[0])}")

    def build_df_fwd_perf_metrics(self, events):
        return self.build_df_perf_metrics(events, bwd=False)
    def build_df_bwd_perf_metrics(self, events):
        return self.build_df_perf_metrics(events, bwd=True)


    @staticmethod
    def summarize_df_perf_metrics(df_perf_metrics, agg_metrics=['mean', 'std']):
        if df_perf_metrics.empty:
            warnings.warn("Input DataFrame is empty. Returning an empty summary DataFrame.")
            return pd.DataFrame()  # Return an empty DataFrame instead of raising an error

        dict_agg = {}
        # first element for GFLOPS and FLOPS/Byte
        dict_agg['GFLOPS'] = 'first'
        dict_agg['Data Moved (MB)'] = 'first'
        dict_agg['FLOPS/Byte'] = 'first'
        dict_agg['TB/s'] = agg_metrics
        dict_agg['TFLOPS/s'] = agg_metrics
        if 'Simulated Time (µs)' in df_perf_metrics.columns:
            # first since it should be same for the group
            dict_agg['Simulated Time (µs)'] = 'first'
            dict_agg['Simulated TFLOPS/s'] = 'first'
        if 'Non-Data-Mov TFLOPS/s' in df_perf_metrics.columns:
            dict_agg['Non-Data-Mov TFLOPS/s'] = agg_metrics
        if 'Non-Data-Mov Kernel Time (µs)' in df_perf_metrics.columns:
            dict_agg['Non-Data-Mov Kernel Time (µs)'] = ['sum']
        # this is a quick fix, we need to veriify it matches in the group
        if 'kernel_details' in df_perf_metrics.columns:
            dict_agg['kernel_details'] = partial(TreePerfAnalyzer._summarize_kernel_stats, agg_metrics=agg_metrics)
        args_cols = ['Input Dims', 'Input type', 'Input Strides', 'Concrete Inputs']
        for arg in args_cols:
            if arg in df_perf_metrics.columns:
                dict_agg[arg] = 'first'
        dict_agg['Kernel Time (µs)'] = agg_metrics + ['sum']
        #dict_agg['Simulated Kernel Time (us)'] = agg_metrics + ['sum']
        dict_agg['name'] = 'count'  # Use the 'name' column as a proxy for counting rows
        dict_agg['UID'] = 'first'

        # Identify parameter columns for grouping
        param_cols = [col for col in df_perf_metrics.columns if col.startswith('param: ')]
        #TODO warn user if nans in the performance metrics
        # Perform the aggregation
        df_perf_metrics_summary = (
            df_perf_metrics
            .groupby(['name'] + param_cols, dropna=False)
            .agg(dict_agg)
        )
        df_perf_metrics_summary.columns = ['_'.join(col).strip() for col in df_perf_metrics_summary.columns.values]
        df_perf_metrics_summary.reset_index(inplace=True)

        df_perf_metrics_summary.sort_values(by='Kernel Time (µs)_sum', ascending=False, inplace=True)
        #df_perf_metrics_summary.sort_values(by='Simulated Kernel Time (us)_sum', ascending=False, inplace=True)
        df_perf_metrics_summary.reset_index(drop=True, inplace=True)

        return df_perf_metrics_summary

    def get_kernel_launchers(self, include_nccl=False):
        # This method traverses the event tree to identify CPU operations that serve as
        # "kernel launchers." These are operations that result in GPU kernel
        # execution without further cpu op calls.
        # Note that kernels are called through runtime events.
        # This is why, this method identifies such cases
        # by checking if grandchildren of CPU operations are kernel events.
        kernel_launchers = []
        for event in self.tree.events:
            if self.event_to_category(event) != 'cpu_op':
                continue

            if event['name'] == 'execute':
                parent = self.tree.get_parent_event(event)
                list_kernel_uids = parent.get('gpu_events', [])
                list_kernels = [self.tree.get_UID2event(uid) for uid in list_kernel_uids]
                parent['total_direct_kernel_time'] = GPUEventAnalyser(list_kernels).compute_metrics()['busy_time']
                parent['direct_kernel_count'] = len(list_kernels)
                parent['kernel_details'] = [{'name': kernel['name'],
                                             'dur': kernel['dur'],
                                             'stream': kernel.get('args', {}).get('stream', None)}
                                            for kernel in list_kernels]
                parent['op category'] = self.op_categorizer(parent)
                kernel_launchers.append(parent)
                continue # no need to check children of this event

            kernel_launcher = False
            # total_direct_kernel_time = 0
            # direct_kernel_count = 0
            list_kernels = []
            for child_UID in event.get('children', []):
                child = self.tree.events_by_uid[child_UID]
                for grand_child_UID in child.get('children', []):
                    grand_child = self.tree.events_by_uid[grand_child_UID]
                    is_kernel = self.event_to_category(grand_child) == 'kernel'
                    is_nccl = 'nccl' in grand_child['name']
                    should_include = is_kernel and (include_nccl or not is_nccl)
                    if should_include:
                        kernel_launcher = True
                        list_kernels.append(grand_child)
            if kernel_launcher:
                event['total_direct_kernel_time'] = GPUEventAnalyser(list_kernels).compute_metrics()['busy_time']
                event['direct_kernel_count'] = len(list_kernels)
                event['kernel_details'] = [{'name': kernel['name'],
                                           'dur': kernel['dur'],
                                           'stream': kernel.get('args', {}).get('stream', None)}
                                          for kernel in list_kernels]
                event['op category'] = self.op_categorizer(event)
                kernel_launchers.append(event)
        return kernel_launchers

    def get_df_kernel_launchers(self, id_cols=False, include_kernel_details=False):

        def list_to_tuple(obj):
            if isinstance(obj, list):
                return tuple(list_to_tuple(item) for item in obj)
            return obj

        kernel_launchers = self.get_kernel_launchers()
        rows = []
        for event in kernel_launchers:
            metrics_event = {'name': event['name'],
                             'op category': event['op category'],
                             'UID': event['UID'],
                            'total_direct_kernel_time': event['total_direct_kernel_time'],
                            'direct_kernel_count': event['direct_kernel_count']}
            for arg in ['Input Dims', 'Input type', 'Input Strides', 'Concrete Inputs']:
                if arg in event['args']:
                    metrics_event[arg] = list_to_tuple(event['args'][arg])
                else:
                    metrics_event[arg] = None

            if id_cols:
                metrics_event['pid'] = event['pid']
                metrics_event['tid'] = event['tid']
                metrics_event['external_id'] = event['args'].get('External id')
            if include_kernel_details:
                if 'kernel_details' in event:
                    metrics_event['kernel_details'] = event['kernel_details']
            rows.append(metrics_event)
        df = pd.DataFrame(rows)
        return df

    @staticmethod
    def get_df_kernel_launchers_summary(df_kernel_launchers):
        df_temp = df_kernel_launchers.copy()
        df_agg = df_temp.groupby('name').agg({'total_direct_kernel_time': ['sum', 'count']})
        df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns.values]
        df_agg.reset_index(inplace=True)
        df_agg.rename(columns={'total_direct_kernel_time_count': 'Count'}, inplace=True)
        df_agg.sort_values(by='total_direct_kernel_time_sum', ascending=False, inplace=True)
        df_agg['total_direct_kernel_time_ms'] = df_agg['total_direct_kernel_time_sum'] / 1000
        total_duration_ms = df_agg['total_direct_kernel_time_ms'].sum()
        df_agg['Percentage (%)'] = (df_agg['total_direct_kernel_time_ms'] / total_duration_ms) * 100
        df_agg['Cumulative Percentage (%)'] = df_agg['Percentage (%)'].cumsum()
        df_agg.reset_index(drop=True, inplace=True)

        return df_agg

    #separate out name wise perf breakdown and shape wise perf breakdown for a given name
    @staticmethod
    def get_df_kernel_launchers_summary_by_shape(df_kernel_launchers, name):
        warnings.warn("get_df_kernel_launchers_summary_by_shape is deprecated. Use get_df_kernel_launchers_unique_args instead.")
        df_temp = df_kernel_launchers.copy()
        df_temp = df_temp[df_temp['name'] == name]
        dict_agg = {'total_direct_kernel_time': ['sum', 'count', 'mean', 'std'],
                    'direct_kernel_count': ['max', 'min']}
        # df_agg = df_temp.groupby(['Input Dims']).agg(dict_agg)
        #check if the input dims and others are present in the df
        df_agg = df_temp.groupby(['Input Dims', 'Input type', 'Input Strides'], dropna=False).agg(dict_agg)
        df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns.values]
        df_agg.reset_index(inplace=True)
        df_agg.rename(columns={'total_direct_kernel_time_sum': 'Total Kernel Time (µs)',
                               'total_direct_kernel_time_count': 'Count',
                               'total_direct_kernel_time_mean': 'Mean Kernel Time (µs)',
                               'total_direct_kernel_time_std': 'Std Kernel Time (µs)',
                               'direct_kernel_count_max': 'Max Direct Kernel Count',
                               'direct_kernel_count_min': 'Min Direct Kernel Count'}, inplace=True)
        df_agg.sort_values(by='Total Kernel Time (µs)', ascending=False, inplace=True)
        df_agg['Total Kernel Time (ms)'] = df_agg['Total Kernel Time (µs)'] / 1000
        total_duration_ms = df_agg['Total Kernel Time (ms)'].sum()
        df_agg['Percentage (%)'] = (df_agg['Total Kernel Time (ms)'] / total_duration_ms) * 100
        df_agg['Cumulative Percentage (%)'] = df_agg['Percentage (%)'].cumsum()
        df_agg.reset_index(drop=True, inplace=True)
        return df_agg

    @staticmethod
    def _summarize_kernel_stats(series_of_kernel_lists, agg_metrics=['mean']):
        """
        Revised implementation for ordered kernel summarization.
        """
        METRIC_MAP = {
            'mean': ('mean_duration_us', np.mean),
            'median': ('median_duration_us', np.median),
            'max': ('max_duration_us', np.max),
            'min': ('min_duration_us', np.min),
            'std': ('std_dev_duration_us', np.std)
        }

        # --- CHANGE: More robust way to get the template ---
        # Find the first valid list in the series to use as a template.
        try:
            template = next(item for item in series_of_kernel_lists if isinstance(item, list) and item)
        except StopIteration:
            return [] # The series was empty or contained no valid lists.

        # --- CHANGE: Collect durations BY INDEX, not by name ---
        all_durations = [[] for _ in template]

        for kernel_list in series_of_kernel_lists:
            if isinstance(kernel_list, list):
                # Basic validation to prevent errors and warn about inconsistencies
                if len(kernel_list) != len(template):
                    warnings.warn(f"Inconsistent kernel list length found. Skipping a row.", UserWarning)
                    continue
                
                for i, kernel in enumerate(kernel_list):
                    try:
                        # Append the duration to the list corresponding to its position
                        all_durations[i].append(kernel['dur'])
                    except (KeyError, IndexError):
                        warnings.warn(f"Malformed kernel event or index issue at index {i}. Skipping kernel: {kernel}", UserWarning)
                        continue

        # --- CHANGE: Create a deep copy to avoid modifying original data ---
        summary_list = copy.deepcopy(template)

        # Now, compute statistics and populate the summary list
        for i, kernel_summary in enumerate(summary_list):
            durations_for_this_index = all_durations[i]
            dur_arr = np.array(durations_for_this_index)
            
            # --- CHANGE: Use consistent naming and clear up original 'dur' key ---
            del kernel_summary['dur']

            kernel_summary['count'] = len(dur_arr)
            kernel_summary['total_duration_us'] = np.sum(dur_arr) # Use consistent key name
            
            if not durations_for_this_index:
                # If no durations were collected (e.g., all rows skipped), skip metric calculation
                continue

            for metric in agg_metrics:
                if metric in METRIC_MAP:
                    metric_name, agg_func = METRIC_MAP[metric]
                    # --- CHANGE: Use the consistent metric name directly ---
                    kernel_summary[metric_name] = agg_func(dur_arr)
                    
        return summary_list

    @staticmethod
    def get_df_kernel_launchers_unique_args(df_kernel_launchers: pd.DataFrame,
                                            event_name=None, agg_metrics=['mean'], include_pct=False) -> pd.DataFrame:
        """
        Generate a DataFrame with unique arguments for each operation in the input DataFrame.

        Args:
            df_kernel_launchers (pd.DataFrame): DataFrame containing kernel launchers.
            event_name (str): Optional name of the event to filter the DataFrame.
            agg_metrics (list): List of aggregation metrics to apply. ex: ['mean', 'std', 'median']
            include_pct (bool): If True, include percentage of total time for each row as well as cumulative percentage.

        Returns:
            pd.DataFrame: DataFrame with unique arguments for each operation.
        """
        grouping_cols_original = ['name', 'Input Dims', 'Input type', 'Input Strides', 'Concrete Inputs']

        # 0. Filter the DataFrame based on the event name if provided
        if event_name is not None:
            df_filtered = df_kernel_launchers[df_kernel_launchers['name'] == event_name].copy()
        else:
            df_filtered = df_kernel_launchers.copy()

        # 1. Create string representations of the grouping columns - so we can group by them
        str_col_names, actual_grouping_cols = [], []
        for col in grouping_cols_original:
            if col not in df_filtered.columns:
                continue
            actual_grouping_cols.append(col)
            str_col_name = f"{col}_str_repr_for_grouping"
            df_filtered[str_col_name] = df_filtered[col].apply(str)
            str_col_names.append(str_col_name)
        if not str_col_names:
            raise ValueError("No valid columns found to group by.")

        # 2. Aggregate the DataFrame by the string representations of the grouping columns
        agg_dict = {}
        if 'total_direct_kernel_time' in df_filtered.columns:
            agg_dict['total_direct_kernel_time'] = agg_metrics + (['sum'] if 'sum' not in agg_metrics else [])
        columns_to_keep_first = []
        if 'UID' in df_filtered.columns:
            agg_dict['UID'] = ['first', 'count']
            columns_to_keep_first.append('UID')
        if 'kernel_details' in df_filtered.columns:
            agg_dict['kernel_details'] = partial(TreePerfAnalyzer._summarize_kernel_stats, agg_metrics=agg_metrics)
            columns_to_keep_first.append('kernel_details')
        for col in actual_grouping_cols:
            agg_dict[col] = 'first'
            columns_to_keep_first.append(col)
        df_unique_args = df_filtered.groupby(str_col_names, dropna=False, sort=False).agg(agg_dict)
        df_unique_args.columns = ['_'.join(col).strip() for col in df_unique_args.columns.values]
        df_unique_args.reset_index(inplace=True)

        # 3. Rename columns for clarity
        rename_map = {'UID_count': 'operation_count'}
        for col in columns_to_keep_first:
            col_first = f'{col}_first'
            if col_first in df_unique_args.columns:
                rename_map[col_first] = col
        # uid needs to be mapped to ex_UID
        if 'UID_first' in df_unique_args.columns:
            rename_map['UID_first'] = 'ex_UID'
        for col in df_unique_args.columns:
            if col.startswith('kernel_details_'):
                rename_map[col] = 'kernel_details_summary'
        df_unique_args.rename(columns=rename_map, inplace=True)

        # 4. Reorder columns: start with grouping + key metrics, then rest
        primary_cols = [col for col in grouping_cols_original if col in df_unique_args.columns]
        metric_cols = [col for col in ['UID', 'operation_count', 'kernel_names', 'total_direct_kernel_time_mean'] if col in df_unique_args.columns]
        other_cols = [col for col in df_unique_args.columns if col not in primary_cols + metric_cols and not col.endswith('_str_repr_for_grouping')]
        df_unique_args = df_unique_args[primary_cols + metric_cols + other_cols]

        # 5. Sort the DataFrame by the sum of total_direct_kernel_time
        if 'total_direct_kernel_time_sum' in df_unique_args.columns:
            df_unique_args = df_unique_args.sort_values(by="total_direct_kernel_time_sum", ascending=False).reset_index(drop=True)

        # 6. Calculate percentage of total time and cumulative percentage if requested
        if include_pct and 'total_direct_kernel_time_sum' in df_unique_args.columns:
            total_duration_ms = df_unique_args['total_direct_kernel_time_sum'].sum()
            df_unique_args['Percentage (%)'] = (df_unique_args['total_direct_kernel_time_sum'] / total_duration_ms) * 100
            df_unique_args['Cumulative Percentage (%)'] = df_unique_args['Percentage (%)'].cumsum()
        return df_unique_args

    @staticmethod
    def get_df_kernel_launchers_summary_by_category(df_kernel_launchers: pd.DataFrame) -> pd.DataFrame:
        """                            
        Generate a DataFrame with breakdown of kernel launchers by category.
        Args:
            df_kernel_launchers (pd.DataFrame): DataFrame containing kernel launchers.
        Returns:
            pd.DataFrame: DataFrame with breakdown of kernel launchers by category.
        """
        df_temp = df_kernel_launchers.copy()
        df_agg = df_temp.groupby('op category').agg({'total_direct_kernel_time': ['sum', 'count']})
        df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns.values]
        df_agg.reset_index(inplace=True)
        df_agg.rename(columns={'total_direct_kernel_time_count': 'Count'}, inplace=True)
        df_agg.sort_values(by='total_direct_kernel_time_sum', ascending=False, inplace=True)
        df_agg['total_direct_kernel_time_ms'] = df_agg['total_direct_kernel_time_sum'] / 1000
        # remove the us col as we will use ms col
        df_agg.drop(columns=['total_direct_kernel_time_sum'], inplace=True)
        total_duration_ms = df_agg['total_direct_kernel_time_ms'].sum()
        df_agg['Percentage (%)'] = (df_agg['total_direct_kernel_time_ms'] / total_duration_ms) * 100
        df_agg['Cumulative Percentage (%)'] = df_agg['Percentage (%)'].cumsum()
        df_agg.reset_index(drop=True, inplace=True)

        return df_agg

    def get_df_gpu_timeline(self):
        kernel_events =  [event for event in self.tree.events if self.event_to_category(event) in {'kernel', 'gpu_memcpy', 'gpu_memset'} and event.get('tree')]
        gpu_event_analyser = self.GPUEventAnalyser(kernel_events)
        df = gpu_event_analyser.get_breakdown_df()
        return df

    def get_kernel_details(self, kernel_event,
                           launcher_detail=False, cpu_op_detail = True, nn_module_detail=False):
        """
        Extract detailed information for a given kernel event.

        This method traces a kernel event's parent relationships to retrieve
        its launcher and CPU operation details, then returns a dictionary of
        relevant information. If any of the necessary links are missing or invalid,
        the function returns None.

        Args:
            kernel_event (dict): The kernel event dictionary.
            launcher_detail (bool): If True, include details of the kernel's launcher.
            cpu_op_detail (bool): If True, include details of the parent CPU operation.
            nn_module_detail (bool): If True, include details of the parent nn.Module event. Only valid if
                `add_python_func` is True. Else, it will be ignored.

        Returns:
            dict or None: A dictionary containing the kernel details, or None if linking fails.
        """
        def list_to_tuple(obj):
            # Recursively convert lists to tuples.
            return tuple(list_to_tuple(item) for item in obj) if isinstance(obj, list) else obj

        # Verify that the event is a kernel event.
        if self.event_to_category(kernel_event) != 'kernel':
            return None

        kernel_details = {
            'UID': kernel_event['UID'],
            'Kernel name': kernel_event['name'],
            'Kernel duration (µs)': kernel_event['dur'],
            'Kernel stream': kernel_event['args'].get('stream')
        }

        # 1. get launcher event
        launcher = self.tree.get_parent_event(kernel_event)

        # add launcher details
        if launcher and launcher_detail:
            kernel_details['Launcher UID'] = launcher['UID']
            kernel_details['Launcher'] = launcher['name']
            kernel_details['Grid'] = list_to_tuple(launcher['args'].get('grid'))
            kernel_details['Block'] = list_to_tuple(launcher['args'].get('block'))

        # 2. get lowest cpu_op event - events of cat 'cpu_op' contain args info
        cpu_op = None
        evt = launcher
        while evt:
            if self.event_to_category(evt) == 'cpu_op':
                cpu_op = evt
                break
            evt = self.tree.get_parent_event(evt)

        # add cpu_op details
        if cpu_op and cpu_op_detail:
            kernel_details['Parent cpu_op UID'] = cpu_op['UID']
            kernel_details['Parent cpu_op'] = cpu_op['name']
            kernel_details['Input dims'] = list_to_tuple(cpu_op['args'].get('Input Dims'))
            kernel_details['Input types'] = list_to_tuple(cpu_op['args'].get('Input type'))
            kernel_details['Input strides'] = list_to_tuple(cpu_op['args'].get('Input Strides'))
            kernel_details['Concrete Inputs'] = list_to_tuple(cpu_op['args'].get('Concrete Inputs'))
            kernel_details['kernel_file'] = cpu_op['args'].get('kernel_file')
            if cpu_op.get('gpu_busy_time') is None:
                # If the cpu_op event does not have GPU busy time, compute it.
                gpu_events = [self.tree.get_UID2event(uid) for uid in cpu_op.get('gpu_events', [])]
                cpu_op['gpu_busy_time'] = GPUEventAnalyser(gpu_events).compute_metrics()['busy_time']
            if cpu_op.get('kernel_count') is None:
                # If the cpu_op event does not have direct kernel count, compute it.
                cpu_op['kernel_count'] = len(cpu_op.get('gpu_events', []))
            kernel_details['Parent cpu_op busy time (µs)'] = cpu_op['gpu_busy_time']
            kernel_details['Parent cpu_op kernel count'] = cpu_op.get('kernel_count', 0)
            if cpu_op['kernel_count'] == 1:
                pct = 100
            else:
                pct = kernel_event['dur'] / cpu_op['gpu_busy_time'] * 100
            kernel_details['Percent of Parent cpu_op busy time (%)'] = pct

        # 3. get nn.Module event
        nn_module_event = None
        if nn_module_detail and self.add_python_func:
            # Attempt to find the nn.Module parent event.
            evt = kernel_event
            while evt:
                if self.event_to_category(evt) == 'python_function' and evt['name'].startswith('nn.Module:'):
                    nn_module_event = evt
                    break
                evt = self.tree.get_parent_event(evt)

        # add nn.Module details
        if nn_module_event and nn_module_detail:
            kernel_details['Parent nn.Module UID'] = nn_module_event['UID']
            kernel_details['Parent nn.Module'] = nn_module_event['name']
            if nn_module_event.get('gpu_busy_time') is None:
                # If the nn.Module event does not have GPU busy time, compute it.
                gpu_events = [self.tree.get_UID2event(uid) for uid in nn_module_event.get('gpu_events', [])]
                nn_module_event['gpu_busy_time'] = GPUEventAnalyser(gpu_events).compute_metrics()['busy_time']
            if nn_module_event.get('kernel_count') is None:
                # If the nn.Module event does not have kernel count, compute it.
                nn_module_event['kernel_count'] = len(nn_module_event.get('gpu_events', []))
            kernel_details['Parent nn.Module kernel count'] = nn_module_event.get('kernel_count', 0)
            kernel_details['Parent nn.Module GPU busy time (µs)'] = nn_module_event.get('gpu_busy_time')
            if nn_module_event['kernel_count'] == 1:
                pct = 100
            else:
                pct = kernel_event['dur'] / nn_module_event['gpu_busy_time'] * 100
            kernel_details['Percent of Parent nn.Module busy time (%)'] = pct
        return kernel_details

    def get_df_kernels(self,
                       launcher_detail=False,
                       cpu_op_detail=True,
                       nn_module_detail=False):
        """
        Build a DataFrame with kernel details augmented with
        additional information such as launcher, CPU operation,
        and nn.Module details.
        Args:
            launcher_detail (bool): If True, include details of the kernel's launcher.
            cpu_op_detail (bool): If True, include details of the parent CPU operation.
            nn_module_detail (bool): If True, include details of the parent nn.Module event.

        Returns:
            pd.DataFrame: A DataFrame containing detailed kernel information and aggregated metrics.
        """
        if self.with_python_stack:
            raise ValueError("This method does not support traces with Python stack events at the moment.")
        kernel_details_list = []

        # Extract details for all kernel events.
        for event in self.tree.events:
            if self.event_to_category(event) != 'kernel':
                continue
            details = self.get_kernel_details(event,
                                                launcher_detail=launcher_detail,
                                                cpu_op_detail=cpu_op_detail,
                                                nn_module_detail=nn_module_detail)
            kernel_details_list.append(details)

        df_kernel_view = pd.DataFrame(kernel_details_list)
        for col in df_kernel_view.columns:
            if 'UID' in col or 'count' in col:
                df_kernel_view[col] = df_kernel_view[col].astype('Int64')
        df_kernel_view.reset_index(drop=True, inplace=True)
        return df_kernel_view

    def build_nn_module_latency_tree(self, root_nn_module: Dict[str, Any]):
        """
        Compute the GPU time metrics for a subtree of nn.Module events rooted at the provided event.
        We populate the nn.Module events with the following metrics:
        - 'GPU Time': the total GPU busy time of the subtree rooted at the nn.Module event.
        - 'nn Parent GPU Time': the total GPU busy time of the parent nn.Module event.
        - 'Non-nn.Module GPU Time': the GPU busy time not attributed to nn.Module children if any.

        """
        if not self.add_python_func:
            raise ValueError("This method requires the trace to include Python function events.")
        if not self.tree._is_nn_module_event(root_nn_module):
            raise ValueError("The provided root event is not an nn.Module event.")
        self._build_nn_modules_subtree_recursive(root_nn_module)

    def _build_nn_modules_subtree_recursive(self, node: Dict[str, Any], parent_gpu_time=None):
        gpu_events_subtree_UIDs = node.get('gpu_events', [])
        gpu_events_subtree = [self.tree.get_UID2event(uid) for uid in gpu_events_subtree_UIDs]
        gpu_time = GPUEventAnalyser(gpu_events_subtree).compute_metrics()['busy_time']
        node['GPU Time'] = gpu_time
        node['nn Parent GPU Time'] = parent_gpu_time

        # nn_module_children = node.get('nn_module_children', [])
        nn_module_children = self.tree.get_nn_module_children(node)
        if not nn_module_children:
            return

        for i, child_UID in enumerate(nn_module_children):
            child = self.tree.get_UID2event(child_UID)
            self._build_nn_modules_subtree_recursive(child, parent_gpu_time=gpu_time)

        # Account for GPU time not attributed to nn.Module children.
        union_gpu_events_childrenUIDs = set()
        for child_UID in nn_module_children:
            union_gpu_events_childrenUIDs.update(self.tree.get_UID2event(child_UID).get('gpu_events', []))
        remaining_gpu_events_UIDs = set(gpu_events_subtree_UIDs) - union_gpu_events_childrenUIDs
        if remaining_gpu_events_UIDs:
            gpu_events_remaining = [self.tree.get_UID2event(uid) for uid in remaining_gpu_events_UIDs]
            gpu_time_remaining = GPUEventAnalyser(gpu_events_remaining).compute_metrics()['busy_time']
            node['Non-nn.Module GPU Time'] = gpu_time_remaining
        return