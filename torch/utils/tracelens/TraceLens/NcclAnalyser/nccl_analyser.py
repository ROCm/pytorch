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

import gzip
import os
import json
import pandas as pd
import warnings
import gzip

from ..util import DataLoader

def list_to_tuple(obj):
    if isinstance(obj, list):
        return tuple(list_to_tuple(item) for item in obj)
    return obj

class NcclAnalyser:
    def __init__(self, list_profile_filepaths, world_size):
        self.list_profile_filepaths = list_profile_filepaths
        self.world_size = world_size

        # Byte sizes per dtype
        self.dtype2bytes = {
            "Float": 4, "Int": 4, "Long": 8, "BFloat16": 2, "Bool": 1,
            "Byte": 1, "Double": 8, "Half": 2, "Short": 2
        }

        # Scaling factors for recognized collectives
        self.collective2scaling_factor = {
            'allreduce':     lambda n: 2 * (n - 1) / n,
            'reducescatter': lambda n: (n - 1) / n,
            'allgather':     lambda n: (n - 1) / n,
            'alltoall':      lambda n: (n - 1) /  n,
        }

        # Known names => "category"
        self.collective_type2name = {
            'allreduce':     ['allreduce', 'allreduce_coalesced'],
            'reducescatter': ['reducescatter', '_reduce_scatter_base', 'reduce_scatter_tensor_coalesced'],
            'allgather':     ['allgather', 'all_gather', '_allgather_base', 'all_gather_into_tensor_coalesced', 'allgather_into_tensor_coalesced'],
            'alltoall':      ['all_to_all'],
            'alltoallv':     ['all_to_allv'],
        }

        self.collective_name2type = {
            name: cat for cat, names in self.collective_type2name.items()
            for name in names
        }
        self.implicit_sync_cat = {'allreduce', 'reducescatter', 'allgather', 'alltoall'}
        # Filter function: keep only kernel events with "nccl" in the name
        self.filter_event_fn = self._nccl_filter_event_fn

        # Internal storage
        self.rank2trace_data = {}  # Stores per-rank data
        self.load_trace_data()

    def _nccl_filter_event_fn(self, event):
        """Filters NCCL kernel events."""
        is_nccl_kernel = event.get('cat') == 'kernel' and 'nccl' in event.get('name', '').lower()
        is_linked = event.get('args', {}).get('External id') is not None
        return is_nccl_kernel and is_linked

    def load_trace_data(self):
        """Loads NCCL JSON trace data and extracts relevant events."""
        print(f"Make sure the rank to file mapping is correct as incorrect mapping may lead to unexpected results.")
        print('Also note that we need all ranks for the analysis. We will add a fallback soon for lesser features for single rank or partial data.')
        self.rank2trace_data.clear()
        for rank, filepath in enumerate(self.list_profile_filepaths):
            print(f"Loading rank {rank} from {filepath}")
            raw_data = DataLoader.load_data(filepath)

            nccl_events = [e for e in raw_data['traceEvents'] if self._nccl_filter_event_fn(e)]

            # Build a dictionary with event data
            rank_dict = {idx: evt for idx, evt in enumerate(nccl_events)}
            self.rank2trace_data[rank] = rank_dict

    # ------------------------------------------------------------------------
    # Step 1: Build a long table where each row is a collective event on a rank
    # ------------------------------------------------------------------------
    def build_df_long(self):
        """Constructs a long table where each row is a collective event on a rank."""
        metadata_fields = ['Process Group Name', 'Process Group Ranks', 'Collective name', 'Group size',
                           'dtype', 'In msg nelems', 'Out msg nelems', 'In split size', 'Out split size',
                            'stream']
        rows = []
        for rank in self.rank2trace_data:
            for cid, evt in self.rank2trace_data[rank].items():
                row = {'ts': evt['ts'], 'dur': evt['dur'], 'rank': rank}
                for field in metadata_fields:
                    if field in evt['args']:
                        field_value = evt['args'][field]
                    else:
                        field_value = None
                    if isinstance(field_value, list):
                        field_value = list_to_tuple(field_value)
                    row[field] = field_value
                bytes_per_elem = self.dtype2bytes[row['dtype']] if row['dtype'] in self.dtype2bytes else None
                if bytes_per_elem is not None and row['In msg nelems'] is not None:
                    row['In msg size (MB)'] = row['In msg nelems'] * bytes_per_elem / 1024**2
                    row['Out msg size (MB)'] = row['Out msg nelems'] * bytes_per_elem / 1024**2
                else:
                    row['In msg size (MB)'] = None
                    row['Out msg size (MB)'] = None
                rows.append(row)

        df_long = pd.DataFrame(rows)
        df_long = df_long.reset_index(drop=True)

        # Assign an index within each process group and rank
        df_long['Process Group Name'] = df_long['Process Group Name'].fillna('Unknown_Group')
        df_long['index_in_group'] = df_long.groupby(['Process Group Name', 'rank'])['ts'].rank(method='first').astype(int) - 1

        # Create a composite collective ID (process group + index)
        df_long['collective_id'] = df_long['Process Group Name'] + '_' + df_long['index_in_group'].astype(str)

        desired_col_order = [
            "collective_id", "index_in_group", "rank",
            "Process Group Name", "Process Group Ranks",
            "Collective name", "Group size", "dtype",
            "In msg nelems", "In msg size (MB)",
            "Out msg nelems", "Out msg size (MB)",
            "In split size", "Out split size",
            "stream", "ts", "dur"
        ]
        df_long = df_long[desired_col_order]
        self.df_per_rank_coll = df_long
        return df_long

    def build_df_summary_long(self, agg_metrics=['mean', 'std', 'min', 'max'], 
                            group_by_cols=['rank', 'Process Group Name', 'Process Group Ranks', 
                                        'Collective name', 'Group size', 'dtype',
                                        'In msg nelems', 'Out msg nelems', 'In split size', 
                                        'Out split size', 'stream'],
                            include_count=True):
        """
        Summarizes df_long based on specified aggregation metrics.
        
        Parameters:
        -----------
        agg_metrics : list
            List of aggregation metrics to apply (e.g., 'mean', 'std', 'min', 'max')
        group_by_cols : list
            Columns to group by
        include_count : bool
            Whether to include count of operations in the summary
        
        Returns:
        --------
        DataFrame
            Summarized communications data
        """
        if not hasattr(self, 'df_per_rank_coll'):
            self.build_df_long()
        
        df = self.df_per_rank_coll
        
        # Define aggregations for different column types
        agg_dict = {
            'In msg size (MB)': 'first',  # should match across instances
            'Out msg size (MB)': 'first',
            'dur': ['sum']+agg_metrics  # Duration in microseconds
        }
        
        # Add count if requested
        if include_count:
            agg_dict['collective_id'] = 'count'
        
        # Group and aggregate
        summary_df = df.groupby(group_by_cols).agg(agg_dict)
        
        # Flatten the multi-index columns
        summary_df.columns = ['_'.join(col).strip() for col in summary_df.columns.values]
        
        # Rename the count column for clarity
        if include_count:
            summary_df = summary_df.rename(columns={'collective_id_count': 'operation_count'})
        
        # Sort by total duration (descending)
        summary_df = summary_df.sort_values(by='dur_sum', ascending=False)
        
        summary_df = summary_df.reset_index()
        self.df_summary_long = summary_df
        
        return summary_df

    # ------------------------------------------------------------------------
    # Step 2: Build a wide table for implicit sync class
    # where each row is a collective operation
    # ------------------------------------------------------------------------
    def build_df_nccl_implicit_sync_cat(self, detailed=False):
        """
        Builds a single DF with one row *per collective ID*, including per-rank ts/dur + metadata.
        Ensures metadata consistency across ranks.
        """
        if not hasattr(self, 'df_per_rank_coll'):
            self.build_df_long()

        df = self.df_per_rank_coll

        metadata_fields = ['Process Group Name', 'Process Group Ranks', 'Collective name', 'Group size',
                           'dtype', 'In msg nelems', 'Out msg nelems', 'In msg size (MB)', 'Out msg size (MB)']
        collective_ids = df['collective_id'].unique()
        rows = []

        for cid in collective_ids:
            rank_events = df[df['collective_id'] == cid]
            rank_events = rank_events.set_index('rank')

            # Skip if the collective type is not in the implicit sync category
            collective_name = rank_events.iloc[0]['Collective name']
            if self.collective_name2type.get(collective_name) not in self.implicit_sync_cat:
                continue

            # **Metadata Consistency Check**
            ref_metadata = {field: rank_events.iloc[0][field] for field in metadata_fields}
            for field in metadata_fields:
                unique_values = rank_events[field].unique()
                if len(unique_values) > 1:
                    raise ValueError(f"Metadata mismatch in '{field}' for collective {cid}: {unique_values}")

            row = {'collective_id': cid, **ref_metadata}

            # Compute per-rank timestamps and durations
            for r in rank_events.index:
                row[f'rank_{r}_ts'] = rank_events.loc[r, 'ts']
                row[f'rank_{r}_dur'] = rank_events.loc[r, 'dur']

            # Compute communication latency
            latest_start = max(row.get(f'rank_{r}_ts', 0) for r in rank_events.index)
            earliest_end = min(row.get(f'rank_{r}_ts', 0) + row.get(f'rank_{r}_dur', 0) for r in rank_events.index)
            row['comm_latency'] = min(row[f'rank_{r}_dur'] for r in rank_events.index)

            # Compute per-rank wait time
            for r in rank_events.index:
                row[f'rank_{r}_wait_time'] = latest_start - row.get(f'rank_{r}_ts', 0)

            # Compute max wait time and rank
            max_wait, max_wait_rank = max((row[f'rank_{r}_wait_time'], r) for r in rank_events.index)
            row['skew in start time'] = max_wait
            row['earliest arrival rank'] = max_wait_rank
            row['avg_wait_time'] = sum(row[f'rank_{r}_wait_time'] for r in rank_events.index) / len(rank_events.index)

            # Compute end time spread
            latest_end = max(row.get(f'rank_{r}_ts', 0) + row.get(f'rank_{r}_dur', 0) for r in rank_events.index)
            row['skew in end time'] = latest_end - earliest_end

            # Compute algorithmic and bus bandwidth
            c_type = self.collective_name2type.get(row['Collective name'])
            row['Full msg size (MB)'] = row['Out msg size (MB)'] if c_type == 'allgather' else row['In msg size (MB)']
            row['algo bw (GB/s)'] = (row['Full msg size (MB)']/1024) / (row['comm_latency'] / 1e6)
            scaling_factor = self.collective2scaling_factor[c_type](row['Group size'])
            row['bus bw (GB/s)'] = row['algo bw (GB/s)'] * scaling_factor

            rows.append(row)

        df = pd.DataFrame(rows).reset_index(drop=True)

        # Separate per-rank columns
        per_rank_cols = [col for col in df.columns if col.startswith('rank_')]
        # Define explicit order for general (non-rank) columns
        general_cols = [
            # Collective Identifier & Metadata
            "collective_id", "Process Group Name", "Process Group Ranks",
            "Collective name", "Group size", "dtype",
            "In msg nelems", "Out msg nelems", "In msg size (MB)", "Out msg size (MB)", "Full msg size (MB)",

            # High-Level Performance Metrics
            "comm_latency", "skew in start time", "earliest arrival rank",
            "avg_wait_time", "skew in end time", "algo bw (GB/s)", "bus bw (GB/s)"
        ]

        # Reorder columns: General metadata + performance metrics + per-rank details
        ordered_cols = general_cols + per_rank_cols
        df = df[ordered_cols]

        self.df_implicit_sync_cat_detailed = df
        self.df_implicit_sync_cat = df.drop(columns=per_rank_cols)

        return self.df_implicit_sync_cat if not detailed else self.df_implicit_sync_cat_detailed


    def build_df_summary_nccl_implicit_sync_cat(self, agg_metrics=['mean', 'std'],
                                                metadata_fields=["Process Group Name", "Group size", "Full msg size (MB)"]):
        """
        Builds a summary DF with one row per collective name, dtype, and msg size.
        Aggregates across all collectives and ranks.
        """
        if not hasattr(self, 'df_implicit_sync_cat'):
            self.df_implicit_sync_cat = self.build_df_nccl_implicit_sync_cat()

        # Aggregation logic

        df = self.df_implicit_sync_cat
        agg_logic = {
            'comm_latency': agg_metrics + ['size', lambda x: x.sum() / 1000],  # Size and sum (convert to ms)
            'skew in start time': agg_metrics,
            'skew in end time': agg_metrics,
            'algo bw (GB/s)': agg_metrics,
            'bus bw (GB/s)': agg_metrics,
        }
        metric_fields = list(agg_logic.keys()).copy()
        for col in metadata_fields:
            agg_logic[col] = 'first'

        groupby_cols = ['Collective name', 'dtype', 'In msg nelems']
        agg_result = df.groupby(groupby_cols).agg(agg_logic)

        # Post-processing: rename columns and sort

        agg_result.columns = [
            f"{col[0]}_{col[1]}" if col[1] != '' else col[0]
            for col in agg_result.columns
        ]
        column_renames = {
            'comm_latency_<lambda_0>': 'Total comm latency (ms)',
            'comm_latency_size': 'count',
        }
        for col in metadata_fields:
            column_renames[col + '_first'] = col

        agg_result.rename(columns=column_renames, inplace=True)
        summary_df = agg_result.reset_index()
        summary_df = summary_df.sort_values(by='Total comm latency (ms)', ascending=False)
        columns_order = groupby_cols + metadata_fields
        for group in metric_fields:
            for agg in agg_metrics:
                columns_order.append(f"{group}_{agg}")
        columns_order.extend(['count', 'Total comm latency (ms)'])
        summary_df = summary_df[columns_order]
        return summary_df

    def build_df_nccl_all2allv(self, detailed=False):
        # this is diff from implicit sync cat
        # first, each rank can send and receive different amount of data
        # as a result they do not respect the implicit sync cat
        # we cannot calculate comm latency as min dur
        # thus we cannot calculate algo bw and bus bw
        # as we discuss and understand the metrics, we can add them
        # for now we expose raw data and leave the calculations to the user
        # we will add some basic metrics for now

        if not hasattr(self, 'df_per_rank_coll'):
            self.build_df_long()

        df = self.df_per_rank_coll

        metadata_fields = ['Process Group Name', 'Process Group Ranks', 'Collective name', 'Group size',
                           'dtype', 'stream']
        collective_ids = df['collective_id'].unique()

        rows = []
        for cid in collective_ids:
            rank_events = df[df['collective_id'] == cid]
            rank_events = rank_events.set_index('rank')

            collective_name = rank_events.iloc[0]['Collective name']
            if collective_name != 'all_to_allv':
                continue

            # **Metadata Consistency Check**
            ref_metadata = {field: rank_events.iloc[0][field] for field in metadata_fields}
            for field in metadata_fields:
                unique_values = rank_events[field].unique()
                if len(unique_values) > 1:
                    raise ValueError(f"Metadata mismatch in '{field}' for collective {cid}")

            # **Common metadata**
            row = {'collective_id': cid, **ref_metadata}

            # Per-rank columns
            per_rank_cols = ['ts', 'dur', 'In msg nelems', 'Out msg nelems', 'In msg size (MB)', 'Out msg size (MB)',
                             'In split size', 'Out split size']
            for r in rank_events.index:
                for col in per_rank_cols:
                    row[f'rank_{r}_{col}'] = rank_events.loc[r, col]

            # agg latency metrics
            earliest_start = min(row[f'rank_{r}_ts'] for r in rank_events.index)
            latest_start = max(row[f'rank_{r}_ts'] for r in rank_events.index)
            earliest_end = min(row[f'rank_{r}_ts'] + row[f'rank_{r}_dur'] for r in rank_events.index)
            latest_end = max(row[f'rank_{r}_ts'] + row[f'rank_{r}_dur'] for r in rank_events.index)
            row['skew in start time'] = latest_start - earliest_start
            row['skew in end time'] = latest_end - earliest_end

            # 1) For the entire cohort, record the earliest starter's start time (S) and the earliest finisher's finish time (F)
            # 2)  For every rank report its start skew w.r.t S and its end skew w.r.t F
            for r in rank_events.index:
                row[f'rank_{r}_skew in start time'] = row[f'rank_{r}_ts'] - earliest_start
                row[f'rank_{r}_skew in end time'] = latest_end - (row[f'rank_{r}_ts'] + row[f'rank_{r}_dur'])

            # data size metrics
            total_in_size = sum(row[f'rank_{r}_In msg size (MB)'] for r in rank_events.index)
            total_in_nelems = sum(row[f'rank_{r}_In msg nelems'] for r in rank_events.index)
            row['total data communicated (MB)'] = total_in_size
            row['total nelems communicated'] = total_in_nelems

            rows.append(row)

        if len(rows) == 0:
            warnings.warn("No all_to_allv collectives found in the trace data.")
            return None

        df = pd.DataFrame(rows).reset_index(drop=True)
        per_rank_cols = [col for col in df.columns if col.startswith('rank_')]
        general_cols = [
            "collective_id", "Process Group Name", "Process Group Ranks",
            "Collective name", "Group size", "dtype", "stream",
            "total data communicated (MB)", "total nelems communicated",
            "skew in start time", "skew in end time"
        ]
        ordered_cols = general_cols + per_rank_cols
        df = df[ordered_cols]
        df = df.sort_values(by='total data communicated (MB)', ascending=False)
        self.df_all2allv_detailed = df
        return self.df_all2allv_detailed if detailed else df.drop(columns=per_rank_cols)
