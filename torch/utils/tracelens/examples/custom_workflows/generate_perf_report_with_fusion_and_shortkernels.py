import argparse
import json
import pandas as pd
import numpy as np
from TraceLens import TreePerfAnalyzer
from TraceLens.PerfModel import dict_cat2names


def get_next_host_op(perf_analyzer, host_op):
    """
    Given a host op we get host op of the gpu event (executed in the same stream)
    after all the gpu events launched by this host op
    """
    gpu_event_uids = host_op.get('gpu_events')
    if gpu_event_uids is None:
        raise ValueError("Host op does not have gpu events")
    gpu_events = [perf_analyzer.tree.get_UID2event(uid) for uid in gpu_event_uids]
    gpu_streams = [e['args']['stream'] for e in gpu_events]
    # as a simplifying assumption we assume all gpu events are in the same stream
    assert len(set(gpu_streams)) == 1, "Not all GPU events are in the same stream"
    gpu_stream = gpu_streams[0]
    sorted_gpu_events = sorted(gpu_events, key=lambda x: x['ts'])
    last_event = sorted_gpu_events[-1]
    stream_index = last_event['args']['stream_index']
    next_index = stream_index + 1
    next_gpu_event = perf_analyzer.tree.dict_stream_index2event.get((gpu_stream, next_index), None)
    assert next_gpu_event is not None, "No next gpu event found in the stream"
    # lets get the parent host op of this next gpu event
    # tree is like host op -> runtime op (cuda/ hip launch) -> gpu op
    next_gpu_event_launcher = perf_analyzer.tree.get_parent_event(next_gpu_event)
    assert next_gpu_event_launcher is not None, "No launcher event found for the next gpu event"
    next_gpu_event_host_op = perf_analyzer.tree.get_parent_event(next_gpu_event_launcher)
    assert next_gpu_event_host_op is not None, "No host op found for the next gpu event"
    return next_gpu_event_host_op


def get_df_fusion_opportunity(perf_analyzer, fusion_op_names):

    fusion_candidates = [e for e in perf_analyzer.tree.events if e['name'] in fusion_op_names]

    list_dicts_info = []

    for evt in fusion_candidates:
        gpu_events = [
            perf_analyzer.tree.get_UID2event(uid) for uid in evt.get('gpu_events', [])
        ]
        last_kernel = gpu_events[-1] if gpu_events else None
        try:
            next_host_op = get_next_host_op(perf_analyzer, evt)
        except Exception as e:
            print(f"Error getting fusion post op for event {evt['UID']}: {e}")
            continue
        next_gpu_events = [
            perf_analyzer.tree.get_UID2event(uid) for uid in next_host_op.get('gpu_events', [])
        ]
        first_next_kernel = next_gpu_events[0] if next_gpu_events else None

        if last_kernel and first_next_kernel:
            dict_info = {
                'fusion candidate': evt['name'],
                'fusion candidate uid': evt['UID'],
                'last kernel by fusion candidate': last_kernel['name'],
                'last kernel by fusion candidate uid': last_kernel['UID'],
                'last kernel by fusion candidate stream': last_kernel['args'].get('stream'),
                'last kernel by fusion candidate stream index': last_kernel['args'].get('stream_index'),
                'last kernel by fusion candidate duration': last_kernel['dur'],
                'fusion post op': next_host_op['name'],
                'fusion post op uid': next_host_op['UID'],
                'fusion post op input dims': next_host_op['args'].get('Input Dims'),
                'first kernel by fusion post op': first_next_kernel['name'],
                'first kernel by fusion post op uid': first_next_kernel['UID'],
                'first kernel by fusion post op stream': first_next_kernel['args'].get('stream'),
                'first kernel by fusion post op stream index': first_next_kernel['args'].get('stream_index'),
                'first kernel by fusion post op duration': first_next_kernel['dur'],
            }
            list_dicts_info.append(dict_info)
        
    # Create DataFrame from the list of dictionaries
    df_fusion_study = pd.DataFrame(list_dicts_info)
    df_fusion_study['combo'] = df_fusion_study['fusion candidate'] + ' -> ' + df_fusion_study['fusion post op']
    df_fusion_study_grouped = df_fusion_study.groupby(['combo']).agg(
        {
            'first kernel by fusion post op duration': ['mean', 'std', 'sum', 'count'],
        }
    ).reset_index()

    df_fusion_study_grouped.columns = ['_'.join(col).strip() for col in df_fusion_study_grouped.columns.values]

    df_fusion_study_grouped['estimated saved time percent'] = (
        df_fusion_study_grouped['first kernel by fusion post op duration_sum'] / (perf_analyzer.total_time_ms * 1e3) * 100
    )
    return df_fusion_study_grouped

def get_dfs_short_kernels(perf_analyzer, short_kernel_threshold_us=10, histogram_bins=100, topk=None):
    df_kernels = perf_analyzer.get_df_kernels()
    df_filtered = df_kernels[df_kernels['Kernel duration (µs)'] < short_kernel_threshold_us]

    # 1. get histogram of these short kernels
    vals = df_filtered['Kernel duration (µs)'].values
    counts, bin_edges = np.histogram(vals, bins=histogram_bins)
    df_hist = pd.DataFrame({
        "bin_start": bin_edges[:-1],
        "bin_end": bin_edges[1:],
        "count": counts
    })

    # 2. get df short kernels topk by total time
    agg_dict = {
        'Kernel duration (µs)': ['sum', 'count', 'mean'],
    }
    df_grouped = df_filtered.groupby(['Parent cpu_op', 'Input dims', 'Input strides', 'Concrete Inputs', 'Kernel name'], sort=False).agg(agg_dict)

    # Flatten multi-level column names
    df_grouped.columns = ['_'.join(col).strip() for col in df_grouped.columns]

    # Rename columns for clarity
    df_grouped.rename(columns={
        'Kernel duration (µs)_sum':  'Short Kernel duration (µs) sum',
        'Kernel duration (µs)_count': 'Short Kernel count',
        'Kernel duration (µs)_mean': 'Short Kernel duration (µs) mean'
    }, inplace=True)

    # Add percentage contribution to total time
    df_grouped['Short Kernel duration (µs) percent of total time'] = (
        df_grouped['Short Kernel duration (µs) sum'] / (perf_analyzer.total_time_ms * 1e3) * 100
    )

    # Sort and format
    df_grouped.sort_values(by='Short Kernel duration (µs) sum', ascending=False, inplace=True)
    df_grouped.reset_index(inplace=True)
    if topk is not None:
        df_grouped = df_grouped.head(topk)
    return df_hist, df_grouped


def main():
    parser = argparse.ArgumentParser(description='Process a JSON trace profile and generate performance report tables.')
    parser.add_argument('--profile_json_path', type=str, required=True, help='Path to the profile.json file')
    parser.add_argument('--output_xlsx_path', type=str, required=True, help='Path to the output Excel file')

    # optional arguments
    parser.add_argument('--topk_ops', type=int, default=None,
                        help='Rows to keep in the unique-args launcher table.')
    parser.add_argument('--short_kernel_threshold_us', type=float, default=10,
                        help='Cut-off (µs) to classify a kernel as "short".')
    parser.add_argument('--short_kernel_histogram_bins', type=int, default=100,
                        help='Number of bins for the short-kernel histogram.')
    parser.add_argument('--topk_short_kernels', type=int, default=None,
                        help='Rows to keep in the short-kernel table.')
    parser.add_argument('--fusion_op_names', type=str, default=
                        'aten::miopen_convolution,aten::miopen_batch_norm',
                        help='Comma-separated op names considered for fusion.')
    parser.add_argument('--topk_roofline_ops', type=int, default=None,
                        help='Rows to keep in the roofline table.')

    args = parser.parse_args()

    perf_analyzer = TreePerfAnalyzer.from_file(args.profile_json_path)

    agg_metrics = ['mean', 'median', 'std', 'min', 'max']

    # Generate base DataFrames
    df_gpu_timeline = perf_analyzer.get_df_gpu_timeline()

    total_time_row = df_gpu_timeline[df_gpu_timeline['type'] == 'total_time']
    total_time_ms = total_time_row['time ms'].values[0]
    perf_analyzer.total_time_ms = total_time_ms


    df_kernel_launchers = perf_analyzer.get_df_kernel_launchers(include_kernel_names=True)
    df_kernel_launchers_summary = perf_analyzer.get_df_kernel_launchers_summary(df_kernel_launchers)
    df_kernel_launchers_unique_args = perf_analyzer.get_df_kernel_launchers_unique_args(df_kernel_launchers, 
                                                                                        agg_metrics=agg_metrics, 
                                                                                        include_pct=True)
    if args.topk_ops is not None:
        df_kernel_launchers_unique_args = df_kernel_launchers_unique_args.head(args.topk_ops)

    # Short kernel study
    df_hist, df_short_kernels = get_dfs_short_kernels(perf_analyzer, short_kernel_threshold_us=args.short_kernel_threshold_us,
                                                      histogram_bins=args.short_kernel_histogram_bins,
                                                      topk=args.topk_short_kernels)
    
    # Fusion opportunity study
    fusion_op_names = [name.strip() for name in args.fusion_op_names.split(',')]
    df_fusion_details = get_df_fusion_opportunity(perf_analyzer, fusion_op_names)

    # Roofline study
    op_dfs = {}
    for op_cat, op_names in dict_cat2names.items():
        # Filter events belonging to the current category
        op_events = [event for event in perf_analyzer.tree.events if event['name'] in op_names]

        if op_cat in ['GEMM', 'UnaryElementwise', 'BinaryElementwise']:
            # For GEMM: create a single table that covers both fwd and bwd.
            df_ops = perf_analyzer.build_df_perf_metrics(op_events, bwd=False, include_kernel_names=True, include_args=True)
            df_ops = perf_analyzer.summarize_df_perf_metrics(df_ops, agg_metrics)
            if args.topk_roofline_ops is not None:
                df_ops = df_ops.head(args.topk_roofline_ops)
            op_dfs[op_cat] = df_ops
        else:
            # For FLASH_ATTN and CONV: create separate tables for forward and backward passes.
            df_ops_fwd = perf_analyzer.build_df_perf_metrics(op_events, bwd=False, include_kernel_names=True, include_args=True)
            df_ops_fwd = perf_analyzer.summarize_df_perf_metrics(df_ops_fwd, agg_metrics)
            if args.topk_roofline_ops is not None:
                df_ops_fwd = df_ops_fwd.head(args.topk_roofline_ops)
            df_ops_bwd = perf_analyzer.build_df_perf_metrics(op_events, bwd=True, include_kernel_names=True, include_args=True)
            df_ops_bwd = perf_analyzer.summarize_df_perf_metrics(df_ops_bwd, agg_metrics)
            if args.topk_roofline_ops is not None:
                df_ops_bwd = df_ops_bwd.head(args.topk_roofline_ops)
            op_dfs[f"{op_cat}_fwd"] = df_ops_fwd
            op_dfs[f"{op_cat}_bwd"] = df_ops_bwd

    # Write all DataFrames to separate sheets in an Excel workbook
    with pd.ExcelWriter(args.output_xlsx_path) as writer:
        df_gpu_timeline.to_excel(writer, sheet_name='gpu_timeline', index=False)
        df_kernel_launchers_summary.to_excel(writer, sheet_name='ops_summary', index=False)
        if args.topk_ops is not None:
            name = f'ops_topk{args.topk_ops}'
        else:
            name = 'ops_all'
        df_kernel_launchers_unique_args.to_excel(writer, sheet_name=name, index=False)   
        df_hist.to_excel(writer, sheet_name='short_kernels_histogram', index=False)
        if args.topk_short_kernels is not None:
            name = f'short_kernels_topk{args.topk_short_kernels}_details'
        else:
            name = 'short_kernels_all_details'
        df_short_kernels.to_excel(writer, sheet_name=name, index=False)
        df_fusion_details.to_excel(writer, sheet_name='fusion_candidates_details', index=False)
        # Write each op category DataFrame
        for sheet_name, df in op_dfs.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"DataFrames successfully written to {args.output_xlsx_path}")

if __name__ == "__main__":
    main()
