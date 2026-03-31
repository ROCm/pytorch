#!/usr/bin/env python3

import re
import sys
import pandas as pd
import argparse
import os.path


def parse_args():
    parser = argparse.ArgumentParser(description = 'Update missed/skipped tests causes/assignees')
    parser.add_argument('-i', '--input', required = True, help = 'input file')
    parser.add_argument('-s', '--source', required = True, help = 'skip reasons original source file')
    parser.add_argument('-o', '--output', required = True, help = 'updated skip reasons output file')
    return parser.parse_args()


def write_csv_file(tests, filename, sep = '\t'):
    exist = os.path.exists(filename)
    with open (filename, 'a') as daily_summary:
        if not exist:
            #the first row (i.e., column title) for the csv
            first_row = ["test_file", "test_name", "test_class", "skip_reason", "assignee", "comments"]
            first_row = sep.join(first_row)
            daily_summary.write('{}\n'.format(first_row))
        for test in tests:
            # test_name may contains "\t" already. Need to keep quotes.
            with_quotes = False
            if "\t" in test['test_name']:
                test['test_name'] = '"' + test['test_name'] + '"'
                with_quotes = True
            test_details = sep.join([test['test_file'], test['test_name'], test['test_class'], test['skip_reason'], test['assignee'], test['comments']])
            if with_quotes:
                daily_summary.write(test_details)
                daily_summary.write("\n")
            else:
                daily_summary.write('{}\n'.format(test_details))


def main():

    # parse args
    global args
    args = parse_args()

    latest_info = pd.read_csv(args.input)
    latest_info = latest_info.to_dict(orient = "records")

    original_csv_info = pd.read_csv(args.source, sep = '\t')
    original_csv_info = original_csv_info.to_dict(orient = "records")

    new_test_cases = []
    for item in latest_info:
        if not item.__contains__('comments'):
            item['comments'] = " "
        # in case it misses some information, just skip it
        if pd.isna(item['test_file']) or pd.isna(item['test_name']) or pd.isna(item['test_class']):
            continue
        matched = False
        for csv_info in original_csv_info:
            # if match
            if item['test_file'] == csv_info['test_file'] and item['test_name'] == csv_info['test_name'] and item['test_class'] == csv_info['test_class']:
                matched = True
                # no skipped_reason, assignee and comments info anymore, remove item in skip_reasons.csv file
                if (pd.isna(item['skip_reason']) or item['skip_reason'] == ' ') and (pd.isna(item['assignee']) or item['assignee'] == ' ') and (pd.isna(item['comments']) or item['comments'] == ' '):
                    original_csv_info.remove(csv_info)
                # update info
                else:
                    if not pd.isna(item['skip_reason']):
                        csv_info['skip_reason'] = item['skip_reason']
                    else:
                        csv_info['skip_reason'] = ' '
                    if not pd.isna(item['assignee']):
                        csv_info['assignee'] = item['assignee']
                    else:
                        csv_info['assignee'] = ' '
                    if not pd.isna(item['comments']):
                        csv_info['comments'] = item['comments']
                    else:
                        csv_info['comments'] = ' '

        if not matched:
            if (pd.isna(item['skip_reason']) or item['skip_reason'] == ' ') and (pd.isna(item['assignee']) or item['assignee'] == ' ') and (pd.isna(item['comments']) or item['comments'] == ' '):
                pass
            else:
                new_test_cases.append(item)

    combined_info = original_csv_info + new_test_cases
    for test in combined_info:
        if pd.isna(test['skip_reason']):
            test['skip_reason'] = ' '
        if pd.isna(test['assignee']):
            test['assignee'] = ' '
        if not test.__contains__('comments') or pd.isna(test['comments']):
            test['comments'] = ' '


    write_csv_file(combined_info, args.output)


if __name__ == "__main__":
    main()
