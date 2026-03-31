#!/usr/bin/env python3

import re
import sys
import pandas as pd
import argparse 
import os.path

#added UNEXFAIL case
all_causes = { 
        'SKIPPED': re.compile(r"SKIPPED: .*"),
        'MISSED': re.compile(r"MISSED: .*"),
        'UNEXFAIL': re.compile(r"UNEXFAIL: .*") }

def debug(*args):
    DEBUG = 0
    if DEBUG:
        print("DEBUG : ", *args, end='')

def find_tests_in_log(filename, causes):

    tests_in_log = []
    for line in open(filename):

        match = False
        for cause in causes:
            match = cause.search(line)
            if match:
                break

        if match:
            debug(line)
            test_cause = line.split()[0]
            test_cause = test_cause[:-1]
            test_file = line.split()[1]
            test_name = line.split()[2:-1]
            test_name = ' '.join(test_name)
            test_class = line.split()[-1]
            tests_in_log.append({ 'test_cause':test_cause, 'test_file':test_file, 'test_name':test_name, 'test_class':test_class})
    return tests_in_log

def write_csv_file(tests, filename, sep='\t'):
    exist=os.path.exists(filename)
    with open (filename, 'a') as daily_summary:
        if not exist:
            daily_summary.write("sep={}\n".format(sep)) #setup delimiter for the csv file
            #the first row (i.e., column title) for the csv
            first_row=["test_cause", "test_file", "test_name", "test_class", "skip_reason", "assignee", "comments"]
            first_row=sep.join(first_row)
            daily_summary.write('{}\n'.format(first_row))
        for test in tests:
            test_details=sep.join([test['test_cause'],test['test_file'],test['test_name'],test['test_class'],test['skip_reason'],test['assignee'],test['comments']])
            daily_summary.write('{}\n'.format(test_details))

def parse_args():
    parser = argparse.ArgumentParser(description='Match skipped tests to skip causes')
    parser.add_argument('-i', '--input', nargs='+', default=[],  \
         help='list of filename') 
    parser.add_argument('-s', '--skip-reasons', required=True, help='skip reasons file')
    parser.add_argument('-c', '--causes', nargs='+', default=["SKIPPED"], \
         help='causes can be SKIPPED, MISSED')
    parser.add_argument('-o', '--output', required=True, help='output file')
    parser.add_argument('--showSelectedTestsWithReason', action='store_true', help="keep model directory after run")
    return parser.parse_args()

def main():

    # parse args
    global args
    args = parse_args()

    known_skips = pd.read_csv(args.skip_reasons,sep='\t')
    known_skips = known_skips.to_dict(orient="records")

    selected_causes = []
    for cause in args.causes:
        selected_causes.append( all_causes[cause] )

    for filename in args.input:

        tests = find_tests_in_log(filename, selected_causes)

        skip_reasons_statistics = dict()
        for test in tests:
            for known_skip in known_skips:
                # Warning: Currently only comparing test_name, not test_file or test_class
                if test['test_name'] == known_skip['test_name'] and test['test_file'] == known_skip['test_file'] and test['test_class'] == known_skip['test_class']:
                    # matching here may introduced by skip_reason with or without assignee
                    if known_skip.__contains__('skip_reason') and not pd.isna(known_skip['skip_reason']):
                        #maybe add stringify or string casting here
                        test['skip_reason'] = str(known_skip['skip_reason'])
                        if not known_skip['skip_reason'] in skip_reasons_statistics:
                            skip_reasons_statistics[ known_skip['skip_reason'] ] = 1
                        else:
                            skip_reasons_statistics[ known_skip['skip_reason'] ] += 1
                    if known_skip.__contains__('assignee'):
                        if pd.isna(known_skip['assignee']):
                            test['assignee'] = "ASSIGNEE_NOT_FOUND"
                        else:
                            test['assignee'] = str(known_skip['assignee'])
                    if known_skip.__contains__('comments') and not pd.isna(known_skip['comments']):
                        test['comments'] = str(known_skip['comments'])
                    break
            if 'skip_reason' not in test.keys() or test['skip_reason'] == " ":
                test['skip_reason'] = "REASON_NOT_FOUND"
            if 'assignee' not in test.keys():
                test['assignee'] = "ASSIGNEE_NOT_FOUND"
            if 'comments' not in test.keys():
                test['comments'] = " "

        skip_reasons_statistics.pop(' ', None)

        print( "SELECTED CAUSES SUMMARY" )
        print( "=====================" )
        sorted_skip_reasons_statistics = sorted(skip_reasons_statistics.keys(), key = lambda x : x.lower())
        for skip_reason_entry in sorted_skip_reasons_statistics:
            print( skip_reason_entry, ": ", skip_reasons_statistics[skip_reason_entry] )
        print( "")
        print( "SELECTED REASON NOT FOUND" )
        print( "---------------------" )
        print( "\n".join( [test['test_cause'] + ", " + test['test_file'] + ", " + test['test_name'] + ", " + test['test_class'] + ", " + test['assignee'] for test in tests if test['skip_reason'] == "REASON_NOT_FOUND"]) )
        if args.showSelectedTestsWithReason:
            print( "TESTS WITH KNOWN SELECT REASON" )
            print( "---------------------" )
            print( "\n".join( [test['test_cause'] + ", " + test['test_file'] + ", " + test['test_name'] + ", " + test['test_class'] + ", " + test['assignee'] + ", " + test['skip_reason'] for test in tests if test['skip_reason'] != "REASON_NOT_FOUND"]) )
        print( "=========================================================" )
        
        for test in tests:
            if test['skip_reason'] == "REASON_NOT_FOUND":
                test['skip_reason'] = " "

            if test['assignee'] == "ASSIGNEE_NOT_FOUND":
                test['assignee'] = " "

        write_csv_file(tests,args.output)
    
if __name__ == "__main__":
    main()
