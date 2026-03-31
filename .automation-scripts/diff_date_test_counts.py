#!/usr/bin/env python3

import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='compare different log from different date')
    parser.add_argument('-o', '--old', required=True, help='older result log. require result of get_test_counts.py') 
    parser.add_argument('-n', '--new', required=True, help='new result log. require result of get_test_counts.py')
    return parser.parse_args()

def process_log(filename):
    tests={}
    for line in open(filename):
        try:
            parts = line.strip().split()
            module = parts[0]
            status = parts[-1]
            _time_taken = parts[-2]
            name = " ".join(parts[1:-2])
        except:
            print(line)
            print(line.strip().split())
            raise
        tests[name] = [module + " " + name, status]
    return tests

def find_diff(old_tests,new_tests):
    print("running comparison")
    for test in old_tests:
        old_test, old = old_tests[test]
        new_test, new = new_tests.get(test, ["", "9"]) # 9 means not found
        if old != new:
            if new == "9":   #new test case run in the newer log
                print(" REMOVED: ", old_test) 
            elif new=="0":                #was passing but skipped in the newer log
                print(" ENABLED: ", old_test) 
            elif new=="1":                #was skipped but unskipped in the newer log
                print(" SKIPPED: ", old_test)
            elif new=="2":
                print(" XFAILED: ", old_test)
            else:
                assert False

def main():
    args = parse_args()
    old_tests = {}
    new_tests = {}
    
    print("parsing old test list file '%s'" % args.old)
    old_tests=process_log(args.old)
    print("OLD_TEST %d tests" % len(old_tests))
    
    print("parsing new test list file '%s'" % args.new)
    new_tests=process_log(args.new)
    print("NEW_TESTS %d tests" % len(new_tests))

    #compare
    find_diff(old_tests, new_tests)

if __name__ == "__main__":
    main()
