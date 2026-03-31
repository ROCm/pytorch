#!/usr/bin/env python3

import sys

old_tests = {}
new_ref_tests = {}
new_tests = {}

print("parsing old test list file '%s'" % sys.argv[1])
for line in open(sys.argv[1]):
    try:
        parts = line.strip().split()
        module = parts[0]
        status = parts[-1]
        name = " ".join(parts[1:-1])
    except:
        print(line)
        print(line.strip().split())
        raise
    old_tests[name] = [module + " " + name, status]
print("REF_CUDA_TESTS %d tests" % len(old_tests))

print("parsing new ref test list file '%s'" % sys.argv[2])
for line in open(sys.argv[2]):
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
    new_ref_tests[name] = [module + " " + name, status]
print("CUR_CUDA_TESTS %d tests" % len(new_ref_tests))


print("parsing new test list file '%s'" % sys.argv[3])
for line in open(sys.argv[3]):
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
    new_tests[name] = [module + " " + name, status]
print("CUR_ROCM_TESTS %d tests" % len(new_tests))

print("running comparison")

for test in old_tests:
    old_test, old = old_tests[test]
    # To use the original code, comment everything out until the original code and uncomment original code
    split_test = test.split()
    split_old_test = old_test.split()
    if len(split_old_test) == 4:
        old_test = split_old_test[0] + " " + split_old_test[1] + " " + split_old_test[2]
        test_name_for_new = split_test[0] + " " + split_test[1]
    else:
        test_name_for_new = test  
    new_ref_test, new_ref = new_ref_tests.get(test_name_for_new, ["", "9"])
    new_test, new = new_tests.get(test_name_for_new, ["", "9"])
    # Original code
    # new_ref_test, new_ref = new_ref_tests.get(test, ["", "9"])
    # new_test, new = new_tests.get(test, ["", "9"]) # 9 means not found
    if old != new and old=="0" and new_ref=="0":
        if new == "9":   #new test case run in the newer log
            print(" MISSED: ", old_test)
        elif new=="0":                #was passing but skipped in the newer log
            print(" ROCMONLY: ", old_test)
        elif new=="1":                #was skipped but unskipped in the newer log
            print(" SKIPPED: ", old_test)
        elif new=="2":
            print(" XFAILED: ", old_test)
        elif new == "3":
            print(" UNEXFAILED: ", old_test)
        else:
            assert False
