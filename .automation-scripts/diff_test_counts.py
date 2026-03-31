#!/usr/bin/env python3

import sys

cuda_tests = {}
rocm_tests = {}

print("parsing cuda test list file '%s'" % sys.argv[1])
for line in open(sys.argv[1]):
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
    cuda_tests[module + " " + name] = status
print("CUDA_TESTS %d tests" % len(cuda_tests))

print("parsing rocm test list file '%s'" % sys.argv[2])
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
    rocm_tests[module + " " + name] = status
print("ROCM_TESTS %d tests" % len(rocm_tests))

print("running comparison")

for test in cuda_tests:
    cuda = cuda_tests[test]
    rocm = rocm_tests.get(test, "9") # 9 means not found
    if cuda != rocm:
        if rocm == "9":
            print("  MISSED: ", test)
        elif rocm == "0":
            print("ROCMONLY: ", test)
        elif rocm == "1":
            print(" SKIPPED: ", test)
        elif rocm == "2":
            print(" XFAILED: ", test)
        elif rocm == "3":
            print(" UNEXFAILED: ", test)
        else:
            assert False
