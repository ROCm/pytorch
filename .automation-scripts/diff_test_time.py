#!/usr/bin/env python3

#the print/output of this script should directly goes to an csv file for now. delimiter is \t

import sys

cuda_tests = {}
rocm_tests = {}

for line in open(sys.argv[1]):
    try:
        parts = line.strip().split()
        module = parts[0]
        status = parts[-1]
        time_taken = parts[-2]
        name = " ".join(parts[1:-2])
    except:
        print(line)
        print(line.strip().split())
        raise
    cuda_tests[module + " " + name] = [float(time_taken), status]
    
for line in open(sys.argv[2]):
    try:
        parts = line.strip().split()
        module = parts[0]
        status = parts[-1]
        time_taken = parts[-2]
        name = " ".join(parts[1:-2])
    except:
        print(line)
        print(line.strip().split())
        raise
    rocm_tests[module + " " + name] = [float(time_taken), status]

test_list=[]

#comparing all tests appeared in rocm ci log to the tests in cuda ci log including skipped, pass, xfailed, rocmonly because this is for debug purpose
#to analyze which tests take more time compared to cuda
for test in rocm_tests:
    status_map={
        '0':"PASSED",
        '1':"SKIPPED",
        '2':"XFAILED",
        '9':"ROCMONLY",
    }
    rocm_time, rocm_status = rocm_tests[test]
    cuda_time, cuda_status = cuda_tests.get(test, [0.0, "9"]) # 9 means not found
    rocm_time=float(rocm_time)
    cuda_time=float(cuda_time)
    diff_percent=0.0
    if cuda_time!=0.0:
        diff_percent=round((rocm_time-cuda_time)/cuda_time*100,2)
        
    test_info=test.split() #test_file, test_name, test_class
    test_info+=[' ']*(3-len(test_info)) #padding in case some test cases have no class name detected
    test_file,test_name,test_class=test_info[0],test_info[1],test_info[-1] 
    test_list.append([
        test_file, test_name, test_class, cuda_time, status_map[cuda_status], 
        rocm_time, status_map[rocm_status], round(rocm_time-cuda_time,2), diff_percent
    ])

print("sep=\t") #setup delimiter for the csv file
#the first row (i.e., column title) for the csv
print("test_file", "test_name", "test_class", "cuda_time", "cuda_status", "rocm_time", "rocm_status", "delta(second)", "delta(%)",sep="\t")
#sort test by their time spent diff in seconds (rocm time - cuda time) from largest to smallest
for test in sorted(test_list, reverse=True, key=lambda x: x[7]):
    spreadsheet_row = '\t'.join(str(e) for e in test)
    print(spreadsheet_row)
