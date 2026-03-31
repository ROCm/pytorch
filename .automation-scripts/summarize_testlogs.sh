#!/bin/bash

TEST_FOLDER=$1
PrWD=$(pwd)

OVERRIDE=False
if [ "X$2" == "Xoverride" ]; then
    OVERRIDE=True
fi

cd $TEST_FOLDER

if [ "$OVERRIDE" == True ] || [ ! -f results_cuda.txt ]; then
    echo "creating results_cuda.txt"
    ( python3 $PrWD/get_test_counts.py cuda1.txt cuda2.txt cuda3.txt cuda4.txt cuda5.txt >& results_cuda.txt ; echo "results_cuda status: $?" ) &
fi

if [ "$OVERRIDE" == True ] || [ ! -f results_rocm.txt ]; then
    echo "creating results_rocm.txt"
    ( python3 $PrWD/get_test_counts.py rocm1.txt rocm2.txt rocm3.txt >& results_rocm.txt ; echo "results_rocm status: $?" ) &
fi

wait

if [ "$OVERRIDE" == True ] || [ ! -f diff.txt ]; then
    echo "diff.txt"
    ( python3 $PrWD/diff_test_counts.py results_cuda.txt results_rocm.txt >& diff.txt ; echo "diff status: $?" ) &
fi

if [ "$OVERRIDE" == True ] || [ ! -f results_cuda_dist.txt ]; then
    echo "creating results_cuda_dist.txt"
    (python3 $PrWD/get_test_counts.py cuda_dist1.txt cuda_dist2.txt cuda_dist3.txt >& results_cuda_dist.txt  ; echo "results_cuda_dist status: $?" ) &
fi


 if [ "$OVERRIDE" == True ] || [ ! -f results_rocm_dist.txt ]; then
     echo "creating results_rocm_dist.txt"
     (python3 $PrWD/get_test_counts.py rocm_dist1.txt rocm_dist2.txt >& results_rocm_dist.txt  ; echo "results_rocm_dist status: $?" ) &
 fi

wait

 if [ "$OVERRIDE" == True ] || [ ! -f diff_dist.txt ]; then
     echo "creating diff_dist.txt"
     (python3 $PrWD/diff_test_counts.py results_cuda_dist.txt results_rocm_dist.txt >& diff_dist.txt  ; echo "diff_dist status: $?" ) &
 fi

 if [ "$OVERRIDE" == True ] || [ ! -f diff_dist_time.csv ]; then
     echo "creating diff_dist_time.csv"
     (python3 $PrWD/diff_test_time.py results_cuda_dist.txt results_rocm_dist.txt >& diff_dist_time.csv  ; echo "diff_dist_time status: $?" ) &
 fi

wait

if [ -f summary_tests.csv ]; then
    echo "summary_tests.csv"
    rm summary_tests.csv
fi

test_skipped=$(grep SKIPPED diff.txt | wc -l)
test_missed=$(grep MISSED diff.txt | wc -l)
test_rocmonly=$(grep ROCMONLY diff.txt | wc -l)
test_failed=$(grep UNEXFAILED diff.txt | wc -l)
test_cuda=$(grep CUDA_TESTS diff.txt | awk -F' ' '{print $2}')
test_rocm=$(grep ROCM_TESTS diff.txt | awk -F' ' '{print $2}')

test_dist_skipped=$(grep SKIPPED diff_dist.txt | wc -l)
test_dist_missed=$(grep MISSED diff_dist.txt | wc -l)
test_dist_rocmonly=$(grep ROCMONLY diff_dist.txt | wc -l)
test_dist_failed=$(grep UNEXFAILED diff_dist.txt | wc -l)
test_dist_cuda=$(grep CUDA_TESTS diff_dist.txt | awk -F' ' '{print $2}')
test_dist_rocm=$(grep ROCM_TESTS diff_dist.txt | awk -F' ' '{print $2}')

echo "____ $TEST_FOLDER ____"
echo "_____________________________________"
echo "Test-default"
echo "==========="
echo "SKIPPED, MISSED, ROCMONLY, UNEXFAIL, CUDA, ROCM"
echo "$test_skipped, $test_missed, $test_rocmonly, $test_failed, $test_cuda, $test_rocm"

$PrWD/find_cause.py -i diff.txt -s $PrWD/ut_test_skip_reasons.csv -c SKIPPED MISSED -o summary_tests.csv

echo "_____________________________________"
echo "Test-distributed"
echo "==========="
echo "SKIPPED, MISSED, ROCMONLY, UNEXFAIL, CUDA, ROCM"
echo "$test_dist_skipped, $test_dist_missed, $test_dist_rocmonly, $test_dist_failed, $test_dist_cuda, $test_dist_rocm"

$PrWD/find_cause.py -i diff_dist.txt -s $PrWD/ut_test_skip_reasons.csv -c SKIPPED MISSED -o summary_tests.csv
echo "_____________________________________"
echo "full list of skipped tests is stored in summary_tests.csv"

cd $PrWD
