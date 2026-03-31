#!/usr/bin/env python3

import re
import sys

DEBUG=False

# These regular expressions assume a timestamp and space preceed each line
RE_START = re.compile(r"(.* |^)Running (.*) \.\.\. .*") #(.* |^)
#python test_custom_ops.py -v
RE_START_PYTHON = re.compile(r"(.* |^)python (.*) -v")
#This is required to find the test suites with missing "Running test_*" headers
RE_START_MISSING_HEADER  = re.compile(r"(.* |^)(##\[group\]PRINTING LOG FILE of (.*) (.*))")
RE_RUNNING = re.compile(r"(.* |^)Running tests\.\.\..*")
RE_TEST_SESSION_STARTS = re.compile(r"(.* |^)=* test session starts =*")
RE_STOP = re.compile(r"(.* |^)Ran (\d+) tests? in (.*)")
RE_OK_S_E = re.compile(r"(.* |^)OK \(skipped=(\d+), expected failures=(\d+)\)")
RE_OK_S = re.compile(r"(.* |^)OK \(skipped=(\d+)\)")
RE_OK_PYTEST_S_D_X_W_R = re.compile(r"(.* |^)=* (((\d+) (passed|skipped|deselected|xfailed|warnings?|rerun)(?:,|) )*)in (.*) =*")
# 7-bit and 8-bit C1 ANSI sequences
RE_ANSI_ESCAPE_8BIT = re.compile(r'(?:\x1B[@-Z\\-_]|[\x80-\x9A\x9C-\x9F]|(?:\x1B\[|\x9B)[0-?]*[ -/]*[@-~])')
RE_CUDA_NOT_AVAILABLE = re.compile(r"(.* |^)CUDA not available, skipping tests")
RE_TEST_OK = re.compile(r"(.* |^)(test.* \((?:__main__|fx|jit|quantization|ao|autograd|test_.*|torch*)\.[\S.]*\)) \.\.\. ok \((.*)\)") #THIS IS NOT CATCHING WHAT NEEDS TO BE CAUGHT
RE_TEST_SKIP = re.compile(r"(.* |^)(test.* \((?:__main__|fx|jit|quantization|ao|autograd|test_.*|torch*)\.[\S.]*\)) \.\.\. skip: .* \((.*)\)")
RE_TEST_SKIP_OLD = re.compile(r"(.* |^)(test.* \((?:__main__|fx|jit|quantization|ao|autograd|test_.*|torch*)\.[\S.]*\)) \.\.\. skip")
RE_TEST_X = re.compile(r"(.* |^)(test.* \((?:__main__|fx|jit|quantization|ao|autograd|test_.*|torch*)\.[\S.]*\)) \.\.\. expected failure \((.*)\)")
RE_TEST_ERROR = re.compile(r"(.* |^)(test.* \((?:__main__|fx|jit|quantization|ao|autograd|test_.*|torch*)\.[\S.]*\)) \.\.\. ERROR \((.*)\)")
RE_TEST_UNEXPECTED_SUCCESS = re.compile(r"(.* |^)(test.* \((?:__main__|fx|jit|quantization|ao|autograd|test_.*|torch*)\.[\S.]*\)) \.\.\. unexpected success \((.*)\)")
#sample line for matching: "test_sparse_qlinear (ao.sparsity.test_kernels.TestQuantizedSparseKernels) ... [W TensorImpl.h:1347] Warning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable. (function operator())"
RE_TEST_NL = re.compile(r".* (test.* \((?:__main__|fx|jit|quantization|ao|autograd|test_.*|torch*)\.[\S.]*\))") #this is catching what should be RE_TEST_NL_OK
RE_TEST_RUNNING = re.compile(r"(.* |^)Running (.* |^)(.*\.py::[\S.]*)")
RE_TEST_NL_OK = re.compile(r"(.* |^)ok \((.*)\)")
RE_TEST_NL_ERROR = re.compile(r"(.* |^)ERROR \((.*)\)")
RE_TEST_NL_SKIP = re.compile(r"(.* |^)skip: .* \((.*)\)")
RE_TEST_NL_SKIP_OLD = re.compile(r"(.* |^)skip")
RE_TEST_NL_X = re.compile(r"(.* |^)expected failure \((.*)\)")
RE_PYTEST_PASSED = re.compile(r"(.* |^)(.*\.py::[\S.]*) PASSED ")
RE_PYTEST_SKIPPED = re.compile(r"(.* |^)(.*\.py::[\S.]*) SKIPPED ")
RE_PYTEST_XFAIL = re.compile(r"(.* |^)(.*\.py::[\S.]*) XFAIL ")
RE_PASSED_WITH_SPACE_FRONT = re.compile(r"(.* |^)PASSED(.*\])(.*::.*::.*)")
RE_SKIPPED_WITH_SPACE_FRONT = re.compile(r"(.* |^)SKIPPED(.*\])(.*::.*::.*)")
RE_XFAIL_WITH_SPACE_FRONT = re.compile(r"(.* |^)XFAIL(.*\])(.*::.*::.*)")
RE_PASSED_WITH_SPACE = re.compile(r"(.* |^)(.*::.*::.*) PASSED ")
RE_SKIPPED_WITH_SPACE = re.compile(r"(.* |^)(.*::.*::.*) SKIPPED ")
RE_XFAIL_WITH_SPACE = re.compile(r"(.* |^)(.*::.*::.*) XFAIL ")
RE_PYTEST_PASSED_WITH_SPACE = re.compile(r"(.* |^)(.*\.py::.*) PASSED ")
RE_PYTEST_SKIPPED_WITH_SPACE = re.compile(r"(.* |^)(.*\.py::.*) SKIPPED ")
RE_PYTEST_XFAIL_WITH_SPACE = re.compile(r"(.* |^)(.*\.py::.*) XFAIL ")
RE_PYTEST_FAILED_WITH_RETRY = re.compile(r"(.* |^)failed - num_retries_left: (.*)")
#sample line for matching: "2022-04-29T20:43:09.2292850Z distributed/pipeline/sync/test_pipe.py::test_parameters libibverbs: Warning: couldn't open config directory '/etc/libibverbs.d'."
RE_PYTEST_NL = re.compile(r"(.* |^)(.*\.py::[\S.]*) ")
RE_PYTEST_NL_PASSED = re.compile(r".*PASSED.* \[.*]")
RE_PYTEST_NL_SKIPPED = re.compile(r".*SKIPPED.* \[.*]")
RE_PYTEST_NL_XFAIL = re.compile(r".*XFAIL.* \[.*]")
RE_PYTEST_NL_PASSED_EXTENDED = re.compile(r"(.* |^)PASSED (.*\.py::[\S.]*)")
RE_PYTEST_NL_SKIPPED_EXTENDED = re.compile(r"(.* |^)SKIPPED (.*\.py::[\S.]*)")
RE_PYTEST_NL_XFAIL_EXTENDED = re.compile(r"(.* |^)XFAIL (.*\.py::[\S.]*)")
RE_PYTEST_NL_PASSED_EXTENDED_END = re.compile(r"(.*\.py::[\S.]*) (.* |^)PASSED")
RE_PYTEST_NL_SKIPPED_EXTENDED_END = re.compile(r"(.*\.py::[\S.]*) (.* |^)SKIPPED")
RE_PYTEST_NL_XFAIL_EXTENDED_END = re.compile(r"(.*\.py::[\S.]*) (.* |^)XFAIL")

RE_TEST_SUMMARY_START = re.compile(r"(.* |^)=* short test summary info =*") #=========================== short test summary info ============================ 
RE_TEST_SUMMARY_STOP = re.compile(r"(.* |^)=* .* =*") #======================== 16 passed, 2 skipped in 5.45s =========================

RE_TEST_IGNORE_ERROR = re.compile(r"(.* |^)ERROR(:| \[.*\]:).*") #ERROR [11.257s]: test_ddp_hook_with_optimizer_parity_adam_optimize_subset_False (__main__.TestDistBackendWithSpawn) or ERROR:

RE_UNEXPECTED_FAIL = re.compile(r"(.* |^)\.\.\. FAIL ")
RE_FAIL_MESSAGE = re.compile(r"(.* |^)FAIL ")

KNOWN_BAD_TEST_FILES = [
"distributed/elastic/timer/api_test",
"distributed/fsdp/test_shard_utils",
"distributed/rpc/test_faulty_agent",
"distributed/rpc/test_tensorpipe_agent",
"distributed/test_pg_wrapper",
"distributed/_shard/sharded_tensor/ops/test_math_ops",
"distributed/_shard/sharding_plan/test_sharding_plan",
"distributed/_shard/test_replicated_tensor",
"distributed/_shard/test_sharder",
"distributed/_sharded_tensor/ops/test_embedding",
"lazy/test_bindings",
"test_deploy",
"test_hub",
"test_masked",
"test_nestedtensor",
"test_prims",
"test_pruning_op",
"lazy/test_extract_compiled_graph",
"test_mps",
"test_proxy_tensor",
"test_transformers",
# test_shape_ops was commented out since the test test_flip_large_tensor_cuda occasionally results in a SIGKILL message which this script cannot handel
"test_shape_ops", # add comment for issue with occasional SIGKILL for test test_flip_large_tensor_cuda
"dynamo/test_dynamic_shapes", #Added this test suite as many tests are repeating in this suite and we do not want to skip
"inductor/test_torchinductor_dynamic_shapes"
]

#The tests belonging to below testsuites are th this format: 2022-10-20T22:16:25.0955546Z test_ops.py::TestCommonCUDA::test_compare_cpu___radd___cuda_float32 SKIPPED (test is slow; run with PYTORCH_TEST_WITH_SLOW to enable test) [  0%]
KNOWN_TESTS_TO_FORMAT = [
"test_decomp",
"test_ops",
"test_ops_fwd_gradients",
"test_ops_gradients",
"test_ops_jit",
"inductor/test_torchinductor_opinfo"
]

#These testsuite tests are divided across the log
KNOWN_TEST_SUITES_WITH_DIVIDED_TESTS = [
"test_decomp",
"test_ops",
"test_ops_gradients",
"test_ops_jit",
"distributed/algorithms/quantization/test_quantization",
"distributed/test_distributed_spawn",
"test_ops_fwd_gradients",
"inductor/test_torchinductor_opinfo",
"functorch/test_ops",
"test_meta",
"test_jit_fuser_te",
"test_quantization",
"test_modules",
"test_foreach",
"inductor/test_torchinductor",
"distributed/fsdp/test_fsdp_core",
"distributed/fsdp/test_fsdp_state_dict",
"distributed/test_c10d_nccl",
"distributed/test_c10d_gloo",
"distributed/rpc/cuda/test_tensorpipe_agent",
"inductor/test_torchinductor_codegen_dynamic_shapes",
"inductor/test_minifier"
]

#These tests need to use a different pattern to match
KNOWN_TEST_CASES_WITH_SPECIAL_TEST_NAME = [
# matched with RE_PYTEST_PASSED_WITH_SPACE/RE_PYTEST_SKIPPED_WITH_SPACE/RE_PYTEST_XFAIL_WITH_SPACE
"test_fx_passes.py::TestFXGraphPasses::test_fuser_util_partition_",
"test_fx_passes.py::TestFXGraphPasses::test_partitioner_fn_",
"test_fx_passes.py::TestFXGraphPasses::test_fuser_util_xfail_partition_",
"test_fx_passes.py::TestFXMatcherUtils::test_subgraph_matcher_test_model_",
"test_jiterator.py::TestPythonJiteratorCUDA::test_all_dtype_contiguous_shape_strides_",
"test_jiterator.py::TestPythonJiteratorCUDA::test_all_dtype_noncontiguous_shape_strides_",
"test_sparse_csr.py::TestSparseCSRCUDA::test_addmv_matrix_shape_",
"test_testing.py::TestMakeTensorCUDA::test_low_ge_high_low_high_",
"test_testing.py::TestMakeTensorCUDA::test_low_high_nan_low_high_",
"test_testing.py::TestMakeTensorCUDA::test_memory_format_memory_format_and_shape_",
"test_testing.py::TestMakeTensorCUDA::test_noncontiguous_noncontiguous_False_shape_",
"test_testing.py::TestMakeTensorCUDA::test_noncontiguous_noncontiguous_True_shape_",
"test_testing.py::TestMakeTensorCUDA::test_smoke_shape_",
"test_jiterator.py::TestPythonJiteratorCUDA::test_invalid_function_name_code_string_template",
# matched with RE_PASSED_WITH_SPACE/RE_SKIPPED_WITH_SPACE/RE_XFAIL_WITH_SPACE or RE_PASSED_WITH_SPACE_FRONT/RE_SKIPPED__WITH_SPACE_FRONT/RE_XFAIL_WITH_SPACE_FRONT
"basic::",
"atest::",
"scalar_test::",
"apply_utils_test::",
"dlconvertor_test::",
"native_test::",
"scalar_tensor_test::",
"lazy_tensor_test::",
"Dimname_test::",
"tensor_iterator_test::",
"Dict_test::",
"NamedTensor_test::",
"cpu_generator_test::",
"operators_test::",
"legacy_vmap_test::",
"test_jit::",
"nvfuser_tests::",
"test_lazy::",
"test_api::",
"test_tensorexpr::",
"test_mobile_nnc::",
"FileStoreTest::",
"HashStoreTest::",
"TCPStoreTest::",
"ProcessGroupGlooTest::",
"ProcessGroupNCCLTest::",
"ProcessGroupNCCLErrorsTest::",
"test_cpp_rpc::",
"broadcast_test::",
"wrapdim_test::",
"undefined_tensor_test::",
"extension_backend_test::",
"Dict_test::",
"FileStoreTest::",
"HashStoreTest::",
"TCPStoreTest::",
"ProcessGroupGlooTest::",
"ProcessGroupNCCLTest::",
"ProcessGroupNCCLErrorsTest::",
"test_cpp_rpc::",
"tensor_iterator_test::"
]


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

all_test_modules = {}

PASS=0
SKIP=1
XFAIL=2
UNEXFAIL = 3
UNEXSUCCESS = 4

def debug_fallthrough(*args):
    if DEBUG:
        #print(*args, end='')
        eprint("DEBUG_FALLTHROUGH :", *args)

def debug_RE(*args):
    if DEBUG:
        #print(*args, end='')
        eprint("DEBUG_RE :", *args)

def debug(*args):
    if DEBUG:
        eprint("DEBUG : ", *args, end='')

def check_valid(module_name, run, tests, cuda_not_avail):
    captured = len(tests)
    if module_name not in KNOWN_BAD_TEST_FILES:
        if int(run) == 0 and captured > 0 and not cuda_not_avail:
            eprint("SOME MISSED TESTS", module_name, int(run), int(captured))
            eprint("EXPECTED =", tests )
            assert int(run) > 0 if captured > 0 else True
        if 'spawn' in module_name or 'fsdp' in module_name or 'quantization' in module_name:
            # it will undercount by a multiple of the number of GPUs used, so we relax the condition
            if int(captured) > 0 and int(run) % int(captured) != 0:
                if DEBUG:
                    eprint("SOME MISSED TESTS", module_name, int(run), int(captured))
                    eprint("EXPECTED =", tests )
                return
        elif int(run) != int(captured):
            if DEBUG:
                eprint("SOME MISSED TESTS", module_name, int(run), int(captured))
                eprint("EXPECTED =", tests )
            # relax this for off by one
            assert abs(int(run) - int(captured)) <= 1

def store_module(tests, last_module_name, run, skipped, expected_fail):
    global all_test_modules
    # remove percentage info (such as 1/1) in last_module_name
    split_names = last_module_name.split()
    if len(split_names) > 1:
        last_module_name = last_module_name.split()[0]
    if last_module_name in all_test_modules and last_module_name not in KNOWN_BAD_TEST_FILES and last_module_name not in KNOWN_TEST_SUITES_WITH_DIVIDED_TESTS:
        print(last_module_name)
        assert(last_module_name not in all_test_modules)
    if last_module_name in all_test_modules and last_module_name in KNOWN_TEST_SUITES_WITH_DIVIDED_TESTS:
        new_list = list([run,skipped,expected_fail,tests])
        all_test_modules[last_module_name].append(new_list)
    else:
        all_test_modules[last_module_name] = [[run,skipped,expected_fail,tests]]

def store_test(tests, name, status, time_taken="0s"):
    if 'Spawn' not in name and 'Quantization' not in name and 'dynamo/test_dynamic_shapes' not in name and 'inductor/test_torchinductor_dynamic_shapes' not in name:
        if name in tests:
            print(name)
        #assert name not in tests
            return
    else:
        if name in tests:
            #Logic to update the test status depending on it's previous results present in tests dict. This is for tests that run with different contexts eg. nccl/gloo backend or env/file init_method
            previous_status = tests.get(name).get("status")
            current_status = status
            if previous_status == SKIP or current_status == SKIP:
                status = SKIP
            elif previous_status == UNEXFAIL or current_status == UNEXFAIL:
                status = UNEXFAIL
            elif previous_status == UNEXSUCCESS:
                status = UNEXSUCCESS
    tests[name] = {"time_taken": float(time_taken[:-1]), "status": status}

def format_test_name(last_test_name, last_module_name):
    split_pattern = last_test_name.split('::')
    # check whether need to format
    if len(split_pattern) == 4:
        test_name = last_test_name.split('::')[2] + "::" + last_test_name.split('::')[3]
        test_suite = last_test_name.split('::')[1]
        entry = test_name + " (__main__." + test_suite +")"
    elif len(split_pattern) == 3:
        test_name = last_test_name.split('::')[2]
        if "<- ../" in test_name:
            index = test_name.find("<- ../") - 1
            test_name = test_name[0 : index]
        test_suite = last_test_name.split('::')[1]
        entry = test_name + " (__main__." + test_suite +")"
    elif len(split_pattern) == 2:
        test_name = last_test_name.split('::')[1]
        if "<- ../" in test_name:
            index = test_name.find("<- ../") - 1
            test_name = test_name[0 : index]
        test_suite = last_test_name.split('::')[0]
        entry = test_name + " (__main__." + test_suite +")"
    else:
        entry = last_test_name
    return entry

def process_log(filename):
    run = 0
    skipped = 0
    expected_fail = 0
    last_module_name = None
    cuda_not_avail = False
    tests = {}
    last_test_name = None
    newline_mode = False
    started = False
    extended_test = False
    summary_mode = False

    if DEBUG:
        eprint("Processing :", filename )

    num_lines = sum(1 for line in open(filename))

    line_count = 0
    for line_ in open(filename):
        line_count = line_count + 1
        line = RE_ANSI_ESCAPE_8BIT.sub('', line_)

        if len(line) > 10000 : 
            # skip
            continue

        match = RE_START_PYTHON.search(line) or RE_START_MISSING_HEADER.search(line)
        if match:
            debug(line)
            debug_RE("RE_START")
            if last_module_name:
                check_valid(last_module_name, run, tests, cuda_not_avail)
                store_module(tests, last_module_name, run, skipped, expected_fail)
            run = 0
            skipped = 0
            expected_fail = 0
            if not RE_START_MISSING_HEADER.search(line):
                last_module_name = match.groups()[1]
            else:
                last_module_name = match.groups()[2]
            cuda_not_avail = False
            tests = {}
            summary_mode = False
            continue

        match = RE_TEST_IGNORE_ERROR.search(line)
        if match:
            debug(line)
            debug_RE("RE_TEST_IGNORE_ERROR")
            continue
 
        match = RE_TEST_SUMMARY_START.search(line) 
        if match:
            debug(line)
            debug_RE("RE_TEST_SUMMARY_START")
            summary_mode = True
            continue
 
        if newline_mode:
            match = RE_TEST_NL_OK.search(line)
            if match:
                debug(line)
                debug_RE("RE_TEST_NL_OK")
                store_test(tests, last_test_name, PASS, match.groups()[-1])
                last_test_name = None
                newline_mode = False
                continue

            match = RE_TEST_NL_ERROR.search(line)
            if match:
                debug(line)
                debug_RE("RE_TEST_NL_ERROR")
                store_test(tests, last_test_name, UNEXFAIL, match.groups()[-1])
                last_test_name = None
                newline_mode = False
                continue

            match = RE_TEST_NL_SKIP.search(line)
            if match:
                debug(line)
                debug_RE("RE_TEST_NL_SKIP")
                store_test(tests, last_test_name, SKIP, match.groups()[-1])
                last_test_name = None
                newline_mode = False
                continue

            match = RE_TEST_NL_SKIP_OLD.search(line)
            if match:
                debug(line)
                debug_RE("RE_TEST_NL_SKIP_OLD")
                store_test(tests, last_test_name, SKIP)
                last_test_name = None
                newline_mode = False
                continue


            match = RE_TEST_NL_X.search(line)
            if match:
                debug(line)
                debug_RE("RE_TEST_NL_X")
                store_test(tests, last_test_name, XFAIL, match.groups()[-1])
                last_test_name = None
                newline_mode = False
                continue

            match = RE_PYTEST_NL_PASSED.search(line)
            if match:
                debug(line)
                debug_RE("RE_PYTEST_NL_PASSED")
                if not summary_mode:
                    entry = format_test_name(last_test_name, last_module_name)
                    store_test(tests, entry, PASS)
                    last_test_name = None
                    newline_mode = False
                else:
                    debug_RE("skipped since in summary_mode")
                continue

            match = RE_PYTEST_NL_SKIPPED.search(line)
            if match:
                debug(line)
                debug_RE("RE_PYTEST_NL_SKIPPED")
                if not summary_mode:
                    entry = format_test_name(last_test_name, last_module_name)
                    store_test(tests, entry, SKIP)
                    last_test_name = None
                    newline_mode = False
                else:
                    debug_RE("skipped since in summary_mode")
                continue

            match = RE_PYTEST_NL_XFAIL.search(line)
            if match:
                debug(line)
                debug_RE("RE_PYTEST_NL_XFAIL")
                if not summary_mode:
                    entry = format_test_name(last_test_name, last_module_name)
                    store_test(tests, entry, XFAIL)
                    last_test_name = None
                    newline_mode = False
                else:
                    debug_RE("skipped since in summary_mode")
                continue

        match = RE_RUNNING.search(line)
        if match:
            debug(line)
            debug_RE("RE_RUNNING")
            started = True
            continue

        match = RE_TEST_SESSION_STARTS.search(line)
        if match:
            debug(line)
            debug_RE("RE_TEST_SESSION_STARTS")
            started = True
            continue

        if not started:
            # parsing takes to long to regex every line when nothing matches
            # try to speed it up by detected when we're in a test block
            debug(line)
            debug_RE("TEST SESSION NOT STARTED; skipped")
            continue

        match = RE_STOP.search(line)
        if match:
            debug(line)
            debug_RE("RE_STOP")
            started = False
            run_ = match.groups()[1]
            run += int(run_)
            continue

        match = RE_OK_S_E.search(line)
        if match:
            debug(line)
            debug_RE("RE_OK_S_E")
            _,skipped_,expected_fail_ = match.groups()
            skipped += int(skipped_)
            expected_fail += int(expected_fail_)
            continue

        match = RE_OK_S.search(line)
        if match:
            debug(line)
            debug_RE("RE_OK_S")
            skipped_ = match.groups()[1]
            skipped += int(skipped_)
            continue

        match = RE_OK_PYTEST_S_D_X_W_R.search(line)
        if match:
            debug(line)
            debug_RE("RE_OK_PYTEST_S_D_X_W_R")
            status_line = match.groups()[1]
            passed = sum( int(a) for a in (re.findall(r"(\d+) passed", match.groups()[1] )) )
            skipped = sum( int(a) for a in (re.findall(r"(\d+) skipped", match.groups()[1] )) )
            deselected = sum( int(a) for a in (re.findall(r"(\d+) deselected", match.groups()[1] )) )
            xfailed = sum( int(a) for a in (re.findall(r"(\d+) xfailed", match.groups()[1] )) )
            warnings = sum( int(a) for a in (re.findall(r"(\d+) warnings?", match.groups()[1] )) )
            rerun = sum( int(a) for a in (re.findall(r"(\d+) rerun", match.groups()[1] )) )

            run += (passed + skipped + xfailed)
            started = False
            continue

        match = RE_CUDA_NOT_AVAILABLE.search(line)
        if match:
            debug(line)
            debug_RE("RE_CUDA_NOT_AVAILABLE")
            cuda_not_avail = True
            continue
        
        match = RE_TEST_OK.search(line)
        if match:
            debug(line)
            debug_RE("RE_TEST_OK")
            store_test(tests, match.groups()[1], PASS, match.groups()[-1])
            continue

        match = RE_TEST_SKIP.search(line)
        if match:
            debug(line)
            debug_RE("RE_TEST_SKIP")
            store_test(tests, match.groups()[1], SKIP, match.groups()[-1])
            continue

        match = RE_TEST_SKIP_OLD.search(line)
        if match:
            debug(line)
            debug_RE("RE_TEST_SKIP_OLD")
            store_test(tests, match.groups()[1], SKIP)
            continue

        match = RE_TEST_X.search(line)
        if match:
            debug(line)
            debug_RE("RE_TEST_XFAIL")
            store_test(tests, match.groups()[1], XFAIL, match.groups()[-1])
            continue

        match = RE_TEST_ERROR.search(line)
        if match:
            debug(line)
            debug_RE("RE_TEST_ERROR")
            store_test(tests, match.groups()[1], UNEXFAIL, match.groups()[-1])
            continue
        
        match = RE_UNEXPECTED_FAIL.search(line)
        if match: 
            debug(line)
            debug_RE("RE_UNEXPECTED_FAIL")
            line_split = line.split()
            temp_name = line_split[1] + line_split[2]
            temp_time = line_split[-1].strip('()')
            store_test(tests, temp_name, UNEXFAIL, temp_time)
            continue

        match = RE_TEST_UNEXPECTED_SUCCESS.search(line)
        if match:
            debug(line)
            debug_RE("RE_TEST_UNEXPECTED_SUCCESS")
            store_test(tests, match.groups()[1], UNEXSUCCESS, match.groups()[-1])
            continue

        match = RE_FAIL_MESSAGE.search(line)
        if match:
            debug(line)
            debug_RE("RE_FAIL_MESSAGE")
            continue

        match = RE_PYTEST_FAILED_WITH_RETRY.search(line)
        if match:
            debug(line)
            debug_RE("RE_PYTEST_FAILED_WITH_RETRY")
            continue

        match = RE_TEST_NL.search(line)
        if match:
            debug(line)
            debug_RE("RE_TEST_NL")
            if not started: continue
            newline_mode = True
            #debug("ENTERING NEWLINE MODE")
            assert last_test_name is None
            last_test_name = match.groups()[0]
            main_index = last_test_name.find('__main__')
            close_paren = last_test_name.find(')', main_index)
            assert close_paren != -1
            last_test_name = last_test_name[:close_paren+1]
            continue

        match = RE_PYTEST_PASSED.search(line)
        if match:
            debug(line)
            debug_RE("RE_PYTEST_PASSED")
            entry = format_test_name(match.groups()[1], last_module_name)
            store_test(tests, entry, PASS)
            continue

        match = RE_PYTEST_SKIPPED.search(line)
        if match:
            debug(line)
            debug_RE("RE_PYTEST_SKIPPED")
            entry = format_test_name(match.groups()[1], last_module_name)
            store_test(tests, entry, SKIP)
            continue

        match = RE_PYTEST_XFAIL.search(line)
        if match:
            debug(line)
            debug_RE("RE_PYTEST_XFAIL")
            entry = format_test_name(match.groups()[1], last_module_name)
            store_test(tests, entry, XFAIL)
            continue

        match = RE_PYTEST_NL_PASSED_EXTENDED.search(line)
        if match:
            debug(line)
            debug_RE("RE_PYTEST_NL_PASSED_EXTENDED")
            if not summary_mode:
                extended_test = False
                test_name = match.groups()[1].split('::')[1]
                test_suite = match.groups()[1].split('::')[0]
                entry = test_name + " (__main__." + test_suite +")"
                store_test(tests, entry, PASS)
                extended_test = True
            else:
                debug_RE("skipped since in summary_mode")
            continue

        match = RE_PYTEST_NL_SKIPPED_EXTENDED.search(line)
        if match:
            debug(line)
            debug_RE("RE_PYTEST_NL_SKIPPED_EXTENDED")
            if not summary_mode:
                extended_test = False
                test_name = match.groups()[1].split('::')[1]
                test_suite = match.groups()[1].split('::')[0]
                entry = test_name + " (__main__." + test_suite +")"
                store_test(tests, entry, SKIP)
                extended_test = True
            else:
                debug_RE("skipped since in summary_mode")
            continue

        match = RE_PYTEST_NL_XFAIL_EXTENDED.search(line)
        if match:
            debug(line)
            debug_RE("RE_PYTEST_NL_XFAIL_EXTENDED")

            if not summary_mode:
                extended_test = False
                test_name = match.groups()[1].split('::')[1]
                test_suite = match.groups()[1].split('::')[0]
                entry = test_name + " (__main__." + test_suite +")"
                store_test(tests, entry, XFAIL)
                extended_test = True
            else:
                debug_RE("skipped since in summary_mode")
            continue

        match = RE_TEST_RUNNING.search(line)
        if match:
            debug(line)
            debug_RE("RE_TEST_RUNNING")
            started = True
            continue

        # special handling for irregular test names with spaces
        special_name_matched = False
        for special_name in KNOWN_TEST_CASES_WITH_SPECIAL_TEST_NAME:
            if special_name in line:
                special_name_matched = True
                break

        if special_name_matched:
            match = RE_PYTEST_PASSED_WITH_SPACE.search(line)
            if match:
                debug(line)
                debug_RE("RE_PYTEST_PASSED_WITH_SPACE")
                if not summary_mode:
                    extended_test = False
                    test_name = match.groups()[1].split('::')[2]
                    test_suite = match.groups()[1].split('::')[1]
                    entry = test_name + " (__main__." + test_suite +")"
                    store_test(tests, entry, PASS)
                    extended_test = True
                else:
                    debug_RE("skipped since in summary_mode")
                continue

            match = RE_PYTEST_SKIPPED_WITH_SPACE.search(line)
            if match:
                debug(line)
                debug_RE("RE_PYTEST_SKIPPED_WITH_SPACE")
                if not summary_mode:
                    extended_test = False
                    test_name = match.groups()[1].split('::')[2]
                    test_suite = match.groups()[1].split('::')[1]
                    entry = test_name + " (__main__." + test_suite +")"
                    store_test(tests, entry, SKIP)
                    extended_test = True
                else:
                    debug_RE("skipped since in summary_mode")
                continue

            match = RE_PYTEST_XFAIL_WITH_SPACE.search(line)
            if match:
                debug(line)
                debug_RE("RE_PYTEST_XFAIL_WITH_SPACE")

                if not summary_mode:
                    extended_test = False
                    test_name = match.groups()[1].split('::')[2]
                    test_suite = match.groups()[1].split('::')[1]
                    entry = test_name + " (__main__." + test_suite +")"
                    store_test(tests, entry, XFAIL)
                    extended_test = True
                else:
                    debug_RE("skipped since in summary_mode")
                continue

            match = RE_PASSED_WITH_SPACE.search(line)
            if match:
                debug(line)
                debug_RE("RE_PASSED_WITH_SPACE")
                if not summary_mode:
                    extended_test = False
                    test_name = match.groups()[1].split('::')[2]
                    test_suite = match.groups()[1].split('::')[1]
                    entry = test_name + " (__main__." + test_suite +")"
                    store_test(tests, entry, PASS)
                    extended_test = True
                else:
                    debug_RE("skipped since in summary_mode")
                continue

            match = RE_SKIPPED_WITH_SPACE.search(line)
            if match:
                debug(line)
                debug_RE("RE_SKIPPED_WITH_SPACE")
                if not summary_mode:
                    extended_test = False
                    test_name = match.groups()[1].split('::')[2]
                    test_suite = match.groups()[1].split('::')[1]
                    entry = test_name + " (__main__." + test_suite +")"
                    store_test(tests, entry, SKIP)
                    extended_test = True
                else:
                    debug_RE("skipped since in summary_mode")
                continue

            match = RE_XFAIL_WITH_SPACE.search(line)
            if match:
                debug(line)
                debug_RE("RE_XFAIL_WITH_SPACE")

                if not summary_mode:
                    extended_test = False
                    test_name = match.groups()[1].split('::')[2]
                    test_suite = match.groups()[1].split('::')[1]
                    entry = test_name + " (__main__." + test_suite +")"
                    store_test(tests, entry, XFAIL)
                    extended_test = True
                else:
                    debug_RE("skipped since in summary_mode")
                continue
            
            match = RE_PASSED_WITH_SPACE_FRONT.search(line)
            if match:
                debug(line)
                debug_RE("RE_PASSED_WITH_SPACE_FRONT")
                if not summary_mode:
                    extended_test = False
                    test_name = match.groups()[2].split('::')[2]
                    test_suite = match.groups()[2].split('::')[1]
                    entry = test_name + " (__main__." + test_suite +")"
                    store_test(tests, entry, PASS)
                    extended_test = True
                else:
                    debug_RE("skipped since in summary_mode")
                continue

            match = RE_SKIPPED_WITH_SPACE_FRONT.search(line)
            if match:
                debug(line)
                debug_RE("RE_SKIPPED_WITH_SPACE_FRONT")
                if not summary_mode:
                    extended_test = False
                    test_name = match.groups()[2].split('::')[2]
                    test_suite = match.groups()[2].split('::')[1]
                    entry = test_name + " (__main__." + test_suite +")"
                    store_test(tests, entry, SKIP)
                    extended_test = True
                else:
                    debug_RE("skipped since in summary_mode")
                continue

            match = RE_XFAIL_WITH_SPACE_FRONT.search(line)
            if match:
                debug(line)
                debug_RE("RE_XFAIL_WITH_SPACE_FRONT")

                if not summary_mode:
                    extended_test = False
                    test_name = match.groups()[2].split('::')[2]
                    test_suite = match.groups()[2].split('::')[1]
                    entry = test_name + " (__main__." + test_suite +")"
                    store_test(tests, entry, XFAIL)
                    extended_test = True
                else:
                    debug_RE("skipped since in summary_mode")
                continue

        # add additional match sections
        match = RE_PYTEST_NL_PASSED_EXTENDED_END.search(line)
        if match:
            debug(line)
            debug_RE("RE_PYTEST_NL_PASSED_EXTENDED_END")
            if not summary_mode:
                extended_test = False
                test_name = match.groups()[0].split('::')[2]
                test_suite = match.groups()[0].split('::')[1]
                entry = test_name + " (__main__." + test_suite +")"
                store_test(tests, entry, PASS)
                extended_test = True
            else:
                debug_RE("skipped since in summary_mode")
            continue

        match = RE_PYTEST_NL_SKIPPED_EXTENDED_END.search(line)
        if match:
            debug(line)
            debug_RE("RE_PYTEST_NL_SKIPPED_EXTENDED_END")
            if not summary_mode:
                extended_test = False
                test_name = match.groups()[0].split('::')[2]
                test_suite = match.groups()[0].split('::')[1]
                entry = test_name + " (__main__." + test_suite +")"
                store_test(tests, entry, SKIP)
                extended_test = True
            else:
                debug_RE("skipped since in summary_mode")
            continue

        match = RE_PYTEST_NL_XFAIL_EXTENDED_END.search(line)
        if match:
            debug(line)
            debug_RE("RE_PYTEST_NL_XFAIL_EXTENDED_END")

            if not summary_mode:
                extended_test = False
                test_name = match.groups()[0].split('::')[2]
                test_suite = match.groups()[0].split('::')[1]
                entry = test_name + " (__main__." + test_suite +")"
                store_test(tests, entry, XFAIL)
                extended_test = True
            else:
                debug_RE("skipped since in summary_mode")
            continue

        match = RE_PYTEST_NL.search(line)
        if match:
            debug(line)
            debug_RE("RE_PYTEST_NL")
            if not started: continue
            newline_mode = True
            last_test_name = format_test_name(match.groups()[1], last_module_name)
            continue

        debug_fallthrough("FAILED MATCH:", line)

    # don't forget the last one
    store_module(tests, last_module_name, run, skipped, expected_fail)
    check_valid(last_module_name, run, tests, cuda_not_avail)

for filename in sys.argv[1:]:
    process_log(filename)

for module in sorted(all_test_modules):
    for i in range(len(all_test_modules[module])):
        run,skipped,expected_fail,tests = all_test_modules[module][i]
        for test in sorted(tests):
                print('{} {} {} {}'.format(module, test, tests[test]["time_taken"], tests[test]["status"]))
