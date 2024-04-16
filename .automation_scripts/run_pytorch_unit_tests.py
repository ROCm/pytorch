#!/usr/bin/env python3

""" The Python PyTorch testing script.
##
# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""

import argparse
import os
import shutil
import subprocess

from collections import namedtuple
from datetime import datetime
from pathlib import Path
from parse_xml_results import (
        parse_xml_report
)
from pprint import pprint

# unit test status list
UT_STATUS_LIST = [
    "PASSED",
    "MISSED",
    "SKIPPED",
    "FAILED",
    "XFAILED",
    "ERROR"
]

DEFAULT_CORE_TESTS = [
    "test_nn",
    "test_torch",
    "test_cuda",
    "test_ops",
    "test_unary_ufuncs",
    "test_autograd",
    "inductor/test_torchinductor"
]

DISTRIBUTED_CORE_TESTS = [
    "distributed/test_c10d_common",
    "distributed/test_c10d_nccl",
    "distributed/test_distributed_spawn"
]

def parse_xml_reports_as_dict(workflow_run_id, workflow_run_attempt, tag, workflow_name, path="."):
    test_cases = {}
    items_list = os.listdir(path)
    for dir in items_list:
        new_dir = path + '/' + dir + '/'
        if os.path.isdir(new_dir):
            for xml_report in Path(new_dir).glob("**/*.xml"):
                test_cases.update(
                    parse_xml_report(
                        tag,
                        xml_report,
                        workflow_run_id,
                        workflow_run_attempt,
                        workflow_name
                    )
                )
    return test_cases

def get_test_status(test_case):
  # In order of priority: S=skipped, F=failure, E=error, P=pass
  if "skipped" in test_case and test_case["skipped"]:
      type_message = test_case["skipped"]
      if type_message.__contains__('type') and type_message['type'] == "pytest.xfail":
          return "XFAILED"
      else:
          return "SKIPPED"
  elif "failure" in test_case and test_case["failure"]:
    return "FAILED"
  elif "error" in test_case and test_case["error"]:
    return "ERROR"
  else:
    return "PASSED"

def get_test_message(test_case, status=None):
  if status == "SKIPPED":
    return test_case["skipped"] if "skipped" in test_case else ""
  elif status == "FAILED":
    return test_case["failure"] if "failure" in test_case else ""
  elif status == "ERROR":
    return test_case["error"] if "error" in test_case else ""
  else:
    if "skipped" in test_case:
      return test_case["skipped"]
    elif "failure" in test_case:
      return test_case["failure"]
    elif "error" in test_case:
      return test_case["error"]
    else:
      return ""

def get_test_running_time(test_case):
  if test_case.__contains__('time'):
    return test_case["time"]
  return ""

def summarize_xml_files(path, workflow_name):
    # statistics
    TOTAL_TEST_NUM = 0
    TOTAL_PASSED_NUM = 0
    TOTAL_SKIPPED_NUM = 0
    TOTAL_XFAIL_NUM = 0
    TOTAL_FAILED_NUM = 0
    TOTAL_ERROR_NUM = 0

    #parse the xml files
    test_cases = parse_xml_reports_as_dict(-1, -1, 'testcase', workflow_name, path)
    test_file_and_status = namedtuple("test_file_and_status", ["file_name", "status"])
    # results dict
    res = {}
    res_item_list = [ "PASSED", "SKIPPED", "XFAILED", "FAILED", "ERROR" ]
    test_file_items = set()
    for (k,v) in list(test_cases.items()):
        file_name = k[0]
        if not file_name in test_file_items:
            test_file_items.add(file_name)
            # initialization
            for item in res_item_list:
                temp_item = test_file_and_status(file_name, item)
                res[temp_item] = {}
            temp_item_statistics = test_file_and_status(file_name, "STATISTICS")
            res[temp_item_statistics] = {'TOTAL': 0, 'PASSED': 0, 'SKIPPED': 0, 'XFAILED': 0, 'FAILED': 0, 'ERROR': 0}

    for (k,v) in list(test_cases.items()):
        file_name = k[0]
        class_name = k[1]
        test_name = k[2]
        combined_name = file_name + "::" + class_name + "::" + test_name
        test_status = get_test_status(v)
        test_running_time = get_test_running_time(v)
        test_message = get_test_message(v, test_status)
        test_info_value = ""
        test_tuple_key_status = test_file_and_status(file_name, test_status)
        test_tuple_key_statistics = test_file_and_status(file_name, "STATISTICS")
        TOTAL_TEST_NUM += 1
        res[test_tuple_key_statistics]["TOTAL"] += 1
        if test_status == "PASSED":
            test_info_value = str(test_running_time)
            res[test_tuple_key_status][combined_name] = test_info_value
            res[test_tuple_key_statistics]["PASSED"] += 1
            TOTAL_PASSED_NUM += 1
        elif test_status == "SKIPPED":
            test_info_value = str(test_running_time)
            res[test_tuple_key_status][combined_name] = test_info_value
            res[test_tuple_key_statistics]["SKIPPED"] += 1
            TOTAL_SKIPPED_NUM += 1
        elif test_status == "XFAILED":
            test_info_value = str(test_running_time)
            res[test_tuple_key_status][combined_name] = test_info_value
            res[test_tuple_key_statistics]["XFAILED"] += 1
            TOTAL_XFAIL_NUM += 1
        elif test_status == "FAILED":
            test_info_value = test_message
            res[test_tuple_key_status][combined_name] = test_info_value
            res[test_tuple_key_statistics]["FAILED"] += 1
            TOTAL_FAILED_NUM += 1
        elif test_status == "ERROR":
            test_info_value = test_message
            res[test_tuple_key_status][combined_name] = test_info_value
            res[test_tuple_key_statistics]["ERROR"] += 1
            TOTAL_ERROR_NUM += 1

    # generate statistics_dict
    statistics_dict = {}
    statistics_dict["TOTAL"] = TOTAL_TEST_NUM
    statistics_dict["PASSED"] = TOTAL_PASSED_NUM
    statistics_dict["SKIPPED"] = TOTAL_SKIPPED_NUM
    statistics_dict["XFAILED"] = TOTAL_XFAIL_NUM
    statistics_dict["FAILED"] = TOTAL_FAILED_NUM
    statistics_dict["ERROR"] = TOTAL_ERROR_NUM
    aggregate_item = workflow_name + "_aggregate"
    total_item = test_file_and_status(aggregate_item, "STATISTICS")
    res[total_item] = statistics_dict

    return res

def run_entire_tests(workflow_name, test_shell_path, overall_logs_path_current_run, test_reports_src):
    if os.path.exists(test_reports_src):
        shutil.rmtree(test_reports_src)

    os.mkdir(test_reports_src)
    copied_logs_path = ""
    if workflow_name == "default":
        os.environ['TEST_CONFIG'] = 'default'
        copied_logs_path = overall_logs_path_current_run + "default_xml_results_entire_tests/"
    elif workflow_name == "distributed":
        os.environ['TEST_CONFIG'] = 'distributed'
        copied_logs_path = overall_logs_path_current_run + "distributed_xml_results_entire_tests/"
    elif workflow_name == "inductor":
        os.environ['TEST_CONFIG'] = 'inductor'
        copied_logs_path = overall_logs_path_current_run + "inductor_xml_results_entire_tests/"
    # use test.sh for tests execution
    subprocess.call(test_shell_path, shell=True)
    copied_logs_path_destination = shutil.copytree(test_reports_src, copied_logs_path)
    entire_results_dict = summarize_xml_files(copied_logs_path_destination, workflow_name)
    return entire_results_dict

def run_priority_tests(workflow_name, test_run_test_path, overall_logs_path_current_run, test_reports_src):
    if os.path.exists(test_reports_src):
        shutil.rmtree(test_reports_src)

    os.mkdir(test_reports_src)
    copied_logs_path = ""
    if workflow_name == "default":
        os.environ['TEST_CONFIG'] = 'default'
        os.environ['HIP_VISIBLE_DEVICES'] = '0'
        copied_logs_path = overall_logs_path_current_run + "default_xml_results_priority_tests/"
        # use run_test.py for tests execution
        default_priority_test_suites = " ".join(DEFAULT_CORE_TESTS)
        command = "python " + test_run_test_path + " --include " + default_priority_test_suites + " --exclude-jit-executor --exclude-distributed-tests --verbose"
        subprocess.call(command, shell=True)
        del os.environ['HIP_VISIBLE_DEVICES']
    elif workflow_name == "distributed":
        os.environ['TEST_CONFIG'] = 'distributed'
        os.environ['HIP_VISIBLE_DEVICES'] = '0,1'
        copied_logs_path = overall_logs_path_current_run + "distributed_xml_results_priority_tests/"
        # use run_test.py for tests execution
        distributed_priority_test_suites = " ".join(DISTRIBUTED_CORE_TESTS)
        command = "python " + test_run_test_path + " --include " + distributed_priority_test_suites + " --distributed-tests --verbose"
        subprocess.call(command, shell=True)
        del os.environ['HIP_VISIBLE_DEVICES']
    copied_logs_path_destination = shutil.copytree(test_reports_src, copied_logs_path)
    priority_results_dict = summarize_xml_files(copied_logs_path_destination, workflow_name)

    return priority_results_dict

def run_selected_tests(workflow_name, test_run_test_path, overall_logs_path_current_run, test_reports_src, selected_list):
    if os.path.exists(test_reports_src):
        shutil.rmtree(test_reports_src)

    os.mkdir(test_reports_src)
    copied_logs_path = ""
    if workflow_name == "default":
        os.environ['TEST_CONFIG'] = 'default'
        os.environ['HIP_VISIBLE_DEVICES'] = '0'
        copied_logs_path = overall_logs_path_current_run + "default_xml_results_selected_tests/"
        # use run_test.py for tests execution
        default_selected_test_suites = " ".join(selected_list)
        command = "python " + test_run_test_path + " --include " + default_selected_test_suites  + " --exclude-jit-executor --exclude-distributed-tests --verbose"
        subprocess.call(command, shell=True)
        del os.environ['HIP_VISIBLE_DEVICES']
    elif workflow_name == "distributed":
        os.environ['TEST_CONFIG'] = 'distributed'
        os.environ['HIP_VISIBLE_DEVICES'] = '0,1'
        copied_logs_path = overall_logs_path_current_run + "distributed_xml_results_selected_tests/"
        # use run_test.py for tests execution
        distributed_selected_test_suites = " ".join(selected_list)
        command = "python " + test_run_test_path + " --include " + distributed_selected_test_suites + " --distributed-tests --verbose"
        subprocess.call(command, shell=True)
        del os.environ['HIP_VISIBLE_DEVICES']
    elif workflow_name == "inductor":
        os.environ['TEST_CONFIG'] = 'inductor'
        copied_logs_path = overall_logs_path_current_run + "inductor_xml_results_selected_tests/"
        inductor_selected_test_suites = ""
        non_inductor_selected_test_suites = ""
        for item in selected_list:
            if "inductor/" in item:
                inductor_selected_test_suites += item
                inductor_selected_test_suites += " "
            else:
                non_inductor_selected_test_suites += item
                non_inductor_selected_test_suites += " "
        if inductor_selected_test_suites != "":
            inductor_selected_test_suites = inductor_selected_test_suites[:-1]
            command = "python " + test_run_test_path + " --include " + inductor_selected_test_suites + " --verbose"
            subprocess.call(command, shell=True)
        if non_inductor_selected_test_suites != "":
            non_inductor_selected_test_suites = non_inductor_selected_test_suites[:-1]
            command = "python " + test_run_test_path + " --inductor --include " + non_inductor_selected_test_suites + " --verbose"
            subprocess.call(command, shell=True)
    copied_logs_path_destination = shutil.copytree(test_reports_src, copied_logs_path)
    selected_results_dict = summarize_xml_files(copied_logs_path_destination, workflow_name)

    return selected_results_dict

def run_test_and_summarize_results(args):
    # copy current environment variables
    _environ = dict(os.environ)
    
    # modify path
    pytorch_root_dir = args.pytorch_root
    test_shell_path = pytorch_root_dir + "/.ci/pytorch/test.sh"
    test_run_test_path = pytorch_root_dir + "/test/run_test.py"
    repo_test_log_folder_path = pytorch_root_dir + "/.automation_logs/"
    test_reports_src = pytorch_root_dir + "/test/test-reports/"

    # change directory to pytorch root
    os.chdir(pytorch_root_dir)

    # all test results dict
    res_all_tests_dict = {}

    # create logs folder
    if not os.path.exists(repo_test_log_folder_path):
        os.mkdir(repo_test_log_folder_path)

    # Set common environment variables for all scenarios
    os.environ['CI'] = '1'
    os.environ['PYTORCH_TEST_WITH_ROCM'] = '1'
    os.environ['HSA_FORCE_FINE_GRAIN_PCIE'] = '1'
    os.environ['PYTORCH_TESTING_DEVICE_ONLY_FOR'] = 'cuda'
    os.environ['CONTINUE_THROUGH_ERROR'] = 'True'

    # Time stamp
    current_datetime = datetime.now().strftime("%Y%m%d_%H-%M-%S")
    print("Current date & time : ", current_datetime)
    # performed as Job ID
    str_current_datetime = str(current_datetime)
    overall_logs_path_current_run = repo_test_log_folder_path + str_current_datetime + "/"
    # Run entire tests for each workflow
    if not args.priority_tests and not args.default_list and not args.distributed_list and not args.inductor_list:
        # run entire tests for default, distributed and inductor workflows → use test.sh
        if not args.test_config:
            # default test process
            res_default_all = run_entire_tests("default", test_shell_path, overall_logs_path_current_run, test_reports_src)
            res_all_tests_dict["default"] = res_default_all
            # distributed test process
            res_distributed_all = run_entire_tests("distributed", test_shell_path, overall_logs_path_current_run, test_reports_src)
            res_all_tests_dict["distributed"] = res_distributed_all
            # inductor test process
            res_inductor_all = run_entire_tests("inductor", test_shell_path, overall_logs_path_current_run, test_reports_src)
            res_all_tests_dict["inductor"] = res_inductor_all
        else:
            workflow_list = []
            for item in args.test_config:
                workflow_list.append(item)
            if "default" in workflow_list:
                res_default_all = run_entire_tests("default", test_shell_path, overall_logs_path_current_run, test_reports_src)
                res_all_tests_dict["default"] = res_default_all
            if "distributed" in workflow_list:
                res_distributed_all = run_entire_tests("distributed", test_shell_path, overall_logs_path_current_run, test_reports_src)
                res_all_tests_dict["distributed"] = res_distributed_all
            if "inductor" in workflow_list:
                res_inductor_all = run_entire_tests("inductor", test_shell_path, overall_logs_path_current_run, test_reports_src)
                res_all_tests_dict["inductor"] = res_inductor_all
    # Run priority test for each workflow
    elif args.priority_tests and not args.default_list and not args.distributed_list and not args.inductor_list:
        if not args.test_config:
            # default test process
            res_default_priority = run_priority_tests("default", test_run_test_path, overall_logs_path_current_run, test_reports_src)
            res_all_tests_dict["default"] = res_default_priority
            # distributed test process
            res_distributed_priority = run_priority_tests("distributed", test_run_test_path, overall_logs_path_current_run, test_reports_src)
            res_all_tests_dict["distributed"] = res_distributed_priority
            # will not run inductor priority tests
            print("Inductor priority tests cannot run since no core tests defined with inductor workflow.")
        else:
            workflow_list = []
            for item in args.test_config:
                workflow_list.append(item)
            if "default" in workflow_list:
                res_default_priority = run_priority_tests("default", test_run_test_path, overall_logs_path_current_run, test_reports_src)
                res_all_tests_dict["default"] = res_default_priority
            if "distributed" in workflow_list:
                res_distributed_priority = run_priority_tests("distributed", test_run_test_path, overall_logs_path_current_run, test_reports_src)
                res_all_tests_dict["distributed"] = res_distributed_priority
            if "inductor" in workflow_list:
                print("Inductor priority tests cannot run since no core tests defined with inductor workflow.")
    # Run specified tests for each workflow
    elif (args.default_list or args.distributed_list or args.inductor_list) and not args.test_config and not args.priority_tests:
        if args.default_list:
            default_workflow_list = []
            for item in args.default_list:
                default_workflow_list.append(item)
            res_default_selected = run_selected_tests("default", test_run_test_path, overall_logs_path_current_run, test_reports_src, default_workflow_list)
            res_all_tests_dict["default"] = res_default_selected
        if args.distributed_list:
            distributed_workflow_list = []
            for item in args.distributed_list:
                distributed_workflow_list.append(item)
            res_distributed_selected = run_selected_tests("distributed", test_run_test_path, overall_logs_path_current_run, test_reports_src, distributed_workflow_list)
            res_all_tests_dict["distributed"] = res_distributed_selected
        if args.inductor_list:
            inductor_workflow_list = []
            for item in args.inductor_list:
                 inductor_workflow_list.append(item)
            res_inductor_selected = run_selected_tests("inductor", test_run_test_path, overall_logs_path_current_run, test_reports_src, inductor_workflow_list)
            res_all_tests_dict["inductor"] = res_inductor_selected
    else:
        raise Exception("Invalid test configurations!")

    # restore environment variables
    os.environ.clear()
    os.environ.update(_environ)

    return res_all_tests_dict

def parse_args():
    parser = argparse.ArgumentParser(description='Run PyTorch unit tests and generate xml results summary')
    parser.add_argument('--test_config', nargs='+', default=[], type=str, help="space-separated list of test workflows to be executed eg. 'default distributed'")
    parser.add_argument('--priority_tests', action='store_true', help="run priority tests only")
    parser.add_argument('--default_list', nargs='+', default=[], help="space-separated list of 'default' config test suites/files to be executed eg. 'test_weak test_dlpack'")
    parser.add_argument('--distributed_list', nargs='+', default=[], help="space-separated list of 'distributed' config test suites/files to be executed eg. 'distributed/test_c10d_common distributed/test_c10d_nccl'")
    parser.add_argument('--inductor_list', nargs='+', default=[], help="space-separated list of 'inductor' config test suites/files to be executed eg. 'inductor/test_torchinductor test_ops'")
    parser.add_argument('--pytorch_root', default='/var/lib/jenkins/pytorch', type=str, help="PyTorch root directory")
    return parser.parse_args()

def main():
    global args
    args = parse_args()
    all_tests_results = run_test_and_summarize_results(args)
    pprint(dict(all_tests_results))

if __name__ == "__main__":
    main()
