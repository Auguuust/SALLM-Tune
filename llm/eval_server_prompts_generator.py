import re
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from options_files.ops_options_file import cleanup_options_file
from eval_server_utility_functions import log_update
from dotenv import load_dotenv

load_dotenv()

def build_massage(system_content, user_contents):
    messages = [{"role": "system", "content": system_content}]
    for content in user_contents:
        messages.append({"role": "user", "content": content})

    return messages

def generate_system_content(device_information, rocksdb_version, trace_result=None):
    content = (
            "You are a RocksDB Expert. "
            "You are being consulted by a company to help improve their RocksDB configuration "
            "by optimizing their options file based on their System information and benchmark output. "
            f"Only provide options files for rocksdb version {rocksdb_version}. Also, Direct IO will always be used for both flush and compaction. "
            "Additionally, compression type is set to none always."   
            f"The Device information is: {device_information}. "
            f"The workload summary of the tracefile is: {trace_result}"
            "IMPORTANT: Only return the parameters that need to be changed, not the complete configuration. "
            "Maximum 10 parameters per response. "
            "This is a strict limit that cannot be exceeded. Count each option carefully. "
            """
            Respond in the following format:
            <reasoning>
            Analysis of why changes are needed. Explicitly list the 10 (or fewer) options you chose and justify each selection.
            </reasoning>
            <config>
            ONLY the modified options (maximum 10 lines, count them carefully):
            option1 = value1
            option2 = value2
            ...
            (NEVER exceed 10 options total)
            </config>
            """
        )
    return content

def generate_default_user_content(device_information, rocksdb_version, chunk_string, 
                                  previous_option_files, average_cpu_used=-1.0, 
                                  average_mem_used=-1.0, test_name="fillrandom"):
    user_contents = []
    for _, benchmark_result, reasoning, _ in previous_option_files[1: -1]:
        benchmark_line = generate_benchmark_info(test_name, benchmark_result, average_cpu_used, average_mem_used)
        user_content = f"The benchmark results are: {benchmark_line}"
        user_contents.append(user_content)

    _, benchmark_result, _, _ = previous_option_files[-1]
    benchmark_line = generate_benchmark_info(test_name, benchmark_result, average_cpu_used, average_mem_used)
    user_content = (
        f"Part of the current option file is:\n###\n{chunk_string}\n###\nThe benchmark results are: {benchmark_line}"
    )
    user_contents.append(user_content + "Based on these information generate a new file in the same format as the options_file (but only give the changed value) to improve my database performance. Enclose the new options file in <config></config>.")
    return user_contents

def generate_benchmark_info(test_name, benchmark_result, average_cpu_used, average_mem_used):
    benchmark_line = (f"The use case for the database is perfectly simulated by the {test_name} test. "
                      f"The db_bench benchmark results for {test_name} are: Write/Read speed: {benchmark_result['data_speed']} "
                      f"{benchmark_result['data_speed_unit']}, Operations per second: {benchmark_result['ops_per_sec']}.")
    
    if average_cpu_used != -1 and average_mem_used != -1:
        benchmark_line += f" CPU used: {average_cpu_used}%, Memory used: {average_mem_used}% during test."
    
    return benchmark_line

def generate_option_file_with_LLM(previous_option_files, device_information, 
                                 average_cpu_used=-1.0, average_mem_used=-1.0, 
                                 test_name="fillrandom", version="8.8.1", trace_result=None):
    log_update("[OG] Generating options file with long option changes")
    print("[OG] Generating options file with long option changes")
    system_content = generate_system_content(device_information, version, trace_result)
    previous_option_file, _, _, _ = previous_option_files[-1]
    user_contents = generate_default_user_content(device_information, version, previous_option_file, 
                                                  previous_option_files, average_cpu_used, average_mem_used, test_name)
    messages = build_massage(system_content, user_contents)
    return messages

