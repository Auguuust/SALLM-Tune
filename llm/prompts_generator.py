import re
from llm.content_generator import *
from utils.constants import VERSION

def generate_option_file_with_llm(metric, previous_option_files, db_bench_args, device_information, trace_result, temperature=0.4, average_cpu_used=-1.0, average_mem_used=-1.0, test_name="fillrandom", version="8.8.1", performance_feedback=""):
    log_update("[OG] Generating options file with metric-based optimization")
    print("[OG] Generating options file with metric-based optimization")
    
    system_content = generate_system_content(device_information, version)
    user_contents = generate_shard_content(previous_option_files, metric, test_name)
    
    clean_options_file = ""
    reasoning = ""
    changed_value_dict = {}

    for index, chunk_string in enumerate(user_contents):
        user_content = generate_default_user_content(metric, chunk_string, previous_option_files, average_cpu_used, average_mem_used, test_name)
        
        if index == 0 and performance_feedback:
            user_content.append(performance_feedback)
        
        if index == 0:
            user_content += user_content_for_db_bench_args(db_bench_args)
        matches = request_llm(system_content, user_content, None, temperature)
        if matches is not None:
            clean_options_file, changed_value_dict_part, db_bench_args = cleanup_options_file(matches[1], db_bench_args)
            reasoning += matches[0] + matches[2]
            changed_value_dict.update(changed_value_dict_part)
    
    return clean_options_file, db_bench_args, reasoning, changed_value_dict
        