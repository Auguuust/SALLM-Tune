import re
from difflib import Differ
from options_files.ops_options_file import cleanup_options_file, parse_db_bench_args_to_dict
from llm.llm_request import request_llm
from utils.filter import DB_BENCH_ARGS
from utils.utils import log_update
from dotenv import load_dotenv
import configparser
from utils.constants import VERSION
from utils.constants import RAG

load_dotenv()

def generate_system_content(device_information, trace_result):
    content = (
        "You are a RocksDB Expert. "
        "You are being consulted by a company to help improve their RocksDB configuration "
        "by optimizing their options file based on their System information and benchmark output. "
        f"Only provide options files for rocksdb version {VERSION}. Also, Direct IO will always be used for both flush and compaction. "
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

def generate_benchmark_info(test_name, benchmark_result, average_cpu_used, average_mem_used):
    benchmark_line = (f"Write/Read speed: {benchmark_result['data_speed']} "
                      f"{benchmark_result['data_speed_unit']}, Operations per second: {benchmark_result['ops_per_sec']}.")
    
    if average_cpu_used != -1 and average_mem_used != -1:
        benchmark_line += f" CPU used: {average_cpu_used}%, Memory used: {average_mem_used}% during test."
    
    return benchmark_line

def user_content_for_db_bench_args(db_bench_args):
    args_dict = {key: "-1" for key in DB_BENCH_ARGS}
    args_dict.update(parse_db_bench_args_to_dict(db_bench_args))
    args = "\n".join(f"--{key}={value}" for key, value in args_dict.items())
    return [(
        "If and only if demanded by the workload, you can also update these arguments:\n"
        f"###\n[DBBenchOptions]\n{args}\n###"
        "to improve the performance of the database. "
        "Put it at the first line of the options file if you want to update it."
    )]

def generate_default_user_content(metric, chunk_string, previous_option_files, average_cpu_used=-1.0, average_mem_used=-1.0, test_name="fillrandom"):
    user_contents = []
    for _, benchmark_result, reasoning, _, _ in previous_option_files[1: -1]:
        benchmark_line = generate_benchmark_info(test_name, benchmark_result, average_cpu_used, average_mem_used)
        user_content = f"The option file changes were:\n###\n{reasoning}\n###\nThe benchmark results are: {benchmark_line}"
        user_contents.append(user_content)

    _, benchmark_result, _, _, _ = previous_option_files[-1]
    benchmark_line = generate_benchmark_info(test_name, benchmark_result, average_cpu_used, average_mem_used)
    user_content = f"The part of the current options document that relates to the {metric} indicator is:\n###\n{chunk_string}\n###\nThe benchmark results are: {benchmark_line}"
    user_contents.append(user_content)
    user_contents.append("Based on these information generate a new file in the same format as the options_file (but only give the changed value) to improve my database performance in terms of the {metric} indicator. Enclose the new options file in <config></config>.")
    return user_contents

def generate_assistant_content(previous_option_files):
    assistant_contents = []

    for _, _, reasoning, changes_dict, _ in previous_option_files[1:]:
        changes_str = "\n".join(f"{k}={v}" for k, v in changes_dict.items())
        assistant_contents.append((
            f"{reasoning}\n"
            "The options changes were:\n"
            f"###\n{changes_str}\n###"
        ))

    return assistant_contents

def generate_user_content_with_difference(previous_option_files, average_cpu_used=-1.0, average_mem_used=-1.0, test_name="fillrandom"):
    result = ""
    user_content = []

    if len(previous_option_files) == 1:
        m1_file, m1_benchmark_result, _, _, _ = previous_option_files[-1]
        benchmark_line = generate_benchmark_info(test_name, m1_benchmark_result, average_cpu_used, average_mem_used)
        user_content = f"The original file is:\n###\n{m1_file}\n###\nThe benchmark results for the original file are: {benchmark_line}"
    
    elif len(previous_option_files) > 1:
        previous_option_file1, _, _, _, _ = previous_option_files[-1]
        previous_option_file2, _, _, _, _ = previous_option_files[-2]

        pattern = re.compile(r'\s*([^=\s]+)\s*=\s*([^=\s]+)\s*')

        file1_lines = pattern.findall(previous_option_file1)
        file2_lines = pattern.findall(previous_option_file2)

        file1_lines = ["{} = {}".format(k, v) for k, v in file1_lines]
        file2_lines = ["{} = {}".format(k, v) for k, v in file2_lines]
        differ = Differ()
        diff = list(differ.compare(file1_lines, file2_lines))
        lst= []
        for line in diff:
            if line.startswith('+'):
                lst.append(line)
        result = '\n'.join(line[2:] for line in lst)
        m1_file, m1_benchmark_result, _, _, _ = previous_option_files[-1]
        benchmark_line = generate_benchmark_info(test_name, m1_benchmark_result, average_cpu_used, average_mem_used)
        user_content = (
            f"The original file is:\n###\n{m1_file}\n###\n"
            f"The benchmark results for the original file are: {benchmark_line}\n"
            f"The previous file modifications are:\n###\n{result}\n###\n"
        )
    
    else:
        _, benchmark_result, _, _, _ = previous_option_files[-1]
        benchmark_line = generate_benchmark_info(test_name, benchmark_result, average_cpu_used, average_mem_used)

        user_content = ("The previous file modifications are: "
                         f"\n###\n{result}\n###\n"
                         f"The benchmark results for the previous file are: {benchmark_line}")
    
    
    user_contents = [user_content, "Based on these information generate a new file in the same format as the options_file (but only give the changed value) to improve my database performance. Enclose the new options file in <config></config>."]
    return user_contents

def midway_options_file_generation(options, db_bench_args, avg_cpu_used, avg_mem_used, last_throughput, device_information, trace_result, options_file):
    system_content = generate_system_content(device_information, trace_result)

    user_content = []
    
    previous_option_file = options_file[-1][0]
    config = configparser.ConfigParser()
    config.read_string(previous_option_file)
    
    ops_per_sec_knobs = [
        'access_hint_on_compaction_start','allow_concurrent_memtable_write','allow_fallocate','allow_ingest_behind',
        'allow_mmap_writes','compaction_readahead_size','db_write_buffer_size','delayed_write_rate','enable_pipelined_write',
        'max_background_compactions','max_background_flushes','max_background_jobs','max_file_opening_threads','max_open_files',
        'max_subcompactions','max_write_batch_group_size_bytes','two_write_queues','unordered_write',
        'use_direct_io_for_flush_and_compaction','writable_file_max_buffer_size','arena_block_size','hard_pending_compaction_bytes_limit',
        'level0_slowdown_writes_trigger','level0_stop_writes_trigger',
        'write_buffer_size', 'max_write_buffer_number', 'min_write_buffer_number_to_merge',
        'target_file_size_base', 'max_bytes_for_level_base', 'level0_file_num_compaction_trigger'
    ]
    
    relevant_options = {}
    for section in config.sections():
        for key, value in config.items(section):
            if key in ops_per_sec_knobs:
                relevant_options[key] = value
    
    content = "Can you generate a new options file for RocksDB based on the following information?\n"
    content += "The previous throughput-related options were:\n"
    content += "###\n"
    
    for key, value in relevant_options.items():
        content += f"{key} = {value}\n"
    
    content += "###\n"
    content += f"The throughput results for the above options were: {options_file[-1][1]['ops_per_sec']} ops/sec.\n"

    user_content.append(content)
    
    if options != previous_option_file:
        content = ""
        content += "We then made the following changes to the options:\n"

        pattern = re.compile(r'\s*([^=\s]+)\s*=\s*([^=\s]+)\s*')
        file1_lines = pattern.findall(options)
        file2_lines = pattern.findall(previous_option_file)

        file1_lines = ["{} = {}".format(k, v) for k, v in file1_lines]
        file2_lines = ["{} = {}".format(k, v) for k, v in file2_lines]
        differ = Differ()
        diff = list(differ.compare(file1_lines, file2_lines))
        lst= []
        for line in diff:
            if line.startswith('+'):
                lst.append(line)
        result = '\n'.join(line[2:] for line in lst)

        content += "###\n"
        content += result
        content += "###\n"
        content += f"The updated configuration resulted in throughput of: {last_throughput} ops/sec.\n\n"
        user_content.append(content)
    
    content = ""
    content += f"Current system utilization: CPU {avg_cpu_used}%, Memory {avg_mem_used}%\n"
    content += "The throughput has decreased significantly. Please generate new throughput-optimized options "
    content += "that can better utilize the available system resources. "
    content += "Focus on parallel processing, memory allocation, and I/O optimization. "
    content += "Enclose the new options in <config></config>. Feel free to use up to 100% of CPU and Memory."
    user_content.append(content)

    log_update("[OG] Generating midway options file with throughput-focused parameters")
    log_update("[OG] Prompt for midway options file generation")
    log_update(content)
    matches = request_llm(system_content, user_content, None, 0.4)

    clean_options_file = ""
    reasoning = ""
    changed_value_dict = {}

    if matches is not None:
        clean_options_file, changed_value_dict, db_bench_args = cleanup_options_file(matches[1], db_bench_args)
        reasoning = matches[0] + matches[2]

    return clean_options_file, db_bench_args, reasoning, changed_value_dict


def dynamic_options_file_generation(prev_options, db_bench_args, avg_cpu_used, avg_mem_used, last_throughput, device_information, trace_result, options_file):
    sys_content = (
        "You are a RocksDB Expert being consulted by a company to help improve their RocksDB performance "
        "by optimizing the mutable options while the workloads is running."
        f"Direct IO will always be used. Additionally, compression type is set to none always. "
        "Respond with the the reasoning first, then show the options in original format."
        f"The Device information is: {device_information}"
    )

    db_options = [
        'max_background_jobs',
        'max_background_compactions',
        'max_subcompactions',
        'avoid_flush_during_shutdown',
        'writable_file_max_buffer_size',
        'delayed_write_rate',
        'max_total_wal_size',
        'delete_obsolete_files_period_micros',
        'stats_dump_period_sec',
        'stats_persist_period_sec',
        'stats_history_buffer_size',
        'max_open_files',
        'bytes_per_sync',
        'wal_bytes_per_sync',
        'strict_bytes_per_sync',
        'compaction_readahead_size',
        'max_background_flushes'
    ]

    user_content = []

    for opt_file in options_file[-3:-1]:
        values = {}
        opt_string = ""
        for line in opt_file[0].split('\n'):
            for var in db_options:
                pattern = rf'\b{var}\b\s*=\s*(\S+)'
                match = re.search(pattern, line)
                if match:
                    values[var] = match[0]
        
        for var, val in values.items():
            opt_string += f"{var} = {val}\n"

        content = "The previous options file is:\n"
        content += "###\n"
        content += opt_string
        content += "###\n"
        content += (
            f"The throughput results for the above options file are: {opt_file[1]['ops_per_sec']}. "
        )
        user_content.append(content)

    content = "Can you generate a new options for RocksDB based on the following information?\n"
    
    content += f"The trace from the last 20 seconds of the workload is as follows:\n"
    content += f"{trace_result}\n"

    content += "The CPU and Memory usage during the last 20 seconds of the workload was: "
    content += f"{avg_cpu_used}% and {avg_mem_used}\n"

    content += "The previous db_options values for each of the MutableDBOptions are as follows:\n"

    values = {}
    for line in options_file[-1][0].split('\n'):
        for var in db_options:
            pattern = rf'\b{var}\b\s*=\s*(\S+)'
            match = re.search(pattern, line)
            if match:
                values[var] = match[0]
    
    for var, val in values.items():
        content += f"{var} = {val}\n"

    content += (
        f"The throughput results for the above options file are: {options_file[-1][1]['ops_per_sec']}. "
    )
    if (len(options_file) > 1):
        if (options_file[-1][1]['ops_per_sec'] > options_file[-2][1]['ops_per_sec']):
            content += (
                "Which is an improvement from the previous throughput of "
                f"{options_file[-2][1]['ops_per_sec']}. "
                "Keep it up!. "
                "Based on this information generate a new file. "
            )
        else:
            content += (
                "Which is a decrease from the previous throughput of "
                f"{options_file[-2][1]['ops_per_sec']}. "
                "Please revert the changes made in the previous file, "
                "and generate a new file but different approach from the previous one. "
            )
            
    content += "Enclose the new options in <config></config>. Feel free to use upto 100% of the CPU and Memory."
    user_content.append(content)

    log_update("[OG] Generating options file with differences")
    log_update("[OG] Prompt for Dynamic options file generation")
    matches = request_llm(sys_content, user_content, None, 0.4)

    clean_options_file = ""
    reasoning = ""
    changed_value_dict = {}

    if matches is not None:
        clean_options_file, changed_value_dict, db_bench_args = cleanup_options_file(matches[1], db_bench_args)
        reasoning = matches[0] + matches[2]
    return clean_options_file, db_bench_args, reasoning, changed_value_dict

def error_correction_options_file_generation(error_options, db_bench_args, reasoning, changed_value_dict, error_reason, iteration):
    system_content = (
        "You are a RocksDB Expert being consulted by a company to help improve their RocksDB performance "
        "by optimizing the options configured for a particular scenario they face."
        "But there was an error in the options file generated. "
        "Respond with the error reasoning first, then show the corrected option file in original format."
        f"Only provide options files for rocksdb version {VERSION}. "
        "Enclose the new options in <config></config>"
    )

    args_dict = parse_db_bench_args_to_dict(db_bench_args)
    args = "\n".join(f"{key}={value}" for key, value in args_dict.items())

    user_content = [(
        "The options file generated had an error. This is the options file that was generated:\n"
        "###\n"
        f"{args}\n"
        f"{error_options}"
        "###\n"
        "The error in the options file was:\n"
        f"{error_reason}"
        "Fix the error and generate a new file (but only give the changed value). Enclose the new options in <config></config>."
    )]

    print("[OG] Generating options file to correct error")
    log_update("[OG] Generating options file to correct error")
    matches = request_llm(system_content, user_content, None, 0.4)

    clean_options_file = ""
    changed_value_dict_part = {}
    
    if matches is not None:
        clean_options_file, changed_value_dict_part, db_bench_args = cleanup_options_file(matches[1], db_bench_args)
        reasoning += "\n"+ matches[0] + matches[2]
    
    changed_value_dict.update(changed_value_dict_part)
    return clean_options_file, db_bench_args, reasoning, changed_value_dict

def generate_resource_usage_content(previous_option_files, average_cpu_used=-1.0, average_mem_used=-1.0, test_name="fillrandom"):
    result =" "
    user_content = []

    previous_option_file1, _, _, _, _ = previous_option_files[-1]
    config = configparser.ConfigParser()
    config.read_string(previous_option_file1)

    resource_usage = {
        'CPU': [
            'max_background_flushes', 'max_background_compactions', 'max_background_jobs', 
            'max_file_opening_threads', 'max_subcompactions', 'enable_thread_tracking',
            'write_thread_max_yield_usec', 'write_thread_slow_yield_usec', 'enable_write_thread_adaptive_yield',
            'two_write_queues', 'compaction_style', 'compaction_pri', 'level0_file_num_compaction_trigger',
            'level0_slowdown_writes_trigger', 'level0_stop_writes_trigger', 'paranoid_checks', 
            'verify_sst_unique_id_in_manifest', 'use_adaptive_mutex'
        ],
        'Storage': [
            'max_open_files', 'compaction_readahead_size', 'wal_bytes_per_sync', 'bytes_per_sync',
            'delete_obsolete_files_period_micros', 'max_total_wal_size', 'strict_bytes_per_sync',
            'writable_file_max_buffer_size', 'log_file_time_to_roll', 'max_log_file_size',
            'manifest_preallocation_size', 'allow_data_in_errors', 'WAL_ttl_seconds', 'recycle_log_file_num',
            'file_checksum_gen_factory', 'keep_log_file_num', 'random_access_max_buffer_size', 
            'access_hint_on_compaction_start', 'manual_wal_flush', 'use_direct_reads', 
            'use_direct_io_for_flush_and_compaction', 'allow_mmap_writes', 'allow_mmap_reads', 
            'advise_random_on_open', 'db_write_buffer_size'
        ],
        'Advanced': [
            'stats_history_buffer_size', 'stats_dump_period_sec', 'stats_persist_period_sec', 
            'info_log_level', 'enable_pipelined_write', 'persist_stats_to_disk', 'WAL_size_limit_MB', 
            'fail_if_options_file_error', 'db_host_id', 'wal_recovery_mode', 'wal_filter', 'allow_2pc', 
            'unordered_write', 'track_and_verify_wals_in_manifest', 'skip_checking_sst_file_sizes_on_db_open', 
            'skip_stats_update_on_db_open', 'force_consistency_checks', 'memtable_whole_key_filtering', 
            'cache_index_and_filter_blocks', 'cache_index_and_filter_blocks_with_high_priority', 
            'pin_l0_filter_and_index_blocks_in_cache', 'allow_ingest_behind', 'avoid_unnecessary_blocking_io', 
            'write_dbid_to_manifest', 'best_efforts_recovery', 'enable_write_thread_adaptive_yield', 
            'flush_verify_memtable_count', 'create_missing_column_families', 'create_if_missing', 
            'is_fd_close_on_exec', 'enforce_single_del_contracts'
        ],
        'Memory': [
            'write_buffer_size', 'max_write_buffer_number', 'arena_block_size', 'max_bytes_for_level_base',
            'max_bytes_for_level_multiplier', 'target_file_size_base', 'max_compaction_bytes', 'block_size',
            'block_restart_interval', 'pin_top_level_index_and_filter', 'max_write_batch_group_size_bytes',
            'write_thread_max_yield_usec', 'db_write_buffer_size'
        ]
    }
    categorized_parameters = {category: {} for category in resource_usage}
    for section in config.sections():
        for key, value in config.items(section):
            for category, params in resource_usage.items():
                if key in params:
                    categorized_parameters[category][key] = value

    result = {category: [] for category in resource_usage}
    for category in ['CPU', 'Storage', 'Advanced', 'Memory']:
        result[category].append(f"{category} Parameters:\n")
        for param, value in categorized_parameters[category].items():
            result[category].append(f"  {param}= {value}\n")
        result[category].append("\n")

    for category in ['CPU', 'Storage', 'Advanced', 'Memory']:
        user_content.append(f"###\n{''.join(result[category])}###\n")
    return user_content

def generate_shard_content(previous_option_files, metric, test_name="fillrandom"):
    result =" "
    user_content = []

    previous_option_file1, _, _, _, _ = previous_option_files[-1]
    config = configparser.ConfigParser()
    config.read_string(previous_option_file1)

    item_knobs= {
        'p99': [
            'advise_random_on_open', 'allow_mmap_reads', 'atomic_flush', 'avoid_unnecessary_blocking_io', 
            'bytes_per_sync', 'enable_write_thread_adaptive_yield', 'lowest_used_cache_tier', 'manual_wal_flush', 
            'strict_bytes_per_sync', 'table_cache_numshardbits', 'use_adaptive_mutex', 'use_direct_reads', 'use_fsync', 
            'wal_bytes_per_sync', 'write_thread_max_yield_usec', 'write_thread_slow_yield_usec', 'max_sequential_skip_in_iterations', 
            'memtable_insert_with_hint_prefix_extractor', 'prefix_extractor', 'block_restart_interval'
        ],
        'ops_per_sec': [
            'access_hint_on_compaction_start','allow_concurrent_memtable_write','allow_fallocate','allow_ingest_behind',
            'allow_mmap_writes','compaction_readahead_size','db_write_buffer_size','delayed_write_rate','enable_pipelined_write',
            'max_background_compactions','max_background_flushes','max_background_jobs','max_file_opening_threads','max_open_files',
            'max_subcompactions','max_write_batch_group_size_bytes','two_write_queues','unordered_write',
            'use_direct_io_for_flush_and_compaction','writable_file_max_buffer_size','arena_block_size','hard_pending_compaction_bytes_limit',
            'level0_slowdown_writes_trigger','level0_stop_writes_trigger'
        ],
        'write_amp': [
            'blob_file_size','blob_file_starting_level','blob_garbage_collection_age_cutoff','blob_garbage_collection_force_threshold',
            'bottommost_compression','bottommost_compression_opts','compaction_filter','compaction_filter_factory','compaction_options_fifo',
            'compaction_options_universal','compaction_pri','compression','compression_opts','disable_auto_compactions','enable_blob_files',
            'enable_blob_garbage_collection','experimental_mempurge_threshold','ignore_max_compaction_bytes_for_input','inplace_update_support',
            'level_compaction_dynamic_file_size','level0_file_num_compaction_trigger','max_bytes_for_level_base',
            'max_bytes_for_level_multiplier','max_bytes_for_level_multiplier_additional','max_compaction_bytes','max_successive_merges',
            'max_write_buffer_number','max_write_buffer_number_to_maintain','max_write_buffer_size_to_maintain','memtable_max_range_deletions',
            'merge_operator','min_blob_size','min_write_buffer_number_to_merge','num_levels','periodic_compaction_seconds',
            'preclude_last_level_data_seconds','preserve_internal_time_seconds','sst_partitioner_factory','target_file_size_base',
            'target_file_size_multiplier','ttl','compaction_style','write_buffer_size','flush_block_policy_factory'
        ],
        'read_amp': [
            'memtable_prefix_bloom_size_ratio','memtable_whole_key_filtering','optimize_filters_for_hits','block_size','block_size_deviation',
            'cache_index_and_filter_blocks','cache_index_and_filter_blocks_with_high_priority','data_block_hash_table_util_ratio',
            'data_block_index_type','enable_index_compression','filter_policy','index_shortening','index_type','metadata_cache_options',
            'no_block_cache','optimize_filters_for_memory','partition_filters','pin_l0_filter_and_index_blocks_in_cache','pin_top_level_index_and_filter',
            'prepopulate_block_cache','read_amp_bytes_per_bit','whole_key_filtering'
        ]
    }
    item_parameters = {item: {} for item in item_knobs}
    for section in config.sections():
        for key, value in config.items(section):
            for item, params in item_knobs.items():
                if key in params:
                    item_parameters[item][key] = value

    user_content.append(f"###\n{item_parameters[metric]}\n###\n")
    return user_content