import utils.constants as constants
from utils.graph import plot, plot_multiple, plot_all_metrics
from utils.system_operations.fio_runner import get_fio_result
from options_files.ops_options_file import parse_option_file_to_dict, get_initial_options_file

import rocksdb.subprocess_manager as spm
from utils.utils import log_update, store_best_option_file, store_best_options_for_all_metrics, path_of_db, store_diff_options_list
from utils.system_operations.get_sys_info import system_info
from llm.prompts_generator import generate_option_file_with_llm
from trace_analyzer.analyzer import analyze_tracefile
import os
import datetime

def check_metric_degradation(current_results, previous_results, metric_name, degradation_threshold=0.1):

    if previous_results is None or current_results is None:
        return False, 0.0
    
    current_value = current_results.get(metric_name)
    previous_value = previous_results.get(metric_name)
    
    if current_value is None or previous_value is None:
        return False, 0.0

    if metric_name == "p99":
        if current_value > previous_value:
            degradation = (current_value - previous_value) / previous_value
            return degradation > degradation_threshold, degradation

    elif metric_name == "ops_per_sec":
        if current_value < previous_value:
            degradation = (previous_value - current_value) / previous_value
            return degradation > degradation_threshold, degradation

    elif metric_name in ["write_amp", "read_amp"]:
        if current_value > previous_value:
            degradation = (current_value - previous_value) / previous_value
            return degradation > degradation_threshold, degradation
    
    return False, 0.0

def check_all_metrics_degradation(current_results, options_files, tune_item, current_tune_index, degradation_threshold=0.1):

    degradation_status = {}
    
    for i, metric in enumerate(itemtune_[:current_tune_index + 1]):
        best_value = None
        best_index = -1

        for j, (_, results, _, _, _) in enumerate(options_files):
            if results and results.get(metric) is not None:
                if best_value is None:
                    best_value = results[metric]
                    best_index = j
                else:

                    if metric in ["p99", "write_amp", "read_amp"]:
                        if results[metric] < best_value:
                            best_value = results[metric]
                            best_index = j

                    elif metric == "ops_per_sec":
                        if results[metric] > best_value:
                            best_value = results[metric]
                            best_index = j
        
        if best_value is not None:
            mock_previous = {metric: best_value}
            is_degraded, degradation_pct = check_metric_degradation(
                current_results, mock_previous, metric, degradation_threshold
            )
            
            degradation_status[metric] = {
                "degraded": is_degraded,
                "degradation_pct": degradation_pct,
                "best_value": best_value,
                "current_value": current_results.get(metric),
                "best_index": best_index
            }
        else:
            degradation_status[metric] = {
                "degraded": False,
                "degradation_pct": 0.0,
                "best_value": None,
                "current_value": current_results.get(metric),
                "best_index": -1
            }
    
    return degradation_status

def check_tuning_conditions(current_results, options_files, tune_item, current_tune_index):
    result = {
        "can_proceed": True,
        "previous_metric_status": None,
        "current_metric_status": None,
        "issues": []
    }
    
    current_metric = tune_item[current_tune_index]

    if len(options_files) >= 1:
        if len(options_files) == 1:
            previous_result = options_files[0][1]
        else:
            previous_result = options_files[-1][1]
        
        current_value = current_results.get(current_metric)
        previous_value = previous_result.get(current_metric)
        
        if current_value is not None and previous_value is not None:
            if current_metric in ["p99", "write_amp", "read_amp"]:
                improvement = (previous_value - current_value) / previous_value
                if current_metric in ["write_amp", "read_amp"]:
                    current_improved = improvement >= 0
                else:
                    current_improved = improvement > 0
            elif current_metric == "ops_per_sec":
                improvement = (current_value - previous_value) / previous_value
                current_improved = improvement > 0
            else:
                current_improved = False
                improvement = 0
            
            result["current_metric_status"] = {
                "improved": current_improved,
                "improvement_pct": improvement,
                "current_value": current_value,
                "previous_value": previous_value
            }
            
            if not current_improved:
                result["can_proceed"] = False
                if current_metric in ["write_amp", "read_amp"]:
                    result["issues"].append(f"Current tuning metric {current_metric} degraded (worsened)")
                else:
                    result["issues"].append(f"Current tuning metric {current_metric} shows no improvement")
    
    if current_tune_index > 0:
        previous_metric = tune_item[current_tune_index - 1]
        
        best_value = None
        for config in options_files:
            if len(config) >= 5:
                _, results, _, _, is_failed_tuning = config
            else:
                _, results, _, _ = config
                
            if results and results.get(previous_metric) is not None:
                if best_value is None:
                    best_value = results[previous_metric]
                else:
                    if previous_metric in ["p99", "write_amp", "read_amp"]:
                        if results[previous_metric] < best_value:
                            best_value = results[previous_metric]
                    elif previous_metric == "ops_per_sec":
                        if results[previous_metric] > best_value:
                            best_value = results[previous_metric]
        
        if best_value is not None:
            current_prev_value = current_results.get(previous_metric)
            if current_prev_value is not None:
                if previous_metric in ["p99", "write_amp", "read_amp"]:
                    if current_prev_value > best_value:
                        degradation = (current_prev_value - best_value) / best_value
                        degraded = degradation > 0.1
                    else:
                        degradation = 0
                        degraded = False
                elif previous_metric == "ops_per_sec":
                    if current_prev_value < best_value:
                        degradation = (best_value - current_prev_value) / best_value
                        degraded = degradation > 0.1
                    else:
                        degradation = 0
                        degraded = False
                else:
                    degraded = False
                    degradation = 0
                
                result["previous_metric_status"] = {
                    "degraded": degraded,
                    "degradation_pct": degradation,
                    "current_value": current_prev_value,
                    "best_value": best_value
                }
                
                if degraded:
                    result["can_proceed"] = False
                    result["issues"].append(f"Previous metric {previous_metric} degraded by more than 10% ({degradation:.2%})")
                    print(f"Previous metric {previous_metric} degraded by more than 10% ({degradation:.2%})")
    
    return result

def main():
    options_files = []
    options_list = []

    output_folder_dir = constants.OUTPUT_PATH
    os.makedirs(output_folder_dir, exist_ok=True)
    db_path = path_of_db()
    fio_result = get_fio_result(constants.FIO_RESULT_PATH)

    log_update(f"[MFN] Starting the program with metric-based optimization")
    print(f"[MFN] Starting the program with metric-based optimization")

    options, reasoning = get_initial_options_file()

    is_error, benchmark_results, average_cpu_usage, average_memory_usage, options = spm.benchmark(
        db_path, options, output_folder_dir, reasoning, None, 0, None, options_files, [])

    if is_error:
        log_update("[MFN] Failed to benchmark with the initial options file. Exiting.")
        print("[MFN] Failed to benchmark with the initial options file. Exiting.")
        exit(1)
    else:
        trace_result = analyze_tracefile(db_path + "/tracefile")
        if trace_result is None:
            print("[MFN] Tracefile analysis failed. Exiting.")
            log_update("[MFN] Tracefile analysis failed. Exiting.")
            exit(1)

        parsed_options = parse_option_file_to_dict(options)
        options_list.append(parsed_options)

        options_files.append((options, benchmark_results, reasoning, "", False))
        db_bench_args = []

        iteration_count = constants.ITERATION_COUNT

        if constants.TEST_NAME == "fillrandom":
            tune_item = ['p99', 'ops_per_sec', 'write_amp']
        elif constants.TEST_NAME == "readrandom":
            tune_item = ['p99', 'ops_per_sec', 'read_amp']
        else:
            tune_item = ['p99', 'ops_per_sec', 'write_amp', 'read_amp']

        for i, item in enumerate(tune_item):

            log_update(f"[MFN] Starting iteration {item}")
            log_update(f"[MFN] Querying LLM for next options file")

            print("-" * 50)
            print(f"[MFN] Starting iteration {item}")

            print("[MFN] Querying LLM for next options file")
            temperature = 0.5
            retry_counter = 4
            generated = False
            tuning_retry_count = 0
            max_tuning_retries = 3
            last_failed_attempt = None
            all_failed_attempts = []

            while tuning_retry_count < max_tuning_retries:
                log_update(f"[MFN] Starting tuning attempt {tuning_retry_count + 1} for {item}")
                print(f"[MFN] Starting tuning attempt {tuning_retry_count + 1} for {item}")
                
                current_retry_counter = retry_counter
                
                for llm_query_count in range(current_retry_counter, 0, -1):
                    performance_feedback = ""
                    
                    if tuning_retry_count > 0 and last_failed_attempt is not None:
                        failed_results = last_failed_attempt[1]
                        tuning_check = check_tuning_conditions(failed_results, options_files, tune_item, i)
                        
                        if not tuning_check["can_proceed"]:
                            feedback_parts = []
                            
                            if tuning_check["current_metric_status"]:
                                current_status = tuning_check["current_metric_status"]
                                if not current_status["improved"]:
                                    improvement_pct = current_status["improvement_pct"]
                                    current_val = current_status["current_value"]
                                    previous_val = current_status["previous_value"]
                                    
                                    if item in ["p99", "write_amp", "read_amp"]:
                                        if item == "p99":
                                            feedback_parts.append(f"Current metric {item} failed to improve (increased from {previous_val:.3f}ms to {current_val:.3f}ms, change: {improvement_pct:.2%})")
                                        else:
                                            feedback_parts.append(f"Current metric {item} degraded (increased from {previous_val:.4f} to {current_val:.4f}, change: {improvement_pct:.2%})")
                                    else:
                                        feedback_parts.append(f"Current metric {item} failed to improve (decreased from {previous_val:.0f} to {current_val:.0f}, change: {improvement_pct:.2%})")
                            
                            if tuning_check["previous_metric_status"]:
                                prev_status = tuning_check["previous_metric_status"]
                                if prev_status["degraded"]:
                                    prev_metric = tune_item[i-1] if i > 0 else "N/A"
                                    degradation_pct = prev_status["degradation_pct"]
                                    current_val = prev_status["current_value"]
                                    best_val = prev_status["best_value"]
                                    
                                    if prev_metric in ["p99", "write_amp", "read_amp"]:
                                        if prev_metric == "p99":
                                            feedback_parts.append(f"Previous metric {prev_metric} degraded significantly (increased from best {best_val:.3f}ms to {current_val:.3f}ms, degradation: {degradation_pct:.2%})")
                                        else:
                                            feedback_parts.append(f"Previous metric {prev_metric} degraded significantly (increased from best {best_val:.4f} to {current_val:.4f}, degradation: {degradation_pct:.2%})")
                                    else:
                                        feedback_parts.append(f"Previous metric {prev_metric} degraded significantly (decreased from best {best_val:.0f} to {current_val:.0f}, degradation: {degradation_pct:.2%})")
                            
                            if feedback_parts:
                                performance_feedback = f"Performance issue feedback: {' AND '.join(feedback_parts)}. Please adjust the configuration to address these specific performance issues while optimizing for {item}."
                            else:
                                performance_feedback = f"Performance issue feedback: Previous attempt failed to meet tuning conditions for metric {item}. Please adjust the configuration."

                    log_update(f"[MFN] LLM generation attempt {current_retry_counter - llm_query_count + 1}/{current_retry_counter}")
                    print(f"[MFN] LLM generation attempt {current_retry_counter - llm_query_count + 1}/{current_retry_counter}")
                    
                    new_options_file, db_bench_args, reasoning, changed_value_dict = generate_option_file_with_llm(item,
                        options_files, db_bench_args,
                        system_info(db_path, fio_result), trace_result, temperature,
                        average_cpu_usage, average_memory_usage, 
                            constants.TEST_NAME, "8.8.1", performance_feedback)
                    if new_options_file is None:
                        log_update(f"[MFN] Failed to generate options file. Retrying. Retries left: {llm_query_count - 1}")
                        print("[MFN] Failed to generate options file. Retrying. Retries left: ", llm_query_count - 1)
                        continue

                    log_update(f"[MFN] Options file generated successfully, starting benchmark")
                    print(f"[MFN] Options file generated successfully, starting benchmark")
                    
                    is_error, benchmark_results, average_cpu_usage, average_memory_usage, new_options_file = spm.benchmark(
                        db_path, new_options_file, output_folder_dir, reasoning, changed_value_dict, iteration_count, benchmark_results, options_files, db_bench_args)
                    if is_error:
                        log_update(f"[MFN] Benchmark failed. Retrying with new options file. Retries left: {llm_query_count - 1}")
                        print("[MFN] Benchmark failed. Retrying with new options file. Retries left: ", llm_query_count - 1)
                        temperature += 0.1
                        continue
                    else:
                        tuning_check = check_tuning_conditions(benchmark_results, options_files, tune_item, i)
                    
                        log_update(f"[MFN] Tuning condition check results:")
                        if tuning_check["current_metric_status"]:
                            current_status = tuning_check["current_metric_status"]
                            if item in ["write_amp", "read_amp"]:
                                status_text = 'acceptable (not degraded)' if current_status['improved'] else 'degraded'
                            else:
                                status_text = 'improved' if current_status['improved'] else 'not improved'
                            log_update(f"[MFN] Current metric {item}: {status_text} "
                                     f"({current_status['improvement_pct']:.2%})")
                            print(f"[MFN] Current metric {item}: {status_text} "
                                  f"({current_status['improvement_pct']:.2%})")
                        
                        if tuning_check["previous_metric_status"]:
                            prev_status = tuning_check["previous_metric_status"]
                            prev_metric = tune_item[i-1] if i > 0 else "N/A"
                            log_update(f"[MFN] Previous metric {prev_metric}: {'degraded by more than 10%' if prev_status['degraded'] else 'degraded by less than 10%'} "
                                     f"({prev_status['degradation_pct']:.2%})")
                            print(f"[MFN] Previous metric {prev_metric}: {'degraded by more than 10%' if prev_status['degraded'] else 'degraded by less than 10%'} "
                                  f"({prev_status['degradation_pct']:.2%})")
                        
                        if tuning_check["can_proceed"]:
                            log_update(f"[MFN] Tuning conditions satisfied, can proceed")
                            print(f"[MFN] Tuning conditions satisfied, can proceed")
                            generated = True
                            break
                        else:
                            log_update(f"[MFN] Tuning conditions not satisfied: {'; '.join(tuning_check['issues'])}")
                            print(f"[MFN] Tuning conditions not satisfied: {'; '.join(tuning_check['issues'])}")
                            
                            last_failed_attempt = (new_options_file, benchmark_results, reasoning, changed_value_dict, False)
                            all_failed_attempts.append((new_options_file, benchmark_results, reasoning, changed_value_dict, False))
                            
                            log_update(f"[MFN] Failed attempt recorded with detailed performance data, preparing to regenerate")
                            print(f"[MFN] Failed attempt recorded with detailed performance data, preparing to regenerate")
                            
                            generated = False
                            break

                log_update(f"[MFN] LLM retry loop ended. Generated: {generated}, LLM retries left: {llm_query_count-1 if 'llm_query_count' in locals() else 'unknown'}")
                print(f"[MFN] LLM retry loop ended. Generated: {generated}")
                
                if generated:
                    break
                else:
                    tuning_retry_count += 1
                    log_update(f"[MFN] Incrementing tuning retry count to {tuning_retry_count}, max allowed: {max_tuning_retries}")
                    print(f"[MFN] Incrementing tuning retry count to {tuning_retry_count}")
                    
                    if tuning_retry_count < max_tuning_retries:
                        log_update(f"[MFN] Retrying tuning for {item}, attempt {tuning_retry_count + 1}")
                        print(f"[MFN] Retrying tuning for {item}, attempt {tuning_retry_count + 1}")
                        temperature += 0.1
                    else:
                        log_update(f"[MFN] Failed to tune {item}, reached maximum retry attempts")
                        print(f"[MFN] Failed to tune {item}, reached maximum retry attempts")

            if generated:
                options = new_options_file
                options_files.append((options, benchmark_results, reasoning, changed_value_dict, False))
                parsed_options = parse_option_file_to_dict(options)
                options_list.append(parsed_options)

                log_update(f"[MFN] Successfully added new configuration to results (total: {len(options_files)} configurations)")
                print(f"[MFN] Successfully added new configuration to results (total: {len(options_files)} configurations)")

                plot([e[1]["ops_per_sec"] for e in options_files], f"OpsPerSec {constants.TEST_NAME}",
                     f"{output_folder_dir}/OpsPerSec.png")
                plot_multiple(options_files, "Ops Per Second",
                              f"{output_folder_dir}/opsM_per_sec.png")
                
                plot_all_metrics(options_files, output_folder_dir, constants.TEST_NAME)
        
            else:
                if all_failed_attempts:
                    best_attempt = None
                    best_value = None
                    
                    for attempt in all_failed_attempts:
                        if len(attempt) == 5:
                            attempt_options, attempt_results, attempt_reasoning, attempt_changed_dict, _ = attempt
                        elif len(attempt) == 4:
                            attempt_options, attempt_results, attempt_reasoning, attempt_changed_dict = attempt
                            log_update(f"[MFN] Warning: Found 4-element tuple in failed attempts, handling gracefully")
                        else:
                            log_update(f"[MFN] Error: Unexpected tuple length {len(attempt)} in failed attempts, skipping")
                            continue
                        
                        current_value = attempt_results.get(item)
                        
                        if current_value is not None:
                            if best_value is None:
                                best_value = current_value
                                best_attempt = attempt
                            else:
                                if item in ["p99", "write_amp", "read_amp"]:
                                    if current_value < best_value:
                                        best_value = current_value
                                        best_attempt = attempt
                                elif item == "ops_per_sec":
                                    if current_value > best_value:
                                        best_value = current_value
                                        best_attempt = attempt
                    
                    if best_attempt:
                        if len(best_attempt) == 5:
                            best_options, best_results, best_reasoning, best_changed_dict, _ = best_attempt
                        elif len(best_attempt) == 4:
                            best_options, best_results, best_reasoning, best_changed_dict = best_attempt
                            log_update(f"[MFN] Warning: best_attempt has 4 elements, handling gracefully")
                        else:
                            log_update(f"[MFN] Error: Unexpected best_attempt length {len(best_attempt)}, skipping")
                            continue
                        
                        options_files.append((best_options, best_results, best_reasoning, best_changed_dict, True))
                        parsed_options = parse_option_file_to_dict(best_options)
                        options_list.append(parsed_options)
                        
                        log_update(f"[MFN] Failed to tune {item} after maximum retries. Added best failed attempt to results.")
                        log_update(f"[MFN] Best failed attempt for {item}: {best_value}")
                        log_update(f"[MFN] Current results count: {len(options_files)} configurations")
                        print(f"[MFN] Failed to tune {item} after maximum retries. Added best failed attempt to results.")
                        print(f"[MFN] Best failed attempt for {item}: {best_value}")
                        print(f"[MFN] This configuration will be marked in red in the plots.")
                        
                        # Graph Ops/Sec
                        plot([e[1]["ops_per_sec"] for e in options_files], f"OpsPerSec {constants.TEST_NAME}",
                             f"{output_folder_dir}/OpsPerSec.png")
                        plot_multiple(options_files, "Ops Per Second",
                                      f"{output_folder_dir}/opsM_per_sec.png")
                        
                        # Generate comprehensive metrics plots for all 4 tuning metrics
                        plot_all_metrics(options_files, output_folder_dir, constants.TEST_NAME)
                    else:
                        log_update(f"[MFN] Failed to tune {item} after maximum retries. No valid failed attempts found.")
                        log_update(f"[MFN] Current results count remains: {len(options_files)} configurations")
                        print(f"[MFN] Failed to tune {item} after maximum retries. No valid failed attempts found.")
                        print(f"[MFN] Continuing to next metric.")
                else:
                    log_update(f"[MFN] Failed to tune {item} after maximum retries. No failed attempts recorded.")
                    log_update(f"[MFN] Current results count remains: {len(options_files)} configurations")
                    print(f"[MFN] Failed to tune {item} after maximum retries. No failed attempts recorded.")
                    print(f"[MFN] Continuing to next metric.")
            
            store_diff_options_list(options_list, output_folder_dir)

        store_best_option_file(options_files, output_folder_dir)
        store_best_options_for_all_metrics(options_files, output_folder_dir)

        # Graph Ops/Sec
        plot([e[1]["ops_per_sec"] for e in options_files], f"OpsPerSec {constants.TEST_NAME}",
             f"{output_folder_dir}/OpsPerSec.png")
        plot_multiple(options_files, "Ops Per Second",
                      f"{output_folder_dir}/opsM_per_sec.png")
        
        # Generate comprehensive metrics plots for all 4 tuning metrics
        plot_all_metrics(options_files, output_folder_dir, constants.TEST_NAME)
        
        store_diff_options_list(options_list, output_folder_dir)
        
        log_update(f"[MFN] Final results: {len(options_files)} successful configurations saved")
        print(f"[MFN] Final results: {len(options_files)} successful configurations saved")


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    time_taken = (end_time - start_time).total_seconds() / 60
    print(f"[MFN] Time taken: {time_taken} minutes")
    log_update(f"[MFN] Time taken: {time_taken} minutes")