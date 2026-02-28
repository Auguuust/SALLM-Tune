#!/usr/bin/env python3

import os
import sys
import subprocess
import shutil
import time
import glob
from datetime import datetime

WORKLOADS = ['readrandom', 'fillrandom', 'readwhilewriting', 'mixgraph']

LIMIT_LISTS = ['244','288','444', '488']
LLM_MODELS = ['qwen3-8b-ft-0829', 'deepseek-ai/DeepSeek-V3', 'qwen-plus-0428']

SHARKTUNE_SCRIPT = "SALLM-Tune.py"

def log_message(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def get_latest_output_folder(device="data"):
    output_base_dir = f"output/output_{device}"
    
    if not os.path.exists(output_base_dir):
        return None
    
    folders = []
    for item in os.listdir(output_base_dir):
        folder_path = os.path.join(output_base_dir, item)
        if os.path.isdir(folder_path):
            folders.append(folder_path)
    
    if not folders:
        return None
    
    folders.sort(key=lambda x: os.path.getctime(x), reverse=True)
    return folders[0]

def cleanup_failed_run(device="data"):
    latest_folder = get_latest_output_folder(device)
    if latest_folder:
        log_message(f"delete failed run folder: {latest_folder}")
        try:
            shutil.rmtree(latest_folder)
            log_message(f"success delete folder: {latest_folder}")
        except Exception as e:
            log_message(f"failed delete folder: {e}")

def run_single_test(workload, llm_model, test_num, total_tests, limit_list):
    log_message(f"start running test {test_num}/{total_tests}: workload={workload}, llm_model={llm_model}, limit_list={limit_list}")
    
    cmd = [
        sys.executable, SHARKTUNE_SCRIPT,
        f"--workload={workload}",
        f"--llm_model={llm_model}",
        f"--limit_list={limit_list}"
    ]
    
    max_retries = 20
    for attempt in range(max_retries):
        try:
            log_message(f"execute command (attempt {attempt + 1}/{max_retries}): {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=False, 
                text=True,
                timeout=3600
            )
            
            if result.returncode == 0:
                log_message(f"test success completed: workload={workload}, llm_mode={llm_mode}")
                return True
            else:
                log_message(f"test failed (exit code: {result.returncode})")
                log_message(f"standard error output: {result.stderr}")
                
                if "Failed to benchmark with the initial options file" in result.stderr or \
                   "Tracefile analysis failed" in result.stderr or \
                   result.returncode == 1:
                    log_message("detected initialization failed, clean output folder")
                    cleanup_failed_run()
                    
                    if attempt < max_retries - 1:
                        log_message(f"wait 5 seconds to retry...")
                        time.sleep(5)
                        continue
                
                return False
                
        except subprocess.TimeoutExpired:
            log_message(f"test timeout: workload={workload}, llm_mode={llm_mode}")
            cleanup_failed_run()
            if attempt < max_retries - 1:
                log_message(f"wait 5 seconds to retry...")
                time.sleep(5)
                continue
            return False
        except Exception as e:
            log_message(f"error occurred when running test: {e}")
            cleanup_failed_run()
            if attempt < max_retries - 1:
                log_message(f"wait 5 seconds to retry...")
                time.sleep(5)
                continue
            return False
    
    log_message(f"test final failed: workload={workload}, llm_mode={llm_mode}")
    return False

def main():
    log_message("start batch testing")
    log_message(f"Workloads: {WORKLOADS}")
    log_message(f"LLM Modes: {LLM_MODES}")
    
    if not os.path.exists(SHARKTUNE_SCRIPT):
        log_message(f"error: script file {SHARKTUNE_SCRIPT} not found")
        sys.exit(1)
    
    test_combinations = []
    for workload in WORKLOADS:
        for llm_mode in LLM_MODES:
            for limit_list in LIMIT_LISTS:
                test_combinations.append((workload, llm_mode, limit_list))
    
    total_tests = len(test_combinations)
    log_message(f"total {total_tests} tests to run")
    
    successful_tests = []
    failed_tests = []
    
    for i, (workload, llm_mode, limit_list) in enumerate(test_combinations, 1):
        subprocess.run(["free", "-m"])
        subprocess.run(["df", "-h"])
        subprocess.run(["swapon", "-s"])
        
        time.sleep(20)
        
        log_message(f"\n{'='*60}")
        log_message(f"progress: {i}/{total_tests}")
        
        success = run_single_test(workload, llm_mode, i, total_tests, limit_list)
        
        if success:
            successful_tests.append((workload, llm_mode))
        else:
            failed_tests.append((workload, llm_mode))
        
        log_message(f"test {i} completed, result: {'success' if success else 'failed'}")

        if i < total_tests:
            log_message("wait 5 seconds to continue the next test...")
            time.sleep(5)
    
    log_message(f"\n{'='*60}")
    log_message("batch testing completed")
    log_message(f"successful tests: {len(successful_tests)}")
    log_message(f"failed tests: {len(failed_tests)}")
    
    if successful_tests:
        log_message("\nsuccessful tests:")
        for workload, llm_mode in successful_tests:
            log_message(f"  - workload={workload}, llm_mode={llm_mode}")
    
    if failed_tests:
        log_message("\nfailed tests:")
        for workload, llm_mode in failed_tests:
            log_message(f"  - workload={workload}, llm_mode={llm_mode}")
    
    if failed_tests:
        log_message("some tests failed, please check the logs")
        sudoPassword = 'embed'
        finish_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        emailCommand = f'sudo /usr/local/bin/sendEmail \
                    -f tranining_notifier@163.com \
                    -t tranining_notifier@163.com \
                    -s smtp.163.com \
                    -u "[L40]_training_finished" \
                    -o message-content-type=html -o message-charset=utf8 \
                    -xu tranining_notifier@163.com \
                    -xp PHgJQLsRakbAy24S \
                    -m "Testing finished at {finish_time} on L40! Success {len(successful_tests)}. Failed {len(failed_tests)}. "'
        state = False
        while not state:
            try:
                os.system('echo %s|sudo -S %s' % (sudoPassword, emailCommand))
                state = True
            except:
                print('Failed! Resending...\n')
                state = False
                time.sleep(5)
        if state == True:
            print('Mail sent successfully')
        sys.exit(1)
    else:
        log_message("all tests completed successfully!")

if __name__ == "__main__":
    main() 