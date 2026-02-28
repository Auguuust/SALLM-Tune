import os
import json
import configparser
from typing import Dict, List, Any, Tuple
from datetime import datetime
from flask import Flask, request, jsonify
import sys
import threading
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

module_paths = [
    os.path.join(current_dir, "utils"),
    os.path.join(current_dir, "rocksdb"),
    os.path.join(current_dir, "options_files"),
    os.path.join(current_dir, "llm"),
    os.path.join(current_dir, "trace_analyzer")
]

for path in module_paths:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

from eval_server_config import (
    setup_environment, 
    validate_dependencies, 
    EVAL_SERVER_HOST, 
    EVAL_SERVER_PORT, 
    LOG_DIR,
    OUTPUT_DIR,
    DB_PATH_DIR
)

setup_environment()
if not validate_dependencies():
    print("Failed to validate dependencies. Exiting.")
    sys.exit(1)

import utils.constants as constants
from utils.system_operations.fio_runner import get_fio_result
from utils.system_operations.get_sys_info import system_info
from options_files.ops_options_file import parse_option_file_to_dict, cleanup_options_file

from eval_server_utility_functions import log_update, path_of_db
from llm.eval_server_prompts_generator import generate_option_file_with_LLM
import rocksdb.eval_server_training_subprocess_manager as spm

from trace_analyzer.analyzer import analyze_tracefile, generate_trace_model, save_model_as_json

app = Flask(__name__)

log_name = f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}_eval_server.log"
log_file = open(os.path.join(LOG_DIR, log_name), "w")

current_cgroup_manager = None
current_config_str = None

class SystemConfigManager:
    
    def __init__(self):
        self.cgroup_managers = {}
        
    def parse_system_config(self, config_str: str) -> Dict[str, int]:
        if len(config_str) != 3:
            raise ValueError(f"System config string must be 3 digits, got: {config_str}")
        
        cpu_cores = int(config_str[0])
        swap_memory_gb = int(config_str[1])
        memory_gb = int(config_str[2])
        
        return {
            "cpu_cores": cpu_cores,
            "swap_memory_gb": swap_memory_gb, 
            "memory_gb": memory_gb
        }
    
    def apply_system_limits(self, config: Dict[str, int], config_str: str) -> bool:
        try:
            print(f"[EVAL SERVER] Applying system limits for config {config_str}: {config}")
            log_file.write(f"[EVAL SERVER] Applying system limits for config {config_str}: {config}\n")
            
            cgroup_name = f"rocksdb_training_{config_str}"
            
            manager = spm.FixedCGroupManager(cgroup_name, helper_script=os.path.abspath("utils/root_cgroup_helper.sh"))
            
            if manager.create_cgroup() != 0:
                print(f"[EVAL SERVER] Failed to create cgroup: {cgroup_name}")
                log_file.write(f"[EVAL SERVER] Failed to create cgroup: {cgroup_name}\n")
                return False
            
            if not manager.set_cpu_limit(config["cpu_cores"]):
                print(f"[EVAL SERVER] Failed to set CPU limit: {config['cpu_cores']}")
                log_file.write(f"[EVAL SERVER] Failed to set CPU limit: {config['cpu_cores']}\n")
                return False
            
            memory_bytes = config["memory_gb"] * 1024 * 1024 * 1024
            if not manager.set_memory_limit(memory_bytes):
                print(f"[EVAL SERVER] Failed to set memory limit: {config['memory_gb']}GB")
                log_file.write(f"[EVAL SERVER] Failed to set memory limit: {config['memory_gb']}GB\n")
                return False
            
            swap_bytes = config["swap_memory_gb"] * 1024 * 1024 * 1024
            if not manager.set_memory_swap_limit(swap_bytes):
                print(f"[EVAL SERVER] Failed to set swap limit: {config['swap_memory_gb']}GB")
                log_file.write(f"[EVAL SERVER] Failed to set swap limit: {config['swap_memory_gb']}GB\n")
                return False
            
            self.cgroup_managers[config_str] = manager
            
            print(f"[EVAL SERVER] Successfully applied system limits for {config_str}:")
            print(f"[EVAL SERVER] - CPU cores: {config['cpu_cores']}")
            print(f"[EVAL SERVER] - Memory: {config['memory_gb']}GB")
            print(f"[EVAL SERVER] - Swap: {config['swap_memory_gb']}GB")
            
            log_file.write(f"[EVAL SERVER] Successfully applied system limits for {config_str}:\n")
            log_file.write(f"[EVAL SERVER] - CPU cores: {config['cpu_cores']}\n")
            log_file.write(f"[EVAL SERVER] - Memory: {config['memory_gb']}GB\n")
            log_file.write(f"[EVAL SERVER] - Swap: {config['swap_memory_gb']}GB\n")
            
            return True
            
        except Exception as e:
            print(f"[EVAL SERVER] Error applying system limits: {e}")
            log_file.write(f"[EVAL SERVER] Error applying system limits: {e}\n")
            return False
    
    def get_cgroup_name(self, config_str: str) -> str:
        return f"rocksdb_training_{config_str}"

system_config_manager = SystemConfigManager()

def get_initial_options_file():
    try:
        from utils.constants import DEFAULT_OPTION_FILE_DIR, INITIAL_OPTIONS_FILE_NAME
        default_option_dir = DEFAULT_OPTION_FILE_DIR
        initial_file_name = INITIAL_OPTIONS_FILE_NAME
    except:
        default_option_dir = "options_files/default_options_files"
        initial_file_name = "dbbench_default_options-8.8.1.ini"
    
    initial_options_file_path = os.path.join(default_option_dir, initial_file_name)
    
    if not os.path.exists(initial_options_file_path):
        initial_options_file_path = os.path.join(current_dir, default_option_dir, initial_file_name)
    
    with open(initial_options_file_path, "r") as f:
        options = f.read()

    reasoning = f"Initial options file: {initial_options_file_path}"

    return options, reasoning

def rocksdb_eval(is_init: bool, new_options: str = None, previous_results: Dict = None, cgroup_name: str = None, workload: str = None) -> Dict:
    try:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        try:
            db_path = constants.DB_PATH
        except:
            db_path = os.path.join(DB_PATH_DIR, "db")
        
        output_folder_dir = OUTPUT_DIR
        options_files = []
        
        try:
            fio_result = get_fio_result(constants.FIO_RESULT_PATH)
        except:
            fio_result = {}
        
        log_update(f"[EVAL SERVER] Starting evaluation", OUTPUT_DIR)
        print(f"[EVAL SERVER] Starting evaluation")
        
        if is_init and new_options is None:
            options, reasoning = get_initial_options_file()
        else:
            options = cleanup_options_file(new_options)
            reasoning = "Evaluate the new options"
        
        is_error, benchmark_results, average_cpu_usage, average_memory_usage, options = spm.benchmark(
            db_path, options, output_folder_dir, reasoning, 0, previous_results, options_files, [], cgroup_name=cgroup_name, workload=workload
        )
        
        if is_error:
            log_update("[EVAL SERVER] Failed to benchmark with the options file.", OUTPUT_DIR)
            log_file.write(f"[EVAL SERVER] ERROR: {benchmark_results['error']}\n")
            print("[EVAL SERVER] Failed to benchmark with the options file.")
            print(f"[EVAL SERVER] ERROR: {benchmark_results['error']}")
            return {
                "error": str(benchmark_results['error']),
                "benchmark_results": {"ops_per_sec": 0},
                "average_cpu_usage": 0,
                "average_memory_usage": 0,
                "options": options,
                "db_path": db_path,
                "fio_result": fio_result,
                "trace_result": None
            }
        
        tracefile_path = db_path + "/tracefile"
        if os.path.exists(tracefile_path):
            trace_result = analyze_tracefile(tracefile_path)
            if trace_result is None:
                print("[EVAL SERVER] Tracefile analysis failed.")
                log_update("[EVAL SERVER] Tracefile analysis failed.", OUTPUT_DIR)
                trace_result = {}
        else:
            print(f"[EVAL SERVER] Tracefile not found at {tracefile_path}")
            log_update(f"[EVAL SERVER] Tracefile not found at {tracefile_path}", OUTPUT_DIR)
            trace_result = {}
        
        options_files.append((options, benchmark_results, reasoning, ""))
        
        return {
            "benchmark_results": benchmark_results,
            "average_cpu_usage": average_cpu_usage,
            "average_memory_usage": average_memory_usage,
            "options_files": options_files,
            "db_path": db_path,
            "fio_result": fio_result,
            "trace_result": trace_result
        }
        
    except Exception as e:
        print(f"[EVAL SERVER] Error in rocksdb_eval: {e}")
        log_file.write(f"[EVAL SERVER] Error in rocksdb_eval: {e}\n")
        return {
            "error": str(e),
            "benchmark_results": {"ops_per_sec": 0},
            "average_cpu_usage": 0,
            "average_memory_usage": 0,
            "options": "",
            "db_path": "",
            "fio_result": {},
            "trace_result": None
        }

def generate_prompt(benchmark_results: Dict, average_cpu_usage: float, average_memory_usage: float, 
                   options_files: List, db_path: str, fio_result: Dict, trace_result: Dict, cgroup_name: str) -> List[Dict[str, str]]:
    try:
        try:
            test_name = constants.TEST_NAME
            version = constants.VERSION
        except:
            test_name = "fillrandom"
            version = "8.8.1"
        
        message = generate_option_file_with_LLM(
            options_files,
            system_info(db_path, fio_result, cgroup_name),
            average_cpu_usage, 
            average_memory_usage,
            test_name, 
            version, 
            trace_result
        )
        return message
    except Exception as e:
        print(f"[EVAL SERVER] Error generating prompt: {e}")
        log_file.write(f"[EVAL SERVER] Error generating prompt: {e}\n")
        return []

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "ok", "message": "Eval server is running"})

@app.route('/apply_config', methods=['POST'])
def apply_config():
    try:
        data = request.json
        config = data.get('config')
        config_str = data.get('config_str')
        
        if not config or not config_str:
            return jsonify({"success": False, "error": "Missing config or config_str"}), 400
        
        success = system_config_manager.apply_system_limits(config, config_str)
        
        if success:
            global current_config_str
            current_config_str = config_str
            return jsonify({"success": True, "message": f"Config {config_str} applied successfully"})
        else:
            return jsonify({"success": False, "error": "Failed to apply system limits"}), 500
            
    except Exception as e:
        print(f"[EVAL SERVER] Error in apply_config: {e}")
        log_file.write(f"[EVAL SERVER] Error in apply_config: {e}\n")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/evaluate', methods=['POST'])
def evaluate():
    try:
        data = request.json
        config_str = data.get('config_str')
        new_options = data.get('new_options')
        is_init = data.get('is_init', False)
        workload = data.get('workload')
        
        if workload is None:
            try:
                workload = constants.TEST_NAME
            except:
                workload = "fillrandom"
        
        if not config_str:
            return jsonify({"error": "Missing config_str"}), 400
        
        cgroup_name = system_config_manager.get_cgroup_name(config_str)
        
        result = rocksdb_eval(is_init, new_options, None, cgroup_name, workload)
        
        if "error" not in result:
            try:
                prompt = generate_prompt(
                    result["benchmark_results"],
                    result["average_cpu_usage"],
                    result["average_memory_usage"],
                    result["options_files"],
                    result["db_path"],
                    result["fio_result"],
                    result["trace_result"],
                    cgroup_name
                )
                result["prompt"] = prompt
            except Exception as e:
                print(f"[EVAL SERVER] Error generating prompt: {e}")
                log_file.write(f"[EVAL SERVER] Error generating prompt: {e}\n")
                result["prompt"] = []
        
        return jsonify(result)
        
    except Exception as e:
        print(f"[EVAL SERVER] Error in evaluate: {e}")
        log_file.write(f"[EVAL SERVER] Error in evaluate: {e}\n")
        return jsonify({"error": str(e)}), 500

@app.route('/shutdown', methods=['POST'])
def shutdown():
    try:
        print("[EVAL SERVER] Shutting down...")
        log_file.write("[EVAL SERVER] Shutting down...\n")
        log_file.close()
        
        for config_str, manager in system_config_manager.cgroup_managers.items():
            try:
                manager.cleanup()
            except:
                pass
        
        func = request.environ.get('werkzeug.server.shutdown')
        if func is None:
            raise RuntimeError('Not running with the Werkzeug Server')
        func()
        
        return jsonify({"message": "Server shutting down..."})
        
    except Exception as e:
        print(f"[EVAL SERVER] Error in shutdown: {e}")
        return jsonify({"error": str(e)}), 500

def main():
    print(f"[EVAL SERVER] Starting eval server on {EVAL_SERVER_HOST}:{EVAL_SERVER_PORT}")
    log_file.write(f"[EVAL SERVER] Starting eval server on {EVAL_SERVER_HOST}:{EVAL_SERVER_PORT}\n")
    
    try:
        app.run(host=EVAL_SERVER_HOST, port=EVAL_SERVER_PORT, debug=False, threaded=True)
    except Exception as e:
        print(f"[EVAL SERVER] Error starting server: {e}")
        log_file.write(f"[EVAL SERVER] Error starting server: {e}\n")
    finally:
        log_file.close()

if __name__ == "__main__":
    main()

