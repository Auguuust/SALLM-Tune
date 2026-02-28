import os
import json
import getpass
from datetime import datetime
from collections import defaultdict
from deepdiff import DeepDiff

def log_update(update_string, output_path=None):
    current_datetime = datetime.now()
    date_time_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    update_string = f"[{date_time_string}] {update_string}"
    
    if output_path is None:
        try:
            from eval_server_config import OUTPUT_DIR
            output_path = OUTPUT_DIR
        except:
            output_path = "."
    
    log_file = os.path.join(output_path, "log.txt")
    with open(log_file, "a+") as f:
        f.write(update_string + "\n")

def log_llm_response(prompt, response, output_path=None):
    if output_path is None:
        try:
            from eval_server_config import OUTPUT_DIR
            output_path = OUTPUT_DIR
        except:
            output_path = "."
    
    llm_response_path = os.path.join(output_path, "llm_response")
    os.makedirs(llm_response_path, exist_ok=True)
    
    file_index = 1
    while os.path.exists(f"{llm_response_path}/response_{file_index}.txt"):
        file_index += 1
    
    file_path = f"{llm_response_path}/response_{file_index}.txt"
    
    with open(file_path, "w") as f:
        f.write("Prompt:\n")
        f.write(json.dumps(prompt, indent=4) + "\n\n")
        f.write("Response:\n")
        f.write(response + "\n")

def store_db_bench_output(output_folder_name, output_file_name,
                          benchmark_results, options_file, reasoning):
    with open(f"{output_folder_name}/{output_file_name}", "a+") as f:
        f.write("# " + json.dumps(benchmark_results) + "\n\n")
        f.write(options_file + "\n")
        for line in reasoning.splitlines():
            f.write("# " + line + "\n")

def store_best_option_file(options_files, output_folder_dir):
    best_result = max(options_files, key=lambda x: x[1]["ops_per_sec"])
    best_options = best_result[0]
    best_reasoning = best_result[2]
    with open(f"{output_folder_dir}/best_options.ini", "w") as f:
        f.write(best_options)
        for line in best_reasoning.splitlines():
            f.write("# " + line + "\n")

def store_diff_options_list(options_list, output_folder_dir):
    differences = calculate_differences(options_list)
    changed_fields_frequency = defaultdict(lambda: 0)

    with open(f"{output_folder_dir}/diffOptions.txt", 'w') as f:
        for i, diff in enumerate(differences, start=1):
            f.write(f"[MFN] Differences between iteration {i} and iteration {i + 1}: \n")
            f.write(json.dumps(diff, indent=4))
            f.write("\n")
            f.write("=" * 50)
            f.write("\n\n")

            for key in diff["values_changed"]:
                changed_fields_frequency[key] += 1

        f.write("\n\n[MFN] Changed Fields Frequency:\n")
        f.write(json.dumps(changed_fields_frequency, indent=4))

def path_of_db(db_path=None):
    if db_path is None:
        try:
            from eval_server_config import DB_PATH_DIR
            db_path = DB_PATH_DIR
        except:
            db_path = "./eval_dbpath"
    
    if not os.path.exists(db_path):
        os.makedirs(db_path, exist_ok=True)
        log_update(f"[UTL] Created database path: {db_path}")
    
    user_name = getpass.getuser()
    db_path_name = os.path.join(db_path, user_name[0].lower())
    db_path_final = os.getenv("DB_PATH", db_path_name)
    print(f"[UTL] Using database path: {db_path_final}")

    return db_path_final

def path_of_output_folder(output_path=None):
    if output_path is None:
        try:
            from eval_server_config import OUTPUT_DIR
            output_path = OUTPUT_DIR
        except:
            output_path = "./eval_output"
    
    current_datetime = datetime.now()
    date_time_string = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    output_folder_dir = os.path.join(output_path, f"output_{date_time_string}")

    os.makedirs(output_folder_dir, exist_ok=True)
    log_update(f"[UTL] Using output folder: {output_folder_dir}")
    print(f"[UTL] Using output folder: {output_folder_dir}")

    return output_folder_dir

def calculate_differences(iterations):
    differences = []
    for i in range(1, len(iterations)):
        diff = DeepDiff(iterations[i-1], iterations[i])
        differences.append(diff)
    return differences

