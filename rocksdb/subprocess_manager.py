import subprocess
import os
import time
from cgroup_monitor import CGroupMonitor, CGroupManager

from llm.content_generator import error_correction_options_file_generation
from utils.utils import log_update, path_of_db
from utils.constants import ERROR_CORRECTION_COUNT, TEST_NAME, DB_BENCH_PATH, OPTIONS_FILE_DIR, NUM_ENTRIES, DURATION, SIDE_CHECKER, FIO_RESULT_PATH, DYNAMIC_OPTION_TUNING, ENABLE_MIDWAY_MONITORING, LIMIT_LIST
from utils.constants import SINE_WRITE_RATE_INTERVAL_MILLISECONDS, SINE_A, SINE_B, SINE_C, SINE_D, OUTPUT_PATH, PRE_LOAD_CMD, NUM_THREADS, PRE_LOAD_DB_PATH
from rocksdb.parse_db_bench_output import parse_db_bench_output
from utils.utils import store_db_bench_output
from utils.graph import plot_2axis
from utils.mmap_utils import add_mmap_file_to_option, create_mmap_file, write_to_mmap_file
from llm.prompts_generator import midway_options_file_generation, dynamic_options_file_generation
from utils.system_operations.fio_runner import get_fio_result
from utils.system_operations.get_sys_info import system_info
from trace_analyzer.analyzer import analyze_tracefile, analyze_last_n_tracefile_windows


def detect_cgroup_version():
    if os.path.exists("/sys/fs/cgroup/cgroup.controllers"):
        return 2
    elif os.path.exists("/sys/fs/cgroup/cpu/") and os.path.exists("/sys/fs/cgroup/memory/"):
        return 1
    else:
        raise RuntimeError("Cannot detect cgroup v1 or v2 environment")


class FixedCGroupManager(CGroupManager):
    def __init__(self, group_name, helper_script=None):
        super().__init__(group_name, helper_script)
        self.cgroup_version = detect_cgroup_version()
        self.group_name = group_name
        if self.cgroup_version == 1:
            self.cpu_path = f"/sys/fs/cgroup/cpu/{group_name}"
            self.mem_path = f"/sys/fs/cgroup/memory/{group_name}"
        else:
            self.v2_path = f"/sys/fs/cgroup/{group_name}"

    def add_process(self, pid, sudo=True):
        try:
            if self.cgroup_version == 1:
                for path in [self.cpu_path, self.mem_path]:
                    procs_file = f"{path}/cgroup.procs"
                    if self.helper_script is None:
                        cmd = ["sudo", "sh", "-c", f"echo {pid} > {procs_file}"]
                        proc = subprocess.run(cmd, check=True)
                    else:
                        cmd = ["sudo", self.helper_script, "write", procs_file, str(pid)]
                        proc = subprocess.run(cmd, check=True)
            else:
                procs_file = f"{self.v2_path}/cgroup.procs"
                if self.helper_script is None:
                    cmd = ["sudo", "sh", "-c", f"echo {pid} > {procs_file}"]
                    proc = subprocess.run(cmd, check=True)
                else:
                    cmd = ["sudo", self.helper_script, "write", procs_file, str(pid)]
                    proc = subprocess.run(cmd, check=True)
            print(f"[FixedCGroupManager] Process {pid} added to cgroup")
            return True
        except Exception as e:
            print(f"[FixedCGroupManager] Error adding process to cgroup: {e}")
            return False

    def create_cgroup(self):
        if self.cgroup_version == 1:
            for path in [self.cpu_path, self.mem_path]:
                if not os.path.exists(path):
                    if self.helper_script is None:
                        proc = subprocess.run(["sudo", "mkdir", "-p", path], check=True)
                    else:
                        proc = subprocess.run(["sudo", self.helper_script, "create", path], check=True)
                    if proc.returncode != 0:
                        raise Exception(f"Failed to create cgroup: {path}")
            for path in [self.cpu_path, self.mem_path]:
                if self.helper_script is None:
                    proc = subprocess.run(["sudo", "chown", "-R", f"{os.getlogin()}:{os.getlogin()}", path], check=True)
                else:
                    proc = subprocess.run(["sudo", self.helper_script, "chown", path], check=True)
                if proc.returncode != 0:
                    raise Exception(f"Failed to take ownership of cgroup: {path}")
        else:
            if not os.path.exists(self.v2_path):
                proc = subprocess.run(["sudo", "mkdir", "-p", self.v2_path], check=True)
                if proc.returncode != 0:
                    raise Exception(f"Failed to create cgroup v2: {self.v2_path}")
            
            try:
                root_subtree_control = "/sys/fs/cgroup/cgroup.subtree_control"
                cmd = ["sudo", "sh", "-c", f"echo '+cpu +memory' > {root_subtree_control}"]
                subprocess.run(cmd, check=False)
                
                if self.group_name != "":
                    parent_path = os.path.dirname(self.v2_path)
                    if parent_path != "/sys/fs/cgroup":
                        parent_subtree_control = f"{parent_path}/cgroup.subtree_control"
                        cmd = ["sudo", "sh", "-c", f"echo '+cpu +memory' > {parent_subtree_control}"]
                        subprocess.run(cmd, check=False)
            except Exception as e:
                print(f"[FixedCGroupManager] Warning enabling controllers: {e}")
            
            proc = subprocess.run(["sudo", "chown", "-R", f"{os.getlogin()}:{os.getlogin()}", self.v2_path], check=True)
            if proc.returncode != 0:
                raise Exception(f"Failed to take ownership of cgroup v2: {self.v2_path}")
        return 0

    def set_cpu_limit(self, num_cpus, sudo=True):
        try:
            if self.cgroup_version == 1:
                quota = num_cpus * 100000
                if self.helper_script is None:
                    cmd = ["sudo", "sh", "-c", f"echo {quota} > {self.cpu_path}/cpu.cfs_quota_us"]
                    proc = subprocess.run(cmd, check=False)
                    cmd = ["sudo", "sh", "-c", f"echo 100000 > {self.cpu_path}/cpu.cfs_period_us"]
                    proc = subprocess.run(cmd, check=False)
                else:
                    cmd = ["sudo", self.helper_script, "write", f"{self.cpu_path}/cpu.cfs_quota_us", f"{quota}"]
                    proc = subprocess.run(cmd, check=False)
                    cmd = ["sudo", self.helper_script, "write", f"{self.cpu_path}/cpu.cfs_period_us", "100000"]
                    proc = subprocess.run(cmd, check=False)
            else:
                quota = num_cpus * 100000
                max_str = f"{quota} 100000"
                cmd = ["sudo", "sh", "-c", f"echo '{max_str}' > {self.v2_path}/cpu.max"]
                proc = subprocess.run(cmd, check=False)
            print(f"[FixedCGroupManager] Trying to set CPU limit to {num_cpus} CPUs")
            return True
        except Exception as e:
            print(f"[FixedCGroupManager] Error setting CPU limit: {e}")
            return False

    def set_memory_limit(self, memory_in_bytes, sudo=True):
        try:
            if self.cgroup_version == 1:
                if self.helper_script is None:
                    cmd = ["sudo", "sh", "-c", f"echo {memory_in_bytes} > {self.mem_path}/memory.limit_in_bytes"]
                    proc = subprocess.run(cmd, check=False)
                else:
                    cmd = ["sudo", self.helper_script, "write", f"{self.mem_path}/memory.limit_in_bytes", f"{memory_in_bytes}"]
                    proc = subprocess.run(cmd, check=False)
            else:
                cmd = ["sudo", "sh", "-c", f"echo {memory_in_bytes} > {self.v2_path}/memory.max"]
                proc = subprocess.run(cmd, check=False)
            print(f"[FixedCGroupManager] Trying to set memory limit to {memory_in_bytes / 1024 / 1024 / 1024:.2f}GB")
            return True
        except Exception as e:
            print(f"[FixedCGroupManager] Error setting memory limit: {e}")
            return False

    def set_memory_swap_limit(self, memory_in_bytes, sudo=True):
        try:
            if self.cgroup_version == 1:
                if self.helper_script is None:
                    cmd = ["sudo", "sh", "-c", f"echo {memory_in_bytes} > {self.mem_path}/memory.memsw.limit_in_bytes"]
                    proc = subprocess.run(cmd, check=False)
                else:
                    cmd = ["sudo", self.helper_script, "write", f"{self.mem_path}/memory.memsw.limit_in_bytes", f"{memory_in_bytes}"]
                    proc = subprocess.run(cmd, check=False)
            else:
                cmd = ["sudo", "sh", "-c", f"echo {memory_in_bytes} > {self.v2_path}/memory.swap.max"]
                proc = subprocess.run(cmd, check=False)
            print(f"[FixedCGroupManager] Trying to set swap limit to {memory_in_bytes / 1024 / 1024 / 1024:.2f}GB")
            return True
        except Exception as e:
            print(f"[FixedCGroupManager] Error setting swap limit: {e}")
            return False

def pre_tasks(database_path, run_count):

    proc = subprocess.run(
        f'rm -rf {database_path}',
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True,
        check=False
    )

    log_update("[SPM] Flushing the cache")
    print("[SPM] Flushing the cache")

    proc = subprocess.run(
        f'sync; echo 3 > /proc/sys/vm/drop_caches',
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True,
        check=False
    )

    print("[SPM] Waiting for 30 seconds to free up memory, IO and other resources")
    time.sleep(30)


def generate_db_bench_command(db_bench_path, database_path, options, run_count, test_name, db_bench_extra_args=[]):
    db_bench_command = [
        db_bench_path,
        f"--db={database_path}",
        f"--options_file={OPTIONS_FILE_DIR}",
        "--use_direct_io_for_flush_and_compaction",
        "--use_direct_reads", "--compression_type=none",
        "--stats_interval_seconds=1", "--histogram", "--statistics",
        f"--dynamic_options_file=/tmp/mmap_file.mmap" if DYNAMIC_OPTION_TUNING else "",
        f"--threads={NUM_THREADS}", f"--trace_file={database_path}/tracefile",
        f"--num={NUM_ENTRIES}", f"--duration={DURATION}"
    ]

    if test_name == "readrandom" or test_name == "mixgraph" or test_name == "tracefile":
        if PRE_LOAD_DB_PATH != "":
            log_update("[SPM] Running Pre-load command")
            print("[SPM] Running Pre-load command")
            tmp_runner_rm = ["rm", "-rf", database_path]
            tmp_proc_rm = subprocess.run(tmp_runner_rm, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
            tmp_runner = ["cp", "-r", PRE_LOAD_DB_PATH, database_path]
            tmp_proc = subprocess.run(tmp_runner, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)

    if test_name == "fillrandom":
        db_bench_command.append("--benchmarks=fillrandom")
    elif test_name == "readrandomwriterandom":
        db_bench_command.append("--benchmarks=readrandomwriterandom")
    elif test_name == "readrandom":
        if PRE_LOAD_DB_PATH == "":
            log_update("[SPM] Running fillrandom to load the database")
            print("[SPM] Running fillrandom to load the database")
            tmp_runner = db_bench_command[:-3] + ["--num=50000000", "--benchmarks=fillrandom", "--max_background_jobs=8"]
            tmp_proc = subprocess.run(tmp_runner, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        new_db_bench = db_bench_command + ["--benchmarks=readrandom", "--use_existing_db", "--reads=5000000"]
        db_bench_command = new_db_bench
    elif test_name == "mixgraph":
        if PRE_LOAD_DB_PATH == "":
            log_update("[SPM] Running fillrandom to load the database")
            print("[SPM] Running fillrandom to load the database")
            tmp_runner = db_bench_command[:-3] + ["--num=50000000", "--benchmarks=fillrandom", "--key_size=48", "--value_size=43"]
            # tmp_runner = db_bench_command[:-3] + ["--num=500", "--benchmarks=fillrandom", "--key_size=48", "--value_size=43"]
            tmp_proc = subprocess.run(tmp_runner, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        new_db_bench = db_bench_command[:-1] + ["--benchmarks=mixgraph", "--use_existing_db", f"--duration={DURATION}", 
                                                "--mix_get_ratio=0.83", "--mix_put_ratio=0.14", "--mix_seek_ratio=0.03", "--key_size=48",
                                                f"--sine_write_rate_interval_milliseconds={SINE_WRITE_RATE_INTERVAL_MILLISECONDS}", "--sine_mix_rate", 
                                                f"--sine_a={SINE_A}", f"--sine_b={SINE_B}", f"--sine_c={SINE_C}", f"--sine_d={SINE_D}"]
        db_bench_command = new_db_bench
    elif test_name == "readwhilewriting":
        db_bench_command.append("--benchmarks=readwhilewriting")
    elif test_name == "sinetest":
        db_bench_command += [
            "--benchmarks=fillrandom", "--sine_write_rate=true",
            f"--sine_write_rate_interval_milliseconds={SINE_WRITE_RATE_INTERVAL_MILLISECONDS}",
            f"--sine_a={SINE_A}", f"--sine_b={SINE_B}", f"--sine_c={SINE_C}", f"--sine_d={SINE_D}",
        ]
    elif test_name == "jsonconfigured":
        db_bench_command += [
            "--benchmarks=jsonconfigured", 
            f"--json_file_path={os.path.join(os.path.dirname(__file__), '../benchy.json')}"
        ]
    elif test_name == "tracefile":
        if PRE_LOAD_CMD != "" and PRE_LOAD_DB_PATH == "":
            tmp_runner = PRE_LOAD_CMD.split(" ")
            tmp_proc = subprocess.run(tmp_runner, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        db_bench_command[:-2] += [
            "--benchmarks=jsonconfigured", "--use_existing_db",
            f"--json_file_path={os.path.join(OUTPUT_PATH, 'trace_model.json')}"
        ]
    else:
        print(f"[SPM] Test name {test_name} not recognized")
        exit(1)

    db_bench_command += db_bench_extra_args

    log_update(f"[SPM] Command: {db_bench_command}")
    return db_bench_command


def db_bench(db_bench_path, database_path, options, run_count, test_name, previous_throughput, options_files, db_bench_args=[], bm_iter=0):
    global proc_out
    with open(f"{OPTIONS_FILE_DIR}", "w") as f:
        f.write(options)

    pre_tasks(database_path, run_count)
    command = generate_db_bench_command(db_bench_path, database_path, options, run_count, test_name, db_bench_args)

    if DYNAMIC_OPTION_TUNING:
        create_mmap_file()

    log_update(f"[SPM] Executing db_bench with command: {command}")
    print("[SPM] Executing db_bench")


    if SIDE_CHECKER and previous_throughput != None:
        cgm = FixedCGroupManager("llm_cgroup", helper_script=os.path.abspath("utils/root_cgroup_helper.sh"))
        cgroup_monitor = CGroupMonitor("llm_cgroup")
        
        if DYNAMIC_OPTION_TUNING:
            saved_optionfile = options_files[-1][0]
            cur_options_file = []

        start_time = time.time()
        cgroup_monitor.start_monitor()

        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True) as proc_out:
            cgm.add_process(proc_out.pid, sudo=True)

            output = ""
            first_check_interval = 100
            first_check_flag = False

            check_interval = 90

            for line in proc_out.stdout:
                output += line
                elapsed_time = time.time() - start_time

                if first_check_flag == False:
                    if (elapsed_time <= first_check_interval):
                        continue
                    else:
                        first_check_flag = True

                if elapsed_time <= check_interval:
                    continue

                if "ops/second" in line:
                    current_avg_throughput = (float(line.split("(")[2].split(",")[1].split(")")[0]))*NUM_THREADS

                    if ENABLE_MIDWAY_MONITORING and (current_avg_throughput < .9 * float(previous_throughput)) and (bm_iter < 3):
                        print("[SQU] Throughput decreased, resetting the benchmark")
                        log_update(f"[SQU] Throughput decreased {previous_throughput}->{current_avg_throughput}, resetting the benchmark")

                        op = cgroup_monitor.stop_monitor()
                        avg_cpu_used = op["average_cpu_usage_percent"]
                        avg_mem_used = op["average_memory_usage_percent"]

                        proc_out.kill()

                        db_path = path_of_db()
                        fio_result = get_fio_result(FIO_RESULT_PATH)
                        device_info = system_info(db_path, fio_result)
                        trace_result = analyze_tracefile(db_path + "/tracefile")

                        new_options, db_bench_args, _, _ = midway_options_file_generation(options, db_bench_args, avg_cpu_used, avg_mem_used, current_avg_throughput, device_info, trace_result, options_files)
                        output, avg_cpu_used, avg_mem_used, options = db_bench(db_bench_path, database_path, new_options, run_count, test_name, previous_throughput, options_files, db_bench_args, bm_iter+1)

                        log_update("[SPM] Finished running db_bench")
                        return output, avg_cpu_used, avg_mem_used, options

                    if DYNAMIC_OPTION_TUNING and current_avg_throughput < 0.6 * float(previous_throughput):
                        print("[SQU] Dynamic option tuning is enabled and now running")
                        log_update("[SQU] Dynamic option tuning is enabled and now running")

                        db_path = path_of_db()
                        fio_result = get_fio_result(FIO_RESULT_PATH)
                        device_info = system_info(db_path, fio_result)

                        op = cgroup_monitor.get_last_n_stats(check_interval)
                        avg_cpu_used = op["average_cpu_usage_percent"]
                        avg_mem_used = op["average_memory_usage_percent"]

                        trace_result = analyze_last_n_tracefile_windows(db_path + "/tracefile", check_interval//10)

                        cur_options_file.append([
                            saved_optionfile,
                            {"ops_per_sec": current_avg_throughput}
                        ])

                        new_options, _, _, _ = dynamic_options_file_generation(None, db_bench_args, avg_cpu_used, avg_mem_used, None, device_info, trace_result, cur_options_file)

                        saved_optionfile = new_options

                        write_to_mmap_file(new_options)
                else:
                    print("[SQU] No throughput found in the output")
                    log_update("[SQU] No throughput found in the output")

                start_time = time.time()

        print("[SPM] Finished running db_bench")
        print("----------------------------------------------------------------------------")

        op = cgroup_monitor.stop_monitor()
        avg_cpu_used = op["average_cpu_usage_percent"]
        avg_mem_used = op["average_memory_usage_percent"]

        if DYNAMIC_OPTION_TUNING:
            options = add_mmap_file_to_option(options, saved_optionfile)

        return output, avg_cpu_used, avg_mem_used, options
    
    else:
        if LIMIT_LIST:
            cpu_limit = int(LIMIT_LIST[0])
            mem_limit = int(LIMIT_LIST[1])
            swap_limit = int(LIMIT_LIST[2])

        cgm = FixedCGroupManager("llm_cgroup", helper_script=os.path.abspath("utils/root_cgroup_helper.sh"))
        cgm.create_cgroup()
        cgm.set_cpu_limit(cpu_limit)
        cgm.set_memory_limit(mem_limit*1024*1024*1024)
        cgm.set_memory_swap_limit(swap_limit*1024*1024*1024)

        cgroup_monitor = CGroupMonitor("llm_cgroup")
        cgroup_monitor.start_monitor()

        proc_out = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        cgm.add_process(proc_out.pid, sudo=True)
        stdout, stderr = proc_out.communicate()

        op = cgroup_monitor.stop_monitor()
        avg_cpu_used = op["average_cpu_usage_percent"]
        avg_mem_used = op["average_memory_usage_percent"]

        print("[SPM] Finished running db_bench")
        print("---------------------------------------------------------------------------")
        
        return stdout, avg_cpu_used, avg_mem_used, options


def benchmark(db_path, options, output_file_dir, reasoning, changed_value_dict, iteration_count, previous_results, options_files, db_bench_args, bm_iter=0):
    if previous_results is None:
        output, average_cpu_usage, average_memory_usage, options = db_bench(
            DB_BENCH_PATH, db_path, options, iteration_count, TEST_NAME, None, options_files, db_bench_args)
    else:
        output, average_cpu_usage, average_memory_usage, options = db_bench(
            DB_BENCH_PATH, db_path, options, iteration_count, TEST_NAME, previous_results['ops_per_sec'], options_files, db_bench_args)

    benchmark_results = parse_db_bench_output(output)

    contents = os.listdir(output_file_dir)
    ini_file_count = len([f for f in contents if f.endswith(".ini")])

    if benchmark_results.get("error") is not None:
        is_error = True
        log_update(f"[SPM] Benchmark failed, the error is: {benchmark_results.get('error')}")
        print("[SPM] Benchmark failed, the error is: ",
              benchmark_results.get("error"))
        store_db_bench_output(output_file_dir,
                              f"{ini_file_count}-incorrect_options.ini",
                              benchmark_results, options, reasoning, changed_value_dict)
        with open(f"{OPTIONS_FILE_DIR}", "w") as f:
            f.write(options_files[-1][0])

        if bm_iter < ERROR_CORRECTION_COUNT:
            print(f"[SPM] Retrying the benchmark with error correction {bm_iter+1}/{ERROR_CORRECTION_COUNT}")
            log_update(f"[SPM] Retrying the benchmark with error correction {bm_iter+1}/{ERROR_CORRECTION_COUNT}")
            new_options, db_bench_args, reasoning, changed_value_dict = error_correction_options_file_generation(options, db_bench_args, reasoning, changed_value_dict, benchmark_results.get('error'), bm_iter)
            return benchmark(db_path, new_options, output_file_dir, reasoning, changed_value_dict, iteration_count, previous_results, options_files, db_bench_args, bm_iter+1)

    elif benchmark_results['data_speed'] is None:
        is_error = True
        log_update(f"[SPM] Benchmark failed, the error is: Data speed is None. Check DB save path")
        print("[SPM] Benchmark failed, the error is: ",
              "Data speed is None. Check DB save path")
        store_db_bench_output(output_file_dir,
                              f"{ini_file_count}-incorrect_options.ini",
                              benchmark_results, options, reasoning, changed_value_dict)
        with open(f"{OPTIONS_FILE_DIR}", "w") as f:
            f.write(options_files[-1][0])

        if bm_iter < ERROR_CORRECTION_COUNT:
            print(f"[SPM] Retrying the benchmark with error correction {bm_iter+1}/{ERROR_CORRECTION_COUNT}")
            log_update(f"[SPM] Retrying the benchmark with error correction {bm_iter+1}/{ERROR_CORRECTION_COUNT}")
            new_options, db_bench_args, reasoning, changed_value_dict = error_correction_options_file_generation(options, db_bench_args, reasoning, changed_value_dict, benchmark_results.get('error'), bm_iter)
            return benchmark(db_path, new_options, output_file_dir, reasoning, changed_value_dict, iteration_count, previous_results, options_files, db_bench_args, bm_iter+1)

    else:
        is_error = False

        store_db_bench_output(output_file_dir, f"{ini_file_count}.ini",
                              benchmark_results, options, reasoning, changed_value_dict)
        plot_2axis(*benchmark_results["ops_per_second_graph"],
                   f"Ops Per Second - {benchmark_results['ops_per_sec']}",
                   f"{output_file_dir}/ops_per_sec_{ini_file_count}.png")
        log_update(f"[SPM] Latest result: {benchmark_results['data_speed']}"
                        f"{benchmark_results['data_speed_unit']} and {benchmark_results['ops_per_sec']} ops/sec.")
        log_update(f"[SPM] Avg CPU and Memory usage: {average_cpu_usage}% and {average_memory_usage}%")
        print(
            f"[SPM] Latest result: {benchmark_results['data_speed']}",
            f"{benchmark_results['data_speed_unit']} and {benchmark_results['ops_per_sec']} ops/sec.",
            f"\n[SPM] Avg CPU and Memory usage: {average_cpu_usage}% and {average_memory_usage}%"
        )

    return is_error, benchmark_results, average_cpu_usage, average_memory_usage, options
