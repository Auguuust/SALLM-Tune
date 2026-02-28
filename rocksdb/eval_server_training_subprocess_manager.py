import subprocess
import os
import time
import json
import re

from utils.constants import (
    TEST_NAME, 
    NUM_ENTRIES, 
    DURATION, 
    NUM_THREADS, 
    SINE_WRITE_RATE_INTERVAL_MILLISECONDS,
    SINE_A, 
    SINE_B, 
    SINE_C, 
    SINE_D, 
    OUTPUT_PATH,
    PRE_LOAD_CMD,
    PRE_LOAD_DB_PATH,
    DB_BENCH_PATH,
    SIDE_CHECKER,
    DYNAMIC_OPTION_TUNING
)

from eval_server_utility_functions import log_update, path_of_db
from rocksdb.parse_db_bench_output import parse_db_bench_output


def detect_cgroup_version():
    if os.path.exists("/sys/fs/cgroup/cgroup.controllers"):
        return 2
    elif (os.path.exists("/sys/fs/cgroup/cpu/") or 
          os.path.exists("/sys/fs/cgroup/cpu,cpuacct/") or 
          os.path.exists("/sys/fs/cgroup/cpuacct/")) and os.path.exists("/sys/fs/cgroup/memory/"):
        return 1
    else:
        raise RuntimeError("Unable to detect cgroup v1 or v2 environment")


class CGroupMonitor:
    import time
    import threading
    import os
    import psutil
    
    def __init__(self, group_name="", helper_script=None):
        super().__init__()
        self._stop_event = self.threading.Event()
        self.monitor_start_time = 0
        self.monitor_end_time = 0
        self.cpu_usage_percent_list = []
        self.memory_usage_percent_list = []
        self.monitor_thread = None
        
        self.cgroup_version = detect_cgroup_version()
        self.group_name = group_name
        self.helper_script = helper_script
        
        if self.cgroup_version == 1:
            cpu_controller_dir = None
            if os.path.exists("/sys/fs/cgroup/cpu,cpuacct/"):
                cpu_controller_dir = "cpu,cpuacct"
            elif os.path.exists("/sys/fs/cgroup/cpu/"):
                cpu_controller_dir = "cpu"
            elif os.path.exists("/sys/fs/cgroup/cpuacct/"):
                cpu_controller_dir = "cpuacct"
            else:
                raise RuntimeError("Unable to find cgroup v1 CPU controller directory")
            
            if group_name:
                self.cpu_path = f"/sys/fs/cgroup/{cpu_controller_dir}/{group_name}"
                self.mem_path = f"/sys/fs/cgroup/memory/{group_name}"
            else:
                self.cpu_path = f"/sys/fs/cgroup/{cpu_controller_dir}"
                self.mem_path = "/sys/fs/cgroup/memory"
            self.CPU_STAT_PATH = f'{self.cpu_path}/cpuacct.usage'
            self.CPU_MAX_PATH = f'{self.cpu_path}/cpu.cfs_quota_us'
            self.MEMORY_CURRENT_PATH = f'{self.mem_path}/memory.usage_in_bytes'
            self.MEMORY_MAX_PATH = f'{self.mem_path}/memory.limit_in_bytes'
        else:
            if group_name:
                self.v2_path = f"/sys/fs/cgroup/{group_name}"
            else:
                self.v2_path = "/sys/fs/cgroup"
            self.CPU_STAT_PATH = f'{self.v2_path}/cpu.stat'
            self.CPU_MAX_PATH = f'{self.v2_path}/cpu.max'
            self.MEMORY_CURRENT_PATH = f'{self.v2_path}/memory.current'
            self.MEMORY_MAX_PATH = f'{self.v2_path}/memory.max'
        
        self.cpu_limit = self.get_cpu_limit()
        self.memory_limit = self.get_memory_limit()

    def get_cpu_usage(self, interval=0.1):
        if self.cgroup_version == 1:
            try:
                with open(self.CPU_STAT_PATH, 'r') as f:
                    start_usage = int(f.read().strip())
                start_time = self.time.time_ns()

                self.time.sleep(interval)

                with open(self.CPU_STAT_PATH, 'r') as f:
                    end_usage = int(f.read().strip())
                end_time = self.time.time_ns()

                usage_delta = end_usage - start_usage  # nanoseconds
                time_delta = (end_time - start_time) / 1e9  # convert to seconds
                cpu_usage = (usage_delta / 1e9) / time_delta  # usage time ratio
                return cpu_usage
            except:
                return 0
        else:
            try:
                with open(self.CPU_STAT_PATH, 'r') as f:
                    start_stat = f.readlines()
                start_usage = self.parse_usage_usec(start_stat)
                start_time = self.time.time_ns()

                self.time.sleep(interval)

                with open(self.CPU_STAT_PATH, 'r') as f:
                    end_stat = f.readlines()
                end_usage = self.parse_usage_usec(end_stat)
                end_time = self.time.time_ns()

                usage_delta = end_usage - start_usage
                time_delta = (end_time - start_time) / 1e9  # convert ns to s
                cpu_usage = usage_delta / (time_delta * 1e6)  # usage_usec to s
                return cpu_usage
            except:
                return 0

    def parse_usage_usec(self, lines):
        usage_value = 0
        for line in lines:
            if 'usage_usec' in line:
                usage_value = int(line.split()[1])
                break
        return usage_value

    def get_cpu_usage_percent(self, interval=1, decimal_places=2):
        cpu_usage = self.get_cpu_usage(interval)
        percent = cpu_usage / self.cpu_limit * 100 if self.cpu_limit else 0
        return round(percent, decimal_places)

    def get_cpu_limit(self):
        if self.cgroup_version == 1:
            try:
                with open(self.CPU_MAX_PATH, 'r') as f:
                    quota = int(f.read().strip())
                if quota == -1:
                    return os.cpu_count()  # unlimited
                else:
                    # Read period
                    period_path = f'{self.cpu_path}/cpu.cfs_period_us'
                    with open(period_path, 'r') as f:
                        period = int(f.read().strip())
                    return quota / period  # number of CPUs
            except:
                return os.cpu_count()
        else:
            try:
                with open(self.CPU_MAX_PATH, 'r') as f:
                    data = f.read().strip().split()
                # data might look like 'max 100000' or '200000 100000'
                if data[0] == 'max':
                    return os.cpu_count()  # unlimited
                else:
                    quota = int(data[0])
                    period = int(data[1])
                    return quota / period  # number of CPUs
            except:
                return os.cpu_count()

    def get_memory_usage(self):
        try:
            with open(self.MEMORY_CURRENT_PATH, 'r') as f:
                return int(f.read().strip())
        except:
            return 0

    def get_memory_limit(self):
        try:
            with open(self.MEMORY_MAX_PATH, 'r') as f:
                data = f.read().strip()
            if self.cgroup_version == 1:
                limit = int(data)
                import psutil
                if limit > psutil.virtual_memory().total:
                    return psutil.virtual_memory().total
                return limit
            else:
                if data == 'max':
                    import psutil
                    return psutil.virtual_memory().total
                return int(data)
        except:
            import psutil
            return psutil.virtual_memory().total

    def get_memory_usage_percent(self, decimal_places=2):
        usage = self.get_memory_usage()
        return round((usage / self.memory_limit) * 100 if self.memory_limit else 0, decimal_places)

    def start_monitor(self):
        self.monitor_start_time = self.time.time_ns()
        print('[CGM] Monitor started')
        print('[CGM] CPU Limit:', self.cpu_limit)
        print('[CGM] Memory Limit:', self.memory_limit)
        self.monitor_thread = self.threading.Thread(target=self.constant_monitoring)
        self.monitor_thread.start()

    def stop_monitor(self):
        self._stop_event.set()
        if self.monitor_thread:
            self.monitor_thread.join()
        self.monitor_end_time = self.time.time_ns()

        if self.cpu_usage_percent_list and self.memory_usage_percent_list:
            avg_cpu_usage = sum(self.cpu_usage_percent_list) / len(self.cpu_usage_percent_list)
            avg_memory_usage = sum(self.memory_usage_percent_list) / len(self.memory_usage_percent_list)
        else:
            avg_cpu_usage = 1
            avg_memory_usage = 1
        
        # Return format needs to be compatible with external packages
        return {
            "average_cpu_usage_percent": round(avg_cpu_usage, 2),
            "average_memory_usage_percent": round(avg_memory_usage, 2)
        }

    def constant_monitoring(self):
        self.monitor_start_time = self.time.time_ns()

        while not self._stop_event.is_set():
            if (self.time.time_ns() - self.monitor_start_time) > 1000000000:
                self.monitor_start_time = self.time.time_ns()
                self.cpu_usage_percent_list.append(self.get_cpu_usage_percent())
                self.memory_usage_percent_list.append(self.get_memory_usage_percent())


class FixedCGroupManager:
    def __init__(self, group_name, helper_script=None):
        self.helper_script = helper_script
        self.cgroup_version = detect_cgroup_version()
        self.group_name = group_name
        if self.cgroup_version == 1:
            cpu_controller_dir = None
            if os.path.exists("/sys/fs/cgroup/cpu,cpuacct/"):
                cpu_controller_dir = "cpu,cpuacct"
            elif os.path.exists("/sys/fs/cgroup/cpu/"):
                cpu_controller_dir = "cpu"
            elif os.path.exists("/sys/fs/cgroup/cpuacct/"):
                cpu_controller_dir = "cpuacct"
            else:
                raise RuntimeError("Unable to find cgroup v1 CPU controller directory")
            
            self.cpu_path = f"/sys/fs/cgroup/{cpu_controller_dir}/{group_name}"
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
            print(f"[FixedCGroupManager] Added process {pid} to cgroup")
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
            print(f"[FixedCGroupManager] Creating cgroup v2: {self.v2_path}")
            if not os.path.exists(self.v2_path):
                proc = subprocess.run(["sudo", "mkdir", "-p", self.v2_path], check=True)
                if proc.returncode != 0:
                    raise Exception(f"Failed to create cgroup v2: {self.v2_path}")
            
            # v2: enable cpu and memory controllers - use more robust method
            try:
                # First check if root cgroup already has controllers enabled
                root_controllers_path = "/sys/fs/cgroup/cgroup.controllers"
                with open(root_controllers_path, 'r') as f:
                    available_controllers = f.read().strip().split()
                print(f"[FixedCGroupManager] Available controllers: {available_controllers}")
                
                # Enable controllers in root cgroup
                root_subtree_control = "/sys/fs/cgroup/cgroup.subtree_control"
                controllers_to_enable = []
                if "cpu" in available_controllers:
                    controllers_to_enable.append("+cpu")
                if "memory" in available_controllers:
                    controllers_to_enable.append("+memory")
                
                if controllers_to_enable:
                    controller_string = " ".join(controllers_to_enable)
                    print(f"[FixedCGroupManager] Enabling controllers in root: {controller_string}")
                    cmd = ["sudo", "tee", root_subtree_control]
                    proc = subprocess.run(cmd, input=controller_string.encode(), check=False)
                    print(f"[FixedCGroupManager] Root controller enable result: {proc.returncode}")
                
                # Enable controllers hierarchically to target cgroup
                path_parts = self.group_name.split('/')
                current_path = "/sys/fs/cgroup"
                
                for part in path_parts:
                    if part:  # skip empty strings
                        parent_path = current_path
                        current_path = os.path.join(current_path, part)
                        
                        # Enable controllers in parent
                        parent_subtree_control = os.path.join(parent_path, "cgroup.subtree_control")
                        if os.path.exists(parent_subtree_control) and controllers_to_enable:
                            print(f"[FixedCGroupManager] Enabling controllers in {parent_path}")
                            cmd = ["sudo", "tee", parent_subtree_control]
                            proc = subprocess.run(cmd, input=controller_string.encode(), check=False)
                            print(f"[FixedCGroupManager] Controller enable result for {parent_path}: {proc.returncode}")
                
            except Exception as e:
                print(f"[FixedCGroupManager] Warning when enabling controllers: {e}")
            
            # Set permissions
            try:
                proc = subprocess.run(["sudo", "chown", "-R", f"{os.getlogin()}:{os.getlogin()}", self.v2_path], check=True)
                if proc.returncode != 0:
                    raise Exception(f"Failed to take ownership of cgroup v2: {self.v2_path}")
                print(f"[FixedCGroupManager] Successfully set ownership of {self.v2_path}")
            except Exception as e:
                print(f"[FixedCGroupManager] Warning when setting permissions: {e}")
                try:
                    critical_files = ["cpu.max", "memory.max", "memory.swap.max", "cgroup.procs"]
                    for file in critical_files:
                        file_path = os.path.join(self.v2_path, file)
                        if os.path.exists(file_path):
                            subprocess.run(["sudo", "chown", f"{os.getlogin()}:{os.getlogin()}", file_path], check=False)
                except:
                    pass
        
        print(f"[FixedCGroupManager] Cgroup creation completed for {self.group_name}")
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
                cmd = ["sudo", "tee", f"{self.v2_path}/cpu.max"]
                proc = subprocess.run(cmd, input=max_str.encode(), check=False)
            print(f"[FixedCGroupManager] Attempted to set CPU limit to {num_cpus} CPUs")
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
                max_str = f"{memory_in_bytes}"
                cmd = ["sudo", "tee", f"{self.v2_path}/memory.max"]
                proc = subprocess.run(cmd, input=max_str.encode(), check=False)
            print(f"[FixedCGroupManager] Attempted to set memory limit to {memory_in_bytes / 1024 / 1024 / 1024:.2f}GB")
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
                max_str = f"{memory_in_bytes}"
                cmd = ["sudo", "tee", f"{self.v2_path}/memory.swap.max"]
                proc = subprocess.run(cmd, input=max_str.encode(), check=False)
            print(f"[FixedCGroupManager] Attempted to set swap limit to {memory_in_bytes / 1024 / 1024 / 1024:.2f}GB")
            return True
        except Exception as e:
            print(f"[FixedCGroupManager] Error setting swap limit: {e}")
            return False

    def cleanup(self):
        try:
            if self.cgroup_version == 1:
                for path in [self.cpu_path, self.mem_path]:
                    if os.path.exists(path):
                        subprocess.run(["sudo", "rmdir", path], check=False)
            else:
                if os.path.exists(self.v2_path):
                    subprocess.run(["sudo", "rmdir", self.v2_path], check=False)
            print(f"[FixedCGroupManager] Cleaned up cgroup: {self.group_name}")
        except Exception as e:
            print(f"[FixedCGroupManager] Error cleaning up cgroup: {e}")


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

    print("[SPM] Waiting for 20 seconds to free up memory, IO and other resources")
    time.sleep(20)


def generate_db_bench_command(db_bench_path, database_path, options_file_dir, run_count, test_name, db_bench_extra_args=[]):
 
    db_bench_command = [
        db_bench_path,
        f"--db={database_path}",
        f"--options_file={options_file_dir}",
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


def db_bench(db_bench_path, database_path, options, options_file_dir, run_count, test_name, previous_throughput, options_files, db_bench_args=[], bm_iter=0, cgroup_name=None):
 
    with open(options_file_dir, "w") as f:
        f.write(options)

    pre_tasks(database_path, run_count)
    command = generate_db_bench_command(db_bench_path, database_path, options_file_dir, run_count, test_name, db_bench_args)

    log_update(f"[SPM] Executing db_bench with command: {command}")
    print("[SPM] Executing db_bench")

    if SIDE_CHECKER and previous_throughput != None:
        cgroup_monitor = CGroupMonitor()
        
        start_time = time.time()
        cgroup_monitor.start_monitor()

        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True) as proc_out:
            if cgroup_name:
                try:
                    manager = FixedCGroupManager(cgroup_name, helper_script=os.path.abspath("utils/root_cgroup_helper.sh"))
                    if manager.add_process(proc_out.pid):
                        print(f"[SPM] DB_bench process {proc_out.pid} added to cgroup {cgroup_name}")
                    else:
                        print(f"[SPM] Failed to add DB_bench process {proc_out.pid} to cgroup {cgroup_name}")
                except Exception as e:
                    print(f"[SPM] Error adding DB_bench process to cgroup: {e}")

            output = ""
            for line in proc_out.stdout:
                output += line

        print("[SPM] Finished running db_bench")
        print("----------------------------------------------------------------------------")

        op = cgroup_monitor.stop_monitor()
        avg_cpu_used = op["average_cpu_usage_percent"]
        avg_mem_used = op["average_memory_usage_percent"]
        log_update("[SPM] Finished running db_bench")
        return output, avg_cpu_used, avg_mem_used, options
    
    else:
        cgroup_monitor = CGroupMonitor()
        cgroup_monitor.start_monitor()

        proc_out = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        if cgroup_name:
            try:
                manager = FixedCGroupManager(cgroup_name, helper_script=os.path.abspath("utils/root_cgroup_helper.sh"))
                if manager.add_process(proc_out.pid):
                    print(f"[SPM] DB_bench process {proc_out.pid} added to cgroup {cgroup_name}")
                else:
                    print(f"[SPM] Failed to add DB_bench process {proc_out.pid} to cgroup {cgroup_name}")
            except Exception as e:
                print(f"[SPM] Error adding DB_bench process to cgroup: {e}")

        stdout, stderr = proc_out.communicate()

        op = cgroup_monitor.stop_monitor()
        avg_cpu_used = op["average_cpu_usage_percent"]
        avg_mem_used = op["average_memory_usage_percent"]

        print("[SPM] Finished running db_bench")
        print("---------------------------------------------------------------------------")
        
        return stdout, avg_cpu_used, avg_mem_used, options


def benchmark(db_path, options, output_file_dir, reasoning, iteration_count, previous_results, options_files, db_bench_args, bm_iter=0, cgroup_name=None, workload=None):
    test_name = workload if workload else TEST_NAME
    
    options_file_dir = os.path.join(output_file_dir, "options_file.ini")
    
    if previous_results is None:
        output, average_cpu_usage, average_memory_usage, options = db_bench(
            DB_BENCH_PATH, db_path, options, options_file_dir, iteration_count, test_name, None, options_files, db_bench_args, cgroup_name=cgroup_name)
    else:
        output, average_cpu_usage, average_memory_usage, options = db_bench(
            DB_BENCH_PATH, db_path, options, options_file_dir, iteration_count, test_name, previous_results['ops_per_sec'], options_files, db_bench_args, cgroup_name=cgroup_name)

    benchmark_results = parse_db_bench_output(output)

    contents = os.listdir(output_file_dir)
    ini_file_count = len([f for f in contents if f.endswith(".ini")])

    options_file_path = os.path.join(output_file_dir, f"options_file_{ini_file_count}.ini")
    with open(options_file_path, "w") as f:
        f.write(options)

    benchmark_file_path = os.path.join(output_file_dir, f"benchmark_results_{ini_file_count}.json")
    benchmark_data = {
        "options": options,
        "reasoning": reasoning,
        "benchmark_results": benchmark_results,
        "average_cpu_usage": average_cpu_usage,
        "average_memory_usage": average_memory_usage,
        "workload": test_name
    }
    
    with open(benchmark_file_path, "w") as f:
        json.dump(benchmark_data, f, indent=2)

    log_update(f"[SPM] Benchmark completed for workload: {test_name}")
    log_update(f"[SPM] Ops per sec: {benchmark_results.get('ops_per_sec', 0)}")
    
    return False, benchmark_results, average_cpu_usage, average_memory_usage, options

