import os
import psutil
import subprocess
import platform
from cpuinfo import get_cpu_info
from cgroup_monitor import CGroupMonitor

def get_system_data(db_path):
    cgroup_monitor = CGroupMonitor('llm_cgroup')
    try:
        quota, period = cgroup_monitor.get_cpu_limit()
        mem = cgroup_monitor.get_memory_limit()
        swap_space = cgroup_monitor.get_swap_limit()

        cpu_count = os.getenv("CPU_COUNT", str(quota//period))
        mem_max = os.getenv("MEMORY_MAX", str(mem/(1024*1024*1024)))

        system_info = platform.uname()
        cpu_op_modes = system_info.processor

        cpu_model = platform.processor()

        cpu_info = get_cpu_info()
        brand_raw_value = cpu_count + " cores of " + cpu_info['brand_raw']

        l1_data_cache_size = cpu_info.get('l1_data_cache_size', 'N/A')
        l1_instruction_cache_size = cpu_info.get(
            'l1_instruction_cache_size', 'N/A')
        l2_cache_size = cpu_info.get('l2_cache_size', 'N/A')
        l3_cache_size = cpu_info.get('l3_cache_size', 'N/A')

        memory_total = float(mem_max)

        memory_used = psutil.virtual_memory().percent

        memeory_remaining = psutil.virtual_memory().available * 100 / \
            psutil.virtual_memory().total

        swap = swap_space

        partitions = psutil.disk_partitions(all=False)
        path = os.path.dirname(db_path)
        total_disk_size = -1
        device = ""
        all_devices = check_drive_type()
        data_directory = path[:5]
        for partition in partitions:
            usage = psutil.disk_usage(partition.mountpoint)
            if (partition.mountpoint[:5] == data_directory):
                total_disk_size = usage.total
                if (partition.device.split('/')[-1] in all_devices):
                    device = all_devices[partition.device.split('/')[-1]]
                elif (partition.device.split('/')[-1][:-1] in all_devices):
                    device = all_devices[partition.device.split('/')[-1][:-1]]

        return brand_raw_value, memory_total, swap, total_disk_size, device

    except Exception as e:
        print(f"[SYS] Error in fetching system data: {e}")
        return None


def check_drive_type():
    sys_block_path = "/sys/block"
    if os.path.exists(sys_block_path):
        devices = os.listdir(sys_block_path)
        drive_types = {}
        for device in devices:
            try:
                with open(f"{sys_block_path}/{device}/queue/rotational", "r") as file:
                    rotational = file.read().strip()
                    if rotational == "0":
                        drive_types[device] = "SSD"
                    else:
                        drive_types[device] = "HDD"
            except IOError:
                pass
        return drive_types
    else:
        return "System block path does not exist."

def system_info(db_path, fio_result):
    system_data = get_system_data(db_path)
    data = (f"{system_data[0]} with {system_data[1]}GiB of Memory and {system_data[1]}GiB of Swap space."
            f"{system_data[4]} size : {system_data[3]/(1024 ** 4):.2f}T. A single instance of RocksDB is the always going to be the only process running. "
            f"{fio_result}")
    return data
