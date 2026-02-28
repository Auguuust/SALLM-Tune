import mmap
import os
import struct
import re
import time

from utils.utils import log_update

mmap_file_path = "/tmp/mmap_file.mmap"
mmap_size = 1024

def create_mmap_file():
    if not(os.path.exists(mmap_file_path)):
        with open(mmap_file_path, "wb") as f:
            f.write(b'\x00' * mmap_size)
    else:
        log_update("MMap file already exists. Setting top bit to 0 to avoid db_bench reads.")
        with open(mmap_file_path, "r+b") as f:
            with mmap.mmap(f.fileno(), mmap_size, access=mmap.ACCESS_WRITE) as m:
                m[0] = 0

def add_mmap_file_to_option(option_file, mmap_str):
    pattern = re.compile(r'(\w+)\s*=\s*([\w\.\-]+)')

    mmap_options = {}
    for match in pattern.finditer(mmap_str):
        key, value = match.groups()
        mmap_options[key] = value
    
    updated_lines = []
    for line in option_file.split('\n'):
        match = pattern.match(line)
        if match:
            key, _ = match.groups()
            if key in mmap_options:
                line = f"{key} = {mmap_options[key]}"
        updated_lines.append(line)
    
    updated_option_file = '\n'.join(updated_lines)
    
    return updated_option_file

def convert_option_string_to_list(data):
    dataKey = [
        'max_open_files', 
        'max_total_wal_size', 
        'delete_obsolete_files_period_micros', # Currently ignored
        'max_background_jobs', 
        'max_background_compactions', 
        'max_subcompactions', 
        'stats_dump_period_sec', 
        'compaction_readahead_size', 
        'writable_file_max_buffer_size', 
        'bytes_per_sync', 
        'wal_bytes_per_sync', 
        'delayed_write_rate', 
        'avoid_flush_during_shutdown', 
        'write_buffer_size', 
        'compression', 
        'level0_file_num_compaction_trigger', 
        'max_bytes_for_level_base', 
        'disable_auto_compactions', 
        'memtable_max_range_deletions', 
    ]
    
    options = {}
    pattern = re.compile(r'(\w+)\s*=\s*([\w\.\-]+)')
    for match in pattern.finditer(data):
        key, value = match.groups()
        options[key] = value
    
    result = []
    for key in dataKey:
        if key in options:
            if options[key].lower() == 'false':
                value = 0
            elif options[key].lower() == 'true':
                value = 1
            elif 'no' in options[key].lower():
                value = 0
            elif 'snappy' in options[key].lower():
                value = 1
            elif 'zlib' in options[key].lower():
                value = 2
            elif 'bzip2' in options[key].lower():
                value = 3
            elif 'lz4' in options[key].lower():
                value = 4
            elif 'lz4hc' in options[key].lower():
                value = 5
            elif 'xpress' in options[key].lower():
                value = 6
            elif 'zstd' in options[key].lower():
                value = 7
            else:
                try:
                    value = int(options[key])
                    if not(-2147483648 <= value <= 2147483647):
                        log_update(f"Value for {key} is out of boundaries: {value}. Clamping applied.")
                        value = max(-2147483648, min(2147483647, value))
                except:
                    log_update(f"Error converting value of {key} to integer: " + options[key])
                    log_update("Forcing value to 0 for key: " + key)
                    value = 0
        else:
            log_update("Error key not found in options file: " + key)
            log_update("Forcing value to 0 for key: " + key)
            value = 0
        
        result.append(value)
    
    return result

def write_to_mmap_file(data):
    if type(data) == str:
        data = convert_option_string_to_list(data)
    
    with open(mmap_file_path, "r+b") as f:
        with mmap.mmap(f.fileno(), mmap_size, access=mmap.ACCESS_WRITE) as m:
            
            m[0] = 0 

            m.seek(1)
            for i in range(len(data)):
                m.write(struct.pack('i', data[i]))

            m[0] = 1
