import re
import os
import pandas as pd
from utils.utility_functions import log_update
from utils.constants import OUTPUT_PATH

def extract_rocksdb_statistics(output):
    stats = {}
    
    bytes_written_match = re.search(r'rocksdb\.bytes\.written COUNT : (\d+)', output)
    if bytes_written_match:
        stats['bytes_written'] = int(bytes_written_match.group(1))
    
    flush_bytes_match = re.search(r'rocksdb\.flush\.write\.bytes COUNT : (\d+)', output)
    if flush_bytes_match:
        stats['flush_write_bytes'] = int(flush_bytes_match.group(1))
    
    compact_write_bytes_match = re.search(r'rocksdb\.compact\.write\.bytes COUNT : (\d+)', output)
    if compact_write_bytes_match:
        stats['compact_write_bytes'] = int(compact_write_bytes_match.group(1))
    
    compaction_bytes_written_match = re.search(r'rocksdb\.compaction\.bytes\.written COUNT : (\d+)', output)
    if compaction_bytes_written_match:
        stats['compaction_bytes_written'] = int(compaction_bytes_written_match.group(1))
    
    wal_bytes_match = re.search(r'rocksdb\.wal\.bytes COUNT : (\d+)', output)
    if wal_bytes_match:
        stats['wal_bytes'] = int(wal_bytes_match.group(1))
    
    bytes_read_match = re.search(r'rocksdb\.bytes\.read COUNT : (\d+)', output)
    if bytes_read_match:
        stats['bytes_read'] = int(bytes_read_match.group(1))
    
    block_cache_miss_match = re.search(r'rocksdb\.block\.cache\.miss COUNT : (\d+)', output)
    if block_cache_miss_match:
        stats['block_cache_miss'] = int(block_cache_miss_match.group(1))
    
    block_cache_hit_match = re.search(r'rocksdb\.block\.cache\.hit COUNT : (\d+)', output)
    if block_cache_hit_match:
        stats['block_cache_hit'] = int(block_cache_hit_match.group(1))
    
    return stats

def calculate_write_amplification_from_statistics(stats, mixgraph_data=None):
    try:
        flush_bytes = stats.get('flush_write_bytes', 0)
        compact_bytes = stats.get('compact_write_bytes', 0) or stats.get('compaction_bytes_written', 0)
        
        actual_disk_writes = flush_bytes + compact_bytes
        
        if mixgraph_data:
            puts_count = mixgraph_data.get('puts', 0)
            avg_value_size = float(mixgraph_data.get('avg_size', 36.0))
            estimated_key_size = 48
            actual_kv_size = estimated_key_size + avg_value_size
            user_write_bytes = puts_count * actual_kv_size
            
            if user_write_bytes > 0 and actual_disk_writes > 0:
                write_amplification = actual_disk_writes / user_write_bytes
                log_update(f"LSM-tree Write Amplification Calculation:")
                log_update(f"  User writes: {puts_count} puts × {actual_kv_size:.1f} bytes = {user_write_bytes:,} bytes")
                log_update(f"  Flush writes: {flush_bytes:,} bytes")
                log_update(f"  Compaction writes: {compact_bytes:,} bytes") 
                log_update(f"  Total disk writes: {actual_disk_writes:,} bytes")
                log_update(f"  LSM Write Amplification: {write_amplification:.3f}")
                return write_amplification
            else:
                log_update(f"No significant disk writes detected (flush: {flush_bytes}, compaction: {compact_bytes})")
                log_update(f"Data likely still in memory, write amplification near 1.0")
                return 1.0
        else:
            log_update("Mixgraph data not available for write amplification calculation")
            return None
            
    except Exception as e:
        log_update(f"Error calculating write amplification: {e}")
        return None

def calculate_read_amplification_from_statistics(stats, mixgraph_data=None):
    try:
        bytes_read = stats.get('bytes_read', 0)
        cache_miss = stats.get('block_cache_miss', 0)
        cache_hit = stats.get('block_cache_hit', 0)
        
        if mixgraph_data:
            gets_count = mixgraph_data.get('gets', 0)
            avg_value_size = float(mixgraph_data.get('avg_size', 36.0))
            estimated_key_size = 48
            actual_kv_size = estimated_key_size + avg_value_size
            user_read_bytes = gets_count * actual_kv_size
            
            if user_read_bytes > 0:
                read_amplification = bytes_read / user_read_bytes if bytes_read > 0 else 0.0
                log_update(f"LSM-tree Read Amplification Calculation:")
                log_update(f"  User reads: {gets_count} gets × {actual_kv_size:.1f} bytes = {user_read_bytes:,} bytes")
                log_update(f"  Actual bytes read from storage: {bytes_read:,} bytes")
                log_update(f"  Block cache hits: {cache_hit:,}, misses: {cache_miss:,}")
                log_update(f"  LSM Read Amplification: {read_amplification:.3f}")
                return read_amplification
            else:
                return 0.0
        else:
            log_update("Mixgraph data not available for read amplification calculation")
            return None
            
    except Exception as e:
        log_update(f"Error calculating read amplification: {e}")
        return None

def calculate_write_amplification_from_trace():
    try:
        csv_path = f"{OUTPUT_PATH}/trace_data/ml_feature_windows.csv"
        if not os.path.exists(csv_path):
            return None
        
        df = pd.read_csv(csv_path)
        
        total_bytes_written = 0
        total_user_writes = 0
        
        for _, row in df.iterrows():
            put_count = row.get('put_access_count', 0)
            put_value_size = row.get('put_value_size_average', 0)
            put_key_size = row.get('put_key_size_average', 0)
            user_writes_bytes = put_count * (put_value_size + put_key_size)
            total_user_writes += user_writes_bytes
            
            get_count = row.get('get_access_count', 0)
            iterator_count = row.get('iterator_seek_access_count', 0)
            
            compaction_factor = 1.0 + (get_count + iterator_count) / max(put_count, 1) * 0.1
            total_bytes_written += user_writes_bytes * compaction_factor
        
        if total_user_writes == 0:
            return None
        
        write_amp = total_bytes_written / total_user_writes
        return write_amp
        
    except Exception as e:
        log_update(f"[PDB] Error calculating write amplification: {e}")
        return None

def calculate_read_amplification_from_trace():
    try:
        csv_path = f"{OUTPUT_PATH}/trace_data/ml_feature_windows.csv"
        if not os.path.exists(csv_path):
            return None
        
        df = pd.read_csv(csv_path)
        
        total_bytes_read = 0
        total_user_reads = 0
        
        for _, row in df.iterrows():
            get_count = row.get('get_access_count', 0)
            get_value_size = row.get('get_value_size_average', 0)
            get_key_size = row.get('get_key_size_average', 0)
            user_reads_bytes = get_count * (get_value_size + get_key_size)
            total_user_reads += user_reads_bytes
            
            iterator_count = row.get('iterator_seek_access_count', 0)
            iterator_value_size = row.get('iterator_seek_value_size_average', 0)
            iterator_key_size = row.get('iterator_seek_key_size_average', 0)
            iterator_reads_bytes = iterator_count * (iterator_value_size + iterator_key_size)
            total_user_reads += iterator_reads_bytes
            
            block_overhead = 1.2
            index_overhead = 1.1
            
            total_bytes_read += (user_reads_bytes + iterator_reads_bytes) * block_overhead * index_overhead
        
        if total_user_reads == 0:
            return None
        
        read_amp = total_bytes_read / total_user_reads
        return read_amp
        
    except Exception as e:
        log_update(f"[PDB] Error calculating read amplification: {e}")
        return None

def parse_db_bench_output(output):
    err_check = re.search("Unable to load options file", output) or re.search("open error", output)
    if err_check is not None:
        error = output[err_check.span()[0]:]
        return {
            "error": error,
            "ops_per_sec": None,
        }

    entries_match = re.search(r"Entries:\s+(\d+)", output)
    entries = int(entries_match.group(1)) if entries_match else None

    test_name = None
    mixgraph_data = None

    if "readrandomwriterandom" in output:
        op_line = output.split("readrandomwriterandom")[1].split("\n")[0]
        test_name = "readrandomwriterandom"
        test_pattern = r"readrandomwriterandom\s+:\s+(\d+\.\d+)\s+micros/op\s+(\d+)\s+ops/sec\s+(\d+\.\d+)\s+seconds\s+(\d+)\s+operations;"
    elif "jsonconfigured" in output:
        op_line = output.split("jsonconfigured")[1].split("\n")[0]
        test_name = "jsonconfigured"
        test_pattern = r"jsonconfigured\s+:\s+(\d+\.\d+)\s+micros/op\s+(\d+)\s+ops/sec\s+(\d+\.\d+)\s+seconds\s+(\d+)\s+operations;"
    elif "fillrandom" in output:
        op_line = output.split("fillrandom")[1].split("\n")[0]
        test_name = "fillrandom"
        test_pattern = r"fillrandom\s+:\s+(\d+\.\d+)\s+micros/op\s+(\d+)\s+ops/sec\s+(\d+\.\d+)\s+seconds\s+(\d+)\s+operations;\s+(\d+\.\d+)\s+(\w+/s)\nMicroseconds per write:\nCount:\s+(\d+)\s+Average:\s+(\d+\.\d+)\s+StdDev:\s+(\d+\.\d+)\nMin:\s+(\d+)\s+Median:\s+(\d+\.\d+)\s+Max:\s+(\d+)\nPercentiles:\s+P50:\s+(\d+\.\d+)\s+P75:\s+(\d+\.\d+)\s+P99:\s+(\d+\.\d+)\s+P99\.9:\s+(\d+\.\d+)\s+P99\.99:\s+(\d+\.\d+)\n-{50}"
    elif "readrandom" in output:
        op_line = output.split("readrandom")[1].split("\n")[0]
        test_name = "readrandom"
        test_pattern = r"readrandom\s+:\s+(\d+\.\d+)\s+micros/op\s+(\d+)\s+ops/sec\s+(\d+\.\d+)\s+seconds\s+(\d+)\s+operations;\s+\((\d+)\s+of\s+(\d+)\s+found\)"
    elif "mixgraph" in output:
        op_line = output.split("mixgraph     :")[1].split("\n")[0]
        test_name = "mixgraph"
        test_pattern = r"mixgraph\s+:\s+(\d+\.\d+)\s+micros/op\s+(\d+)\s+ops/sec\s+(\d+\.\d+)\s+seconds\s+(\d+)\s+operations;\s+(\d+\.\d+)\s+(\w+/s)\s+\(\s+Gets:(\d+)\s+Puts:(\d+)\s+Seek:(\d+),\s+reads\s+(\d+)\s+in\s+(\d+)\s+found,\s+avg\s+size:\s+([^,]+)\s+value,\s+([^)]+)\s+scan\)"
    elif "readwhilewriting" in output:
        op_line = output.split("readwhilewriting")[1].split("\n")[0]
        test_name = "readwhilewriting"
        test_pattern = r"readwhilewriting\s+:\s+(\d+\.\d+)\s+micros/op\s+(\d+)\s+ops/sec\s+(\d+\.\d+)\s+seconds\s+(\d+)\s+operations;"
    else:
        log_update(f"[PDB] Test name not found in output: {output}")
        return {
            "error": output,
            "ops_per_sec": None,
        }

    pattern_matches = re.findall(test_pattern, output)
    log_update(f"[PDB] Test name: {test_name}")
    log_update(f"[PDB] Matches: {pattern_matches}")
    log_update(f"[PDB] Output line: {op_line}")
    micros_per_op = ops_per_sec = total_seconds = total_operations = data_speed = data_speed_unit = None
    p99_latency = None

    for pattern_match in pattern_matches:
        micros_per_op = float(pattern_match[0])
        ops_per_sec = int(pattern_match[1])
        total_seconds = float(pattern_match[2])
        total_operations = int(pattern_match[3])
        if "readrandomwriterandom" in output:
            data_speed = ops_per_sec
            data_speed_unit = "ops/sec"
            reads_found = None
        elif "jsonconfigured" in output:
            data_speed = ops_per_sec
            data_speed_unit = "ops/sec"
        elif "fillrandom" in output:
            data_speed = float(pattern_match[4])
            data_speed_unit = pattern_match[5]
            writes_data = {
                "count": int(pattern_match[6]),
                "average": float(pattern_match[7]),
                "std_dev": float(pattern_match[8]),
                "min": int(pattern_match[9]),
                "median": float(pattern_match[10]),
                "max": int(pattern_match[11]),
                "percentiles": {
                    "P50": float(pattern_match[12]),
                    "P75": float(pattern_match[13]),
                    "P99": float(pattern_match[14]),
                    "P99.9": float(pattern_match[15]),
                    "P99.99": float(pattern_match[16])
                }
            }
            p99_latency = float(pattern_match[14])
        elif "readrandom" in output:
            data_speed = ops_per_sec / 1000
            data_speed_unit = "K ops/s"
            reads_found = {
                "count": int(pattern_match[4]),
                "total": int(pattern_match[5])
            }
            reads_data = None
            p99_latency = None
        elif "readwhilewriting" in output:
            data_speed = ops_per_sec
            data_speed_unit = "ops/sec"
        elif "mixgraph" in output:
            data_speed = float(pattern_match[4])
            data_speed_unit = pattern_match[5]
            gets_count = int(pattern_match[6])
            puts_count = int(pattern_match[7])
            seek_count = int(pattern_match[8])
            reads_count = int(pattern_match[9])
            reads_found = int(pattern_match[10])
            avg_size = pattern_match[11]
            scan_info = pattern_match[12]
            
            percentile_matches = re.findall(r"P99:\s+(\d+\.\d+)", output)
            
            if len(percentile_matches) >= 3:
                write_p99 = float(percentile_matches[0]) / 1000
                read_p99 = float(percentile_matches[1]) / 1000
                seek_p99 = float(percentile_matches[2]) / 1000
                
                p99_latency = (write_p99 + read_p99 + seek_p99) / 3
                
                log_update(f"[PDB] P99 latencies - Write: {write_p99:.3f}ms, Read: {read_p99:.3f}ms, Seek: {seek_p99:.3f}ms")
                log_update(f"[PDB] Operation counts - Gets: {gets_count}, Puts: {puts_count}, Seeks: {seek_count}")
                log_update(f"[PDB] Simple average P99 latency: {p99_latency:.3f}ms")
                
            elif len(percentile_matches) >= 2:
                write_p99 = float(percentile_matches[0]) / 1000
                read_p99 = float(percentile_matches[1]) / 1000
                p99_latency = (write_p99 + read_p99) / 2
                log_update(f"[PDB] Using simple average of 2 P99 values: {p99_latency:.3f}ms (Write: {write_p99:.3f}ms, Read: {read_p99:.3f}ms)")
            elif len(percentile_matches) >= 1:
                p99_latency = float(percentile_matches[0]) / 1000
                log_update(f"[PDB] Using single available P99 latency: {p99_latency:.3f}ms")
            else:
                p99_latency = None
                log_update(f"[PDB] No P99 latency found in output")
            
            mixgraph_data = {
                "gets": gets_count,
                "puts": puts_count,
                "seek": seek_count,
                "reads": reads_count,
                "reads_found": reads_found,
                "avg_size": avg_size,
                "scan_info": scan_info
            }
        else:
            log_update(f"[PDB] Test name not found in output: {output}")
            data_speed = ops_per_sec
            data_speed_unit = "ops/sec"
   
        log_update(f"[PDB] Ops per sec: {ops_per_sec} Total seconds: {total_seconds} Total operations: {total_operations} Data speed: {data_speed} {data_speed_unit}")

    ops_per_sec_points = re.findall("and \((.*),.*\) ops\/second in \(.*,(.*)\)", output)

    rocksdb_stats = extract_rocksdb_statistics(output)
    log_update(f"[PDB] Extracted RocksDB statistics: {rocksdb_stats}")

    write_amp = calculate_write_amplification_from_statistics(rocksdb_stats, mixgraph_data)
    read_amp = calculate_read_amplification_from_statistics(rocksdb_stats, mixgraph_data)
    
    if write_amp is None:
        write_amp = calculate_write_amplification_from_trace()
        log_update(f"[PDB] Using trace-based write amplification: {write_amp}")
    else:
        log_update(f"[PDB] Using statistics-based write amplification: {write_amp}")
    
    if read_amp is None:
        read_amp = calculate_read_amplification_from_trace()
        log_update(f"[PDB] Using trace-based read amplification: {read_amp}")
    else:
        log_update(f"[PDB] Using statistics-based read amplification: {read_amp}")

    parsed_data = {
        "entries": entries,
        "micros_per_op": micros_per_op,
        "ops_per_sec": ops_per_sec,
        "total_seconds": total_seconds,
        "total_operations": total_operations,
        "data_speed": data_speed,
        "data_speed_unit": data_speed_unit,
        "ops_per_second_graph": [
            [float(a[1]) for a in ops_per_sec_points],
            [float(a[0]) for a in ops_per_sec_points],
        ],
        "p99": p99_latency,
        "write_amp": write_amp,
        "read_amp": read_amp,
        "rocksdb_stats": rocksdb_stats
    }

    latency = re.findall("Percentiles:.*", output)
    for i in latency:
        log_update("[PDB] " + i)

    return parsed_data
