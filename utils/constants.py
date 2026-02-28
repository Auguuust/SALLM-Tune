import os
from dotenv import load_dotenv
import argparse
from datetime import datetime

load_dotenv()

def path_of_output_folder(llm_model, test_name, limit_list):
    '''
    Set the output folder directory

    Parameters:
    - None

    Returns:
    - output_folder_dir (str): The output folder directory
    '''
    current_datetime = datetime.now()
    date_time_string = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    if llm_model == "deepseek-ai/DeepSeek-V3":
        dir_name = f"{llm_model.split('/')[-1]}_{test_name}_{limit_list}"
    else:
        dir_name = f"{llm_model}_{test_name}_{limit_list}"
    output_folder_dir = f"output/output_{DEVICE}/{dir_name}_{date_time_string}"

    os.makedirs(output_folder_dir, exist_ok=True)
    print(f"[UTL] Using output folder: {output_folder_dir}")

    return output_folder_dir

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected!')


env_DEVICE = os.getenv("DEVICE", "data")
env_ITERATION_COUNT = os.getenv("ITERATION_COUNT", 3)
env_TEST_NAME = os.getenv("TEST_NAME", "mixgraph")
env_VERSION = os.getenv("VERSION", "8.8.1")
env_OUTPUT_PATH = os.getenv("OUTPUT_PATH", None)
env_NUM_ENTRIES = os.getenv("NUM_ENTRIES", 2500000)
env_NUM_THREADS = os.getenv("NUM_THREADS", 8)
env_DURATION = os.getenv("DURATION", 200)
env_SINE_WRITE_RATE_INTERVAL_MILLISECONDS = os.getenv("SINE_WRITE_RATE_INTERVAL_MILLISECONDS", 1000)
env_SINE_A = os.getenv("SINE_A", 2000000)
env_SINE_B = os.getenv("SINE_B", 2.3873241464)
env_SINE_C = os.getenv("SINE_C", 0)
env_SINE_D = os.getenv("SINE_D", 10000000)

env_SIDE_CHECKER = str2bool(os.getenv("SIDE_CHECKER", True))
env_ERROR_CORRECTION_COUNT = os.getenv("ERROR_CORRECTION_COUNT", 2)
env_DYNAMIC_OPTION_TUNING = os.getenv("DYNAMIC_OPTION_TUNING", True)
env_ENABLE_MIDWAY_MONITORING = str2bool(os.getenv("ENABLE_MIDWAY_MONITORING", True))
env_LLM_MODEL = os.getenv("LLM_MODEL", "qwen-plus-0428") 
env_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
env_RAG = str2bool(os.getenv("RAG", False))
env_TRACEFILE_PATH = os.getenv("TRACEFILE_PATH", None)
env_PRE_LOAD_CMD = os.getenv("PRE_LOAD_CMD", None)
env_PRE_LOAD_DB_PATH = os.getenv("PRE_LOAD_DB_PATH", "")
env_LIMIT_LIST = os.getenv("LIMIT_LIST", None)

# API Keys for different LLM providers
env_SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "<your_siliconflow_api_key>")
env_DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "<your_dashscope_api_key>")
env_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "<your_openai_api_key>")
env_OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")

# Base URLs for different LLM providers
env_SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
env_DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
env_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "<your_ollama_base_url>")


parser = argparse.ArgumentParser(description='Description of your script')
parser.add_argument('-i', '--iteration_count', type=int, default=env_ITERATION_COUNT, help='Specify the number of iterations')
parser.add_argument('-d', '--device', type=str, default=env_DEVICE, help='Specify the device')
parser.add_argument('-t', '--workload', type=str, default=env_TEST_NAME, help='Specify the test name')
parser.add_argument('-v', '--version', type=str, default=env_VERSION, help='Specify the version of RocksDB')
parser.add_argument('-o', '--output', type=str, default=env_OUTPUT_PATH, help='Specify the output path')
parser.add_argument('-n', '--num_entries', type=int, default=env_NUM_ENTRIES, help='Specify the number of entries')
parser.add_argument('-th', '--num_threads', type=int, default=env_NUM_THREADS, help='Specify the number of threads')
parser.add_argument('-u', '--duration', type=int, default=env_DURATION, help='Specify the duration')
parser.add_argument('-s', '--side_checker', type=str2bool, default=env_SIDE_CHECKER, help='Specify if side checker is enabled')
parser.add_argument('-ec', '--error_correction_count', type=int, default=env_ERROR_CORRECTION_COUNT, help='Specify the error correction count')
parser.add_argument('-dt', '--dynamic_option_tuning', type=str2bool, default=env_DYNAMIC_OPTION_TUNING, help='Specify if dynamic option tuning is enabled')
parser.add_argument('-em', '--enable_midway_monitoring', type=str2bool, default=env_ENABLE_MIDWAY_MONITORING, help='Specify if midway performance monitoring is enabled')
parser.add_argument('-m', '--llm_model', type=str, default=env_LLM_MODEL, help='Specify the LLM model to use')
parser.add_argument('-e', '--embedding_model', type=str, default=env_EMBEDDING_MODEL, help='Specify the embedding model to use')
parser.add_argument('-r', '--rag', type=str2bool, default=env_RAG, help='Specify if RAG is enabled')
parser.add_argument('--tracefile_path', type=str, default=env_TRACEFILE_PATH, help='Specify the path of the tracefile')
parser.add_argument('--pre_load_cmd', type=str, default=env_PRE_LOAD_CMD, help='Specify the pre-load command')
parser.add_argument('--pre_load_db_path', type=str, default=env_PRE_LOAD_DB_PATH, help='Specify the pre-load db path')
parser.add_argument('--sine_write_rate_interval_milliseconds', type=int, default=env_SINE_WRITE_RATE_INTERVAL_MILLISECONDS, help='Specify the sine write rate interval in milliseconds')
parser.add_argument('--sine_a', type=float, default=env_SINE_A, help='Specify the sine parameter a')
parser.add_argument('--sine_b', type=float, default=env_SINE_B, help='Specify the sine parameter b')
parser.add_argument('--sine_c', type=float, default=env_SINE_C, help='Specify the sine parameter c')
parser.add_argument('--sine_d', type=float, default=env_SINE_D, help='Specify the sine parameter d')
parser.add_argument('--limit_list', type=str, default=env_LIMIT_LIST, help='Specify the limit list')

args = parser.parse_args()
LIMIT_LIST = args.limit_list
ITERATION_COUNT = args.iteration_count
DEVICE = args.device
TEST_NAME = args.workload
VERSION = args.version
LLM_MODEL = args.llm_model
OUTPUT_PATH = args.output if args.output else path_of_output_folder(LLM_MODEL, TEST_NAME, LIMIT_LIST)
NUM_ENTRIES = args.num_entries
NUM_THREADS = args.num_threads
DURATION = args.duration
SIDE_CHECKER = args.side_checker
ERROR_CORRECTION_COUNT = args.error_correction_count
DYNAMIC_OPTION_TUNING = args.dynamic_option_tuning
ENABLE_MIDWAY_MONITORING = False
EMBEDDING_MODEL = args.embedding_model
RAG = args.rag
TRACEFILE_PATH = args.tracefile_path
PRE_LOAD_CMD = args.pre_load_cmd
PRE_LOAD_DB_PATH = args.pre_load_db_path
SINE_WRITE_RATE_INTERVAL_MILLISECONDS = args.sine_write_rate_interval_milliseconds
SINE_A = args.sine_a
SINE_B = args.sine_b
SINE_C = args.sine_c
SINE_D = args.sine_d

# API Keys
SILICONFLOW_API_KEY = env_SILICONFLOW_API_KEY
DASHSCOPE_API_KEY = env_DASHSCOPE_API_KEY
OPENAI_API_KEY = env_OPENAI_API_KEY
OLLAMA_API_KEY = env_OLLAMA_API_KEY

# Base URLs
SILICONFLOW_BASE_URL = env_SILICONFLOW_BASE_URL
DASHSCOPE_BASE_URL = env_DASHSCOPE_BASE_URL
OLLAMA_BASE_URL = env_OLLAMA_BASE_URL


# Path Constants locally
DB_BENCH_PATH = f"/home/sdu/rocksdb-8.8.1/db_bench"
TRACE_ANALYZER_PATH = f"/home/sdu/rocksdb-8.8.1/trace_analyzer"
DB_PATH = f"/mnt/n1/dbpath/db"
FIO_RESULT_PATH = f"./data/fio/fio_output_{DEVICE}.txt"
DEFAULT_OPTION_FILE_DIR = "options_files/default_options_files"
INITIAL_OPTIONS_FILE_NAME = f"dbbench_default_options-{VERSION}.ini"
OPTIONS_FILE_DIR = f"{OUTPUT_PATH}/options_file.ini"
PRE_LOAD_DB_PATH = f"/mnt/n0/db_preload"
# Path Constants docker
# DB_BENCH_PATH = f"/rocksdb-{VERSION}/db_bench"
# TRACE_ANALYZER_PATH = f"/rocksdb-{VERSION}/trace_analyzer"
# DB_PATH = f"/{DEVICE}/llm_project/db"
# FIO_RESULT_PATH = f"data/fio/fio_output_{DEVICE}.txt"
# DEFAULT_OPTION_FILE_DIR = "options_files/default_options_files"
# INITIAL_OPTIONS_FILE_NAME = f"dbbench_default_options-{VERSION}.ini"
# OPTIONS_FILE_DIR = f"{OUTPUT_PATH}/options_file.ini"
