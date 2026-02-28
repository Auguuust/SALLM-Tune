import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

EVAL_SERVER_HOST = "0.0.0.0"
EVAL_SERVER_PORT = 8000

UTILS_PATH = os.path.join(PROJECT_ROOT, "utils")
ROCKSDB_PATH = os.path.join(PROJECT_ROOT, "rocksdb")
OPTIONS_FILES_PATH = os.path.join(PROJECT_ROOT, "options_files")
LLM_PATH = os.path.join(PROJECT_ROOT, "llm")
TRACE_ANALYZER_PATH_DIR = os.path.join(PROJECT_ROOT, "trace_analyzer")

LOG_DIR = os.path.join(PROJECT_ROOT, "ft_log")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "eval_output")
DB_PATH_DIR = os.path.join(PROJECT_ROOT, "eval_dbpath")

for directory in [LOG_DIR, OUTPUT_DIR, DB_PATH_DIR]:
    os.makedirs(directory, exist_ok=True)

DEFAULT_TIMEOUT = 3000
MAX_RETRIES = 3
RETRY_DELAY = 5

CGROUP_HELPER_SCRIPT = os.path.join(UTILS_PATH, "root_cgroup_helper.sh")

def get_project_path():
    return PROJECT_ROOT

def validate_dependencies():
    required_paths = [
        UTILS_PATH,
        ROCKSDB_PATH,
        OPTIONS_FILES_PATH,
        LLM_PATH,
        TRACE_ANALYZER_PATH_DIR
    ]
    
    missing_paths = []
    for path in required_paths:
        if not os.path.exists(path):
            missing_paths.append(path)
    
    if missing_paths:
        print("Error: Missing required dependencies:")
        for path in missing_paths:
            print(f"  - {path}")
        return False
    
    return True

def setup_environment():
    import sys
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    
    for path in [UTILS_PATH, ROCKSDB_PATH, OPTIONS_FILES_PATH, LLM_PATH, TRACE_ANALYZER_PATH_DIR]:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    return True

