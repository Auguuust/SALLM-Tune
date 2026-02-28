import os
import re
import configparser

from utils.constants import DEFAULT_OPTION_FILE_DIR, INITIAL_OPTIONS_FILE_NAME, OPTIONS_FILE_DIR
from utils.filter import BLACKLIST, DB_BENCH_ARGS
from utils.parse import dict_to_configparser, configparser_to_string
from utils.utility_functions import log_update, log_llm_response
from utils.options_list import RocksDBOptions

def parse_llm_text_to_dict(llm_output_text):
    options_dict = {}

    for line in llm_output_text.split("\n"):
        if not line.startswith('#'):
            parts = line.split(':', 1)
            if len(parts) == 1:
                parts = line.split('=', 1)
            if len(parts) == 2:
                if '{' not in parts[1].strip():
                    if parts[0].strip() not in BLACKLIST:
                        key, value = parts[0].strip().strip("--"), parts[1].strip().split('#')[0].strip()
                        options_dict[key] = value

    return options_dict


def cleanup_options_file(llm_options_text, prev_db_bench_args=None):
    
    clean_output_dict = parse_option_file_to_dict(open(f"{OPTIONS_FILE_DIR}").read())
    
    llm_output_dict = parse_llm_text_to_dict(llm_options_text)
    
    changed_value = {}
    args_dict = {}
    if prev_db_bench_args:
        args_dict = parse_db_bench_args_to_dict(prev_db_bench_args)
    
    for key, value in llm_output_dict.items():
        if key in DB_BENCH_ARGS:
            if value == "-1":
                continue
            if key not in args_dict or args_dict[key] != value:
                args_dict[key] = value
                changed_value[key] = value
            continue
            
        for internal_dict in clean_output_dict:
            if key in clean_output_dict[internal_dict]:
                if clean_output_dict[internal_dict][key] != value:
                    clean_output_dict[internal_dict][key] = value
                    changed_value[key] = value
    
    config_parser = dict_to_configparser(clean_output_dict)
    config_string = configparser_to_string(config_parser)
    
    new_bench_args = [f"--{k}={v}" for k, v in args_dict.items()]
    
    with open(f"{OPTIONS_FILE_DIR}", "w") as file:
        file.write(config_string)
        
    return config_string, changed_value, new_bench_args

def parse_db_bench_args_to_dict(db_bench_args):
    parsed = {}
    for db_bench_arg in db_bench_args:
        key, value = db_bench_arg.strip().strip("--").split("=")
        parsed[key] = value
    return parsed

def get_initial_options_file():
    initial_options_file_path = os.path.join(DEFAULT_OPTION_FILE_DIR,
                                        INITIAL_OPTIONS_FILE_NAME)
    with open(initial_options_file_path, "r") as f:
        options = f.read()

    reasoning = f"Initial options file: {initial_options_file_path}"

    return options, reasoning


def parse_option_file_to_dict(option_file):
    pat = re.compile("(.*)\s*([#].*)?")
    config = configparser.ConfigParser()
    config.read_string(option_file)
    parsed = {section: dict(config.items(section))
              for section in config.sections()}
    for section_name, section in parsed.items():
        for k, v in section.items():
            m = pat.match(v)
            section[k] = m[1]
    return parsed
