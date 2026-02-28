
import configparser
from utils.filter import key_filter

def dict_to_configparser(dictionary):
    config = configparser.ConfigParser()

    for section, options in dictionary.items():
        config[section] = {}
        for key, value in options.items():
            config[section][key] = str(value)

    return config

def configparser_to_string(config_parser):
    string_representation = ''
    for section in config_parser.sections():
        string_representation += f"[{section}]\n"
        for key, value in config_parser[section].items():
            key = key_filter(key)
            string_representation += f"  {key}={value}\n"
        string_representation += '\n'
    return string_representation