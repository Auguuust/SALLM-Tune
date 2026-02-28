import subprocess
import re
import os


def fio_run(test_type, file_path):
    command = [
        "fio",
        "--name=test",
        "--ioengine=posixaio",
        f"--rw={test_type}",
        "--bs=4k",
        "--numjobs=1",
        "--size=10G",
        "--runtime=60",
        "--time_based"
    ]

    print("[FIO] running fio test now", test_type + "\n")
    proc = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    output = proc.stdout.decode()
    print("[FIO] output :", output)

    parsed_res = parse_fio_output(output, test_type)

    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_path, "a") as file:
        file.write(parsed_res + '\n')

    return parsed_res


def get_fio_result(file_path):
    if (os.path.exists(file_path) and os.path.getsize(file_path) != 0):
        print("[FIO] File exists and is not empty. Reading file.")
        with open(file_path, 'r') as file:
            content = file.read()
        return content

    test_types = ["randwrite", "randread", "read", "write"]
    combined_result = []
    for test_type in test_types:
        fio_result = fio_run(test_type, file_path)
        combined_result.append(fio_result)

    combined_result = "\n".join(combined_result)
    print(f"[FIO] result : \n {combined_result}")
    delete_test_file()
    return combined_result


def parse_fio_output(fio_result, test_type):
    if test_type in ["randwrite", "write"]:
        pattern = re.compile(r'WRITE: bw=(.*?)\s\(.*?\),\s(.*?)\s\(.*?\)')
    elif test_type in ["randread", "read"]:
        pattern = re.compile(r'READ: bw=(.*?)\s\(.*?\),\s(.*?)\s\(.*?\)')
    else:
        print(f"[FIO] Unsupported test type: {test_type}")

    match = pattern.search(fio_result)
    if match:
        values_list = [match.group(1), match.group(2)]
        result_string = f"{test_type} bandwidth is {values_list[0]} ({values_list[1]})"
        print(f"[FIO] result string : {result_string}")
    else:
        print("[FIO] Pattern not found in the fio result.")

    return result_string


def delete_test_file():
    proc = subprocess.run(
        f'rm test.0.0',
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True
    )
