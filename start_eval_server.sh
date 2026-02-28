#!/bin/bash

echo "  Starting RocksDB Eval Server"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

if [ ! -d "utils" ]; then
    echo "Error: utils directory not found"
    exit 1
fi

if [ ! -d "rocksdb" ]; then
    echo "Error: rocksdb directory not found"
    exit 1
fi

if [ ! -d "options_files" ]; then
    echo "Error: options_files directory not found"
    exit 1
fi

if [ ! -f "eval_server.py" ]; then
    echo "Error: eval_server.py not found"
    exit 1
fi

if [ -f "utils/root_cgroup_helper.sh" ]; then
    chmod +x utils/root_cgroup_helper.sh
    echo "✓ root_cgroup_helper.sh is ready"
else
    echo "Warning: root_cgroup_helper.sh not found in utils/"
fi

mkdir -p ft_log
mkdir -p eval_output
mkdir -p eval_dbpath

echo ""
echo "Starting eval server..."
echo "Host: 0.0.0.0"
echo "Port: 8000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 eval_server.py

echo ""
echo "Eval server stopped."

