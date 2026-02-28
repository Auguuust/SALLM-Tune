# SALLM-Tune: Leveraging Fine-Tuned LLMs for Semantic-Awaare Knob Tuning in LSM-Based KV Stores

## Prerequisites

Install dependencies
```bash
apt-get update && apt-get install -y build-essential libgflags-dev libsnappy-dev zlib1g-dev libbz2-dev liblz4-dev libzstd-dev git python3 python3-pip wget fio libaio-dev
```

Download SALLM-Tune and RocksDB repositories
```bash
git clone https://github.com/Auguuust/SALLM-Tune.git
wget https://github.com/facebook/rocksdb/archive/refs/tags/v8.8.1.tar.gz
tar -xzf v8.8.1.tar.gz
```
Copy modified trace_analyzer (proposed by [ELMo-Tune-V2](https://github.com/asu-idi/ELMo-Tune-V2)) to RocksDB
```bash
cp ./SALLM-Tune/trace_analyzer/tools/* ./rocksdb-8.8.1/tools/
cp ./SALLM-Tune/db_bench_dynamic_opts/* ./rocksdb-8.8.1/tools/
```

Build RocksDB with trace_analyzer
```bash
cd ../rocksdb-8.8.1
make -j static_lib db_bench trace_analyzer
```

## Model Fine-Tuning
The ```KnobTuner_Fine-Tune``` directory contains scripts and configurations for fine-tuning large language models (LLMs) to predict optimal knob settings based on workload characteristics. This program runs on another server with a GPU and communicates with the evaluation server running RocksDB via Flask. If you need to do fine-tuning on your local machine, make sure your machine is equipped with the right GPU resources and refactor the remote training code to local training.

#### On the evaluation server:
1. Install Flask
```bash
pip install Flask
```
2. Modify the ```eval_server_config.py```.
3. Make sure this server is reachable from the training server.
4. Run the evaluation server
```bash
./start_eval_server.sh
```

#### On the training server:
1. Install Unslash
```bash
pip install unsloth
```
2. Download a pre-trained LLM (e.g., qwen3-8b-instruct) and place it in the ```models/``` directory.
3. Run the trarining program
```bash
python train_server_full.py
```

## Run SALLM-Tune

#### On GPU-enabled server:
1. Install and set up Ollama following the instructions at https://github.com/ollama/ollama.
2. Run the fine-tuned model following the instructions at https://github.com/ollama/ollama?tab=readme-ov-file#customize-a-model.

#### On the evaluation server:
1. Modify the ```utils/constants.py``` file to set the api_url to point to the Ollama server.
2. If you don't have a fine-tuned model, you can use other models by modifying the ```utils/constants.py```:
   - Set the ```env_LLM_MODEL``` variable to the desired model name.
   - Set the ```env_<API_PROVIDER>_API_KEY``` and ```env_<API_PROVIDER>_BASE_URL``` variables to the appropriate values for the chosen API provider.
3. You can run ```run_all_tests.py``` to evaluate SALLM-Tune on all workloads and resource constraints.
4. Or you can run specific workload with specific resource constraint, e.g.:
```bash
python SALLM-Tune.py --workload fillrandom --llm_model qwen3-8b-ft-0829 --limit_list 244
```