import os
import torch
os.environ["WANDB_API_KEY"] = "<your wandb key>"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"
print(f"Available GPUs: {torch.cuda.device_count()}")
import unsloth
import json
import re
import sys
import argparse
import glob
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from datasets import Dataset
from transformers import TrainingArguments, TrainerCallback
from trl import SFTTrainer, SFTConfig, GRPOConfig, GRPOTrainer
import wandb
import configparser
import requests
import time
from unsloth import FastLanguageModel, PatchFastRL

try:
    from generation_config import get_generation_config
    generation_config = get_generation_config()
    print(f"[CONFIG] Loaded generation config: {generation_config}")
except ImportError:
    generation_config = {
        "temperature": 0.6,
    }
    print(f"[CONFIG] Using default generation config: {generation_config}")

PatchFastRL("GRPO", FastLanguageModel)

EVAL_SERVER_HOST = "101.76.209.85"
EVAL_SERVER_PORT = 51022
EVAL_SERVER_URL = f"http://{EVAL_SERVER_HOST}:{EVAL_SERVER_PORT}"

max_seq_length = 8192
lora_rank = 64
model_name = "./model/qwen3-8b"

class OutputDirectoryManager:
    
    def __init__(self, model_name: str):
        self.model_base_name = model_name.split('/')[-1]
        self.timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M')
        self.root_dir = f"{self.model_base_name}_{self.timestamp}"
        
        self.create_directories()
        
    def create_directories(self):
        directories = [
            self.root_dir,
            self.logs_dir,
            self.responses_dir,
            self.checkpoints_dir,
            self.debug_dir,
            self.models_dir
        ]
        
        for dir_path in directories:
            os.makedirs(dir_path, exist_ok=True)
    
    @property
    def logs_dir(self):
        return os.path.join(self.root_dir, "logs")
    
    @property
    def responses_dir(self):
        return os.path.join(self.root_dir, "responses")
    
    @property
    def checkpoints_dir(self):
        return os.path.join(self.root_dir, "checkpoints")
    
    @property
    def debug_dir(self):
        return os.path.join(self.root_dir, "debug")
    
    @property
    def models_dir(self):
        return os.path.join(self.root_dir, "models")
    
    def get_log_path(self, log_type: str = "main"):
        return os.path.join(self.logs_dir, f"{log_type}_{self.timestamp}.log")
    
    def get_response_path(self, response_type: str):
        return os.path.join(self.responses_dir, f"{response_type}_{self.timestamp}.json")
    
    def get_checkpoint_dir(self, checkpoint_type: str):
        return os.path.join(self.checkpoints_dir, checkpoint_type)
    
    def get_model_dir(self, model_type: str):
        return os.path.join(self.models_dir, model_type)
    
    def get_debug_path(self, debug_type: str):
        return os.path.join(self.debug_dir, f"{debug_type}_{self.timestamp}.json")

output_manager = OutputDirectoryManager(model_name)
log_name = output_manager.get_log_path("main")


SYSTEM_CONFIGS = [
    "111",
    "122",
    "222",
    "888",

]

SUPPORTED_WORKLOADS = ["readrandom", "fillrandom", "readrandomwriterandom", "readwhilewriting", "mixgraph"]

def parse_args():
    parser = argparse.ArgumentParser(description="RocksDB Configuration Optimization Training")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume training from"
    )
    parser.add_argument(
        "--auto_resume",
        action="store_true",
        default=False,
        help="Automatically resume from the latest checkpoint in the output directory"
    )
    parser.add_argument(
        "--skip_knowledge_pretraining",
        action="store_true",
        default=False,
        help="Skip knowledge background pretraining phase"
    )
    parser.add_argument(
        "--skip_structure_pretraining",
        action="store_true",
        default=False,
        help="Skip structure pretraining phase and go directly to GRPO training"
    )
    parser.add_argument(
        "--config_index",
        type=int,
        default=0,
        help="Start training from specific config index (0-based)"
    )
    parser.add_argument(
        "--structure_model_path",
        type=str,
        default=None,
        help="Path to structure pretrained model (if skipping structure pretraining)"
    )
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=SUPPORTED_WORKLOADS,
        help="List of workloads to train on (default: all supported workloads)"
    )
    parser.add_argument(
        "--start_workload_index",
        type=int,
        default=0,
        help="Start training from specific workload index (0-based)"
    )
    parser.add_argument(
        "--load_model_only",
        type=str,
        default=None,
        help="Path to model checkpoint to load (only model weights, no training state)"
    )
    
    return parser.parse_args()

args = parse_args()

class CheckpointManager:
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.checkpoint_pattern = os.path.join(output_dir, "checkpoint-*")
        
    def get_all_checkpoints(self) -> List[str]:
        checkpoints = glob.glob(self.checkpoint_pattern)
        checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
        return checkpoints
    
    def get_latest_checkpoint(self) -> Optional[str]:
        checkpoints = self.get_all_checkpoints()
        return checkpoints[-1] if checkpoints else None
    
    def checkpoint_exists(self, checkpoint_path: str) -> bool:
        """检查检查点是否存在且有效"""
        if not os.path.exists(checkpoint_path):
            return False
        
        required_files = [
            "pytorch_model.bin",
            "trainer_state.json",
            "optimizer.pt",
            "scheduler.pt"
        ]
        
        for file in required_files:
            if not os.path.exists(os.path.join(checkpoint_path, file)):
                print(f"[CHECKPOINT] Missing required file: {file}")
                return False
        
        return True
    
    def load_trainer_state(self, checkpoint_path: str) -> Dict:
        trainer_state_path = os.path.join(checkpoint_path, "trainer_state.json")
        try:
            with open(trainer_state_path, 'r') as f:
                trainer_state = json.load(f)
            return trainer_state
        except Exception as e:
            print(f"[CHECKPOINT] Error loading trainer state: {e}")
            return {}
    
    def save_config_manager_state(self, checkpoint_path: str, config_manager: 'SystemConfigManager'):
        state = {
            "current_index": config_manager.current_index,
            "current_workload_index": config_manager.current_workload_index,
            "improvement_count": config_manager.improvement_count,
            "epoch_count": config_manager.epoch_count,
            "max_improvements": config_manager.max_improvements,
            "max_epochs": config_manager.max_epochs,
            "workloads": config_manager.workloads
        }
        
        state_path = os.path.join(checkpoint_path, "config_manager_state.json")
        try:
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)
            print(f"[CHECKPOINT] Saved config manager state to {state_path}")
        except Exception as e:
            print(f"[CHECKPOINT] Error saving config manager state: {e}")
    
    def load_config_manager_state(self, checkpoint_path: str) -> Optional[Dict]:
        state_path = os.path.join(checkpoint_path, "config_manager_state.json")
        try:
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            print(f"[CHECKPOINT] Error loading config manager state: {e}")
            return None

class SystemConfigManager:
    
    def __init__(self, config_list: List[str], workloads: List[str] = None):
        self.config_list = config_list
        self.workloads = workloads or SUPPORTED_WORKLOADS
        self.current_index = 0
        self.current_workload_index = 0
        self.improvement_count = 0
        self.epoch_count = 0
        self.max_improvements = 3
        self.max_epochs = 20
        
    def restore_state(self, state: Dict):
        self.current_index = state.get("current_index", 0)
        self.current_workload_index = state.get("current_workload_index", 0)
        self.improvement_count = state.get("improvement_count", 0)
        self.epoch_count = state.get("epoch_count", 0)
        self.max_improvements = state.get("max_improvements", 3)
        self.max_epochs = state.get("max_epochs", 20)
        self.workloads = state.get("workloads", SUPPORTED_WORKLOADS)
        
        print(f"[CONFIG] Restored config manager state:")
        print(f"  Current config index: {self.current_index}")
        print(f"  Current workload index: {self.current_workload_index}")
        print(f"  Current workload: {self.get_current_workload()}")
        print(f"  Improvement count: {self.improvement_count}")
        print(f"  Epoch count: {self.epoch_count}")
    
    def get_state(self) -> Dict:
        return {
            "current_index": self.current_index,
            "current_workload_index": self.current_workload_index,
            "improvement_count": self.improvement_count,
            "epoch_count": self.epoch_count,
            "max_improvements": self.max_improvements,
            "max_epochs": self.max_epochs,
            "workloads": self.workloads
        }
    
    def get_current_config(self) -> Dict[str, int]:
        if self.current_index < len(self.config_list):
            return self.parse_system_config(self.config_list[self.current_index])
        return {}
    
    def get_current_config_str(self) -> str:
        if self.current_index < len(self.config_list):
            return self.config_list[self.current_index]
        return ""
    
    def get_current_workload(self) -> str:
        if self.current_workload_index < len(self.workloads):
            return self.workloads[self.current_workload_index]
        return ""
    
    def parse_system_config(self, config_str: str) -> Dict[str, int]:
        config = {}
        try:
            if len(config_str) != 3:
                raise ValueError(f"Config string must be 3 digits, got: {config_str}")
            
            cpu_cores = int(config_str[0])
            swap_memory_gb = int(config_str[1])
            memory_gb = int(config_str[2])
            
            config = {
                "cpu_cores": cpu_cores,
                "swap_memory_gb": swap_memory_gb,
                "memory_gb": memory_gb
            }
        except Exception as e:
            print(f"[CONFIG] Error parsing config string: {e}")
        return config
    
    def should_switch_config(self) -> bool:
        improvement_condition = self.improvement_count >= self.max_improvements
        epoch_condition = self.epoch_count >= self.max_epochs
        should_switch = improvement_condition or epoch_condition
        
        print(f"[CONFIG MANAGER] Should switch check: improvements={self.improvement_count}/{self.max_improvements} ({improvement_condition}), epochs={self.epoch_count}/{self.max_epochs} ({epoch_condition}), result={should_switch}")
        if hasattr(log_file, 'write'):
            log_file.write(f"[CONFIG MANAGER] Should switch check: improvements={self.improvement_count}/{self.max_improvements} ({improvement_condition}), epochs={self.epoch_count}/{self.max_epochs} ({epoch_condition}), result={should_switch}\n")
        
        return should_switch
    
    def should_switch_workload(self) -> bool:
        return self.current_index >= len(self.config_list)
    
    def switch_to_next_config(self) -> bool:
        if self.current_index < len(self.config_list) - 1:
            self.current_index += 1
            self.improvement_count = 0
            self.epoch_count = 0
            return True
        return False
    
    def switch_to_next_workload(self) -> bool:
        if self.current_workload_index < len(self.workloads) - 1:
            self.current_workload_index += 1
            self.current_index = 0
            self.improvement_count = 0
            self.epoch_count = 0
            return True
        return False
    
    def record_improvement(self):
        old_count = self.improvement_count
        self.improvement_count += 1
        print(f"[CONFIG MANAGER] Improvement recorded: {old_count} -> {self.improvement_count}")
        if hasattr(log_file, 'write'):
            log_file.write(f"[CONFIG MANAGER] Improvement recorded: {old_count} -> {self.improvement_count}\n")
    
    def record_epoch(self):
        old_count = self.epoch_count
        self.epoch_count += 1
        print(f"[CONFIG MANAGER] Epoch recorded: {old_count} -> {self.epoch_count}")
        if hasattr(log_file, 'write'):
            log_file.write(f"[CONFIG MANAGER] Epoch recorded: {old_count} -> {self.epoch_count}\n")
    
    def is_complete(self) -> bool:
        return (self.current_workload_index >= len(self.workloads) - 1 and 
                self.current_index >= len(self.config_list) - 1 and
                self.should_switch_config())
    
    def get_progress(self) -> str:
        return (f"Workload {self.current_workload_index + 1}/{len(self.workloads)} "
                f"({self.get_current_workload()}), "
                f"Config {self.current_index + 1}/{len(self.config_list)}, "
                f"Improvements: {self.improvement_count}/{self.max_improvements}, "
                f"Epochs: {self.epoch_count}/{self.max_epochs}")

# 检查点加载函数
def load_model_from_checkpoint(checkpoint_path: str, model, tokenizer) -> Tuple[Any, Any]:
    print(f"[CHECKPOINT] Loading model from {checkpoint_path}")
    
    try:
        model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            print(f"[CHECKPOINT] Successfully loaded model weights")
        else:
            print(f"[CHECKPOINT] Model weights not found, using current model")
        
        tokenizer_files = ["tokenizer.json", "tokenizer_config.json"]
        tokenizer_path = checkpoint_path
        
        for file in tokenizer_files:
            if os.path.exists(os.path.join(tokenizer_path, file)):
                print(f"[CHECKPOINT] Found tokenizer file: {file}")
                break
        
        return model, tokenizer
        
    except Exception as e:
        print(f"[CHECKPOINT] Error loading model from checkpoint: {e}")
        print(f"[CHECKPOINT] Continuing with current model")
        return model, tokenizer

def create_grpo_trainer_from_checkpoint(
    model, tokenizer, reward_func, args, train_dataset, 
    callbacks, checkpoint_path: str = None
) -> GRPOTrainer:
    
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_func],
        args=args,
        train_dataset=train_dataset,
        callbacks=callbacks
    )
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[CHECKPOINT] Loading trainer state from {checkpoint_path}")
        try:
            trainer_state_path = os.path.join(checkpoint_path, "trainer_state.json")
            if os.path.exists(trainer_state_path):
                with open(trainer_state_path, 'r') as f:
                    trainer_state = json.load(f)
                
                trainer.state.global_step = trainer_state.get("global_step", 0)
                trainer.state.epoch = trainer_state.get("epoch", 0)
                trainer.state.max_steps = trainer_state.get("max_steps", args.max_steps)
                trainer.state.num_train_epochs = trainer_state.get("num_train_epochs", args.num_train_epochs)
                trainer.state.log_history = trainer_state.get("log_history", [])
                trainer.state.best_metric = trainer_state.get("best_metric", None)
                trainer.state.best_model_checkpoint = trainer_state.get("best_model_checkpoint", None)
                
                print(f"[CHECKPOINT] Restored trainer state:")
                print(f"  Global step: {trainer.state.global_step}")
                print(f"  Epoch: {trainer.state.epoch}")
                print(f"  Max steps: {trainer.state.max_steps}")
            
            optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
            if os.path.exists(optimizer_path) and trainer.optimizer is not None:
                optimizer_state = torch.load(optimizer_path, map_location="cpu")
                trainer.optimizer.load_state_dict(optimizer_state)
                print(f"[CHECKPOINT] Restored optimizer state")
            
            scheduler_path = os.path.join(checkpoint_path, "scheduler.pt")
            if os.path.exists(scheduler_path) and trainer.lr_scheduler is not None:
                scheduler_state = torch.load(scheduler_path, map_location="cpu")
                trainer.lr_scheduler.load_state_dict(scheduler_state)
                print(f"[CHECKPOINT] Restored scheduler state")
            
        except Exception as e:
            print(f"[CHECKPOINT] Error loading trainer state: {e}")
            print(f"[CHECKPOINT] Starting training from beginning")
    
    return trainer

def load_model_and_tokenizer(checkpoint_path: str = None):
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[CHECKPOINT] Loading model from checkpoint: {checkpoint_path}")
        
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=checkpoint_path,
                max_seq_length=max_seq_length,
                load_in_4bit=False,
                full_finetuning=False,
                fast_inference=False,
                max_lora_rank=lora_rank,
                low_cpu_mem_usage=True,
                gpu_memory_utilization=0.7,
                device_map={'':torch.cuda.current_device()},
            )
            print(f"[CHECKPOINT] Successfully loaded model from checkpoint")
        except Exception as e:
            print(f"[CHECKPOINT] Failed to load from checkpoint: {e}")
            print(f"[CHECKPOINT] Loading base model instead")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                load_in_4bit=False,
                full_finetuning=False,
                fast_inference=False,
                max_lora_rank=lora_rank,
                low_cpu_mem_usage=True,
                gpu_memory_utilization=0.7,
                device_map={'':torch.cuda.current_device()},
            )
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=False,
            full_finetuning=False,
            fast_inference=False,
            max_lora_rank=lora_rank,
            low_cpu_mem_usage=True,
            gpu_memory_utilization=0.7,
            device_map={'':torch.cuda.current_device()},
        )
    
    return model, tokenizer

class ResponseSaver:
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.responses = []
        
    def save_responses(self, step: int, config_str: str, prompts: List, completions: List, 
                      rewards: List, improvements: List, eval_scores: List, 
                      configs: List, benchmark_results: List):
        batch_data = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "completions": [],
            "param_counts": [],
            "param_count_distribution": {}
        }
        
        param_counts = []
        for i in range(len(completions) if completions else 0):
            completion = completions[i] if i < len(completions) else ""
            
            if isinstance(completion, dict):
                completion_content = completion.get("content", "")
            elif isinstance(completion, str):
                completion_content = completion
            else:
                completion_content = str(completion)
            
            batch_data["completions"].append(completion_content)
            
            try:
                config_content = extract_config_from_text(completion_content)
                if config_content:
                    config_dict = parse_partial_config(config_content)
                    param_count = len(config_dict)
                else:
                    param_count = 0
                param_counts.append(param_count)
            except:
                param_counts.append(0)
        
        batch_data["param_counts"] = param_counts
        
        from collections import Counter
        param_distribution = Counter(param_counts)
        batch_data["param_count_distribution"] = dict(param_distribution)
        
        self.responses.append(batch_data)
        
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(self.responses, f, ensure_ascii=False, indent=2)
            
            avg_param_count = sum(param_counts) / len(param_counts) if param_counts else 0
            over_10_count = sum(1 for count in param_counts if count > 10)
            optimal_range_count = sum(1 for count in param_counts if 5 <= count <= 8)
            
            print(f"[RESPONSE SAVER] Step {step}: Saved {len(batch_data['completions'])} completions")
            print(f"[PARAM_STATS] Avg params: {avg_param_count:.1f}, Over 10: {over_10_count}/{len(param_counts)}, Optimal (5-8): {optimal_range_count}/{len(param_counts)}")
            print(f"[PARAM_DISTRIBUTION] {dict(param_distribution)}")
            
            log_file.write(f"[RESPONSE SAVER] Step {step}: Saved {len(batch_data['completions'])} completions\n")
            log_file.write(f"[PARAM_STATS] Avg params: {avg_param_count:.1f}, Over 10: {over_10_count}/{len(param_counts)}, Optimal (5-8): {optimal_range_count}/{len(param_counts)}\n")
            log_file.write(f"[PARAM_DISTRIBUTION] {dict(param_distribution)}\n")
            
        except Exception as e:
            print(f"[RESPONSE SAVER] Error saving responses: {e}")
            log_file.write(f"[RESPONSE SAVER] Error saving responses: {e}\n")
    
    def get_stats(self) -> Dict:
        total_completions = sum(len(batch.get("completions", [])) for batch in self.responses)
        all_param_counts = []
        
        for batch in self.responses:
            all_param_counts.extend(batch.get("param_counts", []))
        
        param_stats = {}
        if all_param_counts:
            from collections import Counter
            param_distribution = Counter(all_param_counts)
            
            param_stats = {
                "avg_param_count": sum(all_param_counts) / len(all_param_counts),
                "min_param_count": min(all_param_counts),
                "max_param_count": max(all_param_counts),
                "param_distribution": dict(param_distribution),
                "over_10_count": sum(1 for count in all_param_counts if count > 10),
                "over_10_percentage": (sum(1 for count in all_param_counts if count > 10) / len(all_param_counts)) * 100,
                "optimal_range_count": sum(1 for count in all_param_counts if 5 <= count <= 8),
                "optimal_range_percentage": (sum(1 for count in all_param_counts if 5 <= count <= 8) / len(all_param_counts)) * 100,
                "within_limit_count": sum(1 for count in all_param_counts if count <= 10),
                "within_limit_percentage": (sum(1 for count in all_param_counts if count <= 10) / len(all_param_counts)) * 100
            }
        
        return {
            "total_batches": len(self.responses),
            "total_completions": total_completions,
            "parameter_statistics": param_stats
        }
    
    def save_stats_summary(self):
        stats = self.get_stats()
        summary_path = self.file_path.replace('.jsonl', '_summary.json')
        
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            print(f"[RESPONSE SAVER] Saved summary to {summary_path}")
        except Exception as e:
            print(f"[RESPONSE SAVER] Error saving summary: {e}")

class EvalServerClient:
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.proxies = {
            'http': None,
            'https': None
        }
        self.timeout = 600
    
    def ping(self) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/ping", timeout=60)
            return response.status_code == 200
        except Exception as e:
            print(f"[EVAL CLIENT] Ping failed: {e}")
            return False
    
    def apply_system_config(self, config_str: str) -> bool:
        try:
            config = config_manager.parse_system_config(config_str)
            
            print(f"[EVAL CLIENT] Parsed config {config_str} → {config}")
            
            data = {
                "config": config,
                "config_str": config_str
            }
            response = self.session.post(f"{self.base_url}/apply_config", json=data, timeout=600)
            
            if response.status_code == 200:
                result = response.json()
                print(f"[EVAL CLIENT] Config application result: {result}")
                return result.get("success", False)
            else:
                print(f"[EVAL CLIENT] Config application failed with status: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"[EVAL CLIENT] Apply config failed: {e}")
            return False
    
    def run_evaluation(self, config_str: str, new_options: str = None, is_init: bool = False, workload: str = None) -> Dict:
        try:
            data = {
                "config_str": config_str,
                "new_options": new_options,
                "is_init": is_init,
                "workload": workload
            }
            response = self.session.post(f"{self.base_url}/evaluate", json=data, timeout=self.timeout)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"[EVAL CLIENT] Evaluation failed with status {response.status_code}")
                return {"error": "evaluation_failed", "benchmark_results": {"ops_per_sec": 0}}
        except Exception as e:
            print(f"[EVAL CLIENT] Evaluation request failed: {e}")
            return {"error": str(e), "benchmark_results": {"ops_per_sec": 0}}

config_manager = SystemConfigManager(SYSTEM_CONFIGS, args.workloads)
eval_client = EvalServerClient(EVAL_SERVER_URL)

def extract_config_from_text(text: str) -> str:
    try:
        config_section = re.search(r'<config>(.*?)</config>', text, re.DOTALL)
        if config_section:
            return config_section.group(1).strip()
        return ""
    except Exception as e:
        print(f"Error extracting config: {e}")
        return ""

def extract_reasoning_from_text(text: str) -> str:
    try:
        reasoning_section = re.search(r'<reasoning>(.*?)</reasoning>', text, re.DOTALL)
        if reasoning_section:
            return reasoning_section.group(1).strip()
        return ""
    except Exception as e:
        print(f"Error extracting reasoning: {e}")
        return ""

def create_structured_response(reasoning: str, config: str) -> str:
    return f"<reasoning>\n{reasoning}\n</reasoning>\n\n<config>\n{config}\n</config>"

def wait_for_eval_server(max_retries: int = 10, retry_delay: int = 5) -> bool:
    print(f"[TRAIN SERVER] Waiting for eval server at {EVAL_SERVER_URL}...")
    log_file.write(f"[TRAIN SERVER] Waiting for eval server at {EVAL_SERVER_URL}...\n")
    
    for retry in range(max_retries):
        if eval_client.ping():
            print(f"[TRAIN SERVER] Eval server is ready!")
            log_file.write(f"[TRAIN SERVER] Eval server is ready!\n")
            return True
        else:
            print(f"[TRAIN SERVER] Eval server not ready, retrying in {retry_delay}s... ({retry+1}/{max_retries})")
            log_file.write(f"[TRAIN SERVER] Eval server not ready, retrying in {retry_delay}s... ({retry+1}/{max_retries})\n")
            time.sleep(retry_delay)
    
    print(f"[TRAIN SERVER] Failed to connect to eval server after {max_retries} retries")
    log_file.write(f"[TRAIN SERVER] Failed to connect to eval server after {max_retries} retries\n")
    return False

def generate_synthetic_examples() -> List[Dict[str, str]]:
    examples = []
    
    base_params = {
        "write_buffer_size": [67108864, 134217728, 268435456, 536870912],
        "max_write_buffer_number": [2, 3, 4, 6],
        "min_write_buffer_number_to_merge": [1, 2, 3],
        "compression": ["kNoCompression", "kLZ4Compression", "kZSTD", "kSnappyCompression"],
        "target_file_size_base": [33554432, 67108864, 134217728],
        "max_bytes_for_level_base": [268435456, 536870912, 1073741824],
        "max_background_jobs": [4, 6, 8, 12, 16],
        "max_background_compactions": [2, 4, 6, 8],
        "max_background_flushes": [2, 3, 4, 6],
        "db_write_buffer_size": [268435456, 536870912, 1073741824],
        "max_open_files": [500, 1000, 2000, 5000],
        "level0_file_num_compaction_trigger": [2, 4, 8],
        "level0_slowdown_writes_trigger": [8, 12, 20],
        "level0_stop_writes_trigger": [12, 24, 36],
        "block_size": [4096, 8192, 16384, 32768],
        "cache_size": [134217728, 268435456, 536870912],
        "bloom_filter_bits_per_key": [8, 10, 12, 15],
        "memtable_factory": ["skip_list", "hash_skip_list", "hash_link_list"],
        "write_buffer_manager_size": [536870912, 1073741824, 2147483648]
    }
    
    system_scenarios = [
        {
            "cpu_cores": 4, "memory": "16GB", "storage": "SATA SSD", 
            "workload": "Light mixed workload", "focus": "memory efficiency"
        },
        {
            "cpu_cores": 8, "memory": "32GB", "storage": "NVMe SSD", 
            "workload": "Read-heavy operations", "focus": "read performance"
        },
        {
            "cpu_cores": 16, "memory": "64GB", "storage": "NVMe SSD", 
            "workload": "Write-intensive workload", "focus": "write throughput"
        },
        {
            "cpu_cores": 32, "memory": "128GB", "storage": "High-speed NVMe", 
            "workload": "High-concurrency mixed operations", "focus": "overall performance"
        },
        {
            "cpu_cores": 6, "memory": "24GB", "storage": "SATA SSD", 
            "workload": "Batch processing", "focus": "compaction efficiency"
        },
        {
            "cpu_cores": 12, "memory": "48GB", "storage": "NVMe SSD", 
            "workload": "Real-time analytics", "focus": "latency optimization"
        }
    ]
    
    optimization_scenarios = [
        {
            "problem": "high memory usage",
            "strategy": ["reduce write buffers", "optimize cache", "use compression"],
            "params": ["write_buffer_size", "db_write_buffer_size", "cache_size", "compression"]
        },
        {
            "problem": "slow write performance", 
            "strategy": ["increase write buffers", "optimize background jobs", "adjust compaction"],
            "params": ["write_buffer_size", "max_write_buffer_number", "max_background_jobs", "level0_file_num_compaction_trigger"]
        },
        {
            "problem": "frequent compactions",
            "strategy": ["adjust level triggers", "increase file sizes", "optimize background threads"],
            "params": ["level0_file_num_compaction_trigger", "target_file_size_base", "max_background_compactions"]
        },
        {
            "problem": "high read latency",
            "strategy": ["optimize block cache", "improve bloom filters", "adjust block size"],
            "params": ["cache_size", "bloom_filter_bits_per_key", "block_size", "max_open_files"]
        },
        {
            "problem": "write stalls",
            "strategy": ["adjust slowdown triggers", "increase background jobs", "optimize write buffers"],
            "params": ["level0_slowdown_writes_trigger", "level0_stop_writes_trigger", "max_background_flushes"]
        }
    ]
    
    import random
    random.seed(42)
    
    for i in range(20):
        scenario = random.choice(system_scenarios)
        optimization = random.choice(optimization_scenarios)
        
        prompt = f"""You are an expert database administrator tasked with optimizing RocksDB configuration for better performance.

                    Current system information:
                    - CPU: {scenario['cpu_cores']} cores
                    - Memory: {scenario['memory']} RAM  
                    - Storage: {scenario['storage']}
                    - Workload: {scenario['workload']}

                    Current issue: {optimization['problem']}
                    Current benchmark results: {random.randint(20000, 80000)} ops/sec

                    Please analyze the configuration and provide ONLY the specific parameters that need to be modified to address the {optimization['problem']} issue. 
                    CRITICAL REQUIREMENTS: 
                    1. Return ONLY the parameters that need to be changed, not the complete configuration
                    2. NEVER exceed 10 parameters - this is a hard limit that must be strictly followed
                    3. Ideally use 5-8 parameters for optimal performance
                    4. Each additional parameter beyond 10 will result in severe penalties
                    5. Focus on {scenario['focus']}
                    6. Quality over quantity - choose the most impactful parameters rather than many parameters

                    Please provide your reasoning and the optimized parameters."""

        available_params = optimization['params'] + random.sample(
            [p for p in base_params.keys() if p not in optimization['params']], 
            min(6, len([p for p in base_params.keys() if p not in optimization['params']]))
        )
        param_ranges = [(5, 8, 0.7), (3, 4, 0.15), (9, 10, 0.15)]
        rand = random.random()
        if rand < 0.7:
            num_params = random.randint(5, 8)
        elif rand < 0.85:
            num_params = random.randint(3, 4)
        else:
            num_params = random.randint(9, 10)
        
        num_params = min(num_params, len(available_params), 10)
        selected_params = random.sample(available_params, num_params)
        
        reasoning_parts = []
        
        if "memory" in optimization['problem']:
            reasoning_parts.append("Memory optimization is critical for this configuration.")
            reasoning_parts.append("Reducing buffer sizes will decrease memory footprint.")
        elif "write" in optimization['problem']:
            reasoning_parts.append("Write performance optimization requires careful buffer tuning.")
            reasoning_parts.append("Increasing parallelism will improve write throughput.")
        elif "compaction" in optimization['problem']:
            reasoning_parts.append("Compaction optimization focuses on reducing unnecessary work.")
            reasoning_parts.append("Adjusting triggers will balance performance and space amplification.")
        elif "read" in optimization['problem']:
            reasoning_parts.append("Read performance depends heavily on caching and indexing.")
            reasoning_parts.append("Optimizing cache and bloom filters will reduce read latency.")
        elif "stall" in optimization['problem']:
            reasoning_parts.append("Write stalls occur when the system cannot keep up with writes.")
            reasoning_parts.append("Adjusting triggers and background jobs will prevent stalls.")
        
        for param in selected_params:
            if "write_buffer" in param:
                reasoning_parts.append(f"Adjusting {param} will optimize memory usage for write operations.")
            elif "background" in param:
                reasoning_parts.append(f"Tuning {param} will improve parallelism for background operations.")
            elif "compression" in param:
                reasoning_parts.append(f"Optimizing {param} will balance CPU usage and storage efficiency.")
            elif "cache" in param or "block" in param:
                reasoning_parts.append(f"Modifying {param} will enhance read performance and memory utilization.")
            elif "level0" in param:
                reasoning_parts.append(f"Adjusting {param} will control compaction triggering behavior.")
        
        reasoning = " ".join(reasoning_parts[:6])
        
        config_lines = []
        for param in selected_params:
            if param in base_params:
                value = random.choice(base_params[param])
                config_lines.append(f"{param}={value}")
        
        config = "\n".join(config_lines)
        
        examples.append({
            "prompt": prompt,
            "response": create_structured_response(reasoning, config)
        })
    
    return examples

def prepare_structure_training_dataset(debug_file_path: str = None) -> Dataset:
    print("Preparing structure training dataset with enhanced diversity...")
    log_file.write("Preparing structure training dataset with enhanced diversity...\n")
    
    synthetic_examples = generate_synthetic_examples()
    print(f"Generated {len(synthetic_examples)} diverse synthetic examples")
    log_file.write(f"Generated {len(synthetic_examples)} diverse synthetic examples\n")
    
    augmented_examples = []
    import random
    random.seed(12345)
    
    for example in synthetic_examples:
        augmented_examples.append(example)
        
        original_prompt = example['prompt']
        
        if "CPU: " in original_prompt:
            modified_prompt = original_prompt
            for old_cpu, new_cpu in [("4 cores", "6 cores"), ("8 cores", "10 cores"), ("16 cores", "20 cores")]:
                if old_cpu in modified_prompt:
                    modified_prompt = modified_prompt.replace(old_cpu, new_cpu)
                    break
            
            if modified_prompt != original_prompt:
                augmented_examples.append({
                    "prompt": modified_prompt,
                    "response": example['response']
                })
        
        if "Memory: " in original_prompt:
            modified_prompt = original_prompt
            for old_mem, new_mem in [("16GB", "20GB"), ("32GB", "40GB"), ("64GB", "80GB")]:
                if old_mem in modified_prompt:
                    modified_prompt = modified_prompt.replace(old_mem, new_mem)
                    break
            
            if modified_prompt != original_prompt:
                augmented_examples.append({
                    "prompt": modified_prompt,
                    "response": example['response']
                })
    
    format_emphasis_examples = [
        {
            "prompt": """You are optimizing RocksDB for a specific workload. 
            IMPORTANT: Only return the parameters that need to be changed, not the complete configuration.
            Maximum 10 parameters per response.
            
            Current issue: Memory usage too high
            System: 8 cores, 32GB RAM
            
            Provide reasoning and ONLY the parameters to modify.""",
            "response": create_structured_response(
                "High memory usage requires reducing buffer sizes and optimizing cache allocation. Focus on write buffers and cache settings.",
                "write_buffer_size=134217728\ndb_write_buffer_size=268435456\ncache_size=268435456"
            )
        },
        {
            "prompt": """RocksDB performance optimization task.
            CONSTRAINT: Return ONLY modified parameters (max 10).
            
            Issue: Write performance degradation
            Hardware: 16 cores, 64GB RAM, NVMe SSD
            
            Analyze and provide specific parameter modifications only.""",
            "response": create_structured_response(
                "Write performance can be improved by increasing parallelism and optimizing buffer management. Background job tuning is essential.",
                "max_background_jobs=12\nmax_write_buffer_number=4\nlevel0_file_num_compaction_trigger=4\nmax_background_flushes=4"
            )
        }
    ]
    
    all_examples = augmented_examples + format_emphasis_examples
    print(f"Total examples after augmentation: {len(all_examples)}")
    log_file.write(f"Total examples after augmentation: {len(all_examples)}\n")
    
    if debug_file_path is None:
        debug_file_path = f"./debug/structure_training_data_{datetime.now().strftime('%Y-%m-%d-%H-%M')}.json"
    
    if not os.path.exists(f"./debug/"):
        os.makedirs(f"./debug/")
    
    debug_data = {
        "metadata": {
            "total_examples": len(all_examples),
            "synthetic_examples": len(synthetic_examples), 
            "augmented_examples": len(augmented_examples),
            "format_emphasis_examples": len(format_emphasis_examples),
            "generation_timestamp": datetime.now().isoformat()
        },
        "examples": all_examples
    }
    
    try:
        with open(debug_file_path, 'w', encoding='utf-8') as f:
            json.dump(debug_data, f, ensure_ascii=False, indent=2)
        print(f"[DEBUG] Training data saved to {debug_file_path}")
        log_file.write(f"[DEBUG] Training data saved to {debug_file_path}\n")
    except Exception as e:
        print(f"[DEBUG] Error saving training data: {e}")
        log_file.write(f"[DEBUG] Error saving training data: {e}\n")
    
    formatted_examples = []
    for example in all_examples:
        formatted_examples.append({
            "text": f"<|im_start|>user\n{example['prompt']}<|im_end|>\n<|im_start|>assistant\n{example['response']}<|im_end|>"
        })
    
    random.shuffle(formatted_examples)
    
    dataset = Dataset.from_list(formatted_examples)
    print(f"Created enhanced structure dataset with {len(dataset)} examples")
    log_file.write(f"Created enhanced structure dataset with {len(dataset)} examples\n")
    
    return dataset

def validate_output_format(text: str) -> Dict[str, bool]:
    has_reasoning = bool(re.search(r'<reasoning>.*?</reasoning>', text, re.DOTALL))
    has_config = bool(re.search(r'<config>.*?</config>', text, re.DOTALL))
    
    reasoning_content = extract_reasoning_from_text(text)
    config_content = extract_config_from_text(text)
    
    param_count_valid = True
    param_count = 0
    if config_content:
        config_dict = parse_partial_config(config_content)
        param_count = len(config_dict)
        param_count_valid = 1 <= param_count <= 10
    
    is_partial_config = True
    if config_content:
        standard_params = [
            "write_buffer_size", "max_write_buffer_number", "compression",
            "target_file_size_base", "max_bytes_for_level_base", "max_background_jobs",
            "max_background_compactions", "max_background_flushes", "db_write_buffer_size",
            "max_open_files", "level0_file_num_compaction_trigger"
        ]
        config_lines = [line.strip() for line in config_content.split('\n') if line.strip() and '=' in line]
        config_param_names = []
        for line in config_lines:
            if '=' in line:
                param_name = line.split('=')[0].strip()
                config_param_names.append(param_name)
        
        standard_param_count = sum(1 for param in config_param_names if param in standard_params)
        if standard_param_count > 10:
            is_partial_config = False
    
    return {
        "has_reasoning_tags": has_reasoning,
        "has_config_tags": has_config,
        "reasoning_not_empty": len(reasoning_content.strip()) > 0,
        "config_not_empty": len(config_content.strip()) > 0,
        "param_count_valid": param_count_valid,
        "param_count": param_count,
        "is_partial_config": is_partial_config,
        "overall_valid": (has_reasoning and has_config and 
                         len(reasoning_content.strip()) > 0 and 
                         len(config_content.strip()) > 0 and 
                         param_count_valid and is_partial_config)
    }

def run_rocksdb_knowledge_pretraining(model=None, tokenizer=None):
    print("==================================================")
    print("PHASE 1: RocksDB Knowledge Background Pretraining")
    print("==================================================")
    
    from rocksdb_knowledge_pretraining import run_rocksdb_knowledge_pretraining as run_knowledge_pretraining
    
    knowledge_model_path = run_knowledge_pretraining("./dataset/rocksdb_tuning_dataset_en_100.json", model, tokenizer, output_manager)
    
    if knowledge_model_path:
        print(f"[KNOWLEDGE] Knowledge pretraining completed successfully: {knowledge_model_path}")
        return knowledge_model_path
    else:
        print("[KNOWLEDGE] Knowledge pretraining failed or skipped")
        return None

def run_structure_pretraining(model, tokenizer, output_manager):
    print("Starting structure pretraining with anti-overfitting measures...")
    log_file.write("Starting structure pretraining with anti-overfitting measures...\n")
    
    training_data_debug_file = output_manager.get_debug_path("structure_training_data")
    validation_debug_file = output_manager.get_debug_path("structure_validation_results")
    structure_response_path = output_manager.get_response_path("structure_validation")
    
    train_dataset = prepare_structure_training_dataset(training_data_debug_file)
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=SFTConfig(
            dataset_text_field = "text",
            per_device_train_batch_size = 8,
            gradient_accumulation_steps = 1,
            warmup_steps = 5,
            num_train_epochs = 30,
            learning_rate = 5e-5,
            logging_steps = 5,
            optim = "adamw_8bit",
            weight_decay = 0.05,
            lr_scheduler_type = "cosine",
            seed = 3407,
            report_to = "wandb",
            save_strategy = "epoch",
            dataloader_drop_last = True,
            dataloader_num_workers = 2,
        ),
    )

    print("Starting structure training with enhanced diversity and reduced overfitting risk...")
    log_file.write("Starting structure training with enhanced diversity and reduced overfitting risk...\n")
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Training parameters:")
    print(f"  - Epochs: 30 (increased for better learning)")
    print(f"  - Learning rate: 5e-5 (reduced)")
    print(f"  - Batch size: 4 (reduced)")
    print(f"  - Weight decay: 0.05 (increased)")
    print(f"  - Examples variety: {len(train_dataset)} diverse examples")
    
    log_file.write(f"Training dataset size: {len(train_dataset)}\n")
    log_file.write(f"Training parameters:\n")
    log_file.write(f"  - Epochs: 30 (increased for better learning)\n")
    log_file.write(f"  - Learning rate: 5e-5 (reduced)\n")
    log_file.write(f"  - Batch size: 4 (reduced)\n")
    log_file.write(f"  - Weight decay: 0.05 (increased)\n")
    log_file.write(f"  - Examples variety: {len(train_dataset)} diverse examples\n")
    
    os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
    trainer.train()
    
    structure_output_dir = output_manager.get_model_dir("structure_pretrained")
    trainer.save_model(structure_output_dir)
    print(f"Structure pretrained model saved to {structure_output_dir}")
    log_file.write(f"Structure pretrained model saved to {structure_output_dir}\n")
    
    print("\n" + "="*60)
    print("VALIDATING STRUCTURE PRETRAINING RESULTS")
    print("="*60)
    log_file.write("\n" + "="*60 + "\n")
    log_file.write("VALIDATING STRUCTURE PRETRAINING RESULTS\n")
    log_file.write("="*60 + "\n")
    
    test_cases = [
        {
            "name": "Memory Optimization Test - Config A",
            "category": "memory_optimization",
            "prompt": """You are optimizing RocksDB configuration for better performance.

            Current system information:
            - CPU: 8 cores
            - Memory: 32GB RAM
            - Storage: NVMe SSD
            - Workload: High memory usage issue

            Current issue: high memory usage
            Current benchmark results: 35000 ops/sec

            Please analyze the configuration and provide ONLY the specific parameters that need to be modified to address the high memory usage issue.
            IMPORTANT:
            1. Return ONLY the parameters that need to be changed, not the complete configuration
            2. Modify at most 10 parameters
            3. Focus on memory efficiency

            Please provide your reasoning and the optimized parameters."""
        },
        {
            "name": "Memory Optimization Test - Config B",
            "category": "memory_optimization", 
            "prompt": """You are optimizing RocksDB configuration for better performance.

            Current system information:
            - CPU: 16 cores
            - Memory: 128GB RAM
            - Storage: SATA SSD
            - Workload: Memory usage spike during peak hours

            Current issue: high memory usage
            Current benchmark results: 28000 ops/sec

            Please analyze the configuration and provide ONLY the specific parameters that need to be modified to address the high memory usage issue.
            IMPORTANT:
            1. Return ONLY the parameters that need to be changed, not the complete configuration
            2. Modify at most 10 parameters
            3. Focus on memory efficiency

            Please provide your reasoning and the optimized parameters."""
        },
        {
            "name": "Memory Optimization Test - Config C",
            "category": "memory_optimization",
            "prompt": """You are optimizing RocksDB configuration for better performance.

            Current system information:
            - CPU: 4 cores
            - Memory: 16GB RAM
            - Storage: NVMe SSD
            - Workload: Memory-constrained environment

            Current issue: high memory usage
            Current benchmark results: 22000 ops/sec

            Please analyze the configuration and provide ONLY the specific parameters that need to be modified to address the high memory usage issue.
            IMPORTANT:
            1. Return ONLY the parameters that need to be changed, not the complete configuration
            2. Modify at most 10 parameters
            3. Focus on memory efficiency

            Please provide your reasoning and the optimized parameters."""
        },
        {
            "name": "Write Performance Test - Config A", 
            "category": "write_performance",
            "prompt": """You are optimizing RocksDB configuration for better performance.

            Current system information:
            - CPU: 16 cores
            - Memory: 64GB RAM
            - Storage: High-speed NVMe
            - Workload: Write-intensive workload

            Current issue: slow write performance
            Current benchmark results: 42000 ops/sec

            Please analyze the configuration and provide ONLY the specific parameters that need to be modified to address the slow write performance issue.
            IMPORTANT:
            1. Return ONLY the parameters that need to be changed, not the complete configuration
            2. Modify at most 10 parameters
            3. Focus on write throughput

            Please provide your reasoning and the optimized parameters."""
        },
        {
            "name": "Write Performance Test - Config B",
            "category": "write_performance",
            "prompt": """You are optimizing RocksDB configuration for better performance.

            Current system information:
            - CPU: 32 cores
            - Memory: 256GB RAM
            - Storage: Enterprise NVMe array
            - Workload: High-volume write operations

            Current issue: slow write performance
            Current benchmark results: 38000 ops/sec

            Please analyze the configuration and provide ONLY the specific parameters that need to be modified to address the slow write performance issue.
            IMPORTANT:
            1. Return ONLY the parameters that need to be changed, not the complete configuration
            2. Modify at most 10 parameters
            3. Focus on write throughput

            Please provide your reasoning and the optimized parameters."""
        },
        {
            "name": "Write Performance Test - Config C",
            "category": "write_performance",
            "prompt": """You are optimizing RocksDB configuration for better performance.

            Current system information:
            - CPU: 8 cores
            - Memory: 32GB RAM
            - Storage: SATA SSD
            - Workload: Moderate write workload

            Current issue: slow write performance
            Current benchmark results: 18000 ops/sec

            Please analyze the configuration and provide ONLY the specific parameters that need to be modified to address the slow write performance issue.
            IMPORTANT:
            1. Return ONLY the parameters that need to be changed, not the complete configuration
            2. Modify at most 10 parameters
            3. Focus on write throughput

            Please provide your reasoning and the optimized parameters."""
        },
        {
            "name": "Compaction Optimization Test - Config A",
            "category": "compaction_optimization",
            "prompt": """You are optimizing RocksDB configuration for better performance.

            Current system information:
            - CPU: 12 cores
            - Memory: 48GB RAM
            - Storage: NVMe SSD
            - Workload: Batch processing

            Current issue: frequent compactions
            Current benchmark results: 28000 ops/sec

            Please analyze the configuration and provide ONLY the specific parameters that need to be modified to address the frequent compactions issue.
            IMPORTANT:
            1. Return ONLY the parameters that need to be changed, not the complete configuration
            2. Modify at most 10 parameters
            3. Focus on compaction efficiency

            Please provide your reasoning and the optimized parameters."""
        },
        {
            "name": "Compaction Optimization Test - Config B",
            "category": "compaction_optimization",
            "prompt": """You are optimizing RocksDB configuration for better performance.

            Current system information:
            - CPU: 24 cores
            - Memory: 96GB RAM
            - Storage: High-performance NVMe
            - Workload: Continuous data ingestion

            Current issue: frequent compactions
            Current benchmark results: 35000 ops/sec

            Please analyze the configuration and provide ONLY the specific parameters that need to be modified to address the frequent compactions issue.
            IMPORTANT:
            1. Return ONLY the parameters that need to be changed, not the complete configuration
            2. Modify at most 10 parameters
            3. Focus on compaction efficiency

            Please provide your reasoning and the optimized parameters."""
        }
    ]
    
    FastLanguageModel.for_inference(model)
    
    validation_results = {
        "total_tests": len(test_cases),
        "passed_tests": 0,
        "failed_tests": 0,
        "format_issues": [],
        "param_count_issues": [],
        "overall_success_rate": 0.0,
        "detailed_test_results": [],
        "overfitting_analysis": {
            "categories": {},
            "overfitting_detected": False,
            "overfitting_categories": [],
            "category_similarity_scores": {},
            "overfitting_summary": ""
        }
    }
    
    debug_test_file = validation_debug_file
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest {i+1}/{len(test_cases)}: {test_case['name']}")
        log_file.write(f"\nTest {i+1}/{len(test_cases)}: {test_case['name']}\n")
        
        inputs = tokenizer(
            [f"<|im_start|>user\n{test_case['prompt']}<|im_end|>\n<|im_start|>assistant\n"],
            return_tensors="pt"
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=generation_config["temperature"],
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "<|im_start|>assistant\n" in response:
            assistant_response = response.split("<|im_start|>assistant\n")[-1]
        else:
            assistant_response = response
        
        print(f"Generated response:")
        print(f"{assistant_response[:200]}..." if len(assistant_response) > 200 else assistant_response)
        log_file.write(f"Generated response:\n")
        log_file.write(f"{assistant_response[:200]}...\n" if len(assistant_response) > 200 else f"{assistant_response}\n")
        
        validation = validate_output_format(assistant_response)
        
        print(f"Validation results:")
        print(f"  - Has reasoning tags: {validation['has_reasoning_tags']}")
        print(f"  - Has config tags: {validation['has_config_tags']}")
        print(f"  - Reasoning not empty: {validation['reasoning_not_empty']}")
        print(f"  - Config not empty: {validation['config_not_empty']}")
        print(f"  - Param count valid (≤10): {validation['param_count_valid']} ({validation['param_count']} params)")
        print(f"  - Is partial config: {validation['is_partial_config']}")
        print(f"  - Overall valid: {validation['overall_valid']}")
        
        log_file.write(f"Validation results:\n")
        log_file.write(f"  - Has reasoning tags: {validation['has_reasoning_tags']}\n")
        log_file.write(f"  - Has config tags: {validation['has_config_tags']}\n")
        log_file.write(f"  - Reasoning not empty: {validation['reasoning_not_empty']}\n")
        log_file.write(f"  - Config not empty: {validation['config_not_empty']}\n")
        log_file.write(f"  - Param count valid (≤10): {validation['param_count_valid']} ({validation['param_count']} params)\n")
        log_file.write(f"  - Is partial config: {validation['is_partial_config']}\n")
        log_file.write(f"  - Overall valid: {validation['overall_valid']}\n")
        
        test_result = {
            "test_id": i + 1,
            "test_name": test_case['name'],
            "test_category": test_case['category'],
            "input_prompt": test_case['prompt'],
            "model_response": assistant_response,
            "full_response": response,
            "validation_details": validation,
            "extracted_reasoning": extract_reasoning_from_text(assistant_response),
            "extracted_config": extract_config_from_text(assistant_response),
            "config_parameters": parse_partial_config(extract_config_from_text(assistant_response)) if extract_config_from_text(assistant_response) else {},
            "test_passed": validation['overall_valid'],
            "timestamp": datetime.now().isoformat()
        }
        validation_results["detailed_test_results"].append(test_result)
        
        category = test_case['category']
        if category not in validation_results["overfitting_analysis"]["categories"]:
            validation_results["overfitting_analysis"]["categories"][category] = []
        validation_results["overfitting_analysis"]["categories"][category].append({
            "test_id": i + 1,
            "test_name": test_case['name'],
            "config_parameters": test_result["config_parameters"],
            "test_passed": validation['overall_valid']
        })
        
        if validation['overall_valid']:
            validation_results['passed_tests'] += 1
            print("TEST PASSED")
            log_file.write("TEST PASSED\n")
        else:
            validation_results['failed_tests'] += 1
            print("TEST FAILED")
            log_file.write("TEST FAILED\n")
            
            if not validation['param_count_valid']:
                validation_results['param_count_issues'].append(f"Test {i+1}: {validation['param_count']} params")
            if not (validation['has_reasoning_tags'] and validation['has_config_tags'] and 
                    validation['reasoning_not_empty'] and validation['config_not_empty'] and 
                    validation['is_partial_config']):
                validation_results['format_issues'].append(f"Test {i+1}: Format issues")
    
    validation_results['overall_success_rate'] = validation_results['passed_tests'] / validation_results['total_tests']
    
    print(f"\n" + "="*60)
    print("OVERFITTING ANALYSIS")
    print("="*60)
    log_file.write(f"\n" + "="*60 + "\n")
    log_file.write("OVERFITTING ANALYSIS\n")
    log_file.write("="*60 + "\n")
    
    def calculate_parameter_similarity(params1: dict, params2: dict) -> float:
        if not params1 and not params2:
            return 1.0
        if not params1 or not params2:
            return 0.0
        
        # 获取所有参数名
        all_params = set(params1.keys()) | set(params2.keys())
        if not all_params:
            return 1.0
        
        # 计算相同参数的比例
        same_params = 0
        for param in all_params:
            if param in params1 and param in params2:
                if params1[param] == params2[param]:
                    same_params += 1
            # 如果参数只在一个配置中存在，认为不同
        
        return same_params / len(all_params)
    
    def analyze_category_overfitting(category_name: str, category_tests: list) -> dict:
        analysis = {
            "category": category_name,
            "total_tests": len(category_tests),
            "passed_tests": sum(1 for test in category_tests if test["test_passed"]),
            "parameter_sets": [],
            "similarity_matrix": [],
            "max_similarity": 0.0,
            "min_similarity": 1.0,
            "avg_similarity": 0.0,
            "overfitting_detected": False,
            "identical_configs": []
        }
        
        passed_tests = [test for test in category_tests if test["test_passed"]]
        if len(passed_tests) < 2:
            analysis["overfitting_detected"] = False
            analysis["reason"] = f"Not enough passed tests ({len(passed_tests)}) for comparison"
            return analysis
        
        for test in passed_tests:
            analysis["parameter_sets"].append({
                "test_name": test["test_name"],
                "parameters": test["config_parameters"]
            })
        
        similarities = []
        identical_pairs = []
        
        for i in range(len(passed_tests)):
            for j in range(i + 1, len(passed_tests)):
                similarity = calculate_parameter_similarity(
                    passed_tests[i]["config_parameters"], 
                    passed_tests[j]["config_parameters"]
                )
                similarities.append(similarity)
                
                analysis["similarity_matrix"].append({
                    "test1": passed_tests[i]["test_name"],
                    "test2": passed_tests[j]["test_name"],
                    "similarity": similarity
                })
                
                if similarity >= 0.9:
                    identical_pairs.append({
                        "test1": passed_tests[i]["test_name"],
                        "test2": passed_tests[j]["test_name"],
                        "similarity": similarity,
                        "params1": passed_tests[i]["config_parameters"],
                        "params2": passed_tests[j]["config_parameters"]
                    })
        
        if similarities:
            analysis["max_similarity"] = max(similarities)
            analysis["min_similarity"] = min(similarities)
            analysis["avg_similarity"] = sum(similarities) / len(similarities)
            analysis["identical_configs"] = identical_pairs
            
            if analysis["avg_similarity"] > 0.8 or len(identical_pairs) > 0:
                analysis["overfitting_detected"] = True
                analysis["reason"] = f"High similarity detected (avg: {analysis['avg_similarity']:.2f}, identical pairs: {len(identical_pairs)})"
            else:
                analysis["overfitting_detected"] = False
                analysis["reason"] = f"Good diversity (avg similarity: {analysis['avg_similarity']:.2f})"
        
        return analysis
    
    overfitting_detected = False
    overfitting_categories = []
    
    for category_name, category_tests in validation_results["overfitting_analysis"]["categories"].items():
        print(f"\nAnalyzing category: {category_name}")
        log_file.write(f"\nAnalyzing category: {category_name}\n")
        
        category_analysis = analyze_category_overfitting(category_name, category_tests)
        validation_results["overfitting_analysis"]["category_similarity_scores"][category_name] = category_analysis
        
        print(f"  Tests in category: {category_analysis['total_tests']}")
        print(f"  Passed tests: {category_analysis['passed_tests']}")
        
        log_file.write(f"  Tests in category: {category_analysis['total_tests']}\n")
        log_file.write(f"  Passed tests: {category_analysis['passed_tests']}\n")
        
        if category_analysis['passed_tests'] >= 2:
            print(f"  Average similarity: {category_analysis['avg_similarity']:.2f}")
            print(f"  Max similarity: {category_analysis['max_similarity']:.2f}")
            print(f"  Min similarity: {category_analysis['min_similarity']:.2f}")
            print(f"  Identical configs: {len(category_analysis['identical_configs'])}")
            
            log_file.write(f"  Average similarity: {category_analysis['avg_similarity']:.2f}\n")
            log_file.write(f"  Max similarity: {category_analysis['max_similarity']:.2f}\n")
            log_file.write(f"  Min similarity: {category_analysis['min_similarity']:.2f}\n")
            log_file.write(f"  Identical configs: {len(category_analysis['identical_configs'])}\n")
            
            if category_analysis['overfitting_detected']:
                print(f"OVERFITTING DETECTED: {category_analysis['reason']}")
                log_file.write(f"OVERFITTING DETECTED: {category_analysis['reason']}\n")
                overfitting_detected = True
                overfitting_categories.append(category_name)
                
                for pair in category_analysis['identical_configs']:
                    print(f"    Identical pair: {pair['test1']} <-> {pair['test2']} (similarity: {pair['similarity']:.2f})")
                    log_file.write(f"    Identical pair: {pair['test1']} <-> {pair['test2']} (similarity: {pair['similarity']:.2f})\n")
            else:
                print(f"GOOD DIVERSITY: {category_analysis['reason']}")
                log_file.write(f"GOOD DIVERSITY: {category_analysis['reason']}\n")
        else:
            print(f"{category_analysis.get('reason', 'Not enough tests for analysis')}")
            log_file.write(f"{category_analysis.get('reason', 'Not enough tests for analysis')}\n")
    
    validation_results["overfitting_analysis"]["overfitting_detected"] = overfitting_detected
    validation_results["overfitting_analysis"]["overfitting_categories"] = overfitting_categories
    
    if overfitting_detected:
        summary = f"Overfitting detected in {len(overfitting_categories)} categories: {', '.join(overfitting_categories)}"
        validation_results["overfitting_analysis"]["overfitting_summary"] = summary
        print(f"\nOVERALL OVERFITTING DETECTED!")
        print(f"   Categories with overfitting: {', '.join(overfitting_categories)}")
        print(f"   Consider: reducing epochs, adding more diverse training data, or increasing regularization")
        
        log_file.write(f"\nOVERALL OVERFITTING DETECTED!\n")
        log_file.write(f"   Categories with overfitting: {', '.join(overfitting_categories)}\n")
        log_file.write(f"   Consider: reducing epochs, adding more diverse training data, or increasing regularization\n")
    else:
        summary = "No significant overfitting detected. Model shows good parameter diversity across different system configurations."
        validation_results["overfitting_analysis"]["overfitting_summary"] = summary
        print(f"\nNO OVERFITTING DETECTED!")
        print(f"   Model shows good adaptation to different system configurations")
        
        log_file.write(f"\nNO OVERFITTING DETECTED!\n")
        log_file.write(f"   Model shows good adaptation to different system configurations\n")
    
    print("="*60)
    log_file.write("="*60 + "\n")
    
    validation_results["validation_metadata"] = {
        "model_name": model_name,
        "test_timestamp": datetime.now().isoformat(),
        "training_epochs": 30,
        "training_parameters": {
            "learning_rate": 5e-5,
            "batch_size": 4,
            "weight_decay": 0.05
        },
        "generation_config": generation_config
    }
    
    try:
        with open(debug_test_file, 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, ensure_ascii=False, indent=2)
        print(f"[DEBUG] Validation debug results saved to {debug_test_file}")
        log_file.write(f"[DEBUG] Validation debug results saved to {debug_test_file}\n")
        
        response_data = {
            "validation_type": "structure_pretraining",
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": validation_results['total_tests'],
                "passed_tests": validation_results['passed_tests'],
                "success_rate": validation_results['overall_success_rate'],
                "overfitting_detected": validation_results["overfitting_analysis"]["overfitting_detected"]
            },
            "test_responses": [
                {
                    "test_name": result["test_name"],
                    "test_category": result["test_category"], 
                    "input_prompt": result["input_prompt"],
                    "model_response": result["model_response"],
                    "test_passed": result["test_passed"]
                }
                for result in validation_results["detailed_test_results"]
            ]
        }
        
        with open(structure_response_path, 'w', encoding='utf-8') as f:
            json.dump(response_data, f, ensure_ascii=False, indent=2)
        print(f"[STRUCTURE VALIDATION] Response results saved to {structure_response_path}")
        log_file.write(f"[STRUCTURE VALIDATION] Response results saved to {structure_response_path}\n")
        
    except Exception as e:
        print(f"[DEBUG] Error saving validation results: {e}")
        log_file.write(f"[DEBUG] Error saving validation results: {e}\n")
    
    print(f"\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Total tests: {validation_results['total_tests']}")
    print(f"Passed tests: {validation_results['passed_tests']}")
    print(f"Failed tests: {validation_results['failed_tests']}")
    print(f"Success rate: {validation_results['overall_success_rate']*100:.1f}%")
    
    log_file.write(f"\n" + "="*60 + "\n")
    log_file.write("VALIDATION SUMMARY\n")
    log_file.write("="*60 + "\n")
    log_file.write(f"Total tests: {validation_results['total_tests']}\n")
    log_file.write(f"Passed tests: {validation_results['passed_tests']}\n")
    log_file.write(f"Failed tests: {validation_results['failed_tests']}\n")
    log_file.write(f"Success rate: {validation_results['overall_success_rate']*100:.1f}%\n")
    
    if validation_results['format_issues']:
        print(f"Format issues in: {', '.join(validation_results['format_issues'])}")
        log_file.write(f"Format issues in: {', '.join(validation_results['format_issues'])}\n")
    
    if validation_results['param_count_issues']:
        print(f"Parameter count issues in: {', '.join(validation_results['param_count_issues'])}")
        log_file.write(f"Parameter count issues in: {', '.join(validation_results['param_count_issues'])}\n")
    
    success_rate_threshold = 0.7
    format_success = validation_results['overall_success_rate'] >= success_rate_threshold
    no_overfitting = not validation_results["overfitting_analysis"]["overfitting_detected"]
    
    print(f"\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    print(f"Format Learning Success: {'SUCCESS' if format_success else 'FAILED'} ({validation_results['overall_success_rate']*100:.1f}% ≥ {success_rate_threshold*100:.0f}%)")
    print(f"Overfitting Check: {'SUCCESS' if no_overfitting else 'FAILED'} ({'No overfitting' if no_overfitting else 'Overfitting detected'})")
    
    log_file.write(f"\n" + "="*60 + "\n")
    log_file.write("FINAL EVALUATION\n")
    log_file.write("="*60 + "\n")
    log_file.write(f"Format Learning Success: {'SUCCESS' if format_success else 'FAILED'} ({validation_results['overall_success_rate']*100:.1f}% ≥ {success_rate_threshold*100:.0f}%)\n")
    log_file.write(f"Overfitting Check: {'SUCCESS' if no_overfitting else 'FAILED'} ({'No overfitting' if no_overfitting else 'Overfitting detected'})\n")
    
    if format_success and no_overfitting:
        print("STRUCTURE PRETRAINING FULLY SUCCESSFUL! Model learned the format correctly without overfitting.")
        log_file.write("STRUCTURE PRETRAINING FULLY SUCCESSFUL! Model learned the format correctly without overfitting.\n")
    elif format_success and not no_overfitting:
        print("STRUCTURE PRETRAINING PARTIALLY SUCCESSFUL: Good format learning but overfitting detected.")
        print("   Recommendations: Reduce epochs, increase training data diversity, or add regularization.")
        log_file.write("STRUCTURE PRETRAINING PARTIALLY SUCCESSFUL: Good format learning but overfitting detected.\n")
        log_file.write("   Recommendations: Reduce epochs, increase training data diversity, or add regularization.\n")
    elif not format_success and no_overfitting:
        print("STRUCTURE PRETRAINING NEEDS IMPROVEMENT: Good diversity but poor format learning.")
        print("   Recommendations: Increase epochs, adjust learning rate, or improve training examples.")
        log_file.write("STRUCTURE PRETRAINING NEEDS IMPROVEMENT: Good diversity but poor format learning.\n")
        log_file.write("   Recommendations: Increase epochs, adjust learning rate, or improve training examples.\n")
    else:
        print("STRUCTURE PRETRAINING FAILED: Both poor format learning and overfitting detected.")
        print("   Recommendations: Redesign training approach - check data quality, training parameters, and model capacity.")
        log_file.write("STRUCTURE
        log_file.write("   Recommendations: Redesign training approach - check data quality, training parameters, and model capacity.\n")
    
    print(f"\nDEBUG FILES SAVED:")
    print(f"  - Training data: {training_data_debug_file}")
    print(f"  - Validation results: {validation_debug_file}")
    print(f"These files contain detailed information for debugging and analysis.")
    
    log_file.write(f"\nDEBUG FILES SAVED:\n")
    log_file.write(f"  - Training data: {training_data_debug_file}\n")
    log_file.write(f"  - Validation results: {validation_debug_file}\n")
    log_file.write(f"These files contain detailed information for debugging and analysis.\n")
    
    print("="*60)
    log_file.write("="*60 + "\n")
    
    return structure_output_dir


def parse_config_to_dict(config_str: str) -> Dict[str, str]:
    try:
        config_dict = {}
        config = configparser.ConfigParser()
        config.read_string(config_str)
        
        for section_name in config.sections():
            section = config[section_name]
            for key, value in section.items():
                config_dict[key] = value
        
        return config_dict
    except Exception as e:
        print(f"Error parsing config to dict: {e}")
        return {}

def parse_partial_config(config_text: str) -> Dict[str, str]:
    config_dict = {}
    
    try:
        import json
        json_data = json.loads(config_text.strip())
        if isinstance(json_data, dict):
            for key, value in json_data.items():
                config_dict[str(key)] = str(value)
    except (json.JSONDecodeError, ValueError):
        try:
            config = configparser.ConfigParser()
            config.read_string(config_text)
            
            for section_name in config.sections():
                section = config[section_name]
                for key, value in section.items():
                    config_dict[key] = value
        except:
            lines = config_text.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line and '=' in line and not line.startswith('[') and not line.startswith('#'):
                    try:
                        key, value = line.split('=', 1)
                        config_dict[key.strip()] = value.strip()
                    except:
                        continue
    
    config_dict = {k: v for k, v in config_dict.items() if k and v}
    
    return config_dict

def merge_configs(original_config: str, partial_config: str) -> str:

    try:
        original_dict = parse_config_to_dict(original_config)
        partial_dict = parse_partial_config(partial_config)
        
        invalid_params = []
        valid_updates = {}
        
        for key, value in partial_dict.items():
            if key in original_dict:
                valid_updates[key] = value
            else:
                invalid_params.append(key)
                print(f"[CONFIG MERGE] Warning: {key} not found in original config")
        
        if invalid_params:
            print(f"[CONFIG MERGE] Invalid parameters found: {invalid_params}")
        
        if not valid_updates:
            return original_config
        
        merged_dict = original_dict.copy()
        merged_dict.update(valid_updates)
        
        merged_config = original_config
        lines = merged_config.split('\n')
        
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            if '=' in stripped_line and not stripped_line.startswith('[') and not stripped_line.startswith('#'):
                try:
                    key_part, value_part = stripped_line.split('=', 1)
                    key = key_part.strip()
                    if key in valid_updates:
                        indent = line[:len(line) - len(line.lstrip())]
                        comment_match = re.search(r'(#.*)', line)
                        comment = comment_match.group(1) if comment_match else ""
                        lines[i] = f"{indent}{key}={valid_updates[key]}{' ' + comment if comment else ''}"
                except:
                    continue
        
        merged_config = '\n'.join(lines)
        return merged_config
        
    except Exception as e:
        print(f"[CONFIG MERGE] Error merging configs: {e}")
        return original_config

def extract_config(text: str) -> str:
    try:
        config_section = re.search(r'<config>(.*?)</config>', text, re.DOTALL)
        if not config_section:
            return ""
        config_text = config_section.group(1).strip()
        return config_text
    except Exception as e:
        print(f"Error extracting config: {e}")
        return ""

def evaluate_benchmark(config_str: str, merged_config: str, current_config_str: str) -> Tuple[float, Dict]:
    """
    """
    eval_result = eval_client.run_evaluation(config_str, merged_config, is_init=False)
    
    if "error" in eval_result:
        print(f"[EVAL] Benchmark evaluation failed: {eval_result['error']}")
        return 0.0, {"ops_per_sec": 0}
    
    benchmark_results = eval_result.get("benchmark_results", {"ops_per_sec": 0})
    
    if isinstance(benchmark_results, dict):
        ops_per_second = float(benchmark_results.get('ops_per_sec', 0))
    else:
        try:
            ops_per_second = float(benchmark_results.split('Operations per second: ')[1].split('.')[0])
        except:
            ops_per_second = 0.0
    
    return ops_per_second, benchmark_results

def create_dataset_from_eval_server(config_str: str, config: str = None, is_init: bool = False, workload: str = None, eval_result: Dict = None) -> Dataset:
    if eval_result is None:
        eval_result = eval_client.run_evaluation(config_str, config, is_init, workload)
    try:    
        if "error" in eval_result:
            print(f"[DATASET] Evaluation failed: {eval_result['error']}")
            return Dataset.from_dict({"prompt": [], "completion": []})
        
        benchmark_results = eval_result.get("benchmark_results", {})
        ops_per_sec = benchmark_results.get("ops_per_sec", 0)
        
        prompt = f"""You are a RocksDB expert. Based on the following system configuration and benchmark results, generate an optimized RocksDB configuration file.

System Configuration: {config_str}
Current Performance: {ops_per_sec} ops/sec
Workload Type: {workload or 'mixgraph'}

Please provide:
1. Your reasoning for the optimization strategy
2. An optimized RocksDB configuration file

Format your response as:
<reasoning>
Your reasoning here
</reasoning>

<config>
Your optimized configuration here
</config>"""

        completion = f"""<reasoning>
Based on the system configuration and current performance, I will optimize the RocksDB configuration for {workload or 'mixgraph'} workload to improve performance.
</reasoning>

<config>
{config or '# Default RocksDB configuration'}
</config>"""

        dataset = Dataset.from_dict({
            "prompt": [prompt],
            "completion": [completion]
        })
        
        print(f"[DATASET] Created dataset with {len(dataset)} examples for workload: {workload}")
        return dataset
        
    except Exception as e:
        print(f"[DATASET] Error creating dataset: {e}")
        return Dataset.from_dict({"prompt": [], "completion": []})

def run_grpo_training(config_manager: SystemConfigManager, model, tokenizer, resume_checkpoint: str = None, output_manager=None):
    
    if args.start_workload_index > 0:
        config_manager.current_workload_index = args.start_workload_index
        config_manager.current_index = 0
        config_manager.improvement_count = 0
        config_manager.epoch_count = 0
        print(f"[TRAIN] Starting from workload index: {args.start_workload_index}")
    
    while not config_manager.is_complete():
        current_workload = config_manager.get_current_workload()
        current_config = config_manager.get_current_config_str()
        
        print(f"\n{'='*80}")
        print(f"[TRAIN] Starting training for workload: {current_workload}")
        print(f"[TRAIN] Current config: {current_config}")
        print(f"[TRAIN] Progress: {config_manager.get_progress()}")
        print(f"{'='*80}\n")
        
        log_file.write(f"\n{'='*80}\n")
        log_file.write(f"[TRAIN] Starting training for workload: {current_workload}\n")
        log_file.write(f"[TRAIN] Current config: {current_config}\n")
        log_file.write(f"[TRAIN] Progress: {config_manager.get_progress()}\n")
        log_file.write(f"{'='*80}\n\n")
        
        print(f"[TRAIN] Applying system config: {current_config}")
        log_file.write(f"[TRAIN] Applying system config: {current_config}\n")
        
        if not eval_client.apply_system_config(current_config):
            print(f"[TRAIN] Warning: Failed to apply system config {current_config}")
            log_file.write(f"[TRAIN] Warning: Failed to apply system config {current_config}\n")
        else:
            print(f"[TRAIN] Successfully applied system config {current_config}")
            log_file.write(f"[TRAIN] Successfully applied system config {current_config}\n")
        
        if output_manager:
            workload_output_dir = output_manager.get_checkpoint_dir(f"grpo_{current_workload}")
        else:
            workload_output_dir = f"{model_name.split('/')[-1]}_grpo_checkpoints_{current_workload}"
        checkpoint_manager = CheckpointManager(workload_output_dir)
        
        latest_checkpoint = checkpoint_manager.get_latest_checkpoint()
        if latest_checkpoint and checkpoint_manager.checkpoint_exists(latest_checkpoint):
            print(f"[TRAIN] Found existing checkpoint for workload {current_workload}: {latest_checkpoint}")
            resume_checkpoint = latest_checkpoint
        
        if output_manager:
            response_path = output_manager.get_response_path(f"grpo_{current_workload}")
        else:
            response_path = f"responses_{current_workload}.jsonl"
        response_saver = ResponseSaver(response_path)
        
        base_config = config_manager.get_current_config_str()
        base_score = 0.0
        base_results = {"ops_per_sec": 0}
        
        print(f"[TRAIN] Running initial evaluation for workload: {current_workload}")
        eval_result = eval_client.run_evaluation(base_config, is_init=True, workload=current_workload)
        
        if "error" not in eval_result:
            # print(f"[TRAIN] Initial evaluation result: {eval_result}")
            base_results = eval_result.get("benchmark_results", {"ops_per_sec": 0})
            base_score = float(base_results.get("ops_per_sec", 0))
            print(f"[TRAIN] Initial performance for {current_workload}: {base_score} ops/sec")
        else:
            print(f"[TRAIN] Initial evaluation failed for {current_workload}: {eval_result['error']}")
        
        train_dataset = create_dataset_from_eval_server(
            base_config, 
            is_init=True, 
            workload=current_workload,
            eval_result=eval_result
        )
        
        if len(train_dataset) == 0:
            print(f"[TRAIN] No training data available for workload {current_workload}, skipping...")
            if config_manager.switch_to_next_workload():
                continue
            else:
                break
        
        training_args = GRPOConfig(
            learning_rate=1e-6,
            adam_beta1=0.9,
            adam_beta2=0.999,
            weight_decay=0.1,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            optim="paged_adamw_8bit",
            logging_steps=1,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=1,
            num_generations=8,
            max_prompt_length=4096,
            max_completion_length=4096,
            max_steps=300,
            save_steps=1,
            max_grad_norm=0.1,
            report_to="wandb",
            output_dir=workload_output_dir,
            log_completions=True,
            sync_ref_model=True,
            save_strategy='steps'
        )
        
        class SharedState:
            def __init__(self, base_score):
                self.current_base_score = base_score
        
        shared_state = SharedState(base_score)
        
        class RocksDBCallback(TrainerCallback):
            def __init__(self, base_config, base_score, base_results, config_manager, current_config_str, shared_state, checkpoint_manager=None, response_saver=None):
                self.base_config = base_config
                self.base_score = base_score
                self.base_results = base_results
                self.config_manager = config_manager
                self.current_config_str = current_config_str
                self.checkpoint_manager = checkpoint_manager
                self.response_saver = response_saver
                self.current_workload = config_manager.get_current_workload()
                self.step_count = 0
                self.config_switched = False
                self.shared_state = shared_state
                
            def on_step_end(self, args, state, control, model=None, **kwargs):
                self.step_count += 1
                
                if self.step_count % args.gradient_accumulation_steps == 0:
                    self.config_manager.record_epoch()
                    print(f"[CALLBACK] Epoch recorded: {self.config_manager.epoch_count}/{self.config_manager.max_epochs}")
                    log_file.write(f"[CALLBACK] Epoch recorded: {self.config_manager.epoch_count}/{self.config_manager.max_epochs}\n")
                
                completions = performance_improvement_reward.completions_container
                rewards = performance_improvement_reward.rewards_container
                configs = performance_improvement_reward.configs_container
                improvements = performance_improvement_reward.improvements_container
                eval_scores = performance_improvement_reward.eval_score_container
                benchmark_results = performance_improvement_reward.benchmark_results_container
                
                if self.response_saver and completions and len(completions) > 0:
                    try:
                        self.response_saver.save_responses(
                            step=state.global_step,
                            config_str=self.current_config_str,
                            prompts=kwargs.get('prompts', []),
                            completions=completions,
                            rewards=rewards,
                            improvements=improvements,
                            eval_scores=eval_scores,
                            configs=configs,
                            benchmark_results=benchmark_results
                        )
                    except Exception as e:
                        print(f"[CALLBACK] Error saving responses: {e}")
                
                improvement_found = False
                if completions and rewards and len(completions) > 0 and len(rewards) > 0:
                    if (len(improvements) > 0 and len(configs) > 0 and 
                        all(isinstance(imp, (int, float)) for imp in improvements) and 
                        max(improvements) > 0):
                        
                        try:
                            best_idx = np.argmax(improvements)
                            if best_idx < len(configs) and configs[best_idx]:
                                best_config = configs[best_idx]

                                self.base_config = best_config
                                print(f"New base configuration found for config {self.current_config_str}!")
                                log_file.write(f"New base configuration found for config {self.current_config_str}!\n")

                                old_improvement_count = self.config_manager.improvement_count
                                self.config_manager.record_improvement()
                                new_improvement_count = self.config_manager.improvement_count
                                print(f"[CALLBACK] Improvement recorded: {old_improvement_count} -> {new_improvement_count}")
                                log_file.write(f"[CALLBACK] Improvement recorded: {old_improvement_count} -> {new_improvement_count}\n")
                                improvement_found = True

                                new_dataset = create_dataset_from_eval_server(
                                    self.current_config_str, 
                                    config=best_config, 
                                    is_init=True,
                                    workload=self.current_workload
                                )
                                model_trainer.train_dataset = new_dataset
                                self.base_results = new_dataset[0]["benchmark_results"]
                                self.base_score = new_dataset[0]["benchmark_score"]
                                self.shared_state.current_base_score = self.base_score
                                print(f"[CALLBACK] Updated shared base score to: {self.base_score}")
                                log_file.write(f"[CALLBACK] Updated shared base score to: {self.base_score}\n")
                            else:
                                print(f"[CALLBACK] Warning: Invalid best_idx {best_idx} or empty config")
                        except Exception as e:
                            print(f"[CALLBACK] Error processing improvements: {e}")
                    else:
                        print(f"[CALLBACK] No valid improvements found or empty containers")

                should_switch = self.config_manager.should_switch_config()
                print(f"[CALLBACK] Step {state.global_step}: Improvements={self.config_manager.improvement_count}/{self.config_manager.max_improvements}, Epochs={self.config_manager.epoch_count}/{self.config_manager.max_epochs}, Should switch: {should_switch}")
                log_file.write(f"[CALLBACK] Step {state.global_step}: Improvements={self.config_manager.improvement_count}/{self.config_manager.max_improvements}, Epochs={self.config_manager.epoch_count}/{self.config_manager.max_epochs}, Should switch: {should_switch}\n")
                
                if should_switch and not self.config_switched:
                    print(f"\n[CONFIG] Switching condition met for config {self.current_config_str}")
                    print(f"[CONFIG] Improvements: {self.config_manager.improvement_count}/{self.config_manager.max_improvements}")
                    print(f"[CONFIG] Epochs: {self.config_manager.epoch_count}/{self.config_manager.max_epochs}")
                    
                    log_file.write(f"\n[CONFIG] Switching condition met for config {self.current_config_str}\n")
                    log_file.write(f"[CONFIG] Improvements: {self.config_manager.improvement_count}/{self.config_manager.max_improvements}\n")
                    log_file.write(f"[CONFIG] Epochs: {self.config_manager.epoch_count}/{self.config_manager.max_epochs}\n")
                    
                    self.config_switched = True
                    
                    control.should_training_stop = True

                kl = None
                if hasattr(model_trainer.state, "log_history") and model_trainer.state.log_history:
                    latest_log = model_trainer.state.log_history[-1]
                    for key in latest_log:
                        if 'kl' in key.lower():
                            kl = latest_log[key]
                            break
                
                avg_eval_score = (lambda lst: sum(filter(None, lst)) / len(list(filter(None, lst))) if any(filter(None, lst)) else 0)(eval_scores)
                
                wandb.log({
                    "step": state.global_step,
                    "system_config": self.current_config_str,
                    "config_improvements": self.config_manager.improvement_count,
                    "config_epochs": self.config_manager.epoch_count,
                    "base_score": self.base_score,
                    'best_reward': max(rewards) if rewards else 0,
                    'best_improvement': max(improvements) if improvements else 0,
                    'improvement_found': improvement_found,
                    'kl': kl,
                    'avg_eval_score': avg_eval_score
                })
        
        def performance_improvement_reward(prompts, completions, **kwargs) -> List[float]:
            
            self = performance_improvement_reward
            self.completions_container = []
            self.rewards_container = []
            self.configs_container = []
            self.benchmark_results_container = []
            self.improvements_container = []
            self.eval_score_container = []
            
            if isinstance(completions[0], str):
                responses = completions
            elif isinstance(completions[0], dict):
                responses = [completion.get("content", "") for completion in completions]
            elif isinstance(completions[0], list):
                responses = [completion[0].get("content", "") if isinstance(completion[0], dict) else str(completion[0]) for completion in completions]
            else:
                responses = [str(completion) for completion in completions]
            
            current_config = ""
            if isinstance(prompts, list):
                if len(prompts) > 0:
                    if isinstance(prompts[0], list):
                        for prompt_list in prompts:
                            for item in prompt_list:
                                if isinstance(item, dict) and item.get('role') == 'user':
                                    user_content = item.get('content', '')
                                    current_config = user_content.split('###')[1] if '###' in user_content else ""
                                    break
                    elif isinstance(prompts[0], dict):
                        for item in prompts:
                            if item.get('role') == 'user':
                                user_content = item.get('content', '')
                                current_config = user_content.split('###')[1] if '###' in user_content else ""
                                break
                    elif isinstance(prompts[0], str):
                        user_content = prompts[0]
                        current_config = user_content.split('###')[1] if '###' in user_content else ""
            elif isinstance(prompts, str):
                current_config = prompts.split('###')[1] if '###' in prompts else ""
            
            current_score = float(shared_state.current_base_score)
            
            rewards = []
            configs = []
            results = []
            improvements = []
            eval_scores = []
            
            for response in responses:
                new_config = extract_config(response)
                configs.append(new_config)
                format_reward = 0.0
                pattern = r"<reasoning>.*?</reasoning>\s*<config>.*?</config>"

                if re.search(pattern, response, re.DOTALL) and new_config:
                    format_reward = 0.1

                if not new_config:
                    rewards.append(format_reward)
                    results.append(None)
                    improvements.append(0.0)
                    eval_scores.append(0.0)
                    continue

                try:
                    param_count_reward = 0.0
                    if new_config:
                        config_dict = parse_partial_config(new_config)
                        param_count = len(config_dict)
                        
                        if 1 <= param_count <= 10:
                            param_count_reward = 0.1
                        else:
                            param_count_reward = 0.0
                        
                        print(f"[PARAM_VALIDATION] Config has {param_count} parameters, reward: {param_count_reward:.3f}")
                        log_file.write(f"[PARAM_VALIDATION] Config has {param_count} parameters, reward: {param_count_reward:.3f}\n")

                        if param_count_reward == 0.0:
                            total_reward = format_reward + param_count_reward
                            rewards.append(total_reward)
                            results.append(None)
                            improvements.append(0.0)
                            eval_scores.append(0.0)
                            print(f"[PARAM_VALIDATION] Skipping evaluation due to parameter count violation (count: {param_count})")
                            log_file.write(f"[PARAM_VALIDATION] Skipping evaluation due to parameter count violation (count: {param_count})\n")
                            continue
                    
                    config_struct_result = eval_client.run_evaluation(
                        config_manager.get_current_config_str(),
                        new_config,
                        is_init=False,
                        workload=config_manager.get_current_workload()
                    )
                    
                    if "error" in config_struct_result:
                        rewards.append(format_reward)
                        results.append(None)
                        improvements.append(0.0)
                        eval_scores.append(0.0)
                        continue
                    
                    benchmark_results = config_struct_result.get("benchmark_results", {"ops_per_sec": 0})
                    new_score = float(benchmark_results.get("ops_per_sec", 0))
                    
                    if new_score == 0:
                        rewards.append(format_reward)
                        results.append(benchmark_results)
                        improvements.append(0.0)
                        eval_scores.append(0.0)
                        continue

                    eval_scores.append(new_score)
                    improvement = (new_score - current_score) / current_score
                    print(f"Config {config_manager.get_current_config_str()} - Improvement:{improvement}\n")

                    performance_reward = 0.0
                    baseline_score = current_score * 0.9
                    
                    if new_score >= baseline_score:
                        if new_score > current_score:
                            performance_reward = min(4.0, (new_score - baseline_score) / baseline_score * 10.0)
                        else:
                            performance_reward = min(1.0, (new_score - baseline_score) / baseline_score * 5.0)
                    else:
                        performance_reward = max(-1.0, (new_score - baseline_score) / baseline_score * 2.0)

                    total_reward = format_reward + param_count_reward + performance_reward

                    rewards.append(total_reward)
                    results.append(benchmark_results)
                    improvements.append(improvement)
                    print(f"Config {config_manager.get_current_config_str()} - Current score: {current_score:.2f}, Eval score: {new_score:.2f}, "
                          f"Baseline: {baseline_score:.2f}, Improvement: {improvement:.2%}, Format reward: {format_reward:.2f}, "
                          f"Param count reward: {param_count_reward:.2f}, Performance reward: {performance_reward:.2f}, Total reward: {total_reward:.2f}")
                    log_file.write(f"Config {config_manager.get_current_config_str()} - Current score: {current_score:.2f}, Eval score: {new_score:.2f}, "
                                   f"Baseline: {baseline_score:.2f}, Improvement: {improvement:.2%}, Format reward: {format_reward:.2f}, "
                                   f"Param count reward: {param_count_reward:.2f}, Performance reward: {performance_reward:.2f}, Total reward: {total_reward:.2f}\n")
                    
                except Exception as e:
                    print(f"[REWARD] Error during evaluation: {e}")
                    param_count_reward = 0.0
                    if new_config:
                        try:
                            config_dict = parse_partial_config(new_config)
                            param_count = len(config_dict)
                            if 1 <= param_count <= 10:
                                param_count_reward = 0.1
                            else:
                                param_count_reward = 0.0
                            print(f"[PARAM_VALIDATION] Exception case - Config has {param_count} parameters, reward: {param_count_reward:.3f}")
                        except:
                            param_count_reward = 0.0
                    
                    total_reward = format_reward + param_count_reward
                    rewards.append(total_reward)
                    results.append(None)
                    improvements.append(0.0)
                    eval_scores.append(0.0)

            self.completions_container = completions
            self.rewards_container = rewards
            self.configs_container = configs
            self.benchmark_results_container = results
            self.improvements_container = improvements
            self.eval_score_container = eval_scores

            return rewards
        
        model_trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[
                lambda prompts, completions, **kwargs: performance_improvement_reward(
                    prompts, completions, **kwargs
                )
            ],
            args=training_args,
            train_dataset=train_dataset,
            callbacks=[RocksDBCallback(base_config, base_score, base_results, config_manager, config_manager.get_current_config_str(), shared_state, checkpoint_manager, response_saver)]
        )
        
        print(f"[TRAIN] Starting GRPO training for workload: {current_workload}")
        model_trainer.train(resume_from_checkpoint=resume_checkpoint)
        
        final_checkpoint_path = os.path.join(workload_output_dir, "final_model")
        model_trainer.save_model(final_checkpoint_path)
        print(f"[TRAIN] Saved final model for workload {current_workload} to {final_checkpoint_path}")
        
        if checkpoint_manager:
            latest_checkpoint = checkpoint_manager.get_latest_checkpoint()
            if latest_checkpoint:
                checkpoint_manager.save_config_manager_state(latest_checkpoint, config_manager)
        
        if config_manager.should_switch_workload():
            if config_manager.switch_to_next_workload():
                print(f"[TRAIN] Switching to next workload: {config_manager.get_current_workload()}")
                continue
            else:
                print(f"[TRAIN] All workloads completed!")
                break
        else:
            if config_manager.switch_to_next_config():
                print(f"[TRAIN] Switching to next config: {config_manager.get_current_config_str()}")
                continue
            else:
                print(f"[TRAIN] All configs completed for current workload!")
                if config_manager.switch_to_next_workload():
                    print(f"[TRAIN] Switching to next workload: {config_manager.get_current_workload()}")
                    continue
                else:
                    print(f"[TRAIN] All workloads completed!")
                    break
    
    print(f"[TRAIN] Training completed for all workloads!")
    log_file.write(f"[TRAIN] Training completed for all workloads!\n")

def main():
    global log_file, output_manager
    
    log_file = open(log_name, 'w', encoding='utf-8')
    
    print(f"[MAIN] Training output directory: {output_manager.root_dir}")
    log_file.write(f"[MAIN] Training output directory: {output_manager.root_dir}\n")
    
    print(f"[MAIN] Starting RocksDB Configuration Optimization Training")
    print(f"[MAIN] Model: {model_name}")
    print(f"[MAIN] Workloads: {args.workloads}")
    print(f"[MAIN] Start workload index: {args.start_workload_index}")
    print(f"[MAIN] System configs: {SYSTEM_CONFIGS}")
    print(f"[MAIN] Load model only: {args.load_model_only}")
    print(f"[MAIN] Resume from checkpoint: {args.resume_from_checkpoint}")
    print(f"[MAIN] Auto resume: {args.auto_resume}")
    
    log_file.write(f"[MAIN] Starting RocksDB Configuration Optimization Training\n")
    log_file.write(f"[MAIN] Model: {model_name}\n")
    log_file.write(f"[MAIN] Workloads: {args.workloads}\n")
    log_file.write(f"[MAIN] Start workload index: {args.start_workload_index}\n")
    log_file.write(f"[MAIN] System configs: {SYSTEM_CONFIGS}\n")
    log_file.write(f"[MAIN] Load model only: {args.load_model_only}\n")
    log_file.write(f"[MAIN] Resume from checkpoint: {args.resume_from_checkpoint}\n")
    log_file.write(f"[MAIN] Auto resume: {args.auto_resume}\n")

    if not wait_for_eval_server():
        print("[MAIN] Failed to connect to eval server, exiting...")
        log_file.write("[MAIN] Failed to connect to eval server, exiting...\n")
        log_file.close()
        return
    
    checkpoint_path = None
    model_only_path = None
    load_training_state = True
    
    if args.load_model_only:
        model_only_path = args.load_model_only
        load_training_state = False
        print(f"[CHECKPOINT] Load model only from: {model_only_path}")
        log_file.write(f"[CHECKPOINT] Load model only from: {model_only_path}\n")
    elif args.resume_from_checkpoint:
        checkpoint_path = args.resume_from_checkpoint
        load_training_state = True
        print(f"[CHECKPOINT] Resume from specified checkpoint: {checkpoint_path}")
        log_file.write(f"[CHECKPOINT] Resume from specified checkpoint: {checkpoint_path}\n")
    elif args.auto_resume:
        output_dir = f"{model_name.split('/')[-1]}_grpo_checkpoints"
        checkpoint_manager = CheckpointManager(output_dir)
        latest_checkpoint = checkpoint_manager.get_latest_checkpoint()
        if latest_checkpoint:
            checkpoint_path = latest_checkpoint
            load_training_state = True
            print(f"[CHECKPOINT] Auto-resume from latest checkpoint: {checkpoint_path}")
            log_file.write(f"[CHECKPOINT] Auto-resume from latest checkpoint: {checkpoint_path}\n")
        else:
            print(f"[CHECKPOINT] No checkpoint found in {output_dir}")
            log_file.write(f"[CHECKPOINT] No checkpoint found in {output_dir}\n")
    
    if model_only_path:
        model, tokenizer = load_model_and_tokenizer(model_only_path)
    elif checkpoint_path:
        model, tokenizer = load_model_and_tokenizer(checkpoint_path)
    else:
        model, tokenizer = load_model_and_tokenizer(None)
    
    print(f"Model device: {model.device}")
    log_file.write(f"Model device: {model.device}\n")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=lora_rank*2,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing=True,
        random_state=3407,
    )
    
    config_manager = SystemConfigManager(SYSTEM_CONFIGS, args.workloads)
    
    if checkpoint_path and load_training_state:
        checkpoint_manager = CheckpointManager(f"{model_name.split('/')[-1]}_grpo_checkpoints")
        config_state = checkpoint_manager.load_config_manager_state(checkpoint_path)
        if config_state:
            config_manager.restore_state(config_state)
            print(f"[CHECKPOINT] Restored config manager from {checkpoint_path}")
            log_file.write(f"[CHECKPOINT] Restored config manager from {checkpoint_path}\n")
        else:
            print(f"[CHECKPOINT] No config manager state found, starting from workload index {args.start_workload_index}")
            log_file.write(f"[CHECKPOINT] No config manager state found, starting from workload index {args.start_workload_index}\n")
            config_manager.current_workload_index = args.start_workload_index
            config_manager.current_index = 0
    else:
        if args.start_workload_index > 0:
            config_manager.current_workload_index = args.start_workload_index
            config_manager.current_index = 0
            print(f"[CONFIG] Starting from workload index {args.start_workload_index}")
            log_file.write(f"[CONFIG] Starting from workload index {args.start_workload_index}\n")
        
        if model_only_path:
            print(f"[CONFIG] Model loaded from {model_only_path}, starting fresh training (no training state restored)")
            log_file.write(f"[CONFIG] Model loaded from {model_only_path}, starting fresh training (no training state restored)\n")
    
    if not args.skip_knowledge_pretraining:
        print("[MAIN] ==================================================")
        print("[MAIN] PHASE 1: RocksDB Knowledge Background Pretraining")
        print("[MAIN] ==================================================")
        log_file.write("[MAIN] Starting Phase 1: Knowledge Background Pretraining\n")
        knowledge_model_path = run_rocksdb_knowledge_pretraining(model, tokenizer)
        
        if knowledge_model_path:
            print(f"[MAIN] Phase 1 completed successfully: {knowledge_model_path}")
            log_file.write(f"[MAIN] Phase 1 completed: {knowledge_model_path}\n")
        else:
            print("[MAIN] Phase 1 failed or skipped")
            log_file.write("[MAIN] Phase 1 failed or skipped\n")
    else:
        print("[MAIN] Skipping Phase 1: Knowledge Background Pretraining")
        log_file.write("[MAIN] Skipping Phase 1: Knowledge Background Pretraining\n")
        knowledge_model_path = None
    
    if not args.skip_structure_pretraining:
        print("[MAIN] ==================================================")
        print("[MAIN] PHASE 2: Structure Output Pretraining")
        print("[MAIN] ==================================================")
        log_file.write("[MAIN] Starting Phase 2: Structure Output Pretraining\n")
        run_structure_pretraining(model, tokenizer, output_manager)
        print("[MAIN] Phase 2 completed successfully")
        log_file.write("[MAIN] Phase 2 completed successfully\n")
    else:
        print("[MAIN] Skipping Phase 2: Structure Output Pretraining")
        log_file.write("[MAIN] Skipping Phase 2: Structure Output Pretraining\n")
    
    print("[MAIN] ==================================================")
    print("[MAIN] PHASE 3: GRPO Reinforcement Learning Training")
    print("[MAIN] ==================================================")
    log_file.write("[MAIN] Starting Phase 3: GRPO Reinforcement Learning Training\n")
    grpo_checkpoint_path = checkpoint_path if load_training_state else None
    run_grpo_training(config_manager, model, tokenizer, grpo_checkpoint_path, output_manager)
    print("[MAIN] Phase 3 completed successfully")
    log_file.write("[MAIN] Phase 3 completed successfully\n")
    
    print("[MAIN] Training completed!")
    log_file.write("[MAIN] Training completed!\n")
    log_file.close()
    
    print("[MAIN] Saving final statistics...")
    for workload in args.workloads:
        try:
            response_path = output_manager.get_response_path(f"grpo_{workload}")
            response_saver = ResponseSaver(response_path)
            response_saver.save_stats_summary()
            print(f"[MAIN] Saved statistics for workload: {workload}")
        except Exception as e:
            print(f"[MAIN] Error saving statistics for workload {workload}: {e}")
    
    sudoPassword = 'embed'
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    emailCommand = f'sudo /usr/local/bin/sendEmail \
                -f tranining_notifier@163.com \
                -t tranining_notifier@163.com \
                -s smtp.163.com \
                -u "[4U4090]_training_finished" \
                -o message-content-type=html -o message-charset=utf8 \
                -xu tranining_notifier@163.com \
                -xp PHgJQLsRakbAy24S \
                -m "Taining of qwen3-8b is finished at {time} on 4U4090!"'
    state = False
    while not state:
        try:
            os.system('echo %s|sudo -S %s' % (sudoPassword, emailCommand))
            state = True
        except:
            print('Failed! Resending...\n')
            state = False
            time.sleep(5)
    if state == True:
        print('Mail sent successfully')

if __name__ == "__main__":
    main() 