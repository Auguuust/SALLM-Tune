#!/usr/bin/env python3

import os
import json
import re
import sys
import argparse
import glob
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import numpy as np
from datasets import Dataset
from transformers import TrainingArguments, TrainerCallback
from trl import SFTTrainer, SFTConfig, GRPOConfig, GRPOTrainer
# import wandb
import configparser
import requests
import time
from unsloth import FastLanguageModel, PatchFastRL

os.environ["WANDB_API_KEY"] = "<you wandb key>"

def load_rocksdb_knowledge_dataset(dataset_path: str = "./dataset/rocksdb_tuning_dataset_en_100.json") -> Dataset:
    print(f"[KNOWLEDGE] Loading RocksDB knowledge dataset from {dataset_path}")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    train_data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                item = json.loads(line)

                if "prompt" in item and "completion" in item:
                    prompt = item["prompt"].strip()
                    response = item["completion"].strip()

                    train_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
                    train_data.append({"text": train_text})
                    
            except json.JSONDecodeError as e:
                print(f"[KNOWLEDGE] Warning: Invalid JSON at line {line_num}: {e}")
                continue
    
    print(f"[KNOWLEDGE] Loaded {len(train_data)} training examples")

    if train_data:
        print(f"[KNOWLEDGE] Sample training text (first 200 chars):")
        print(f"'{train_data[0]['text'][:200]}...'")

        invalid_count = 0
        for i, item in enumerate(train_data):
            if not isinstance(item, dict) or "text" not in item or not item["text"].strip():
                invalid_count += 1
                print(f"[KNOWLEDGE] Warning: Invalid item at index {i}: {item}")
        
        if invalid_count > 0:
            print(f"[KNOWLEDGE] Found {invalid_count} invalid items")
        else:
            print(f"[KNOWLEDGE] All {len(train_data)} items are valid")
    
    return Dataset.from_list(train_data)

def run_rocksdb_knowledge_pretraining(dataset_path: str = "./dataset/rocksdb_tuning_dataset_en_100.json", model=None, tokenizer=None, output_manager=None):
    print("==================================================")
    print("PHASE 1: RocksDB Knowledge Background Pretraining")
    print("==================================================")
    
    try:
        train_dataset = load_rocksdb_knowledge_dataset(dataset_path)
    except Exception as e:
        print(f"[KNOWLEDGE] Error loading dataset: {e}")
        print("[KNOWLEDGE] Skipping knowledge pretraining...")
        return None
    
    if len(train_dataset) == 0:
        print("[KNOWLEDGE] No training data available, skipping knowledge pretraining...")
        return None
    
    if output_manager:
        knowledge_output_dir = output_manager.get_model_dir("knowledge_pretrained")
        knowledge_log_path = output_manager.get_log_path("knowledge_pretraining")
        knowledge_response_path = output_manager.get_response_path("knowledge_validation")
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        knowledge_output_dir = f"knowledge_pretrained_model_{timestamp}"
        os.makedirs(knowledge_output_dir, exist_ok=True)
        knowledge_log_path = f"knowledge_pretraining_{timestamp}.log"
        knowledge_response_path = f"knowledge_validation_{timestamp}.json"
    
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from trl import SFTConfig
    
    import sys
    sys.path.append('.')
    from train_server_full import model_name, max_seq_length, lora_rank
    
    if model is None or tokenizer is None:
        print("[KNOWLEDGE] Loading model and tokenizer...")
        import torch
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=False,
            full_finetuning=False,
            fast_inference=False,
            max_lora_rank=lora_rank,
            low_cpu_mem_usage=True,
            gpu_memory_utilization=0.3,
            device_map={'':torch.cuda.current_device()},
        )
    else:
        print("[KNOWLEDGE] Using provided model and tokenizer...")
    
    training_args = SFTConfig(
        output_dir=knowledge_output_dir,
        num_train_epochs=6,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=1,
        save_steps=5,
        save_strategy="steps",
        load_best_model_at_end=False,
        report_to="wandb",
        remove_unused_columns=False,
        push_to_hub=False,
        dataloader_pin_memory=False,
        max_grad_norm=0.3,
        optim="adamw_torch",
        torch_compile=False,
        bf16=True,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        fp16=False,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("[KNOWLEDGE] Testing tokenizer with sample data...")
    if len(train_dataset) > 0:
        sample_text = train_dataset[0]["text"]
        try:
            tokens = tokenizer(sample_text, truncation=True, padding=True, max_length=max_seq_length, return_tensors="pt")
            print(f"[KNOWLEDGE] Tokenization test successful. Input length: {len(tokens['input_ids'][0])}")
        except Exception as e:
            print(f"[KNOWLEDGE] Tokenization test failed: {e}")
            raise e
    
    print("[KNOWLEDGE] Testing batch tokenization...")
    batch_texts = [train_dataset[i]["text"] for i in range(min(4, len(train_dataset)))]
    try:
        batch_tokens = tokenizer(batch_texts, truncation=True, padding=True, max_length=max_seq_length, return_tensors="pt")
        print(f"[KNOWLEDGE] Batch tokenization test successful. Shape: {batch_tokens['input_ids'].shape}")
    except Exception as e:
        print(f"[KNOWLEDGE] Batch tokenization test failed: {e}")
    
    print("[KNOWLEDGE] Force preprocessing dataset to avoid SFTTrainer data processing issues...")
    def preprocess_function(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_seq_length,
        )
        result["labels"] = result["input_ids"].copy()
        return result
    
    tokenized_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=["text"])
    train_dataset = tokenized_dataset
    use_dataset_text_field = False
    
    print(f"[KNOWLEDGE] Preprocessed dataset columns: {train_dataset.column_names}")
    print(f"[KNOWLEDGE] Sample tokenized data shape: input_ids={len(train_dataset[0]['input_ids'])}, labels={len(train_dataset[0]['labels'])}")
    
    trainer_kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
        packing=False,
    )
    print(f"[KNOWLEDGE] Creating SFTTrainer with preprocessed data...")
    trainer = SFTTrainer(**trainer_kwargs)
    
    print("[KNOWLEDGE] Starting knowledge background pretraining...")
    trainer.train()
    
    trainer.save_model()
    print(f"[KNOWLEDGE] Knowledge background pretraining completed! Model saved to {knowledge_output_dir}")
    
    validation_results = validate_knowledge_pretraining(model, tokenizer, knowledge_response_path)
    
    print(f"[KNOWLEDGE] Knowledge validation completed! Results saved to {knowledge_response_path}")
    
    return knowledge_output_dir

def validate_knowledge_pretraining(model, tokenizer, response_save_path: str):
    import torch
    print("\n" + "="*60)
    print("VALIDATING KNOWLEDGE PRETRAINING RESULTS")
    print("="*60)
    
    test_cases = [
        {
            "name": "Basic RocksDB Knowledge Test 1",
            "category": "basic_knowledge",
            "prompt": "List the three amplification factors in RocksDB and explain what each one means."
        },
        {
            "name": "Basic RocksDB Knowledge Test 2", 
            "category": "basic_knowledge",
            "prompt": "What is write_buffer_size and how does it influence flush frequency?"
        },
        {
            "name": "Configuration Knowledge Test 1",
            "category": "config_knowledge", 
            "prompt": "What does max_write_buffer_number control?"
        },
        {
            "name": "Configuration Knowledge Test 2",
            "category": "config_knowledge",
            "prompt": "Explain the purpose of level0_file_num_compaction_trigger."
        },
        {
            "name": "Performance Tuning Test 1",
            "category": "performance_tuning",
            "prompt": "When write amplification is very high, which resource is most likely saturated?"
        },
        {
            "name": "Performance Tuning Test 2", 
            "category": "performance_tuning",
            "prompt": "What immediate problems does high read amplification cause and how can you mitigate it?"
        }
    ]
    
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)
    
    validation_results = {
        "total_tests": len(test_cases),
        "test_results": [],
        "validation_timestamp": datetime.now().isoformat(),
        "model_output_samples": []
    }
    
    for i, test_case in enumerate(test_cases):
        print(f"\nKnowledge Test {i+1}/{len(test_cases)}: {test_case['name']}")
        
        inputs = tokenizer(
            [f"<|im_start|>user\n{test_case['prompt']}<|im_end|>\n<|im_start|>assistant\n"],
            return_tensors="pt"
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.6,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "<|im_start|>assistant\n" in response:
            assistant_response = response.split("<|im_start|>assistant\n")[-1]
        else:
            assistant_response = response
        
        print(f"Generated response (first 200 chars):")
        print(f"{assistant_response[:200]}..." if len(assistant_response) > 200 else assistant_response)
        
        test_result = {
            "test_id": i + 1,
            "test_name": test_case['name'],
            "test_category": test_case['category'],
            "input_prompt": test_case['prompt'],
            "model_response": assistant_response,
            "full_response": response,
            "response_length": len(assistant_response),
            "timestamp": datetime.now().isoformat()
        }
        validation_results["test_results"].append(test_result)
        
        if i < 3:
            validation_results["model_output_samples"].append({
                "test_name": test_case['name'],
                "prompt": test_case['prompt'],
                "response": assistant_response
            })
    
    try:
        with open(response_save_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(validation_results, f, ensure_ascii=False, indent=2)
        print(f"[KNOWLEDGE VALIDATION] Results saved to {response_save_path}")
    except Exception as e:
        print(f"[KNOWLEDGE VALIDATION] Error saving results: {e}")
    
    print(f"\n" + "="*60)
    print("KNOWLEDGE VALIDATION SUMMARY")
    print("="*60)
    print(f"Total tests: {validation_results['total_tests']}")
    print(f"All tests completed successfully")
    print(f"Average response length: {sum(r['response_length'] for r in validation_results['test_results']) / len(validation_results['test_results']):.0f} characters")
    print("="*60)
    
    return validation_results

if __name__ == "__main__":
    try:
        dataset = load_rocksdb_knowledge_dataset()
        print(f"Successfully loaded dataset with {len(dataset)} examples")
    except Exception as e:
        print(f"Error: {e}") 
