import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import logging
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from configs.moe_config import MoEModelConfig
from training.trainer import train_moe_model
from utils.helpers import set_seed
from utils.logger import setup_logging

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_dataloaders(config):
    """Reuse data loading logic from train_moe.py"""
    print("Loading dataset: SmolLM Cosmopedia...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolLM-135M",
        cache_dir="./hf_cache"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config.vocab_size = tokenizer.vocab_size
    
    raw_dataset = load_dataset(
        "HuggingFaceTB/smollm-corpus",
        "cosmopedia-v2",
        split="train",
        cache_dir="./hf_cache",
        streaming=True,
    )
    
    # Use 5000 docs for sweep
    num_docs = 5000 
    print(f"Taking {num_docs} documents for sweep...")
    
    raw_samples = list(raw_dataset.take(num_docs))
    num_val = int(len(raw_samples) * 0.1)
    raw_train = Dataset.from_list(raw_samples[:len(raw_samples) - num_val])
    raw_val = Dataset.from_list(raw_samples[len(raw_samples) - num_val:])
    
    def tokenize_and_prepare(dataset):
        def tokenize_fn(examples):
            return tokenizer(examples["text"], truncation=False, padding=False)
        
        tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
        
        def group_texts(examples):
            concatenated = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated["input_ids"])
            block_size = config.max_seq_len
            total_length = (total_length // block_size) * block_size
            result = {
                k: [concatenated[k][i:i + block_size] for i in range(0, total_length, block_size)]
                for k in concatenated.keys()
            }
            result["labels"] = result["input_ids"].copy()
            return result
        
        grouped = tokenized.map(group_texts, batched=True)
        grouped.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return grouped
    
    print("Tokenizing datasets...")
    train_ds = tokenize_and_prepare(raw_train)
    val_ds = tokenize_and_prepare(raw_val)
    
    loader_args = dict(
        batch_size=config.batch_size,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    
    return train_loader, val_loader

def main():
    logger = setup_logging(log_dir="./logs/sweep_experts")
    logger.info("Starting Expert Configuration Sweep")
    
    # Sweep configurations: Experts & Aux Loss Weight
    configs_to_test = [
        {"name": "8e_2k_w0.01",   "num_experts": 8,  "expert_top_k": 2, "load_balancing_weight": 0.01},
        {"name": "8e_2k_w0.001",  "num_experts": 8,  "expert_top_k": 2, "load_balancing_weight": 0.001},
        {"name": "16e_2k_w0.01",  "num_experts": 16, "expert_top_k": 2, "load_balancing_weight": 0.01},
        {"name": "16e_2k_w0.001", "num_experts": 16, "expert_top_k": 2, "load_balancing_weight": 0.001},
    ]
    
    results = []
    
    # Load data once
    base_config = MoEModelConfig()
    train_loader, val_loader = get_dataloaders(base_config)
    
    print("\n" + "="*50)
    print("STARTING EXPERT & AUX LOSS SWEEP")
    print("="*50)
    
    for run_cfg in configs_to_test:
        print(f"\nRunning experiment: {run_cfg['name']}")
        
        # Setup config
        config = MoEModelConfig()
        config.num_experts = run_cfg["num_experts"]
        config.expert_top_k = run_cfg["expert_top_k"]
        config.load_balancing_weight = run_cfg["load_balancing_weight"]
        config.max_steps = 300  # Short run for sweep
        config.vocab_size = base_config.vocab_size
        
        # Train
        set_seed(42)
        model, metrics = train_moe_model(config, train_loader, val_loader)
        
        # Record results
        result = {
            "name": run_cfg["name"],
            "val_loss": metrics["val_loss"],
            "val_ppl": metrics["val_perplexity"],
            "config": run_cfg
        }
        results.append(result)
        
        print(f"Finished {run_cfg['name']}: Val Loss = {metrics['val_loss']:.4f}, PPL = {metrics['val_perplexity']:.2f}")

    print("\n" + "="*50)
    print("SWEEP RESULTS")
    print("="*50)
    print(f"{'Experiment':<20} | {'Val Loss':<10} | {'Val PPL':<10} | {'Config'}")
    print("-" * 80)
    for res in results:
        cfg_str = f"E={res['config']['num_experts']}, W={res['config']['load_balancing_weight']}"
        print(f"{res['name']:<20} | {res['val_loss']:<10.4f} | {res['val_ppl']:<10.2f} | {cfg_str}")
    
    # Save results to file
    import json
    with open("experiments/expert_aux_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to experiments/expert_aux_sweep_results.json")

if __name__ == "__main__":
    main()
