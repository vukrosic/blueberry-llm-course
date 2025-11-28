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
    logger = setup_logging(log_dir="./logs/sweep_optimizers")
    logger.info("Starting Optimizer Sweep")
    
    # Sweep configurations: Optimizer Hyperparams (Momentum & Weight Decay)
    # Baseline: momentum=0.9, wd=0.2
    configs_to_test = [
        {"name": "high_mom_low_wd", "muon_momentum": 0.95, "weight_decay": 0.1},
        {"name": "low_mom_high_wd", "muon_momentum": 0.85, "weight_decay": 0.3},
        {"name": "baseline",        "muon_momentum": 0.90, "weight_decay": 0.2},
    ]
    
    results = []
    
    # Load data once
    base_config = MoEModelConfig()
    train_loader, val_loader = get_dataloaders(base_config)
    
    print("\n" + "="*50)
    print("STARTING OPTIMIZER HYPERPARAM SWEEP")
    print("="*50)
    
    for run_cfg in configs_to_test:
        print(f"\nRunning experiment: {run_cfg['name']}")
        
        # Setup config
        config = MoEModelConfig()
        config.muon_momentum = run_cfg["muon_momentum"]
        config.weight_decay = run_cfg["weight_decay"]
        config.max_steps = 200  # Short run
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
        cfg_str = f"Mom={res['config']['muon_momentum']}, WD={res['config']['weight_decay']}"
        print(f"{res['name']:<20} | {res['val_loss']:<10.4f} | {res['val_ppl']:<10.2f} | {cfg_str}")
    
    # Save results to file
    import json
    with open("experiments/optimizer_hyperparam_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to experiments/optimizer_hyperparam_sweep_results.json")

if __name__ == "__main__":
    main()
