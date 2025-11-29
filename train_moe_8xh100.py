"""
Simple Data Parallel Training for 8x H100 GPUs
Uses PyTorch's DataParallel for simple multi-GPU training
"""
import time
import os
import torch
import torch.nn as nn
import logging
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from configs.moe_config_8xh100 import MoEModelConfig8xH100
from training.trainer_8xh100 import train_moe_model_8xh100
from utils.helpers import set_seed
from utils.logger import setup_logging


def print_system_info():
    """Print GPU information"""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"\n{'='*70}")
        print(f"🖥️  8x H100 GPU SETUP")
        print(f"{'='*70}")
        print(f"Number of GPUs available: {num_gpus}")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
        print(f"PyTorch version: {torch.__version__}")
        print(f"{'='*70}\n")
    else:
        raise RuntimeError("No CUDA GPUs available! This script requires GPUs.")


def main():
    logger = setup_logging(log_dir="./logs")
    logger.info("Starting 8xH100 MoE training")

    print_system_info()
    set_seed(42)
    
    # Check GPU availability
    if torch.cuda.device_count() < 8:
        print(f"⚠️  WARNING: Only {torch.cuda.device_count()} GPU(s) available")
        print(f"⚠️  This script is optimized for 8 GPUs but will run on available GPUs")
    
    config = MoEModelConfig8xH100()
    # Update num_gpus based on actual availability
    config.num_gpus = min(torch.cuda.device_count(), config.num_gpus)
    config.effective_batch_size = config.batch_size * config.num_gpus

    # Load dataset
    print("Loading dataset: SmolLM Cosmopedia...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolLM-135M",
        cache_dir="./hf_cache"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config.vocab_size = tokenizer.vocab_size
    
    # Load dataset and split documents
    raw_dataset = load_dataset(
        "HuggingFaceTB/smollm-corpus",
        "cosmopedia-v2",
        split="train",
        cache_dir="./hf_cache",
        streaming=True,
    )
    
    # Take samples and split into train/val (90/10 split)
    raw_samples = list(raw_dataset.take(config.num_documents))
    num_val = int(len(raw_samples) * 0.1)
    raw_train = Dataset.from_list(raw_samples[:len(raw_samples) - num_val])
    raw_val = Dataset.from_list(raw_samples[len(raw_samples) - num_val:])
    logger.info(f"Split into {len(raw_train):,} train docs and {len(raw_val):,} val docs")
    
    # Tokenize and prepare datasets
    def tokenize_and_prepare(dataset, split_name):
        # Tokenize with caching
        def tokenize_fn(examples):
            return tokenizer(examples["text"], truncation=False, padding=False)
        
        cache_file = f"./data_cache/tokenized_{split_name}_{config.num_documents}_{config.max_seq_len}.cache"
        tokenized = dataset.map(
            tokenize_fn, 
            batched=True, 
            remove_columns=dataset.column_names,
            cache_file_name=cache_file,
            desc=f"Tokenizing {split_name}"
        )
        
        # Group into fixed-length sequences
        def group_texts(examples):
            concatenated = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated["input_ids"])
            block_size = config.max_seq_len
            
            # Drop last incomplete block
            total_length = (total_length // block_size) * block_size
            result = {
                k: [concatenated[k][i:i + block_size] for i in range(0, total_length, block_size)]
                for k in concatenated.keys()
            }
            result["labels"] = result["input_ids"].copy()
            return result
        
        cache_file_grouped = f"./data_cache/grouped_{split_name}_{config.num_documents}_{config.max_seq_len}.cache"
        grouped = tokenized.map(
            group_texts, 
            batched=True,
            cache_file_name=cache_file_grouped,
            desc=f"Grouping {split_name}"
        )
        grouped.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return grouped
    
    print("Tokenizing datasets (cached)...")
    os.makedirs("./data_cache", exist_ok=True)
    train_ds = tokenize_and_prepare(raw_train, "train")
    val_ds = tokenize_and_prepare(raw_val, "val")
    logger.info(f"Train sequences: {len(train_ds):,}, Val sequences: {len(val_ds):,}")

    # Data loaders - increase num_workers for multi-GPU
    loader_args = dict(
        batch_size=config.batch_size,  # Per GPU batch size
        num_workers=4,  # More workers for multi-GPU
        pin_memory=True,
        persistent_workers=True,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)

    print("\nModel Configuration")
    print("-" * 70)
    print(f"d_model: {config.d_model}, layers: {config.n_layers}, heads: {config.n_heads}")
    print(f"ff dim: {config.d_ff}")
    print(f"experts: {config.num_experts}, top‑k: {config.expert_top_k}")
    print(f"steps: {config.max_steps}")
    print(f"batch size per GPU: {config.batch_size}")
    print(f"effective batch size: {config.effective_batch_size} ({config.num_gpus} GPUs)")
    print(f"vocab size: {config.vocab_size}\n")
    logger.info(f"Model configuration: {vars(config)}")

    print("Starting 8xH100 Training...")
    print("-" * 70)
    start = time.time()

    model, metrics = train_moe_model_8xh100(config, train_loader, val_loader)
    elapsed = (time.time() - start) / 60
    logger.info("Training complete")

    print("\nResults")
    print("-" * 70)
    print(f"Training time: {elapsed:.2f} min")
    print(f"Val loss:       {metrics['val_loss']:.4f}")
    print(f"Val accuracy:   {metrics['val_accuracy']:.4f}")
    print(f"Val perplexity: {metrics['val_perplexity']:.2f}")
    logger.info(f"Final metrics: {metrics}")

    # Save checkpoint
    ckpt_path = "./checkpoints/final_model_8xh100.pt"
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    
    # Extract model from DataParallel wrapper if needed
    model_to_save = model.module if isinstance(model, nn.DataParallel) else model
    
    torch.save(
        {"model_state_dict": model_to_save.state_dict(),
         "config": config,
         "metrics": metrics},
        ckpt_path,
    )
    print(f"Model checkpoint saved to {ckpt_path}")
    logger.info(f"Model saved to {ckpt_path}")


if __name__ == "__main__":
    main()
