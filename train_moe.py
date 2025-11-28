import time
import os
import torch
import logging
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

# Fix tokenizer parallelism warning when using DataLoader workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from configs.moe_config import MoEModelConfig
from training.trainer import train_moe_model
from utils.helpers import set_seed
from utils.logger import setup_logging


def print_system_info():
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    print(f"PyTorch: {torch.__version__}\n")


def main():
    logger = setup_logging(log_dir="./logs")
    logger.info("Starting MoE training")

    print_system_info()
    set_seed(42)
    config = MoEModelConfig()

    # Hardcoded data loading for SmolLM Cosmopedia
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
    def tokenize_and_prepare(dataset):
        # Tokenize
        def tokenize_fn(examples):
            return tokenizer(examples["text"], truncation=False, padding=False)
        
        tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
        
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
        
        grouped = tokenized.map(group_texts, batched=True)
        grouped.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return grouped
    
    print("Tokenizing datasets...")
    train_ds = tokenize_and_prepare(raw_train)
    val_ds = tokenize_and_prepare(raw_val)
    logger.info(f"Train sequences: {len(train_ds):,}, Val sequences: {len(val_ds):,}")

    loader_args = dict(
        batch_size=config.batch_size,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)

    print("\nModel configuration")
    print("-" * 70)
    print(f"d_model: {config.d_model}, layers: {config.n_layers}, heads: {config.n_heads}")
    print(f"ff dim: {config.d_ff}")
    print(f"experts: {config.num_experts}, top‑k: {config.expert_top_k}")
    print(f"steps: {config.max_steps}, batch size: {config.batch_size}")
    print(f"vocab size: {config.vocab_size}\n")
    logger.info(f"Model configuration: {vars(config)}")

    print("Starting training...")
    print("-" * 70)
    start = time.time()

    model, metrics = train_moe_model(config, train_loader, val_loader)
    elapsed = (time.time() - start) / 60
    logger.info("Training complete")

    print("\nResults")
    print("-" * 70)
    print(f"Training time: {elapsed:.2f} min")
    print(f"Val loss:       {metrics['val_loss']:.4f}")
    print(f"Val accuracy:   {metrics['val_accuracy']:.4f}")
    print(f"Val perplexity: {metrics['val_perplexity']:.2f}")
    logger.info(f"Final metrics: {metrics}")

    ckpt_path = "./checkpoints/final_model.pt"
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(),
         "config": config,
         "metrics": metrics},
        ckpt_path,
    )
    print(f"Model checkpoint saved to {ckpt_path}")
    logger.info(f"Model saved to {ckpt_path}")


if __name__ == "__main__":
    main()
