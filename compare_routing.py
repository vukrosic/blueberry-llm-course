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


def prepare_data(config, logger):
    """Prepare training and validation datasets"""
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
    
    return train_loader, val_loader


def run_experiment(use_credal, max_steps=50):
    """Run a single experiment with the given configuration"""
    experiment_name = "credal" if use_credal else "standard"
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {experiment_name.upper()} MoE ROUTING")
    print(f"{'='*70}\n")
    
    # Setup logging
    log_dir = f"./logs/compare_{experiment_name}"
    logger = setup_logging(log_dir=log_dir)
    logger.info(f"Starting {experiment_name} MoE training")

    print_system_info()
    set_seed(42)  # Same seed for fair comparison
    
    # Configure model
    config = MoEModelConfig()
    config.max_steps = max_steps
    config.use_credal_routing = use_credal
    config.credal_lambda = 5.0  # Adjusted for normalized uncertainty
    
    # Prepare data (same data for both)
    train_loader, val_loader = prepare_data(config, logger)

    print("\nModel configuration")
    print("-" * 70)
    print(f"d_model: {config.d_model}, layers: {config.n_layers}, heads: {config.n_heads}")
    print(f"ff dim: {config.d_ff}")
    print(f"experts: {config.num_experts}, top‑k: {config.expert_top_k}")
    print(f"steps: {config.max_steps}, batch size: {config.batch_size}")
    print(f"vocab size: {config.vocab_size}")
    print(f"Credal routing: {config.use_credal_routing}")
    if config.use_credal_routing:
        print(f"Credal lambda: {config.credal_lambda}")
    print()
    logger.info(f"Model configuration: {vars(config)}")

    print(f"Starting {experiment_name} training...")
    print("-" * 70)
    start = time.time()

    model, metrics, metrics_history = train_moe_model(config, train_loader, val_loader)
    elapsed = (time.time() - start) / 60
    logger.info("Training complete")

    print("\nResults")
    print("-" * 70)
    print(f"Training time: {elapsed:.2f} min")
    if 'train_loss' in metrics:
        print(f"Final train loss: {metrics['train_loss']:.4f}")
    print(f"Val loss:         {metrics['val_loss']:.4f}")
    print(f"Val accuracy:     {metrics['val_accuracy']:.4f}")
    print(f"Val perplexity:   {metrics['val_perplexity']:.2f}")
    logger.info(f"Final metrics: {metrics}")
    
    # Collect credal statistics if using credal routing
    credal_stats = None
    if use_credal:
        # Extract credal router statistics from model
        expert_counts_all = []
        uncertainties_all = []
        selected_experts_all = []
        for block in model.transformer_blocks:
            if hasattr(block.feed_forward.router, 'expert_counts'):
                expert_counts_all.extend(block.feed_forward.router.expert_counts)
                uncertainties_all.extend(block.feed_forward.router.uncertainties)
                if hasattr(block.feed_forward.router, 'selected_experts'):
                    selected_experts_all.extend(block.feed_forward.router.selected_experts)
        
        if expert_counts_all:
            import numpy as np
            
            # Count expert usage
            expert_usage = [0] * 8  # 8 experts
            for experts in selected_experts_all:
                for expert_id in experts:
                    expert_usage[expert_id] += 1
            
            credal_stats = {
                'expert_counts': expert_counts_all,
                'uncertainties': uncertainties_all,
                'selected_experts': selected_experts_all,
                'expert_usage': expert_usage,
                'mean_experts': np.mean(expert_counts_all),
                'std_experts': np.std(expert_counts_all),
                'min_experts': np.min(expert_counts_all),
                'max_experts': np.max(expert_counts_all),
                'mean_uncertainty': np.mean(uncertainties_all),
            }
            print(f"\nCredal Routing Statistics:")
            print(f"  Mean experts selected: {credal_stats['mean_experts']:.2f} ± {credal_stats['std_experts']:.2f}")
            print(f"  Range: [{credal_stats['min_experts']}, {credal_stats['max_experts']}]")
            print(f"  Mean uncertainty: {credal_stats['mean_uncertainty']:.4f}")
            print(f"  Expert usage distribution: {expert_usage}")

    # Save model
    ckpt_path = f"./checkpoints/{experiment_name}_model.pt"
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "metrics": metrics,
            "metrics_history": metrics_history,
            "credal_stats": credal_stats,
        },
        ckpt_path,
    )
    print(f"Model checkpoint saved to {ckpt_path}")
    logger.info(f"Model saved to {ckpt_path}")
    
    return metrics, metrics_history, credal_stats


def main():
    print("\n" + "="*70)
    print("CREDAL VS STANDARD MoE ROUTING COMPARISON")
    print("="*70)
    
    # Run both experiments
    standard_metrics, standard_history, _ = run_experiment(use_credal=False, max_steps=50)
    credal_metrics, credal_history, credal_stats = run_experiment(use_credal=True, max_steps=50)
    
    # Compare results
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print("\nStandard MoE:")
    if 'train_loss' in standard_metrics:
        print(f"  Train Loss:    {standard_metrics['train_loss']:.4f}")
    print(f"  Val Loss:      {standard_metrics['val_loss']:.4f}")
    print(f"  Val Accuracy:  {standard_metrics['val_accuracy']:.4f}")
    print(f"  Val Perplexity: {standard_metrics['val_perplexity']:.2f}")
    
    print("\nCredal MoE:")
    if 'train_loss' in credal_metrics:
        print(f"  Train Loss:    {credal_metrics['train_loss']:.4f}")
    print(f"  Val Loss:      {credal_metrics['val_loss']:.4f}")
    print(f"  Val Accuracy:  {credal_metrics['val_accuracy']:.4f}")
    print(f"  Val Perplexity: {credal_metrics['val_perplexity']:.2f}")
    
    print("\nDifferences (Credal - Standard):")
    if 'train_loss' in credal_metrics and 'train_loss' in standard_metrics:
        train_diff = credal_metrics['train_loss'] - standard_metrics['train_loss']
        print(f"  Train Loss:    {train_diff:+.4f}")
    val_diff = credal_metrics['val_loss'] - standard_metrics['val_loss']
    acc_diff = credal_metrics['val_accuracy'] - standard_metrics['val_accuracy']
    ppl_diff = credal_metrics['val_perplexity'] - standard_metrics['val_perplexity']
    
    print(f"  Val Loss:      {val_diff:+.4f}")
    print(f"  Val Accuracy:  {acc_diff:+.4f}")
    print(f"  Val Perplexity: {ppl_diff:+.2f}")
    
    print("\n" + "="*70)
    
    # Save comparison results with history
    differences = {
        "val_loss": float(val_diff),
        "val_accuracy": float(acc_diff),
        "val_perplexity": float(ppl_diff)
    }
    
    if 'train_loss' in credal_metrics and 'train_loss' in standard_metrics:
        differences["train_loss"] = float(credal_metrics['train_loss'] - standard_metrics['train_loss'])
    
    comparison_results = {
        "standard": standard_metrics,
        "credal": credal_metrics,
        "differences": differences,
        "standard_history": standard_history,
        "credal_history": credal_history,
        "credal_stats": credal_stats,
    }
    
    results_path = "./checkpoints/comparison_results.pt"
    torch.save(comparison_results, results_path)
    print(f"Comparison results saved to {results_path}\n")


if __name__ == "__main__":
    main()
