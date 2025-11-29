"""
8x H100 GPU trainer using PyTorch DataParallel
Simple and straightforward data parallelism across 8 H100 GPUs
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from configs.moe_config_8xh100 import MoEModelConfig8xH100
from models.moe_llm import MoEMinimalLLM
from optimizers.muon import Muon
from training.evaluation import evaluate_model
from utils.helpers import set_seed


def setup_muon_optimizer(model: nn.Module, config: MoEModelConfig8xH100):
    """Setup Muon optimizer with hybrid approach"""
    muon_params = []
    adamw_params = []

    # Get the base model (unwrap DataParallel if needed)
    base_model = model.module if isinstance(model, nn.DataParallel) else model

    for name, param in base_model.named_parameters():
        if (param.ndim == 2 and 
            'token_embedding' not in name and 
            'norm' not in name and 
            param.requires_grad):
            muon_params.append(param)
        else:
            adamw_params.append(param)

    print(f"  Muon parameters: {sum(p.numel() for p in muon_params):,}")
    print(f"  AdamW parameters: {sum(p.numel() for p in adamw_params):,}")

    muon_optimizer = Muon(muon_params, lr=config.muon_lr, momentum=config.muon_momentum)
    adamw_optimizer = torch.optim.AdamW(
        adamw_params,
        lr=config.adamw_lr,
        weight_decay=config.weight_decay
    )

    return [muon_optimizer, adamw_optimizer]


def train_moe_model_8xh100(
    config: MoEModelConfig8xH100, 
    train_loader: DataLoader, 
    val_loader: DataLoader
):
    """
    Train the MoE model using simple DataParallel across 8x H100 GPUs.
    
    This is the simplest multi-GPU approach:
    - Uses nn.DataParallel to split batches across GPUs
    - All GPUs train on different data in parallel
    - Gradients are automatically averaged across GPUs
    """
    print(f"\n🚀 Training MoE model with DataParallel on {config.num_gpus} H100 GPUs")
    print(f"   Experts: {config.num_experts}, Top-k: {config.expert_top_k}")
    
    # Initialize model on CPU first
    set_seed(42)
    model = MoEMinimalLLM(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    active_params = sum(p.numel() for n, p in model.named_parameters()
                       if 'expert' not in n)
    expert_params = total_params - active_params
    
    print(f"  📊 Total parameters: {total_params:,}")
    print(f"  📊 Active parameters: {active_params:,}")
    print(f"  📊 Expert parameters: {expert_params:,}")
    print(f"  📊 Parameter efficiency: {active_params/total_params:.1%} active per forward pass")
    
    # Move to GPU and wrap with DataParallel
    device = torch.device('cuda:0')
    model = model.to(device)
    
    # Wrap model with DataParallel - this handles multi-GPU automatically
    if config.num_gpus > 1:
        gpu_ids = list(range(config.num_gpus))
        model = nn.DataParallel(model, device_ids=gpu_ids)
        print(f"  🔧 DataParallel enabled on GPUs: {gpu_ids}")
    
    # Setup optimizers
    optimizers = setup_muon_optimizer(model, config)
    
    # Learning rate schedule with cosine decay
    schedulers = []
    warmup_steps = max(1, int(config.max_steps * config.warmup_ratio))
    for optimizer in optimizers:
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (config.max_steps - warmup_steps)
                return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        schedulers.append(scheduler)
    
    scaler = GradScaler() if config.use_amp else None
    
    # Training loop
    model.train()
    step = 0
    pbar = tqdm(total=config.max_steps, desc="Training 8xH100")
    train_start_time = time.time()
    
    while step < config.max_steps:
        for batch_idx, batch in enumerate(train_loader):
            if step >= config.max_steps:
                break
            
            # Handle different batch formats
            if isinstance(batch, dict):
                x = batch["input_ids"]
                y = batch["labels"]
                attention_mask = batch.get("attention_mask")
            elif isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    x, attention_mask, y = batch
                elif len(batch) == 2:
                    x, y = batch
                    attention_mask = None
                else:
                    raise ValueError(f"Unexpected batch structure with {len(batch)} elements.")
            else:
                raise TypeError(f"Unsupported batch type: {type(batch)}")
            
            # Move to device - DataParallel will handle distribution across GPUs
            x, y = x.to(device), y.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # Forward pass with AMP
            if config.use_amp:
                with autocast('cuda', dtype=torch.float16):
                    logits, aux_loss = model(x, return_aux_loss=True)
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = y[:, 1:].contiguous()
                    ce_loss = F.cross_entropy(
                        shift_logits.view(-1, config.vocab_size),
                        shift_labels.view(-1)
                    )
                    
                    total_loss = ce_loss
                    if aux_loss is not None:
                        total_loss = total_loss + aux_loss
                    
                    # No gradient accumulation needed with 8 GPUs
                    loss = total_loss
                
                # DataParallel returns a vector of losses, take mean for scalar
                scaler.scale(loss.mean()).backward()
            else:
                logits, aux_loss = model(x, return_aux_loss=True)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = y[:, 1:].contiguous()
                ce_loss = F.cross_entropy(
                    shift_logits.view(-1, config.vocab_size),
                    shift_labels.view(-1)
                )
                
                total_loss = ce_loss
                if aux_loss is not None:
                    total_loss = total_loss + aux_loss
                
                loss = total_loss
                # DataParallel returns a vector of losses, take mean for scalar
                loss.mean().backward()
            
            # Optimizer step
            if config.use_amp:
                for optimizer in optimizers:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                
                for optimizer in optimizers:
                    scaler.step(optimizer)
                    optimizer.zero_grad()
                for scheduler in schedulers:
                    scheduler.step()
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                for optimizer in optimizers:
                    optimizer.step()
                    optimizer.zero_grad()
                for scheduler in schedulers:
                    scheduler.step()
            
            # Logging
            if step % 100 == 0:
                with torch.no_grad():
                    predictions = logits.argmax(dim=-1)
                    accuracy = (predictions == y).float().mean().item()
                    current_loss = ce_loss.mean().item()
                    perplexity = math.exp(min(current_loss, 20))
                    current_lr = schedulers[0].get_last_lr()[0] if schedulers else optimizers[0].param_groups[0]['lr']
                
                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'aux': f'{aux_loss.mean().item() if aux_loss is not None else 0:.4f}',
                    'acc': f'{accuracy:.3f}',
                    'ppl': f'{perplexity:.1f}',
                    'lr': f'{current_lr:.5f}'
                })
            
            # Evaluation and Checkpointing
            if step % config.eval_every == 0 and step > 0:
                eval_metrics = evaluate_model(model, val_loader, config)
                elapsed_time = (time.time() - train_start_time) / 60
                current_lr = schedulers[0].get_last_lr()[0] if schedulers else optimizers[0].param_groups[0]['lr']
                
                print(f"\nStep {step}: Val Loss: {eval_metrics['val_loss']:.4f}, "
                      f"Val Acc: {eval_metrics['val_accuracy']:.4f}, "
                      f"Val PPL: {eval_metrics['val_perplexity']:.2f}, "
                      f"LR: {current_lr:.5f}")
                
                # Save checkpoint
                os.makedirs("./checkpoints", exist_ok=True)
                ckpt_path = f"./checkpoints/step_{step}.pt"
                model_to_save = model.module if isinstance(model, nn.DataParallel) else model
                torch.save({
                    'step': step,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_states': [opt.state_dict() for opt in optimizers],
                    'scheduler_states': [sch.state_dict() for sch in schedulers],
                    'config': config,
                    'metrics': eval_metrics,
                }, ckpt_path)
                print(f"💾 Checkpoint saved: {ckpt_path}")
                
                # Generate sample text
                print("\n🎯 Generating sample text...")
                model.eval()
                with torch.no_grad():
                    prompt = "The future of artificial intelligence"
                    # Simple greedy generation
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", cache_dir="./hf_cache")
                    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                    generated = input_ids
                    
                    for _ in range(50):  # Generate 50 tokens
                        output = model(generated)
                        if isinstance(output, tuple):
                            logits = output[0]
                        else:
                            logits = output
                        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        generated = torch.cat([generated, next_token], dim=1)
                    
                    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
                    print(f"Prompt: {prompt}")
                    print(f"Generated: {generated_text}")
                    print()
                model.train()
            
            step += 1
            if step % 20 == 0:
                pbar.update(20)
    
    pbar.close()
    
    # Final evaluation
    final_eval = evaluate_model(model, val_loader, config)
    total_time = (time.time() - train_start_time) / 60
    
    print(f"\n📊 Final Results:")
    print(f"   Val Loss: {final_eval['val_loss']:.4f}")
    print(f"   Val Accuracy: {final_eval['val_accuracy']:.4f}")
    print(f"   Val Perplexity: {final_eval['val_perplexity']:.2f}")
    print(f"   Total Time: {total_time:.2f} min")
    print(f"   Throughput: ~{config.effective_batch_size * step / (total_time * 60):.0f} samples/sec")
    
    return model, final_eval
