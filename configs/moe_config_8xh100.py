from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class MoEModelConfig8xH100:
    """Configuration for 8x H100 GPUs with simple data parallelism"""
    
    # Model architecture - ~300M total parameters
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 12
    d_ff: int = 2048     # 4x d_model
    use_mla: bool = False
    qk_rope_dim: int | None = 32
    qk_nope_dim: int | None = 192
    kv_lora_rank: int | None = 128
    v_dim: int | None = 192
    
    # Training parameters - optimized for 8x H100
    batch_size: int = 96  # Per GPU batch size (768 effective across 8 GPUs)
    max_steps: int = 400  # Training steps
    gradient_accumulation_steps: int = 1  # Not needed with 8 GPUs
    
    # Optimizer settings
    muon_lr: float = 0.06
    muon_momentum: float = 0.85
    adamw_lr: float = 0.006
    warmup_ratio: float = 0.05
    
    # Data parameters - balanced for fast training
    max_seq_len: int = 768  # Moderate sequence length
    num_documents: int = 20000  # Reasonable dataset size
    max_tokens: int = 2000000  # 2M tokens
    
    # Evaluation
    eval_every: int = 100  # Evaluate and save checkpoint every 100 steps
    eval_steps: int = 200
    
    # Regularization
    weight_decay: float = 0.3
    dropout: float = 0.1
    grad_clip: float = 1.0
    
    # Technical
    use_amp: bool = True
    vocab_size: Optional[int] = None
    log_milestones: Tuple[int, ...] = (2000, 5000, 10000)
    
    # MoE specific parameters
    num_experts: int = 8  # 1 per GPU
    expert_top_k: int = 2
    load_balancing_weight: float = 0.01
    
    # Multi-GPU settings
    num_gpus: int = 8  # Number of GPUs to use
    
    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        # Effective batch size across all GPUs
        self.effective_batch_size = self.batch_size * self.num_gpus
        print(f"Effective batch size: {self.effective_batch_size} (8 GPUs × {self.batch_size} per GPU)")
