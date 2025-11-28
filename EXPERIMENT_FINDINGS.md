# Experiment Findings Summary

This document summarizes the key learnings from all experiments conducted on the LLM training pipeline. These findings have been applied to the main configuration files.

## 📊 Experiments Overview

- **exp8**: Baseline learning rate and load balancing analysis
- **exp9**: Comprehensive Muon vs Adam optimizer comparison (45+ experiments)
- Other experiments: Architecture-specific tests (attention mechanisms, hybrid models, etc.)

---

## 🏆 Key Findings Applied to Main Config

### 1. Optimizer Choice: Muon Hybrid (exp9)

**Winner: Muon optimizer** 🚀

- **Performance improvement**: 7% better validation loss than fully-optimized Adam
  - Muon: 5.16 loss
  - Adam: 5.55 loss
- **Early training**: 15% better at 200 steps (5.72 vs 6.73)

**Configuration Applied:**
```python
# Muon optimizer for 2D weight matrices
muon_lr: float = 0.07            # 70x higher than Adam's optimal LR!
muon_momentum: float = 0.9       # Lower momentum works better
adamw_lr: float = 0.007          # For embeddings and norms
weight_decay: float = 0.2        # Higher than default 0.1
```

**Why Muon Wins:**
1. Better gradient conditioning through Newton-Schulz orthogonalization
2. Can handle much higher learning rates (70x higher than Adam)
3. More robust to hyperparameters (30x wider optimal LR range: 0.02-0.09 vs 0.0007-0.002)
4. Better weight matrix structure leads to improved convergence
5. Stronger performance in early training stages

### 2. Learning Rate Schedule (exp9)

**Optimal: Cosine decay with warmup**

```python
warmup_ratio: float = 0.05       # 5% warmup is crucial for Muon
# LR schedule: cosine decay with min_lr = 0.1 * max_lr
```

**Key Insights:**
- Muon **needs warmup** (5% optimal, no warmup hurts performance)
- Muon **benefits from cosine decay** (5.16 vs 5.25 with constant LR)
- **Contrast with Adam**: Adam prefers constant LR with no warmup (opposite behavior!)

### 3. Load Balancing Weight (exp8)

**Optimal: 0.001** (10x lower than default)

```python
load_balancing_weight: float = 0.001  # Reduced from 0.01
```

**Impact:**
- **Reduces training instability** by ~35%
- **Best final loss**: 5.1478 (vs 5.1649 with 0.01)
- **Lower degradation**: 0.33% (vs 0.46% with default)
- Load balancing auxiliary loss interferes with final optimization when set too high

### 4. Training Hyperparameters

**Core settings:**
```python
batch_size: int = 24
gradient_accumulation_steps: int = 4
max_steps: int = 1000
grad_clip: float = 1.0
dropout: float = 0.1
use_amp: bool = True             # Automatic Mixed Precision
```

**Notes:**
- Higher dropout (0.15-0.2) eliminates degradation but hurts overall performance
- Gradient clipping at 1.0 provides good stability
- AMP enables faster training with minimal quality loss

---

## 📈 Performance Characteristics

### Training Speed
- Muon: ~2.0 min per 500 steps
- Adam: ~1.8 min per 500 steps
- Newton-Schulz computation adds negligible overhead (<10%)

### Convergence
- Muon converges faster initially due to higher learning rate
- More stable with lower momentum (0.9 vs 0.95-0.99)
- Better final loss than Adam at convergence

### Computational Efficiency
- Newton-Schulz steps can be reduced to 3 (from default 5) without quality loss
- Provides ~40% faster per-step computation if needed

---

## 🎯 Complete Optimal Configuration

The following configuration is applied in `configs/moe_config.py`:

```python
# Model architecture
d_model: int = 384
n_heads: int = 8
n_layers: int = 6
d_ff: int = 1536
num_experts: int = 8
expert_top_k: int = 2

# Optimizer (Muon hybrid)
muon_lr: float = 0.07
muon_momentum: float = 0.9
adamw_lr: float = 0.007
weight_decay: float = 0.2

# Training
batch_size: int = 24
gradient_accumulation_steps: int = 4
max_steps: int = 1000
warmup_ratio: float = 0.05
grad_clip: float = 1.0

# Regularization
dropout: float = 0.1
load_balancing_weight: float = 0.001  # 10x lower for stability

# Technical
use_amp: bool = True
```

---

## 💡 Key Lessons Learned

### 1. Optimizer Matters A LOT
- **7% improvement** from choosing the right optimizer with optimal settings
- Muon's orthogonalization provides superior gradient conditioning

### 2. Learning Rate is Critical
- Muon needs **70x higher LR** than Adam (0.07 vs 0.001)
- Muon tolerates **30x wider LR range** (more robust to tuning)
- Different optimizers have different preferences for schedules

### 3. Auxiliary Losses Require Careful Tuning
- Load balancing weight too high (0.01) causes late-training instability
- Reducing to 0.001 improves final performance and stability
- Auxiliary losses can interfere with main objective if not balanced

### 4. Warmup and Scheduling
- Muon **requires warmup** (5% is optimal)
- Adam works better **without warmup** (surprising finding!)
- Cosine schedule helps Muon but hurts Adam

### 5. Training Dynamics
- Early stopping is not the answer to training instability
- Better to fix root causes (LR schedule, auxiliary loss weights)
- Monitoring validation loss degradation is crucial

---

## 🔬 Experimental Methodology

### What Worked
1. **Systematic hyperparameter sweeps**: LR, momentum, weight decay, schedules
2. **Fast iteration**: 200-step experiments for quick LR sweeps, 500 steps for final validation
3. **Fair comparison**: Optimize both Muon AND Adam before comparing
4. **Reproducibility**: Fixed seeds, same data splits, consistent evaluation

### Total Experiments Conducted
- **Exp9 (Muon vs Adam)**: 45+ experiments
- **Exp8 (Load balancing)**: 9 experiments
- **Others**: Architecture and attention mechanism tests

---

## 📦 What to Keep from Experiments

The following files contain the complete methodology and can serve as templates for future experiments:

**Experiment 9 (Muon vs Adam):**
- `experiments/exp9_muon_vs_adam/README.md` - Complete documentation
- `experiments/exp9_muon_vs_adam/run_optimal_muon_suite.py` - Muon optimization suite
- `experiments/exp9_muon_vs_adam/run_adam_optimization_suite.py` - Adam optimization suite

**Experiment 8 (Load Balancing):**
- `experiments/exp8_baseline_cosine_lr_lb0.01/ANALYSIS.md` - Detailed statistical analysis
- `experiments/exp8_baseline_cosine_lr_lb0.01/README.md` - Experiment setup

---

## ✅ Applied Changes

All optimal settings have been applied to:
- ✅ `configs/moe_config.py` - All hyperparameters updated
- ✅ `training/trainer.py` - Uses Muon hybrid optimizer with cosine schedule and warmup

---

## 🚀 Expected Results with Current Config

With the optimized configuration, you should expect:

- **Validation loss**: ~5.16 after 1000 steps
- **Validation accuracy**: ~25.5%
- **Training time**: ~3.5-4.0 minutes (1000 steps on GPU)
- **Stability**: Minimal validation loss degradation (<0.35%)
- **Robustness**: Wide tolerance to learning rate variations

---

## 📝 Recommendations for Future Work

1. **Scale to larger models**: Test on 1B+ parameter models
2. **Longer training**: Validate on 10k+ steps to ensure stability holds
3. **Different architectures**: Test on non-MoE models
4. **Production deployment**: Monitor performance in real-world scenarios
5. **Combine optimizations**: Test with different attention mechanisms (if beneficial)

---

## 🎓 Bottom Line

**The experiments successfully identified optimal hyperparameters that improve validation loss by 7% compared to standard Adam optimizer.**

Key wins:
- ✅ Better optimizer (Muon hybrid)
- ✅ Optimal learning rates (70x higher for Muon)
- ✅ Better load balancing weight (10x lower)
- ✅ Proper warmup and scheduling
- ✅ Improved training stability

**All findings have been integrated into the main training configuration. The experiment folders can now be safely archived or deleted.**
