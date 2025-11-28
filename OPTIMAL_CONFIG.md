# Optimal Configuration Summary

This is a quick reference for the **optimal LLM training configuration** based on comprehensive experiments (45+ runs).

## 🎯 Current Configuration (Optimized)

All settings in [`configs/moe_config.py`](file:///Users/vukrosic/AI%20Science%20Projects/blueberry-llm-course/configs/moe_config.py) are now optimized based on experimental findings.

### Model Architecture
```python
d_model = 384
n_heads = 8
n_layers = 6
d_ff = 1536
num_experts = 8
expert_top_k = 2
```

### Optimizer: Muon Hybrid ⚡
```python
muon_lr = 0.07              # For 2D weight matrices
muon_momentum = 0.9         # Lower momentum works better
adamw_lr = 0.007           # For embeddings and norms
weight_decay = 0.2         # Higher than typical 0.1
```

### Training Schedule
```python
batch_size = 24
gradient_accumulation_steps = 4
max_steps = 1000
warmup_ratio = 0.05        # 5% warmup (crucial!)
# Schedule: Cosine decay with min_lr = 0.1 * max_lr
```

### Regularization
```python
dropout = 0.1
grad_clip = 1.0
load_balancing_weight = 0.001   # 10x lower than default!
```

---

## 📊 Performance vs Baseline

| Metric | Old (Adam) | New (Muon Optimized) | Improvement |
|--------|-----------|---------------------|-------------|
| **Val Loss** | 5.55 | **5.16** | **7% better** ✅ |
| **200-step Loss** | 6.73 | **5.72** | **15% better** ✅ |
| **LR Robustness** | 0.0007-0.002 | **0.02-0.09** | **30x wider range** ✅ |
| **Training Stability** | 0.46% degradation | **0.33% degradation** | **35% better** ✅ |

---

## 🔑 Key Changes Applied

### 1. Optimizer Switch
- **Before**: Pure Adam (LR=0.001)
- **After**: Muon hybrid (Muon LR=0.07 for weights, AdamW LR=0.007 for embeddings)
- **Reason**: 7% better validation loss, 70x higher tolerable learning rate

### 2. Load Balancing Weight
- **Before**: 0.01
- **After**: 0.001 (10x lower)
- **Reason**: Reduces late-training instability and improves final loss

### 3. Learning Rate
- **Before**: 0.001 (Adam)
- **After**: 0.07 (Muon)
- **Reason**: Muon's Newton-Schulz orthogonalization enables much higher LRs

### 4. Weight Decay
- **Before**: 0.1
- **After**: 0.2
- **Reason**: Better regularization with higher learning rates

### 5. Momentum
- **Before**: Not explicitly set (Adam default ~0.999)
- **After**: 0.9 (explicit for Muon)
- **Reason**: Lower momentum provides better convergence for Muon

---

## ✅ Verification Checklist

- [x] Muon hybrid optimizer configured in `training/trainer.py`
- [x] Optimal learning rates applied (Muon: 0.07, AdamW: 0.007)
- [x] Cosine schedule with 5% warmup configured
- [x] Load balancing weight reduced to 0.001
- [x] Weight decay increased to 0.2
- [x] Momentum set to 0.9
- [x] All settings documented in `EXPERIMENT_FINDINGS.md`

---

## 🚀 Expected Results

With the current optimized configuration, you should expect:

- **Validation loss**: ~5.16 (after 1000 steps)
- **Validation accuracy**: ~25.5%
- **Training time**: ~3.5-4 minutes (1000 steps on GPU)
- **Stability**: Minimal loss degradation (<0.35%)
- **Robustness**: Works across wide LR range (0.02-0.09)

---

## 📖 Reference Documents

- **Full experimental analysis**: [`EXPERIMENT_FINDINGS.md`](file:///Users/vukrosic/AI%20Science%20Projects/blueberry-llm-course/EXPERIMENT_FINDINGS.md)
- **Exp9 (Muon vs Adam)**: `experiments/exp9_muon_vs_adam/README.md`
- **Exp8 (Load Balancing)**: `experiments/exp8_baseline_cosine_lr_lb0.01/ANALYSIS.md`

---

## 🎓 One-Liner Summary

**Muon optimizer (LR=0.07, momentum=0.9) + Lower load balancing (0.001) = 7% better loss than Adam**
