# Blueberry LLM Course - Completion Plan (Revised)

This plan focuses strictly on the existing codebase and the specific lessons requested.

## New Modules

### 5_tokenization
- **500_tokenization.ipynb**: Using `AutoTokenizer` with `SmolLM-135M`. Explaining how text becomes numbers.

### 6_data_pipeline
- **600_data_pipeline.ipynb**: Loading `cosmopedia-v2`, using `DataLoader`, and preparing batches.

### 4_transformers (Extensions)
- **700_rmsnorm.ipynb**: Implementation of Root Mean Square Normalization.
- **710_silu_activation.ipynb**: The SiLU activation function (used in our Experts).
- **720_rope.ipynb**: Rotary Positional Embeddings using `torchtune`.
- **730_moe_router.ipynb**: The `TopKRouter` implementation.
- **740_moe_experts.ipynb**: The `Expert` block and `MixtureOfExperts` layer.

## Content Strategy
- **Step-by-step**: Small code blocks with explanations.
- **No huge dumps**: Break down classes into methods and explain logic.
- **Strictly Codebase**: Only teach what is in `models/` and `train_moe.py`.
