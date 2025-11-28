import matplotlib.pyplot as plt
import torch
import numpy as np

# Load results
results = torch.load('./checkpoints/comparison_results.pt')

standard = results['standard']
credal = results['credal']

# Create comparison plots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Metrics to compare
metrics = ['val_loss', 'val_accuracy', 'val_perplexity']
titles = ['Validation Loss', 'Validation Accuracy', 'Validation Perplexity']
colors = ['#3b82f6', '#ef4444']  # Blue for standard, red for credal

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[idx]
    
    values = [standard[metric], credal[metric]]
    bars = ax.bar(['Standard MoE', 'Credal MoE'], values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Value', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add difference annotation
    diff = results['differences'][metric]
    diff_text = f'Δ: {diff:+.4f}'
    ax.text(0.5, 0.95, diff_text, transform=ax.transAxes,
            ha='center', va='top', fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

plt.suptitle('Credal vs Standard MoE Routing Comparison (50 Steps)', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('./comparison_plot.png', dpi=150, bbox_inches='tight')
print("Plot saved to ./comparison_plot.png")
