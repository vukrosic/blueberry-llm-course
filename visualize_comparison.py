"""
Visualization script for comparing Standard vs Credal MoE routing.
Plots loss curves and credal routing statistics.
"""
import matplotlib.pyplot as plt
import torch
import numpy as np
from pathlib import Path

# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid')

def plot_comparison():
    # Load results
    results = torch.load('./checkpoints/comparison_results.pt', weights_only=False)
    
    standard_hist = results['standard_history']
    credal_hist = results['credal_history']
    credal_stats = results['credal_stats']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # ==================== Plot 1: Loss Comparison ====================
    ax1 = fig.add_subplot(gs[0, :2])
    
    # Plot both loss curves on same graph
    steps_std = standard_hist['steps']
    steps_cred = credal_hist['steps']
    loss_std = standard_hist['val_losses']
    loss_cred = credal_hist['val_losses']
    
    ax1.plot(steps_std, loss_std, 'o-', color='#3b82f6', linewidth=2.5, 
             markersize=8, label='Standard Top-2 Routing', alpha=0.9)
    ax1.plot(steps_cred, loss_cred, 's-', color='#ef4444', linewidth=2.5, 
             markersize=8, label='Credal Dynamic Routing', alpha=0.9)
    
    ax1.set_xlabel('Training Steps', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Validation Loss', fontsize=13, fontweight='bold')
    ax1.set_title('Validation Loss Comparison', fontsize=15, fontweight='bold', pad=15)
    ax1.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Highlight final values
    final_std = loss_std[-1]
    final_cred = loss_cred[-1]
    ax1.annotate(f'{final_std:.3f}', 
                xy=(steps_std[-1], final_std), 
                xytext=(10, -15), textcoords='offset points',
                fontsize=10, color='#3b82f6', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    ax1.annotate(f'{final_cred:.3f}', 
                xy=(steps_cred[-1], final_cred), 
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, color='#ef4444', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # ==================== Plot 2: Final Metrics Bar Chart ====================
    ax2 = fig.add_subplot(gs[0, 2])
    
    metrics = ['Loss', 'Accuracy', 'Perplexity']
    std_vals = [results['standard']['val_loss'], 
                results['standard']['val_accuracy'] * 10,  # Scale for visibility
                results['standard']['val_perplexity'] / 100]  # Scale for visibility
    cred_vals = [results['credal']['val_loss'], 
                 results['credal']['val_accuracy'] * 10, 
                 results['credal']['val_perplexity'] / 100]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, std_vals, width, label='Standard', 
                    color='#3b82f6', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax2.bar(x + width/2, cred_vals, width, label='Credal', 
                    color='#ef4444', alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax2.set_ylabel('Normalized Values', fontsize=11, fontweight='bold')
    ax2.set_title('Final Metrics Comparison', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add note about scaling
    ax2.text(0.5, -0.15, 'Note: Accuracy ×10, Perplexity ÷100', 
             transform=ax2.transAxes, ha='center', fontsize=8, style='italic', alpha=0.7)
    
    # ==================== Plot 3: Expert Selection Distribution ====================
    if credal_stats and 'expert_counts' in credal_stats:
        ax3 = fig.add_subplot(gs[1, 0])
        
        expert_counts = credal_stats['expert_counts']
        unique, counts = np.unique(expert_counts, return_counts=True)
        
        bars = ax3.bar(unique, counts, color='#10b981', alpha=0.7, edgecolor='black', linewidth=1.2)
        
        # Highlight bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax3.set_xlabel('Number of Experts Selected', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax3.set_title('Distribution of Expert Selection', fontsize=13, fontweight='bold', pad=10)
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add statistics text
        stats_text = f"Mean: {credal_stats['mean_experts']:.2f}\nStd: {credal_stats['std_experts']:.2f}"
        ax3.text(0.95, 0.95, stats_text, transform=ax3.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10, fontweight='bold')
    
    # ==================== Plot 4: Expert Usage Distribution (Which Experts) ====================
    if credal_stats and 'expert_usage' in credal_stats:
        ax4 = fig.add_subplot(gs[1, 1])
        
        expert_usage = credal_stats['expert_usage']
        expert_ids = list(range(len(expert_usage)))
        total_selections = sum(expert_usage)
        expert_percentages = [count / total_selections * 100 for count in expert_usage]
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(expert_usage)))
        bars = ax4.bar(expert_ids, expert_usage, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Add percentage labels on bars
        for i, (bar, count, pct) in enumerate(zip(bars, expert_usage, expert_percentages)):
            height = bar.get_height()
            if count > 0:
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{count}\n({pct:.1f}%)',
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax4.set_xlabel('Expert ID', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Selection Count', fontsize=11, fontweight='bold')
        ax4.set_title('Individual Expert Usage Distribution', fontsize=13, fontweight='bold', pad=10)
        ax4.set_xticks(expert_ids)
        ax4.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add average line
        avg_usage = np.mean(expert_usage)
        ax4.axhline(y=avg_usage, color='red', linestyle='--', linewidth=2, 
                   label=f'Average: {avg_usage:.0f}', alpha=0.7)
        ax4.legend(fontsize=9)
    else:
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.text(0.5, 0.5, 'Expert usage data\nnot available', 
                transform=ax4.transAxes, ha='center', va='center', fontsize=12)
        ax4.axis('off')
    
    # ==================== Plot 5: Uncertainty Over Time ====================
    if credal_stats and 'uncertainties' in credal_stats:
        ax5 = fig.add_subplot(gs[1, 2])
        
        uncertainties = credal_stats['uncertainties'][:500]  # First 500 samples
        ax5.plot(uncertainties, color='#f59e0b', linewidth=1.5, alpha=0.7)
        ax5.axhline(y=credal_stats['mean_uncertainty'], color='#dc2626', 
                   linestyle='--', linewidth=2, label=f"Mean: {credal_stats['mean_uncertainty']:.4f}")
        
        ax5.set_xlabel('Forward Pass (First 500)', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Epistemic Uncertainty', fontsize=11, fontweight='bold')
        ax5.set_title('Uncertainty Evolution', fontsize=13, fontweight='bold', pad=10)
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3, linestyle='--')
    
    # ==================== Plot 6: Accuracy Comparison ====================
    ax6 = fig.add_subplot(gs[2, 0])
    
    acc_std = standard_hist['val_accuracies']
    acc_cred = credal_hist['val_accuracies']
    
    ax6.plot(steps_std, acc_std, 'o-', color='#3b82f6', linewidth=2, 
             markersize=6, label='Standard', alpha=0.8)
    ax6.plot(steps_cred, acc_cred, 's-', color='#ef4444', linewidth=2, 
             markersize=6, label='Credal', alpha=0.8)
    
    ax6.set_xlabel('Training Steps', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Validation Accuracy', fontsize=11, fontweight='bold')
    ax6.set_title('Accuracy Comparison', fontsize=13, fontweight='bold', pad=10)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3, linestyle='--')
    
    # ==================== Plot 7: Perplexity Comparison ====================
    ax7 = fig.add_subplot(gs[2, 1])
    
    ppl_std = standard_hist['val_perplexities']
    ppl_cred = credal_hist['val_perplexities']
    
    ax7.plot(steps_std, ppl_std, 'o-', color='#3b82f6', linewidth=2, 
             markersize=6, label='Standard', alpha=0.8)
    ax7.plot(steps_cred, ppl_cred, 's-', color='#ef4444', linewidth=2, 
             markersize=6, label='Credal', alpha=0.8)
    
    ax7.set_xlabel('Training Steps', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Validation Perplexity', fontsize=11, fontweight='bold')
    ax7.set_title('Perplexity Comparison', fontsize=13, fontweight='bold', pad=10)
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3, linestyle='--')
    
    # ==================== Plot 8: Summary Statistics ====================
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    # Create summary table
    summary_text = f"""
    COMPARISON SUMMARY
    {'='*35}
    
    Standard MoE (Top-2):
      Val Loss:      {results['standard']['val_loss']:.4f}
      Val Accuracy:  {results['standard']['val_accuracy']:.4f}
      Val Perplexity: {results['standard']['val_perplexity']:.2f}
    
    Credal MoE (Dynamic):
      Val Loss:      {results['credal']['val_loss']:.4f}
      Val Accuracy:  {results['credal']['val_accuracy']:.4f}
      Val Perplexity: {results['credal']['val_perplexity']:.2f}
    
    Differences (Credal - Standard):
      Δ Loss:        {results['differences']['val_loss']:+.4f}
      Δ Accuracy:    {results['differences']['val_accuracy']:+.4f}
      Δ Perplexity:  {results['differences']['val_perplexity']:+.2f}
    """
    
    if credal_stats:
        summary_text += f"""
    Credal Routing Stats:
      Mean Experts:  {credal_stats['mean_experts']:.2f} ± {credal_stats['std_experts']:.2f}
      Range:         [{credal_stats['min_experts']}, {credal_stats['max_experts']}]
      Mean Uncertainty: {credal_stats['mean_uncertainty']:.4f}
    """
    
    ax8.text(0.1, 0.95, summary_text, transform=ax8.transAxes,
            verticalalignment='top', fontfamily='monospace', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
    
    # Overall title
    fig.suptitle('Credal vs Standard MoE Routing - Comprehensive Comparison (50 Steps)', 
                fontsize=17, fontweight='bold', y=0.995)
    
    # Save figure
    output_path = './comparison_plots.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✅ Comprehensive comparison plots saved to {output_path}")
    
    plt.show()

if __name__ == "__main__":
    plot_comparison()
