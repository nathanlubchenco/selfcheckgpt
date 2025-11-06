#!/usr/bin/env python3
"""
Investigate normalization strategies for multi-sentence coherence scoring.

This script compares different normalization approaches to understand if min-max
normalization is washing out important differences in coherence scores.

Usage:
    python scripts/investigate_normalization.py --num-passages 50 --variant shogenji
"""

import argparse
import numpy as np
from datasets import load_dataset
from scipy.stats import pearsonr
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from selfcheckgpt.modeling_coherence import SelfCheckShogenji, SelfCheckFitelson, SelfCheckOlsson
from selfcheckgpt.utils_coherence import normalize_coherence_scores


def normalize_minmax(scores):
    """Current min-max normalization (what we use now)."""
    return normalize_coherence_scores(scores)


def normalize_percentile(scores, lower=0.1, upper=0.9):
    """
    Percentile-based normalization (robust to outliers).

    Maps lower_percentile -> 0.0 and upper_percentile -> 1.0
    """
    scores = np.asarray(scores)
    p_lower = np.percentile(scores, lower * 100)
    p_upper = np.percentile(scores, upper * 100)

    if abs(p_upper - p_lower) < 1e-12:
        return np.full_like(scores, 0.5, dtype=np.float64)

    normalized = (scores - p_lower) / (p_upper - p_lower)
    return np.clip(normalized, 0.0, 1.0)


def normalize_zscore(scores):
    """
    Z-score normalization with sigmoid transformation.

    Converts to z-scores then applies sigmoid to map to [0, 1].
    """
    scores = np.asarray(scores)
    mean = np.mean(scores)
    std = np.std(scores)

    if std < 1e-12:
        return np.full_like(scores, 0.5, dtype=np.float64)

    z_scores = (scores - mean) / std
    # Apply sigmoid: 1 / (1 + exp(-z))
    normalized = 1.0 / (1.0 + np.exp(-z_scores))
    return normalized


def normalize_rank(scores):
    """
    Rank-based normalization.

    Converts scores to ranks, then normalizes ranks to [0, 1].
    """
    scores = np.asarray(scores)
    # Get ranks (0-indexed)
    ranks = np.argsort(np.argsort(scores))
    # Normalize to [0, 1]
    if len(scores) == 1:
        return np.array([0.5])
    normalized = ranks / (len(scores) - 1)
    return normalized


def normalize_softmax(scores, temperature=1.0):
    """
    Softmax-based normalization (emphasizes differences).

    Higher temperature = more uniform, lower = more extreme.
    """
    scores = np.asarray(scores)
    # Shift scores to prevent overflow
    scores_shifted = scores - np.max(scores)
    exp_scores = np.exp(scores_shifted / temperature)
    softmax = exp_scores / np.sum(exp_scores)

    # Convert to cumulative for ranking effect
    sorted_indices = np.argsort(scores)
    cumulative = np.zeros_like(softmax)
    cumulative[sorted_indices] = np.cumsum(softmax[sorted_indices])

    return cumulative


def normalize_none(scores):
    """No normalization - use raw coherence scores."""
    # For Shogenji (unbounded), clip to reasonable range
    # For Olsson (already [0,1]), return as-is
    scores = np.asarray(scores)
    # Apply exponential mapping for unbounded scores (like our single-sentence fix)
    return np.exp(-scores)


def evaluate_with_normalization(variant, dataset, num_passages, normalization_fn,
                                normalization_name, num_samples=3):
    """
    Evaluate coherence variant with a specific normalization strategy.

    Args:
        variant: Coherence variant instance
        dataset: HuggingFace dataset
        num_passages: Number of passages to evaluate
        normalization_fn: Function to normalize scores
        normalization_name: Name for display
        num_samples: Number of stochastic samples to use

    Returns:
        dict with metrics and raw data
    """
    all_scores = []
    all_labels = []
    all_raw_coherence = []  # Before normalization

    print(f"\nEvaluating with {normalization_name}...")

    for idx in tqdm(range(num_passages), desc=f"{normalization_name}"):
        passage = dataset[idx]
        sentences = passage['gpt3_sentences']
        sampled_passages = passage['gpt3_text_samples'][:num_samples]
        annotations = passage['annotation']
        labels = [0 if ann == 'accurate' else 1 for ann in annotations]

        # Get raw coherence scores (before final normalization)
        # We'll monkey-patch the normalization function temporarily
        import selfcheckgpt.modeling_coherence as coherence_module

        # Store original normalization function
        original_normalize = coherence_module.normalize_coherence_scores

        # Replace with identity function to get raw coherence
        coherence_module.normalize_coherence_scores = lambda x: np.asarray(x)

        try:
            # Get raw coherence scores
            raw_scores = variant.predict(sentences, sampled_passages, verbose=False)
            all_raw_coherence.extend(raw_scores.tolist())

            # Now apply our normalization strategy
            # For multi-sentence, we normalize within passage
            if len(raw_scores) > 1:
                normalized = normalization_fn(raw_scores)
                # Invert to hallucination scores
                hallucination_scores = 1.0 - normalized
            else:
                # Single sentence: use exponential mapping
                hallucination_scores = np.exp(-raw_scores)

            all_scores.extend(hallucination_scores.tolist())
            all_labels.extend(labels)
        finally:
            # Restore original normalization function
            coherence_module.normalize_coherence_scores = original_normalize

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    all_raw_coherence = np.array(all_raw_coherence)

    # Calculate metrics
    if len(np.unique(all_labels)) == 2:
        auc_pr = average_precision_score(all_labels, all_scores)
        auc_roc = roc_auc_score(all_labels, all_scores)
        pcc, _ = pearsonr(all_scores, all_labels)
    else:
        auc_pr = auc_roc = pcc = 0.0

    return {
        'name': normalization_name,
        'auc_pr': auc_pr,
        'auc_roc': auc_roc,
        'pcc': pcc,
        'scores': all_scores,
        'labels': all_labels,
        'raw_coherence': all_raw_coherence
    }


def main():
    parser = argparse.ArgumentParser(description="Investigate normalization strategies")
    parser.add_argument('--num-passages', type=int, default=50,
                       help='Number of passages to evaluate (default: 50)')
    parser.add_argument('--variant', type=str, default='shogenji',
                       choices=['shogenji', 'fitelson', 'olsson'],
                       help='Coherence variant to test (default: shogenji)')
    parser.add_argument('--num-samples', type=int, default=3,
                       help='Number of stochastic samples per passage (default: 3)')
    parser.add_argument('--output', type=str, default=None,
                       help='Save plots to file (optional)')

    args = parser.parse_args()

    print("="*80)
    print("NORMALIZATION INVESTIGATION")
    print("="*80)
    print(f"Variant: {args.variant}")
    print(f"Passages: {args.num_passages}")
    print(f"Samples per passage: {args.num_samples}")
    print()

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination")["evaluation"]
    num_passages = min(args.num_passages, len(dataset))

    # Initialize variant
    if args.variant == 'shogenji':
        variant = SelfCheckShogenji(model="gpt-4o-mini")
    elif args.variant == 'fitelson':
        variant = SelfCheckFitelson(model="gpt-4o-mini")
    else:
        variant = SelfCheckOlsson(model="gpt-4o-mini")

    # Test different normalization strategies
    strategies = {
        'Min-Max (current)': normalize_minmax,
        'Percentile (10-90)': normalize_percentile,
        'Z-Score + Sigmoid': normalize_zscore,
        'Rank-Based': normalize_rank,
        'Softmax (T=1.0)': normalize_softmax,
        'None (exponential)': normalize_none,
    }

    results = []
    for name, norm_fn in strategies.items():
        result = evaluate_with_normalization(
            variant, dataset, num_passages, norm_fn, name, args.num_samples
        )
        results.append(result)

    # Print comparison table
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    print(f"{'Strategy':<25} {'AUC-PR':<12} {'AUC-ROC':<12} {'PCC':<12}")
    print("-"*80)

    for result in results:
        print(f"{result['name']:<25} {result['auc_pr']:<12.4f} {result['auc_roc']:<12.4f} {result['pcc']:<12.4f}")

    # Find best strategy
    best_by_auc_pr = max(results, key=lambda x: x['auc_pr'])
    best_by_pcc = max(results, key=lambda x: x['pcc'])

    print("-"*80)
    print(f"Best by AUC-PR: {best_by_auc_pr['name']} ({best_by_auc_pr['auc_pr']:.4f})")
    print(f"Best by PCC: {best_by_pcc['name']} ({best_by_pcc['pcc']:.4f})")

    # Calculate improvement over current method
    current_result = next(r for r in results if r['name'] == 'Min-Max (current)')
    print(f"\nImprovement over current min-max normalization:")
    print(f"  AUC-PR: {(best_by_auc_pr['auc_pr'] - current_result['auc_pr'])*100:.2f}% ({best_by_auc_pr['name']})")
    print(f"  PCC: {(best_by_pcc['pcc'] - current_result['pcc'])*100:.2f}% ({best_by_pcc['name']})")

    # Visualizations
    print("\nGenerating visualizations...")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Metrics comparison
    ax1 = axes[0, 0]
    x = np.arange(len(results))
    width = 0.25
    ax1.bar(x - width, [r['auc_pr'] for r in results], width, label='AUC-PR', alpha=0.8)
    ax1.bar(x, [r['auc_roc'] for r in results], width, label='AUC-ROC', alpha=0.8)
    ax1.bar(x + width, [r['pcc'] for r in results], width, label='PCC', alpha=0.8)
    ax1.set_ylabel('Score', fontweight='bold')
    ax1.set_title('Metrics Comparison', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([r['name'] for r in results], rotation=45, ha='right', fontsize=9)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # 2. Score distributions
    ax2 = axes[0, 1]
    for result in results[:3]:  # Show first 3 to avoid clutter
        ax2.hist(result['scores'], bins=30, alpha=0.5, label=result['name'], density=True)
    ax2.set_xlabel('Hallucination Score', fontweight='bold')
    ax2.set_ylabel('Density', fontweight='bold')
    ax2.set_title('Score Distributions (Top 3)', fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    # 3. Raw coherence distribution
    ax3 = axes[0, 2]
    raw_coherence = results[0]['raw_coherence']  # Same for all strategies
    ax3.hist(raw_coherence, bins=30, color='purple', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Raw Coherence Score', fontweight='bold')
    ax3.set_ylabel('Count', fontweight='bold')
    ax3.set_title('Raw Coherence Distribution (Before Normalization)', fontweight='bold')
    ax3.axvline(np.mean(raw_coherence), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(raw_coherence):.3f}')
    ax3.axvline(np.median(raw_coherence), color='orange', linestyle='--', linewidth=2,
               label=f'Median: {np.median(raw_coherence):.3f}')
    ax3.legend()
    ax3.grid(alpha=0.3)

    # 4. Score separation (mean inaccurate - mean accurate)
    ax4 = axes[1, 0]
    separations = []
    for result in results:
        scores = result['scores']
        labels = result['labels']
        mean_accurate = np.mean(scores[labels == 0])
        mean_inaccurate = np.mean(scores[labels == 1])
        separation = mean_inaccurate - mean_accurate
        separations.append(separation)

    colors = ['green' if s > 0 else 'red' for s in separations]
    ax4.barh(range(len(results)), separations, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_yticks(range(len(results)))
    ax4.set_yticklabels([r['name'] for r in results], fontsize=9)
    ax4.set_xlabel('Separation (Inaccurate - Accurate)', fontweight='bold')
    ax4.set_title('Score Separation by Strategy', fontweight='bold')
    ax4.axvline(0, color='black', linestyle='-', linewidth=1)
    ax4.grid(axis='x', alpha=0.3)

    # 5. Correlation with labels
    ax5 = axes[1, 1]
    # Show scatter for best method
    best_result = best_by_auc_pr
    ax5.scatter(best_result['labels'], best_result['scores'], alpha=0.3, s=20)
    ax5.set_xlabel('Ground Truth (0=Accurate, 1=Inaccurate)', fontweight='bold')
    ax5.set_ylabel('Hallucination Score', fontweight='bold')
    ax5.set_title(f'Best Method: {best_result["name"]}\n(PCC={best_result["pcc"]:.3f})', fontweight='bold')
    ax5.grid(alpha=0.3)

    # Add jitter for visibility
    jittered_labels = best_result['labels'] + np.random.normal(0, 0.05, len(best_result['labels']))
    ax5.scatter(jittered_labels, best_result['scores'], alpha=0.3, s=20, color='blue')

    # 6. Improvement heatmap
    ax6 = axes[1, 2]
    metrics_matrix = np.array([
        [r['auc_pr'] for r in results],
        [r['auc_roc'] for r in results],
        [r['pcc'] for r in results]
    ])
    im = ax6.imshow(metrics_matrix, cmap='RdYlGn', aspect='auto')
    ax6.set_xticks(range(len(results)))
    ax6.set_xticklabels([r['name'] for r in results], rotation=45, ha='right', fontsize=9)
    ax6.set_yticks(range(3))
    ax6.set_yticklabels(['AUC-PR', 'AUC-ROC', 'PCC'], fontsize=10)
    ax6.set_title('Metrics Heatmap', fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax6)
    cbar.set_label('Score', fontweight='bold')

    # Add values to heatmap
    for i in range(3):
        for j in range(len(results)):
            text = ax6.text(j, i, f'{metrics_matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=8)

    plt.suptitle(f'Normalization Strategy Investigation ({args.variant.capitalize()}, {num_passages} passages)',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {args.output}")
    else:
        plt.show()

    # Detailed analysis of raw coherence scores
    print("\n" + "="*80)
    print("RAW COHERENCE SCORE ANALYSIS")
    print("="*80)
    print(f"Mean: {np.mean(raw_coherence):.4f}")
    print(f"Median: {np.median(raw_coherence):.4f}")
    print(f"Std Dev: {np.std(raw_coherence):.4f}")
    print(f"Min: {np.min(raw_coherence):.4f}")
    print(f"Max: {np.max(raw_coherence):.4f}")
    print(f"Range: {np.max(raw_coherence) - np.min(raw_coherence):.4f}")
    print(f"\nPercentiles:")
    for p in [10, 25, 50, 75, 90]:
        print(f"  {p}th: {np.percentile(raw_coherence, p):.4f}")

    # Check if scores are very similar (might explain poor separation)
    unique_scores = len(np.unique(np.round(raw_coherence, 3)))
    print(f"\nUnique scores (rounded to 3 decimals): {unique_scores} / {len(raw_coherence)}")
    if unique_scores < len(raw_coherence) * 0.5:
        print("⚠️  WARNING: Many duplicate scores detected!")
        print("   This suggests limited granularity in coherence measurements.")

    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)

    improvement = (best_by_auc_pr['auc_pr'] - current_result['auc_pr']) * 100
    if improvement > 1.0:
        print(f"✅ SIGNIFICANT IMPROVEMENT FOUND!")
        print(f"   Switching to '{best_by_auc_pr['name']}' could improve AUC-PR by {improvement:.2f}%")
        print(f"   This is worth implementing!")
    elif improvement > 0.1:
        print(f"✓ Modest improvement found")
        print(f"   '{best_by_auc_pr['name']}' improves AUC-PR by {improvement:.2f}%")
        print(f"   Consider A/B testing this strategy")
    else:
        print(f"⚠️  No significant improvement found")
        print(f"   Current min-max normalization appears optimal for this variant")
        print(f"   The performance gap likely comes from other factors")

    print("="*80)


if __name__ == "__main__":
    main()
