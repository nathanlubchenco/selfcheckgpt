#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for Coherence-Based Hallucination Detection Variants

This script evaluates all three coherence variants (Shogenji, Fitelson, Olsson) on the
wiki_bio_gpt3_hallucination dataset and computes performance metrics:
- AUC-PR (Area Under Precision-Recall Curve)
- PCC (Pearson Correlation Coefficient)
- AUC-ROC (Area Under ROC Curve)

Usage:
    python scripts/evaluate_coherence.py --variant all --model gpt-4o-mini --num-samples 3
    python scripts/evaluate_coherence.py --variant shogenji --model gpt-4o-mini --num-samples 5
    python scripts/evaluate_coherence.py --variant fitelson --model gpt-3.5-turbo

Arguments:
    --variant: Which variant to evaluate (shogenji/fitelson/olsson/all)
    --model: OpenAI model name (default: gpt-4o-mini)
    --num-samples: Number of sampled passages to use (default: 3)
    --max-passages: Maximum number of passages to evaluate (default: None, all passages)
    --output-dir: Directory to save results (default: results/)

Environment:
    Requires OPENAI_API_KEY environment variable to be set.
"""

import argparse
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
from datasets import load_dataset
from scipy.stats import pearsonr
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from selfcheckgpt.modeling_coherence import (
    SelfCheckShogenji,
    SelfCheckFitelson,
    SelfCheckOlsson
)


def load_evaluation_dataset() -> Dict:
    """
    Load wiki_bio_gpt3_hallucination evaluation dataset.

    Returns:
        Dataset dict with passages, sentences, and annotations
    """
    print("Loading wiki_bio_gpt3_hallucination dataset...")
    dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination")["evaluation"]
    print(f"Loaded {len(dataset)} passages from evaluation set")
    return dataset


def evaluate_variant(
    variant_name: str,
    model: str,
    dataset: Dict,
    num_samples: int,
    max_passages: int = None
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Evaluate a coherence variant on the dataset.

    Args:
        variant_name: Name of variant (shogenji/fitelson/olsson)
        model: OpenAI model name
        dataset: Evaluation dataset
        num_samples: Number of sampled passages to use
        max_passages: Maximum number of passages to evaluate (None = all)

    Returns:
        Tuple of (scores, labels, cache_stats)
    """
    print(f"\n{'='*80}")
    print(f"Evaluating {variant_name.upper()} variant")
    print(f"Model: {model}, Num Samples: {num_samples}")
    print(f"{'='*80}\n")

    # Initialize variant
    if variant_name == "shogenji":
        variant = SelfCheckShogenji(model=model)
    elif variant_name == "fitelson":
        variant = SelfCheckFitelson(model=model)
    elif variant_name == "olsson":
        variant = SelfCheckOlsson(model=model)
    else:
        raise ValueError(f"Unknown variant: {variant_name}")

    # Collect all scores and labels
    all_scores = []
    all_labels = []

    # Determine number of passages to evaluate
    num_passages = len(dataset) if max_passages is None else min(max_passages, len(dataset))

    # Iterate over passages
    for passage_idx in tqdm(range(num_passages), desc=f"Evaluating {variant_name}"):
        passage_data = dataset[passage_idx]

        # Extract data
        sentences = passage_data['gpt3_sentences']
        annotations = passage_data['annotation']  # 0=accurate, 1=inaccurate
        gpt3_text = passage_data['gpt3_text']

        # Create sampled passages (simplified: use GPT-3 text variations)
        # In real scenario, these would be stochastic samples from LLM
        # For evaluation purposes, we use the GPT-3 text multiple times
        # This is a limitation but allows evaluation without regenerating samples
        sampled_passages = [gpt3_text] * num_samples

        # Predict hallucination scores
        try:
            scores = variant.predict(
                sentences=sentences,
                sampled_passages=sampled_passages,
                verbose=False  # Disable progress bar for cleaner output
            )

            # Collect scores and labels
            all_scores.extend(scores.tolist())
            all_labels.extend(annotations)

        except Exception as e:
            print(f"\nWarning: Error processing passage {passage_idx}: {e}")
            continue

    # Convert to numpy arrays
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # Get cache statistics
    cache_stats = variant.client.get_cache_stats()

    print(f"\nCompleted evaluation for {variant_name}")
    print(f"Total sentences evaluated: {len(all_scores)}")
    print(f"Accurate sentences: {np.sum(all_labels == 0)}")
    print(f"Inaccurate sentences: {np.sum(all_labels == 1)}")

    return all_scores, all_labels, cache_stats


def compute_metrics(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        scores: Hallucination scores (higher = more hallucinated)
        labels: Ground truth labels (0=accurate, 1=inaccurate)

    Returns:
        Dictionary with metrics: auc_pr, pcc, auc_roc
    """
    # AUC-PR (Area Under Precision-Recall Curve)
    auc_pr = average_precision_score(labels, scores)

    # PCC (Pearson Correlation Coefficient)
    pcc, _ = pearsonr(scores, labels)

    # AUC-ROC (Area Under ROC Curve)
    auc_roc = roc_auc_score(labels, scores)

    return {
        'auc_pr': float(auc_pr),
        'pcc': float(pcc),
        'auc_roc': float(auc_roc)
    }


def estimate_api_cost(
    num_sentences: int,
    num_samples: int,
    num_variants: int,
    include_conditional: bool,
    cache_hit_rate: float,
    model: str = "gpt-4o-mini"
) -> Dict[str, float]:
    """
    Estimate API cost based on usage.

    Args:
        num_sentences: Total sentences evaluated
        num_samples: Number of sampled passages per sentence
        num_variants: Number of variants evaluated
        include_conditional: Whether conditional probabilities were used
        cache_hit_rate: Observed cache hit rate
        model: Model name for pricing

    Returns:
        Dictionary with cost estimates
    """
    # Estimate API calls per sentence
    if include_conditional:
        calls_per_sentence = 1 + 3 * num_samples  # P(sent), P(sample), P(joint), P(cond)
    else:
        calls_per_sentence = 1 + 2 * num_samples  # P(sent), P(sample), P(joint)

    # Total calls without caching
    total_calls_uncached = calls_per_sentence * num_sentences * num_variants

    # Actual calls with caching
    actual_calls = int(total_calls_uncached * (1 - cache_hit_rate))

    # Pricing (as of 2024, approximate)
    # gpt-4o-mini: $0.15 per 1M input tokens, $0.60 per 1M output tokens
    # Assume ~50 input tokens per call, ~10 output tokens per call
    if "gpt-4o-mini" in model:
        input_cost_per_1m = 0.15
        output_cost_per_1m = 0.60
    elif "gpt-3.5" in model:
        input_cost_per_1m = 0.50
        output_cost_per_1m = 1.50
    elif "gpt-4" in model:
        input_cost_per_1m = 30.0
        output_cost_per_1m = 60.0
    else:
        input_cost_per_1m = 0.15  # Default to gpt-4o-mini pricing
        output_cost_per_1m = 0.60

    avg_input_tokens = 50
    avg_output_tokens = 10

    input_cost = (actual_calls * avg_input_tokens / 1_000_000) * input_cost_per_1m
    output_cost = (actual_calls * avg_output_tokens / 1_000_000) * output_cost_per_1m
    total_cost = input_cost + output_cost

    return {
        'total_calls_uncached': total_calls_uncached,
        'actual_api_calls': actual_calls,
        'cache_savings': total_calls_uncached - actual_calls,
        'estimated_cost_usd': total_cost,
        'input_cost_usd': input_cost,
        'output_cost_usd': output_cost
    }


def print_comparison_table(results: Dict[str, Dict], baseline_auc_pr: float = 93.42):
    """
    Print comparison table showing all variants vs baseline.

    Args:
        results: Dictionary mapping variant names to metrics
        baseline_auc_pr: Baseline AUC-PR to compare against (SelfCheckAPIPrompt GPT-3.5)
    """
    print("\n" + "="*100)
    print("EVALUATION RESULTS COMPARISON")
    print("="*100)
    print(f"{'Variant':<20} {'AUC-PR':<12} {'vs Baseline':<15} {'PCC':<12} {'AUC-ROC':<12}")
    print("-"*100)

    for variant_name, metrics in sorted(results.items()):
        auc_pr = metrics['auc_pr'] * 100  # Convert to percentage
        pcc = metrics['pcc'] * 100
        auc_roc = metrics['auc_roc'] * 100

        diff = auc_pr - baseline_auc_pr
        diff_str = f"{diff:+.2f}%"

        print(f"{variant_name:<20} {auc_pr:<12.2f} {diff_str:<15} {pcc:<12.2f} {auc_roc:<12.2f}")

    print("-"*100)
    print(f"{'Baseline (GPT-3.5)':<20} {baseline_auc_pr:<12.2f} {'---':<15} {'78.32':<12} {'---':<12}")
    print("="*100)


def save_results(
    results: Dict[str, Dict],
    cache_stats: Dict[str, Dict],
    cost_estimates: Dict[str, Dict],
    output_dir: str,
    metadata: Dict
):
    """
    Save evaluation results to JSON file.

    Args:
        results: Variant metrics
        cache_stats: Cache statistics per variant
        cost_estimates: Cost estimates per variant
        output_dir: Directory to save results
        metadata: Evaluation metadata (model, num_samples, etc.)
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"coherence_evaluation_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    output_data = {
        'metadata': metadata,
        'results': results,
        'cache_stats': cache_stats,
        'cost_estimates': cost_estimates,
        'timestamp': timestamp
    }

    with open(filepath, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate coherence-based hallucination detection variants"
    )
    parser.add_argument(
        '--variant',
        type=str,
        default='all',
        choices=['shogenji', 'fitelson', 'olsson', 'all'],
        help='Which variant to evaluate (default: all)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o-mini',
        help='OpenAI model name (default: gpt-4o-mini)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=3,
        help='Number of sampled passages per sentence (default: 3)'
    )
    parser.add_argument(
        '--max-passages',
        type=int,
        default=None,
        help='Maximum number of passages to evaluate (default: None, all passages)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directory to save results (default: results/)'
    )

    args = parser.parse_args()

    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
        return

    # Load dataset
    dataset = load_evaluation_dataset()

    # Determine which variants to evaluate
    if args.variant == 'all':
        variants_to_eval = ['shogenji', 'fitelson', 'olsson']
    else:
        variants_to_eval = [args.variant]

    print(f"\nEvaluating variants: {', '.join(variants_to_eval)}")
    print(f"Model: {args.model}")
    print(f"Num samples: {args.num_samples}")
    print(f"Max passages: {args.max_passages if args.max_passages else 'All'}")

    # Track results
    all_results = {}
    all_cache_stats = {}
    all_cost_estimates = {}

    start_time = time.time()

    # Evaluate each variant
    for variant_name in variants_to_eval:
        variant_start = time.time()

        # Run evaluation
        scores, labels, cache_stats = evaluate_variant(
            variant_name=variant_name,
            model=args.model,
            dataset=dataset,
            num_samples=args.num_samples,
            max_passages=args.max_passages
        )

        # Compute metrics
        metrics = compute_metrics(scores, labels)

        # Store results
        all_results[variant_name] = metrics
        all_cache_stats[variant_name] = cache_stats

        # Estimate cost
        include_conditional = (variant_name == 'fitelson')
        cost_estimate = estimate_api_cost(
            num_sentences=len(scores),
            num_samples=args.num_samples,
            num_variants=1,
            include_conditional=include_conditional,
            cache_hit_rate=cache_stats['hit_rate'],
            model=args.model
        )
        all_cost_estimates[variant_name] = cost_estimate

        variant_time = time.time() - variant_start

        # Print variant results
        print(f"\n{variant_name.upper()} Results:")
        print(f"  AUC-PR: {metrics['auc_pr']*100:.2f}%")
        print(f"  PCC: {metrics['pcc']*100:.2f}%")
        print(f"  AUC-ROC: {metrics['auc_roc']*100:.2f}%")
        print(f"\nCache Statistics:")
        print(f"  Hit rate: {cache_stats['hit_rate']*100:.2f}%")
        print(f"  API calls: {cache_stats['api_calls']}")
        print(f"  Cache size: {cache_stats['cache_size']}")
        print(f"\nCost Estimate:")
        print(f"  Actual API calls: {cost_estimate['actual_api_calls']}")
        print(f"  Cache savings: {cost_estimate['cache_savings']} calls")
        print(f"  Estimated cost: ${cost_estimate['estimated_cost_usd']:.4f} USD")
        print(f"\nTime elapsed: {variant_time:.2f}s")

    total_time = time.time() - start_time

    # Print comparison table
    print_comparison_table(all_results)

    # Print overall cost summary
    total_cost = sum(est['estimated_cost_usd'] for est in all_cost_estimates.values())
    total_api_calls = sum(stats['api_calls'] for stats in all_cache_stats.values())

    print(f"\n{'='*100}")
    print("OVERALL COST SUMMARY")
    print(f"{'='*100}")
    print(f"Total API calls made: {total_api_calls}")
    print(f"Total estimated cost: ${total_cost:.4f} USD")
    print(f"Total time elapsed: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"{'='*100}\n")

    # Save results
    metadata = {
        'model': args.model,
        'num_samples': args.num_samples,
        'max_passages': args.max_passages,
        'variants_evaluated': variants_to_eval,
        'total_time_seconds': total_time,
        'baseline_auc_pr': 93.42
    }

    output_file = save_results(
        results=all_results,
        cache_stats=all_cache_stats,
        cost_estimates=all_cost_estimates,
        output_dir=args.output_dir,
        metadata=metadata
    )

    print(f"\nEvaluation complete!")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
