#!/usr/bin/env python3
"""
Compare baseline SelfCheckAPIPrompt with coherence methods.
Runs all methods on the same dataset and shows side-by-side comparison.
"""
import argparse
import numpy as np
from datasets import load_dataset
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import pearsonr
from tqdm import tqdm
import json
from datetime import datetime

from selfcheckgpt.modeling_selfcheck_apiprompt import SelfCheckAPIPrompt
from selfcheckgpt import SelfCheckShogenji, SelfCheckFitelson, SelfCheckOlsson


def evaluate_method(method_name, variant, dataset, num_passages, verbose=False):
    """Evaluate a single method and return results."""
    all_scores = []
    all_labels = []

    for idx in tqdm(range(num_passages), desc=f"Evaluating {method_name}", disable=not verbose):
        passage = dataset[idx]

        sentences = passage['gpt3_sentences']
        sampled_passages = passage['gpt3_text_samples']
        annotations = passage['annotation']
        labels = [0 if ann == 'accurate' else 1 for ann in annotations]

        scores = variant.predict(
            sentences=sentences,
            sampled_passages=sampled_passages,
            verbose=False  # Disable inner progress bars
        )

        all_scores.extend(scores.tolist())
        all_labels.extend(labels)

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # Compute metrics
    auc_pr = average_precision_score(all_labels, all_scores)
    auc_roc = roc_auc_score(all_labels, all_scores)
    pcc, _ = pearsonr(all_scores, all_labels)

    return {
        'method': method_name,
        'auc_pr': auc_pr,
        'auc_roc': auc_roc,
        'pcc': pcc,
        'num_sentences': len(all_scores),
        'num_accurate': int(np.sum(all_labels == 0)),
        'num_inaccurate': int(np.sum(all_labels == 1))
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline and coherence methods side-by-side"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o-mini',
        help='OpenAI model name (default: gpt-4o-mini)'
    )
    parser.add_argument(
        '--num-passages',
        type=int,
        default=10,
        help='Number of passages to evaluate (default: 10, use 238 for full evaluation)'
    )
    parser.add_argument(
        '--methods',
        type=str,
        default='all',
        help='Methods to evaluate: all, baseline-only, coherence-only, or comma-separated list'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show progress bars'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file path (optional)'
    )

    args = parser.parse_args()

    # Load dataset
    print("Loading wiki_bio_gpt3_hallucination dataset...")
    dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination")["evaluation"]
    num_passages = min(args.num_passages, len(dataset))
    print(f"Loaded {len(dataset)} passages, evaluating on {num_passages}\n")

    # Determine which methods to run
    if args.methods == 'all':
        methods_to_run = ['baseline', 'shogenji', 'fitelson', 'olsson']
    elif args.methods == 'baseline-only':
        methods_to_run = ['baseline']
    elif args.methods == 'coherence-only':
        methods_to_run = ['shogenji', 'fitelson', 'olsson']
    else:
        methods_to_run = [m.strip() for m in args.methods.split(',')]

    results = []

    # Evaluate baseline
    if 'baseline' in methods_to_run:
        print(f"\n{'='*60}")
        print(f"Evaluating SelfCheckAPIPrompt (baseline) with {args.model}")
        print(f"{'='*60}")
        baseline = SelfCheckAPIPrompt(client_type="openai", model=args.model)
        result = evaluate_method('SelfCheckAPIPrompt', baseline, dataset, num_passages, args.verbose)
        results.append(result)

    # Evaluate coherence variants
    if 'shogenji' in methods_to_run:
        print(f"\n{'='*60}")
        print(f"Evaluating SelfCheckShogenji with {args.model}")
        print(f"{'='*60}")
        shogenji = SelfCheckShogenji(model=args.model)
        result = evaluate_method('SelfCheckShogenji', shogenji, dataset, num_passages, args.verbose)
        if hasattr(shogenji, 'client'):
            result['cache_stats'] = shogenji.client.get_cache_stats()
        results.append(result)

    if 'fitelson' in methods_to_run:
        print(f"\n{'='*60}")
        print(f"Evaluating SelfCheckFitelson with {args.model}")
        print(f"{'='*60}")
        fitelson = SelfCheckFitelson(model=args.model)
        result = evaluate_method('SelfCheckFitelson', fitelson, dataset, num_passages, args.verbose)
        if hasattr(fitelson, 'client'):
            result['cache_stats'] = fitelson.client.get_cache_stats()
        results.append(result)

    if 'olsson' in methods_to_run:
        print(f"\n{'='*60}")
        print(f"Evaluating SelfCheckOlsson with {args.model}")
        print(f"{'='*60}")
        olsson = SelfCheckOlsson(model=args.model)
        result = evaluate_method('SelfCheckOlsson', olsson, dataset, num_passages, args.verbose)
        if hasattr(olsson, 'client'):
            result['cache_stats'] = olsson.client.get_cache_stats()
        results.append(result)

    # Print comparison table
    print(f"\n\n{'='*80}")
    print(f"COMPARISON RESULTS ({num_passages} passages, {args.model})")
    print(f"{'='*80}")
    print(f"{'Method':<25} {'AUC-PR':<12} {'AUC-ROC':<12} {'PCC':<12}")
    print(f"{'-'*80}")

    for result in results:
        print(f"{result['method']:<25} {result['auc_pr']:<12.4f} {result['auc_roc']:<12.4f} {result['pcc']:<12.4f}")

    # Show published baseline for reference
    print(f"\n{'-'*80}")
    print(f"{'Published Baseline*':<25} {'93.42':<12} {'67.09':<12} {'78.32':<12}")
    print(f"*SelfCheckAPIPrompt gpt-3.5-turbo on 238 passages (from paper)")

    # Cache stats for coherence methods
    coherence_results = [r for r in results if r['method'].startswith('SelfCheck') and r['method'] != 'SelfCheckAPIPrompt']
    if coherence_results and 'cache_stats' in coherence_results[0]:
        print(f"\n{'='*80}")
        print(f"CACHE STATISTICS (Coherence Methods)")
        print(f"{'='*80}")
        for result in coherence_results:
            if 'cache_stats' in result:
                stats = result['cache_stats']
                print(f"{result['method']}:")
                print(f"  Hit rate: {stats['hit_rate']:.1%}, API calls: {stats['api_calls']}/{stats['total_requests']}")

    # Save to JSON if requested
    if args.output:
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'model': args.model,
            'num_passages': num_passages,
            'results': results
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
