#!/usr/bin/env python3
"""
Simple evaluation using core SelfCheckGPT pattern.
Uses the pre-generated samples from the dataset.

Matches the usage pattern from README.md
"""
import argparse
import numpy as np
from datasets import load_dataset
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import pearsonr
from tqdm import tqdm

from selfcheckgpt import SelfCheckShogenji, SelfCheckFitelson, SelfCheckOlsson


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate coherence variants (matches SelfCheckGPT usage pattern)"
    )
    parser.add_argument(
        '--variant',
        type=str,
        default='shogenji',
        choices=['shogenji', 'fitelson', 'olsson'],
        help='Which variant to evaluate (default: shogenji)'
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
        '--verbose',
        action='store_true',
        help='Show progress bar during prediction (matches SelfCheckGPT pattern)'
    )

    args = parser.parse_args()

    # Load dataset
    print("Loading wiki_bio_gpt3_hallucination dataset...")
    dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination")["evaluation"]
    print(f"Loaded {len(dataset)} passages\n")

    # Initialize variant (matches README.md pattern)
    print(f"Initializing SelfCheck{args.variant.capitalize()}...")
    if args.variant == 'shogenji':
        variant = SelfCheckShogenji(model=args.model)
    elif args.variant == 'fitelson':
        variant = SelfCheckFitelson(model=args.model)
    elif args.variant == 'olsson':
        variant = SelfCheckOlsson(model=args.model)

    # Collect scores and labels
    all_scores = []
    all_labels = []

    # Determine number of passages
    num_passages = min(args.num_passages, len(dataset))

    # Evaluate passages
    for idx in tqdm(range(num_passages), desc="Evaluating passages", disable=not args.verbose):
        passage = dataset[idx]

        # Extract data (matches core SelfCheckGPT pattern from README)
        sentences = passage['gpt3_sentences']
        sampled_passages = passage['gpt3_text_samples']  # Use pre-generated samples
        annotations = passage['annotation']

        # Convert text labels to binary
        labels = [0 if ann == 'accurate' else 1 for ann in annotations]

        # Predict (matches README.md pattern: predict(sentences, sampled_passages, verbose))
        scores = variant.predict(
            sentences=sentences,
            sampled_passages=sampled_passages,
            verbose=args.verbose  # Matches SelfCheckGPT's verbose parameter
        )

        # Collect results
        all_scores.extend(scores.tolist())
        all_labels.extend(labels)

    # Convert to arrays
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # Compute metrics
    auc_pr = average_precision_score(all_labels, all_scores)
    auc_roc = roc_auc_score(all_labels, all_scores)
    pcc, _ = pearsonr(all_scores, all_labels)

    # Print results
    print(f"\n{'='*50}")
    print(f"Results (on {num_passages} passages):")
    print(f"{'='*50}")
    print(f"Total sentences: {len(all_scores)}")
    print(f"Accurate: {np.sum(all_labels == 0)}")
    print(f"Inaccurate: {np.sum(all_labels == 1)}")
    print(f"\nMetrics:")
    print(f"  AUC-PR:  {auc_pr:.4f}")
    print(f"  AUC-ROC: {auc_roc:.4f}")
    print(f"  PCC:     {pcc:.4f}")
    print(f"\nBaseline (SelfCheckAPIPrompt gpt-3.5-turbo): 93.42 AUC-PR")

    # Cache stats
    cache_stats = variant.client.get_cache_stats()
    print(f"\nCache Statistics:")
    print(f"  Hit rate: {cache_stats['hit_rate']:.1%}")
    print(f"  Cache size: {cache_stats['cache_size']}")
    print(f"  Total requests: {cache_stats['total_requests']}")
    print(f"  API calls: {cache_stats['api_calls']}")


if __name__ == "__main__":
    main()
