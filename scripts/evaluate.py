#!/usr/bin/env python3
"""
Unified evaluation script for all SelfCheckGPT methods.
Supports traditional, API-based, and coherence variants with optional parallelization.

Examples:
    # Just baseline
    python scripts/evaluate.py --methods apiprompt --num-passages 238

    # Baseline + coherence (apples-to-apples comparison)
    python scripts/evaluate.py --methods apiprompt,shogenji,fitelson,olsson --num-passages 238 --workers 4

    # All methods
    python scripts/evaluate.py --methods all --num-passages 238 --workers 4

    # Traditional methods only (no API cost)
    python scripts/evaluate.py --methods nli,mqag,bertscore,ngram --num-passages 238

    # Single coherence method
    python scripts/evaluate.py --methods shogenji --num-passages 238 --workers 8
"""
import argparse
import numpy as np
from datasets import load_dataset
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import pearsonr
from tqdm import tqdm
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from functools import partial

from selfcheckgpt.modeling_selfcheck import SelfCheckNLI, SelfCheckMQAG, SelfCheckBERTScore, SelfCheckNgram
from selfcheckgpt.modeling_selfcheck_apiprompt import SelfCheckAPIPrompt
from selfcheckgpt import SelfCheckShogenji, SelfCheckFitelson, SelfCheckOlsson


def evaluate_single_passage(idx, dataset, variant, needs_passage=False, supports_verbose=False, predict_kwargs=None):
    """Evaluate a single passage - used for parallel processing."""
    passage = dataset[idx]

    sentences = passage['gpt3_sentences']
    sampled_passages = passage['gpt3_text_samples']
    annotations = passage['annotation']
    labels = [0 if ann == 'accurate' else 1 for ann in annotations]

    # Merge default predict_kwargs
    if predict_kwargs is None:
        predict_kwargs = {}

    if needs_passage:
        passage_text = ' '.join(sentences)
        if supports_verbose:
            scores = variant.predict(
                sentences=sentences,
                passage=passage_text,
                sampled_passages=sampled_passages,
                verbose=False,
                **predict_kwargs
            )
        else:
            scores = variant.predict(
                sentences=sentences,
                passage=passage_text,
                sampled_passages=sampled_passages,
                **predict_kwargs
            )
    else:
        if supports_verbose:
            scores = variant.predict(
                sentences=sentences,
                sampled_passages=sampled_passages,
                verbose=False,
                **predict_kwargs
            )
        else:
            scores = variant.predict(
                sentences=sentences,
                sampled_passages=sampled_passages,
                **predict_kwargs
            )

    return {
        'idx': idx,
        'scores': scores.tolist(),
        'labels': labels
    }


def evaluate_method(method_name, variant, dataset, num_passages, num_workers=1,
                   verbose=False, needs_passage=False, supports_verbose=False, predict_kwargs=None):
    """
    Evaluate a method with optional parallelization.

    Args:
        method_name: Display name of the method
        variant: The SelfCheck variant instance
        dataset: HuggingFace dataset
        num_passages: Number of passages to evaluate
        num_workers: Number of parallel workers (1 = sequential)
        verbose: Show progress bars
        needs_passage: Whether method needs full passage text
        supports_verbose: Whether method's predict() accepts verbose parameter
        predict_kwargs: Additional kwargs to pass to variant.predict()
    """
    all_scores = []
    all_labels = []

    # Determine if method is API-based (can benefit from parallelization)
    is_api_based = any(x in method_name for x in ['API', 'Shogenji', 'Fitelson', 'Olsson'])

    eval_func = partial(evaluate_single_passage, dataset=dataset,
                       variant=variant, needs_passage=needs_passage, supports_verbose=supports_verbose,
                       predict_kwargs=predict_kwargs)

    if is_api_based and num_workers > 1:
        # Parallel processing for API-based methods
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(eval_func, idx) for idx in range(num_passages)]

            for future in tqdm(as_completed(futures), total=num_passages,
                             desc=f"Evaluating {method_name}", disable=not verbose):
                result = future.result()
                all_scores.extend(result['scores'])
                all_labels.extend(result['labels'])
    else:
        # Sequential processing (for local models or when workers=1)
        for idx in tqdm(range(num_passages), desc=f"Evaluating {method_name}", disable=not verbose):
            result = eval_func(idx)
            all_scores.extend(result['scores'])
            all_labels.extend(result['labels'])

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
        description="Unified evaluation script for all SelfCheckGPT methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Just baseline
  %(prog)s --methods apiprompt --num-passages 238

  # Apples-to-apples comparison (all use same API model)
  %(prog)s --methods apiprompt,shogenji,fitelson,olsson --num-passages 238 --workers 4

  # All methods (traditional + API + coherence)
  %(prog)s --methods all --num-passages 238 --workers 4

  # Traditional methods only (no API cost)
  %(prog)s --methods nli,mqag,bertscore,ngram --num-passages 238

  # Quick test
  %(prog)s --methods shogenji --num-passages 10 --workers 4 --verbose

Available methods:
  Traditional (local models): nli, mqag, bertscore, ngram
  API-based: apiprompt
  Coherence (API-based): shogenji, fitelson, olsson
  Shortcuts: all, traditional, api-only, coherence
        """
    )

    # Required arguments
    parser.add_argument(
        '--methods',
        type=str,
        required=True,
        help='Methods to evaluate. Use comma-separated list or shortcuts: all, traditional, api-only, coherence'
    )

    # Optional arguments
    parser.add_argument(
        '--num-passages',
        type=int,
        default=238,
        help='Number of passages to evaluate (default: 238 for full dataset, use 10 for quick test)'
    )
    parser.add_argument(
        '--api-model',
        type=str,
        default='gpt-4o-mini',
        help='OpenAI model for API-based methods (default: gpt-4o-mini)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of parallel workers for API methods (default: 1, recommended: 4-8)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device for traditional methods: cuda, cpu, or auto (default: auto)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Save results to JSON file (optional)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show progress bars and detailed output'
    )

    args = parser.parse_args()

    # Determine device for local models
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Print configuration
    print(f"{'='*80}")
    print(f"SelfCheckGPT Evaluation")
    print(f"{'='*80}")
    print(f"Device (local models): {device}")
    print(f"API model: {args.api_model}")
    print(f"Workers: {args.workers} (parallelization {'enabled' if args.workers > 1 else 'disabled'})")
    print(f"")

    # Load dataset
    print("Loading wiki_bio_gpt3_hallucination dataset...")
    dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination")["evaluation"]
    num_passages = min(args.num_passages, len(dataset))
    print(f"Dataset: {len(dataset)} passages total, evaluating on {num_passages}")
    print("")

    # Parse methods to run
    method_groups = {
        'traditional': ['nli', 'mqag', 'bertscore', 'ngram'],
        'api-only': ['apiprompt'],
        'coherence': ['shogenji', 'fitelson', 'olsson'],
        'all': ['nli', 'mqag', 'bertscore', 'ngram', 'apiprompt', 'shogenji', 'fitelson', 'olsson']
    }

    if args.methods.lower() in method_groups:
        methods_to_run = method_groups[args.methods.lower()]
    else:
        methods_to_run = [m.strip().lower() for m in args.methods.split(',')]

    # Validate methods
    valid_methods = ['nli', 'mqag', 'bertscore', 'ngram', 'apiprompt', 'shogenji', 'fitelson', 'olsson']
    invalid_methods = [m for m in methods_to_run if m not in valid_methods]
    if invalid_methods:
        print(f"Error: Invalid methods: {', '.join(invalid_methods)}")
        print(f"Valid methods: {', '.join(valid_methods)}")
        print(f"Shortcuts: all, traditional, api-only, coherence")
        return

    print(f"Methods to evaluate: {', '.join(methods_to_run)}")
    print(f"{'='*80}\n")

    results = []

    # Evaluate traditional methods (local models)
    if 'nli' in methods_to_run:
        print(f"\n{'='*60}")
        print(f"Evaluating SelfCheckNLI (DeBERTa-v3-large)")
        print(f"{'='*60}")
        nli = SelfCheckNLI(device=device)
        result = evaluate_method('SelfCheckNLI', nli, dataset, num_passages,
                                num_workers=1, verbose=args.verbose)  # Sequential for GPU
        results.append(result)

    if 'mqag' in methods_to_run:
        print(f"\n{'='*60}")
        print(f"Evaluating SelfCheckMQAG (T5 + Longformer)")
        print(f"{'='*60}")
        mqag = SelfCheckMQAG(device=device)
        result = evaluate_method('SelfCheckMQAG', mqag, dataset, num_passages,
                                num_workers=1, verbose=args.verbose, needs_passage=True,
                                predict_kwargs={'beta1': 0.95, 'beta2': 0.95})
        results.append(result)

    if 'bertscore' in methods_to_run:
        print(f"\n{'='*60}")
        print(f"Evaluating SelfCheckBERTScore (RoBERTa)")
        print(f"{'='*60}")
        bertscore = SelfCheckBERTScore()
        result = evaluate_method('SelfCheckBERTScore', bertscore, dataset, num_passages,
                                num_workers=1, verbose=args.verbose)
        results.append(result)

    if 'ngram' in methods_to_run:
        print(f"\n{'='*60}")
        print(f"Evaluating SelfCheckNgram (Unigram)")
        print(f"{'='*60}")
        ngram = SelfCheckNgram(n=1, device=device)
        result = evaluate_method('SelfCheckNgram', ngram, dataset, num_passages,
                                num_workers=1, verbose=args.verbose, needs_passage=True)
        results.append(result)

    # Evaluate API-based baseline
    if 'apiprompt' in methods_to_run:
        print(f"\n{'='*60}")
        print(f"Evaluating SelfCheckAPIPrompt ({args.api_model})")
        print(f"{'='*60}")
        apiprompt = SelfCheckAPIPrompt(client_type="openai", model=args.api_model)
        result = evaluate_method('SelfCheckAPIPrompt', apiprompt, dataset, num_passages,
                                num_workers=args.workers, verbose=args.verbose, supports_verbose=True)
        results.append(result)

    # Evaluate coherence variants
    if 'shogenji' in methods_to_run:
        print(f"\n{'='*60}")
        print(f"Evaluating SelfCheckShogenji ({args.api_model})")
        print(f"{'='*60}")
        shogenji = SelfCheckShogenji(model=args.api_model)
        result = evaluate_method('SelfCheckShogenji', shogenji, dataset, num_passages,
                                num_workers=args.workers, verbose=args.verbose, supports_verbose=True)
        if hasattr(shogenji, 'client'):
            result['cache_stats'] = shogenji.client.get_cache_stats()
        results.append(result)

    if 'fitelson' in methods_to_run:
        print(f"\n{'='*60}")
        print(f"Evaluating SelfCheckFitelson ({args.api_model})")
        print(f"{'='*60}")
        fitelson = SelfCheckFitelson(model=args.api_model)
        result = evaluate_method('SelfCheckFitelson', fitelson, dataset, num_passages,
                                num_workers=args.workers, verbose=args.verbose, supports_verbose=True)
        if hasattr(fitelson, 'client'):
            result['cache_stats'] = fitelson.client.get_cache_stats()
        results.append(result)

    if 'olsson' in methods_to_run:
        print(f"\n{'='*60}")
        print(f"Evaluating SelfCheckOlsson ({args.api_model})")
        print(f"{'='*60}")
        olsson = SelfCheckOlsson(model=args.api_model)
        result = evaluate_method('SelfCheckOlsson', olsson, dataset, num_passages,
                                num_workers=args.workers, verbose=args.verbose, supports_verbose=True)
        if hasattr(olsson, 'client'):
            result['cache_stats'] = olsson.client.get_cache_stats()
        results.append(result)

    # Print comparison table
    print(f"\n\n{'='*80}")
    print(f"RESULTS ({num_passages} passages)")
    print(f"{'='*80}")
    print(f"{'Method':<30} {'Type':<20} {'AUC-PR':<10} {'AUC-ROC':<10} {'PCC':<10}")
    print(f"{'-'*80}")

    for result in results:
        method = result['method']
        if method in ['SelfCheckNLI', 'SelfCheckMQAG', 'SelfCheckBERTScore', 'SelfCheckNgram']:
            method_type = 'Traditional (local)'
        elif method == 'SelfCheckAPIPrompt':
            method_type = f'API ({args.api_model})'
        else:
            method_type = f'Coherence ({args.api_model})'

        print(f"{method:<30} {method_type:<20} {result['auc_pr']:<10.4f} {result['auc_roc']:<10.4f} {result['pcc']:<10.4f}")

    # Show published baselines for reference
    print(f"\n{'-'*80}")
    print(f"Published Baselines (238 passages from EMNLP 2023 paper):")
    print(f"  SelfCheckAPIPrompt (gpt-3.5-turbo): AUC-PR=93.42, AUC-ROC=67.09, PCC=78.32")
    print(f"  SelfCheckNLI (DeBERTa):              AUC-PR=92.50, AUC-ROC=N/A,   PCC=74.14")

    # Cache statistics for coherence methods
    coherence_results = [r for r in results if 'cache_stats' in r]
    if coherence_results:
        print(f"\n{'='*80}")
        print(f"CACHE STATISTICS (API Cost Reduction)")
        print(f"{'='*80}")
        for result in coherence_results:
            stats = result['cache_stats']
            saved_calls = stats['cache_hits']
            total_without_cache = stats['total_requests']
            savings_pct = (saved_calls / total_without_cache * 100) if total_without_cache > 0 else 0
            print(f"{result['method']}:")
            print(f"  Cache hits: {stats['cache_hits']:,} / {stats['total_requests']:,} ({stats['hit_rate']:.1%})")
            print(f"  Actual API calls: {stats['api_calls']:,}")
            print(f"  Cost savings: ~{savings_pct:.0f}%")

    # Save results if requested
    if args.output:
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'api_model': args.api_model,
                'device': str(device),
                'num_workers': args.workers,
                'num_passages': num_passages,
                'methods': methods_to_run
            },
            'results': results
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n{'='*80}")
        print(f"Results saved to: {args.output}")

    print(f"\n{'='*80}")
    print(f"Evaluation complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
