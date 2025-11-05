"""
Benchmark Runner for Probability Extraction Quality Evaluation

This module implements the benchmark execution framework that:
1. Loads test cases from test_cases.json
2. Executes probability extraction using CoherenceAPIClient
3. Collects predictions and invokes evaluation metrics
4. Generates comparative analysis reports

Reference:
- agent-os/specs/2025-11-03-coherence-improvements/tasks.md (Task Group 1.3)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
from datetime import datetime

from ..modeling_coherence_api import CoherenceAPIClient
from .metrics import compute_all_metrics


class BenchmarkRunner:
    """
    Benchmark execution framework for probability extraction quality evaluation.

    Features:
    - Load 40 test cases from structured JSON
    - Execute probability extraction with configurable prompt strategies
    - Collect predictions and compute 5 evaluation metrics
    - Generate comparative analysis reports
    - Track API costs and cache statistics

    Example:
        >>> runner = BenchmarkRunner(model="gpt-4o-mini")
        >>> results = runner.run_benchmark(prompt_strategy="baseline")
        >>> runner.print_report(results)
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        test_cases_path: Optional[str] = None
    ):
        """
        Initialize BenchmarkRunner.

        Args:
            model: OpenAI model name (default: "gpt-4o-mini")
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
            test_cases_path: Path to test_cases.json (defaults to package location)
        """
        self.model = model
        self.api_key = api_key

        # Load test cases
        if test_cases_path is None:
            # Use default path relative to this file
            test_cases_path = Path(__file__).parent / "test_cases.json"

        self.test_cases = self._load_test_cases(test_cases_path)
        print(f"Loaded {len(self.test_cases)} test cases from {test_cases_path}")

        # Initialize API client (will be configured with prompt strategy later)
        self.client = None

    def _load_test_cases(self, path: Path) -> List[Dict]:
        """Load test cases from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return data['test_cases']

    def _configure_prompts(self, prompt_strategy: str):
        """
        Configure prompt templates based on strategy.

        Args:
            prompt_strategy: One of "baseline", "cot", "few_shot", "axiom_aware", "hybrid"
        """
        if prompt_strategy == "baseline":
            # Use default simple prompts
            individual_template = "Rate the probability that this statement is true: {statement}"
            joint_template = "Rate the probability that both statements are true: {statement1} AND {statement2}"
            conditional_template = "Rate the probability that statement A is true: {statement1} GIVEN that {statement2} is true"

        elif prompt_strategy == "cot":
            # Chain of Thought prompts
            individual_template = (
                "Let's think step-by-step about the probability that this statement is true: {statement}\n"
                "Consider the evidence, common knowledge, and logical consistency. "
                "What is the probability (between 0.0 and 1.0) that this statement is true?"
            )
            joint_template = (
                "Let's analyze the probability that both statements are simultaneously true:\n"
                "Statement 1: {statement1}\n"
                "Statement 2: {statement2}\n\n"
                "First, consider each statement individually. Then, consider their relationship and any dependencies. "
                "What is the probability (between 0.0 and 1.0) that BOTH statements are true?"
            )
            conditional_template = (
                "Let's reason about conditional probability:\n"
                "Statement A: {statement1}\n"
                "Statement B (given condition): {statement2}\n\n"
                "Assuming that statement B is true, how does this affect the probability of statement A? "
                "What is P(A|B), the probability that A is true GIVEN that B is true?"
            )

        elif prompt_strategy == "axiom_aware":
            # Axiom-aware prompts with probability theory guidance
            individual_template = (
                "Rate the probability that this statement is true: {statement}\n\n"
                "Remember: Probabilities must be between 0.0 (impossible) and 1.0 (certain). "
                "Consider evidence and uncertainty carefully."
            )
            joint_template = (
                "Rate the probability that both statements are true: {statement1} AND {statement2}\n\n"
                "Remember: P(A∧B) must be ≤ min(P(A), P(B)). "
                "Joint probability cannot exceed either individual probability."
            )
            conditional_template = (
                "Rate the probability that statement A is true GIVEN that statement B is true:\n"
                "A: {statement1}\n"
                "B: {statement2}\n\n"
                "Remember: P(A|B) = P(A∧B) / P(B). "
                "Conditional probability measures how the condition affects the hypothesis."
            )

        elif prompt_strategy == "few_shot":
            # Few-shot prompts with examples
            individual_template = (
                "Rate the probability that the statement is true (0.0 = impossible, 1.0 = certain).\n\n"
                "Examples:\n"
                "Statement: 'The sky is blue during clear daytime weather.' → Probability: 0.95\n"
                "Statement: 'It will rain somewhere on Earth tomorrow.' → Probability: 0.99\n"
                "Statement: 'A coin flip will land on heads.' → Probability: 0.5\n\n"
                "Now rate this statement: {statement}"
            )
            joint_template = (
                "Rate the probability that BOTH statements are simultaneously true.\n\n"
                "Example:\n"
                "Statement 1: 'It is raining.' Statement 2: 'The ground is wet.'\n"
                "These are positively correlated. P(both) ≈ 0.35\n\n"
                "Now rate these statements:\n"
                "Statement 1: {statement1}\n"
                "Statement 2: {statement2}"
            )
            conditional_template = (
                "Rate P(A|B) - the probability that A is true GIVEN that B is true.\n\n"
                "Example:\n"
                "A: 'The ground is wet.' B: 'It has been raining.'\n"
                "P(ground wet | rain) ≈ 0.9 (rain strongly implies wet ground)\n\n"
                "Now rate:\n"
                "A: {statement1}\n"
                "B: {statement2}"
            )

        elif prompt_strategy == "hybrid":
            # Hybrid approach combining axiom awareness with CoT
            individual_template = (
                "Evaluate the probability that this statement is true: {statement}\n\n"
                "Think carefully about:\n"
                "1. Available evidence and common knowledge\n"
                "2. Uncertainty and exceptions\n"
                "3. Logical consistency\n\n"
                "Remember: Use 0.0 for impossible, 1.0 for certain, and values in between for uncertain claims. "
                "What is the probability?"
            )
            joint_template = (
                "Evaluate the probability that BOTH statements are simultaneously true:\n"
                "Statement 1: {statement1}\n"
                "Statement 2: {statement2}\n\n"
                "Consider:\n"
                "1. Are these statements independent or related?\n"
                "2. Does one imply or contradict the other?\n"
                "3. P(A∧B) must be ≤ min(P(A), P(B))\n\n"
                "What is the joint probability?"
            )
            conditional_template = (
                "Evaluate the conditional probability P(A|B):\n"
                "A: {statement1}\n"
                "B (given condition): {statement2}\n\n"
                "Think about:\n"
                "1. How does knowing B affects the likelihood of A?\n"
                "2. Is there a causal or logical relationship?\n"
                "3. P(A|B) = P(A∧B) / P(B)\n\n"
                "What is P(A|B)?"
            )

        else:
            raise ValueError(f"Unknown prompt_strategy: {prompt_strategy}")

        # Create new client with configured prompts
        self.client = CoherenceAPIClient(model=self.model, api_key=self.api_key)
        self.client.individual_prob_template = individual_template
        self.client.joint_prob_template = joint_template
        self.client.conditional_prob_template = conditional_template

    def run_benchmark(
        self,
        prompt_strategy: str = "baseline",
        verbose: bool = True
    ) -> Dict:
        """
        Run the full benchmark with 40 test cases.

        Args:
            prompt_strategy: Prompt strategy to use ("baseline", "cot", "few_shot", "axiom_aware", "hybrid")
            verbose: Show progress bars

        Returns:
            Dictionary containing:
            - 'predictions': List of predicted probabilities for each test case
            - 'metrics': All evaluation metric results
            - 'cache_stats': API cache statistics
            - 'test_case_results': Detailed per-test-case results
            - 'config': Benchmark configuration
        """
        print(f"\n{'='*80}")
        print(f"Running Probability Extraction Benchmark")
        print(f"{'='*80}")
        print(f"Model: {self.model}")
        print(f"Prompt Strategy: {prompt_strategy}")
        print(f"Test Cases: {len(self.test_cases)}")
        print(f"")

        # Configure prompts
        self._configure_prompts(prompt_strategy)

        # Estimate API costs
        estimate = CoherenceAPIClient.estimate_api_calls(
            num_sentences=len(self.test_cases),
            num_samples=0,  # No samples needed for benchmark
            num_variants=1,
            include_conditional=True  # Worst case
        )
        print(f"Estimated API calls: {estimate['total_calls_uncached']} (uncached)")
        print(f"")

        # Execute benchmark
        predictions = []
        test_case_results = []

        iterator = tqdm(self.test_cases, desc="Extracting probabilities") if verbose else self.test_cases

        for test_case in iterator:
            tc_type = test_case['type']

            if tc_type == 'individual':
                pred_prob = self.client.extract_individual_probability(
                    test_case['statement'],
                    verbose=False
                )
                predictions.append(pred_prob)

                test_case_results.append({
                    'id': test_case['id'],
                    'type': tc_type,
                    'predicted': pred_prob,
                    'ground_truth': test_case['ground_truth_probability'],
                    'error': abs(pred_prob - test_case['ground_truth_probability']),
                    'domain': test_case['domain']
                })

            elif tc_type == 'joint':
                pred_prob = self.client.extract_joint_probability(
                    test_case['statement1'],
                    test_case['statement2'],
                    verbose=False
                )
                predictions.append(pred_prob)

                test_case_results.append({
                    'id': test_case['id'],
                    'type': tc_type,
                    'predicted': pred_prob,
                    'ground_truth': test_case['ground_truth_joint'],
                    'error': abs(pred_prob - test_case['ground_truth_joint']),
                    'domain': test_case['domain']
                })

            elif tc_type == 'conditional':
                pred_prob = self.client.extract_conditional_probability(
                    test_case['statement1'],
                    test_case['statement2'],
                    verbose=False
                )
                predictions.append(pred_prob)

                test_case_results.append({
                    'id': test_case['id'],
                    'type': tc_type,
                    'predicted': pred_prob,
                    'ground_truth': test_case['ground_truth_conditional'],
                    'error': abs(pred_prob - test_case['ground_truth_conditional']),
                    'domain': test_case['domain']
                })

        # Collect data for metrics calculation
        predicted_probs = np.array(predictions)

        # Extract ground truth probability based on test case type
        ground_truth_probs = []
        for tc in self.test_cases:
            if tc['type'] == 'individual':
                gt = tc.get('ground_truth_probability')
            elif tc['type'] == 'joint':
                gt = tc.get('ground_truth_joint')
            elif tc['type'] == 'conditional':
                gt = tc.get('ground_truth_conditional')
            else:
                gt = None

            if gt is None:
                raise ValueError(f"Missing ground truth for test case {tc.get('id', 'unknown')}")
            ground_truth_probs.append(gt)

        ground_truth_probs = np.array(ground_truth_probs)

        # Convert to binary outcomes for Brier score (threshold at 0.5)
        actual_outcomes = (ground_truth_probs >= 0.5).astype(float)

        # Prepare data for coherence compliance checking
        individual_cases = [tc for tc in self.test_cases if tc['type'] == 'joint']
        joint_cases = [tc for tc in self.test_cases if tc['type'] == 'joint']

        if len(joint_cases) > 0:
            individual_probs_for_compliance = [
                tuple(tc['ground_truth_individual']) for tc in joint_cases
            ]
            joint_probs_for_compliance = [
                tc['ground_truth_joint'] for tc in joint_cases
            ]
        else:
            individual_probs_for_compliance = None
            joint_probs_for_compliance = None

        # Compute metrics
        metrics = compute_all_metrics(
            predicted_probs=predicted_probs,
            actual_outcomes=actual_outcomes,
            individual_probs=individual_probs_for_compliance,
            joint_probs=joint_probs_for_compliance,
            conditional_probs=None,  # Could add if we have test cases with all three
            paraphrase_groups=None,  # Not yet implemented in test cases
            num_bins=10
        )

        # Get cache statistics
        cache_stats = self.client.get_cache_stats()

        # Compile results
        results = {
            'predictions': predictions,
            'metrics': metrics,
            'cache_stats': cache_stats,
            'test_case_results': test_case_results,
            'config': {
                'model': self.model,
                'prompt_strategy': prompt_strategy,
                'num_test_cases': len(self.test_cases),
                'timestamp': datetime.now().isoformat()
            }
        }

        return results

    def compare_strategies(
        self,
        strategies: List[str],
        verbose: bool = True
    ) -> Dict[str, Dict]:
        """
        Run benchmark with multiple prompt strategies and compare results.

        Args:
            strategies: List of prompt strategies to compare
            verbose: Show progress bars

        Returns:
            Dictionary mapping strategy names to their benchmark results
        """
        comparison = {}

        for strategy in strategies:
            print(f"\n{'='*80}")
            print(f"Evaluating Strategy: {strategy}")
            print(f"{'='*80}\n")

            results = self.run_benchmark(
                prompt_strategy=strategy,
                verbose=verbose
            )

            comparison[strategy] = results

        return comparison

    def print_report(self, results: Dict):
        """
        Print human-readable benchmark report.

        Args:
            results: Benchmark results from run_benchmark()
        """
        print(f"\n{'='*80}")
        print(f"BENCHMARK REPORT")
        print(f"{'='*80}")

        config = results['config']
        print(f"Model: {config['model']}")
        print(f"Prompt Strategy: {config['prompt_strategy']}")
        print(f"Test Cases: {config['num_test_cases']}")
        print(f"Timestamp: {config['timestamp']}")
        print(f"")

        # Metrics
        metrics = results['metrics']
        print(f"{'='*80}")
        print(f"EVALUATION METRICS")
        print(f"{'='*80}")

        print(f"\n1. Brier Score (Calibration Quality)")
        print(f"   Score: {metrics['brier_score']:.4f}")
        print(f"   Interpretation: {metrics['summary']['brier_interpretation']}")
        print(f"   (Lower is better, 0.0 = perfect)")

        print(f"\n2. Expected Calibration Error (ECE)")
        print(f"   ECE: {metrics['ece']['ece']:.4f}")
        print(f"   Interpretation: {metrics['summary']['ece_interpretation']}")
        print(f"   (Lower is better, 0.0 = perfect)")

        print(f"\n3. Probability Coherence Compliance")
        compliance = metrics['coherence_compliance']
        print(f"   Compliance Rate: {compliance['compliance_rate']:.1%}")
        print(f"   Interpretation: {metrics['summary']['compliance_interpretation']}")
        print(f"   Violations:")
        print(f"     - Kolmogorov: {compliance['kolmogorov_violations']}")
        print(f"     - Joint: {compliance['joint_violations']}")
        print(f"     - Conditional: {compliance['conditional_violations']}")

        if metrics['consistency']['mean_variance'] is not None:
            print(f"\n4. Probability Consistency Score")
            consistency = metrics['consistency']
            print(f"   Mean Variance: {consistency['mean_variance']:.6f}")
            print(f"   Consistency Score: {consistency['consistency_score']:.4f}")
            print(f"   (Higher is better, 1.0 = perfect)")

        print(f"\n5. Sharpness (Decisiveness)")
        print(f"   Score: {metrics['sharpness']:.4f}")
        print(f"   (Higher is better, max 0.5)")

        print(f"\n{'='*80}")
        print(f"OVERALL ASSESSMENT")
        print(f"{'='*80}")
        print(f"Quality: {metrics['summary']['overall_quality']}")

        # Cache statistics
        print(f"\n{'='*80}")
        print(f"API COST STATISTICS")
        print(f"{'='*80}")
        cache_stats = results['cache_stats']
        print(f"Total Requests: {cache_stats['total_requests']}")
        cache_hits = cache_stats['total_requests'] - cache_stats['api_calls']
        print(f"Cache Hits: {cache_hits} ({cache_stats['hit_rate']:.1%})")
        print(f"API Calls: {cache_stats['api_calls']}")
        print(f"Cache Size: {cache_stats['cache_size']}")
        print(f"")

        # Worst performing test cases
        print(f"{'='*80}")
        print(f"WORST PERFORMING TEST CASES (Top 5)")
        print(f"{'='*80}")

        test_case_results = results['test_case_results']
        sorted_cases = sorted(test_case_results, key=lambda x: x['error'], reverse=True)

        for i, tc in enumerate(sorted_cases[:5], 1):
            print(f"\n{i}. {tc['id']} ({tc['domain']} - {tc['type']})")
            print(f"   Predicted: {tc['predicted']:.4f}")
            print(f"   Ground Truth: {tc['ground_truth']:.4f}")
            print(f"   Error: {tc['error']:.4f}")

        print(f"\n{'='*80}\n")

    def save_results(self, results: Dict, output_path: str):
        """
        Save benchmark results to JSON file.

        Args:
            results: Benchmark results from run_benchmark()
            output_path: Path to save JSON file
        """
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj

        results_serializable = convert_types(results)

        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        print(f"Results saved to: {output_path}")

    def print_comparison_report(self, comparison: Dict[str, Dict]):
        """
        Print comparative analysis of multiple prompt strategies.

        Args:
            comparison: Results from compare_strategies()
        """
        print(f"\n{'='*80}")
        print(f"COMPARATIVE ANALYSIS")
        print(f"{'='*80}\n")

        strategies = list(comparison.keys())

        # Create comparison table
        print(f"{'Strategy':<20} {'Brier':<10} {'ECE':<10} {'Compliance':<12} {'Sharpness':<10}")
        print(f"{'-'*80}")

        for strategy in strategies:
            metrics = comparison[strategy]['metrics']
            print(f"{strategy:<20} "
                  f"{metrics['brier_score']:<10.4f} "
                  f"{metrics['ece']['ece']:<10.4f} "
                  f"{metrics['coherence_compliance']['compliance_rate']:<12.1%} "
                  f"{metrics['sharpness']:<10.4f}")

        # Identify best performer for each metric
        print(f"\n{'='*80}")
        print(f"BEST PERFORMERS")
        print(f"{'='*80}")

        best_brier = min(strategies, key=lambda s: comparison[s]['metrics']['brier_score'])
        best_ece = min(strategies, key=lambda s: comparison[s]['metrics']['ece']['ece'])
        best_compliance = max(strategies, key=lambda s: comparison[s]['metrics']['coherence_compliance']['compliance_rate'])
        best_sharpness = max(strategies, key=lambda s: comparison[s]['metrics']['sharpness'])

        print(f"Best Brier Score: {best_brier} ({comparison[best_brier]['metrics']['brier_score']:.4f})")
        print(f"Best ECE: {best_ece} ({comparison[best_ece]['metrics']['ece']['ece']:.4f})")
        print(f"Best Compliance: {best_compliance} ({comparison[best_compliance]['metrics']['coherence_compliance']['compliance_rate']:.1%})")
        print(f"Best Sharpness: {best_sharpness} ({comparison[best_sharpness]['metrics']['sharpness']:.4f})")

        # Calculate improvement percentages vs baseline
        if 'baseline' in strategies:
            print(f"\n{'='*80}")
            print(f"IMPROVEMENT vs BASELINE")
            print(f"{'='*80}")

            baseline_brier = comparison['baseline']['metrics']['brier_score']
            baseline_ece = comparison['baseline']['metrics']['ece']['ece']
            baseline_compliance = comparison['baseline']['metrics']['coherence_compliance']['compliance_rate']

            for strategy in strategies:
                if strategy == 'baseline':
                    continue

                brier_improvement = ((baseline_brier - comparison[strategy]['metrics']['brier_score']) / baseline_brier) * 100
                ece_improvement = ((baseline_ece - comparison[strategy]['metrics']['ece']['ece']) / baseline_ece) * 100
                compliance_improvement = ((comparison[strategy]['metrics']['coherence_compliance']['compliance_rate'] - baseline_compliance) / baseline_compliance) * 100

                print(f"\n{strategy}:")
                print(f"  Brier Score: {brier_improvement:+.1f}% (lower is better)")
                print(f"  ECE: {ece_improvement:+.1f}% (lower is better)")
                print(f"  Compliance: {compliance_improvement:+.1f}% (higher is better)")

        print(f"\n{'='*80}\n")
