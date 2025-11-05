#!/usr/bin/env python3
"""
Run probability extraction benchmark for SelfCheckGPT coherence variants.

This script executes the 40-test-case benchmark to evaluate probability extraction
quality across different prompt strategies.

Examples:
    # Run with baseline prompts
    python scripts/run_benchmark.py --strategy baseline

    # Compare multiple strategies
    python scripts/run_benchmark.py --strategies baseline,cot,axiom_aware --compare

    # Run on subset for testing
    python scripts/run_benchmark.py --strategy baseline --limit 5

    # Save results to file
    python scripts/run_benchmark.py --strategy hybrid --output results.json
"""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from selfcheckgpt.benchmark import BenchmarkRunner


def main():
    parser = argparse.ArgumentParser(
        description="Run probability extraction benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single strategy evaluation
  %(prog)s --strategy baseline
  %(prog)s --strategy cot
  %(prog)s --strategy hybrid

  # Comparative evaluation
  %(prog)s --strategies baseline,cot,axiom_aware --compare

  # Quick test on subset
  %(prog)s --strategy baseline --limit 10

  # Save results
  %(prog)s --strategy hybrid --output results.json

Available Strategies:
  baseline      - Simple probability rating prompts (current default)
  cot           - Chain of Thought reasoning prompts
  few_shot      - Few-shot examples with prompts
  axiom_aware   - Prompts with probability theory guidance
  hybrid        - Combined CoT + axiom awareness
        """
    )

    # Strategy selection
    strategy_group = parser.add_mutually_exclusive_group(required=True)
    strategy_group.add_argument(
        '--strategy',
        type=str,
        choices=['baseline', 'cot', 'few_shot', 'axiom_aware', 'hybrid'],
        help='Single prompt strategy to evaluate'
    )
    strategy_group.add_argument(
        '--strategies',
        type=str,
        help='Comma-separated list of strategies for comparison'
    )

    # Comparison mode
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Generate comparative analysis report (requires --strategies)'
    )

    # Configuration
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o-mini',
        help='OpenAI model to use (default: gpt-4o-mini)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of test cases (for quick testing)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Save results to JSON file'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress bars'
    )

    args = parser.parse_args()

    # Validate compare mode
    if args.compare and not args.strategies:
        parser.error("--compare requires --strategies")

    # Initialize runner
    print(f"Initializing benchmark runner with model: {args.model}")
    runner = BenchmarkRunner(model=args.model)

    # Limit test cases if requested
    if args.limit:
        print(f"Limiting to first {args.limit} test cases")
        runner.test_cases = runner.test_cases[:args.limit]

    # Run benchmark
    if args.strategy:
        # Single strategy evaluation
        results = runner.run_benchmark(
            prompt_strategy=args.strategy,
            verbose=not args.quiet
        )

        # Print report
        runner.print_report(results)

        # Save results if requested
        if args.output:
            runner.save_results(results, args.output)

    elif args.strategies:
        # Multiple strategy comparison
        strategies = [s.strip() for s in args.strategies.split(',')]

        print(f"Comparing {len(strategies)} strategies: {', '.join(strategies)}")

        comparison = runner.compare_strategies(
            strategies=strategies,
            verbose=not args.quiet
        )

        # Print comparison report
        runner.print_comparison_report(comparison)

        # Save individual results if requested
        if args.output:
            for strategy, results in comparison.items():
                output_path = args.output.replace('.json', f'_{strategy}.json')
                runner.save_results(results, output_path)


if __name__ == "__main__":
    main()
