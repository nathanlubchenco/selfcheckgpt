"""
Probability Extraction Benchmark for SelfCheckGPT Coherence Variants

This module provides a comprehensive benchmark for evaluating probability extraction
quality using 40 carefully designed test cases and 5 evaluation metrics.

Main components:
- test_cases.json: 40 test cases covering various probability ranges and domains
- metrics.py: 5 evaluation metrics (Brier, ECE, Compliance, Consistency, Sharpness)
- runner.py: Benchmark execution framework with comparative analysis

Usage:
    from selfcheckgpt.benchmark import BenchmarkRunner

    runner = BenchmarkRunner(model="gpt-4o-mini")
    results = runner.run_benchmark(prompt_strategy="baseline")
    runner.print_report(results)

Reference:
    agent-os/specs/2025-11-03-coherence-improvements/
"""

from .runner import BenchmarkRunner
from .metrics import (
    brier_score,
    expected_calibration_error,
    probability_coherence_compliance,
    probability_consistency_score,
    sharpness,
    compute_all_metrics
)

__all__ = [
    'BenchmarkRunner',
    'brier_score',
    'expected_calibration_error',
    'probability_coherence_compliance',
    'probability_consistency_score',
    'sharpness',
    'compute_all_metrics'
]
