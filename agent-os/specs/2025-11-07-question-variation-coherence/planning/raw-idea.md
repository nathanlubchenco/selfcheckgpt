# Raw Idea

## Feature Description

Exploring a novel approach to applying coherence-based hallucination detection to benchmarks by generating and evaluating stochastic variations of QUESTIONS rather than answers.

## Core Concept

Instead of generating stochastic answer samples and checking their coherence (the traditional SelfCheckGPT approach), this feature will:

1. Take an original question from a benchmark
2. Generate stochastic variations of that question
3. Use SelfCheckGPT methodology with coherence measures to evaluate the original question against these variations
4. Use that coherence information to guide or improve the answering process

## Context

- This differs from previous work where coherence metrics were applied to benchmark ANSWERS
- Changes assumptions from previous SimpleQA evaluation attempts
- Research-oriented exploratory feature
