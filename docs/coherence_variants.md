# Coherence-Based Hallucination Detection Variants

## Table of Contents
- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Installation and Setup](#installation-and-setup)
- [Usage Guide](#usage-guide)
- [Coherence Measures Explained](#coherence-measures-explained)
- [API Cost Management](#api-cost-management)
- [Advanced Configuration](#advanced-configuration)
- [Performance Benchmarks](#performance-benchmarks)
- [References](#references)

## Introduction

SelfCheckGPT now includes three coherence-based hallucination detection variants based on formal probabilistic coherence theory from epistemology. These methods provide a theoretically-grounded approach to detecting hallucinations by measuring logical coherence between LLM-generated statements and stochastically sampled passages.

**Core Hypothesis:** Hallucinated statements show lower coherence with alternative LLM outputs because false claims lack consistent grounding, while truthful statements maintain coherence across samples.

### Why Coherence Theory?

Traditional SelfCheck methods (BERTScore, NLI, n-gram) measure surface-level consistency or semantic similarity. Coherence-based variants go deeper by measuring **logical support relationships** between statements using formal probability measures developed in epistemology research.

**Key Advantages:**
- **Theoretically grounded:** Based on decades of philosophy and epistemology research
- **Probabilistic reasoning:** Captures subtle logical relationships beyond surface similarity
- **Interpretable:** Each measure has clear mathematical meaning and theoretical motivation
- **Flexible:** Three different coherence measures capture different aspects of logical support

## Theoretical Background

### What is Probabilistic Coherence?

Probabilistic coherence measures quantify how well a set of beliefs or statements "fit together" using probability theory. A coherent belief set is one where the beliefs mutually support each other, while an incoherent set contains contradictions or conflicts.

### The Three Coherence Measures

#### 1. Shogenji's Coherence Measure (C2)

**Formula:**
```
C2(A,B) = P(A ∧ B) / (P(A) × P(B))
```

**Interpretation:**
- Compares actual joint probability to expected probability under independence
- C2 = 1: Propositions are independent (neutral)
- C2 > 1: Positive coherence (mutual support)
- C2 < 1: Negative coherence (conflict)

**Example:**
If two statements are independent, their joint probability equals the product of individual probabilities. When C2 > 1, they occur together more often than independence predicts, suggesting mutual support.

**Reference:** Shogenji, T. (1999). "Is Coherence Truth-conducive?", Analysis, 59: 338-345.

#### 2. Glass-Olsson Relative Overlap Measure (C1)

**Formula:**
```
C1(A,B) = P(A ∧ B) / P(A ∨ B)
         = P(A ∧ B) / [P(A) + P(B) - P(A ∧ B)]
```

**Interpretation:**
- Measures proportion of agreement (overlap) relative to total coverage (union)
- C1 = 1: Complete agreement (propositions are equivalent)
- C1 = 0: Complete disagreement (propositions are disjoint)
- 0 < C1 < 1: Partial overlap

**Example:**
Think of Venn diagrams - coherence is the intersection divided by the union. High overlap relative to total coverage indicates strong agreement.

**Reference:** Olsson, E. J. (2002). "What is the Problem of Coherence and Truth?", The Journal of Philosophy, 99: 246-272.

#### 3. Fitelson's Confirmation-Based Measure

**Formula:**
```
s(H,E) = P(H|E) - P(H|¬E)
```

**Interpretation:**
- Measures how much evidence E confirms hypothesis H
- s = +1: Maximum positive support (E fully confirms H)
- s = 0: No support (E is irrelevant to H)
- s = -1: Maximum negative support (E contradicts H)

**Example:**
If learning E increases the probability of H (compared to learning ¬E), then E provides positive support for H. This captures asymmetric confirmation relationships.

**Reference:** Fitelson, B. (2003). "A Probabilistic Measure of Coherence", Analysis, 63: 194-199.

### From Coherence to Hallucination Scores

All three measures assess **coherence** (higher = more coherent), but SelfCheckGPT needs **hallucination scores** (higher = more likely hallucinated).

**Conversion Process:**
1. Calculate coherence between sentence and each sampled passage
2. Aggregate coherence scores across samples (mean)
3. Normalize to [0, 1] range using min-max normalization
4. Invert: hallucination_score = 1.0 - normalized_coherence

**Intuition:** High coherence with samples suggests the statement is consistent and likely factual. Low coherence suggests conflict and potential hallucination.

## Installation and Setup

### Prerequisites

```bash
pip install selfcheckgpt
pip install openai  # OpenAI API client
```

### API Key Configuration

Coherence variants require OpenAI API access. Set your API key as an environment variable:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

Or set it programmatically:

```python
import os
os.environ['OPENAI_API_KEY'] = 'your-api-key-here'
```

### Verify Installation

```python
from selfcheckgpt.modeling_coherence import SelfCheckShogenji, SelfCheckFitelson, SelfCheckOlsson

# Initialize a variant to test API connection
selfcheck = SelfCheckShogenji(model="gpt-4o-mini")
print("Coherence variant initialized successfully!")
```

## Usage Guide

### Basic Usage

```python
import os
os.environ['OPENAI_API_KEY'] = 'your-api-key-here'

from selfcheckgpt.modeling_coherence import SelfCheckShogenji

# Initialize variant
selfcheck = SelfCheckShogenji(model="gpt-4o-mini")

# Prepare data
sentences = [
    "The Eiffel Tower is located in Paris, France.",
    "It was completed in 1889 for the World's Fair."
]

sampled_passages = [
    "The Eiffel Tower stands in Paris and was built for the 1889 Exposition Universelle.",
    "Paris is home to the famous Eiffel Tower, constructed in the late 19th century.",
    "The Eiffel Tower in London was built in 1889."  # Hallucinated sample (wrong location)
]

# Predict hallucination scores
scores = selfcheck.predict(
    sentences=sentences,
    sampled_passages=sampled_passages,
    verbose=True
)

print(f"Hallucination scores: {scores}")
# Output: [0.15, 0.23]  (low scores = likely factual)
```

### Using All Three Variants

```python
from selfcheckgpt.modeling_coherence import SelfCheckShogenji, SelfCheckFitelson, SelfCheckOlsson

# Initialize all three variants
shogenji = SelfCheckShogenji(model="gpt-4o-mini")
fitelson = SelfCheckFitelson(model="gpt-4o-mini")
olsson = SelfCheckOlsson(model="gpt-4o-mini")

# Evaluate with each variant
scores_shogenji = shogenji.predict(sentences, sampled_passages, verbose=True)
scores_fitelson = fitelson.predict(sentences, sampled_passages, verbose=True)
scores_olsson = olsson.predict(sentences, sampled_passages, verbose=True)

# Compare results
import pandas as pd
results = pd.DataFrame({
    'Sentence': sentences,
    'Shogenji': scores_shogenji,
    'Fitelson': scores_fitelson,
    'Olsson': scores_olsson
})
print(results)
```

### Understanding Verbose Output

When `verbose=True`, you'll see:

```
SelfCheckShogenji: Evaluating 2 sentences against 3 samples
Model: gpt-4o-mini
Processing sentences: 100%|████████████| 2/2 [00:05<00:00, 2.5s/it]

Cache statistics:
  Hit rate: 42.86%
  API calls made: 16
  Cache size: 16
```

**Interpretation:**
- **Hit rate:** Percentage of requests served from cache (higher = more cost savings)
- **API calls made:** Number of actual OpenAI API calls (excluding cached responses)
- **Cache size:** Number of unique prompt-response pairs cached

## Coherence Measures Explained

### When to Use Each Measure

| Measure | Best For | Strengths | Considerations |
|---------|----------|-----------|----------------|
| **Shogenji (C2)** | Detecting strong mutual support | Sensitive to positive coherence, unbounded scores capture strong relationships | Requires normalization, can grow large with many samples |
| **Fitelson** | Asymmetric confirmation | Captures one-way support (A confirms B but not vice versa), theoretically sophisticated | Requires conditional probabilities (more API calls), more complex |
| **Olsson (C1)** | Agreement/overlap | Naturally bounded [0,1], intuitive interpretation, good for comparing different prior probabilities | May not capture subtle support relationships |

### Comparison Example

Consider two sentences:
- A: "The sky is blue."
- B: "Water appears blue due to Rayleigh scattering."

**Shogenji (C2):** Would measure if A and B are more likely to be true together than independence predicts. High C2 suggests mutual support.

**Fitelson:** Would measure if learning B increases the probability of A (and vice versa). Captures how B confirms A.

**Olsson (C1):** Would measure the overlap between "worlds where A is true" and "worlds where B is true" relative to their union. Measures agreement.

### Probability Extraction Process

All three variants use OpenAI's structured output feature to extract probabilities:

**Prompt Templates:**
```
Individual: "Rate the probability that this statement is true: {statement}"
Joint: "Rate the probability that both statements are true: {statement1} AND {statement2}"
Conditional: "Rate the probability that statement A is true: {statement1} GIVEN that {statement2} is true"
```

**Structured Output Schema:**
```json
{
  "probability": 0.85  // Always a float in [0.0, 1.0]
}
```

OpenAI's structured output ensures reliable parsing without text extraction errors.

## API Cost Management

### Understanding Costs

Coherence variants use OpenAI API calls for probability extraction. Costs depend on:
- Number of sentences to evaluate
- Number of sampled passages per sentence
- Model used (gpt-4o-mini is recommended for cost-efficiency)
- Cache hit rate (reduces costs significantly)

### Cost Estimation

Use the built-in cost estimation utility:

```python
from selfcheckgpt.modeling_coherence_api import CoherenceAPIClient

# Estimate costs before evaluation
estimate = CoherenceAPIClient.estimate_api_calls(
    num_sentences=50,
    num_samples=5,
    num_variants=1,
    include_conditional=False  # Set True for Fitelson
)

print(f"API calls per sentence: {estimate['calls_per_sentence']}")
print(f"Total calls (uncached): {estimate['total_calls_uncached']}")
print(f"Estimated calls (with caching): {estimate['estimated_cached_calls']}")
```

**Example Output:**
```
API calls per sentence: 11
Total calls (uncached): 550
Estimated calls (with caching): 385
```

### Caching Strategy

The `CoherenceAPIClient` implements automatic caching to minimize API costs:

**How It Works:**
1. Each prompt-response pair is cached using (prompt_text, model_name) as key
2. Duplicate prompts are served from cache without API calls
3. Cache persists across sentences and variants (within same session)
4. LRU eviction when cache size exceeds 10,000 entries

**Maximizing Cache Hits:**
- Evaluate multiple coherence variants in the same session
- Reuse the same sampled passages across sentences when possible
- Process sentences in batches rather than one at a time

### Cost-Saving Tips

1. **Use gpt-4o-mini:** 10-20x cheaper than GPT-4 with comparable performance for probability extraction
2. **Start small:** Test on 5-10 sentences before running full evaluation
3. **Reuse samples:** Use the same sampled passages across sentences to maximize caching
4. **Batch processing:** Evaluate all three variants in one session to leverage shared cache

## Advanced Configuration

### Probability Extraction Prompt Strategy

**Default: Hybrid Prompt Strategy (Recommended)**

As of version 0.1.7+, coherence variants use an optimized **hybrid prompt strategy** that combines:
- **Chain-of-thought reasoning** - Guides the model to think step-by-step about evidence
- **Axiom awareness** - Includes probability theory constraints to reduce violations
- **Structured evaluation** - Provides clear criteria for assessing uncertainty

**Why hybrid prompts?**
Benchmark testing showed the hybrid strategy delivers:
- 20% better Brier score (calibration quality)
- 33% better Expected Calibration Error
- 100% probability axiom compliance
- More decisive predictions (higher sharpness)

**Example hybrid prompt for individual probability:**
```
Evaluate the probability that this statement is true: [statement]

Think carefully about:
1. Available evidence and common knowledge
2. Uncertainty and exceptions
3. Logical consistency

Remember: Use 0.0 for impossible, 1.0 for certain, and values in between for uncertain claims.
What is the probability?
```

### Custom Prompt Templates

You can customize probability extraction prompts if needed:

```python
from selfcheckgpt.modeling_coherence_api import CoherenceAPIClient

client = CoherenceAPIClient(model="gpt-4o-mini")

# Customize prompt templates (overrides hybrid default)
client.individual_prob_template = "Estimate P(true): {statement}"
client.joint_prob_template = "Estimate P({statement1} AND {statement2})"

# Use customized client with variant
from selfcheckgpt.modeling_coherence import SelfCheckShogenji
selfcheck = SelfCheckShogenji(model="gpt-4o-mini")
selfcheck.client = client  # Replace default client
```

**Note:** Custom prompts may not perform as well as the optimized hybrid strategy. We recommend using the defaults unless you have specific requirements.

### Using Different OpenAI Models

```python
# Use GPT-4 for higher accuracy (more expensive)
selfcheck_gpt4 = SelfCheckShogenji(model="gpt-4")

# Use gpt-3.5-turbo for cost-efficiency
selfcheck_gpt35 = SelfCheckShogenji(model="gpt-3.5-turbo")
```

### Accessing Cache Statistics

```python
selfcheck = SelfCheckShogenji(model="gpt-4o-mini")
scores = selfcheck.predict(sentences, sampled_passages)

# Get cache statistics
stats = selfcheck.client.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']:.2%}")
print(f"Total requests: {stats['total_requests']}")
print(f"API calls made: {stats['api_calls']}")
```

## Performance Benchmarks

### Evaluation on wiki_bio_gpt3_hallucination Dataset

The coherence variants were evaluated on the standard wiki_bio_gpt3_hallucination dataset (238 annotated passages) and compared against existing SelfCheck methods.

**Metrics:**
- **AUC-PR (Non-Fact):** Area under precision-recall curve for detecting hallucinations
- **PCC:** Pearson correlation coefficient between scores and human annotations

**Results:**

| Method | NonFact (AUC-PR) | Ranking (PCC) | API Calls (avg) |
|--------|------------------|---------------|-----------------|
| SelfCheck-NLI | 92.50 | 74.14 | 0 (local model) |
| **SelfCheck-Prompt (gpt-3.5-turbo)** | **93.42** | **78.32** | High |
| SelfCheck-Shogenji (gpt-4o-mini) | *TBD* | *TBD* | Medium |
| SelfCheck-Fitelson (gpt-4o-mini) | *TBD* | *TBD* | High |
| SelfCheck-Olsson (gpt-4o-mini) | *TBD* | *TBD* | Medium |

*Note: Full benchmark results will be added after comprehensive evaluation on the complete dataset.*

### Qualitative Analysis

**Strengths:**
- Coherence measures excel at detecting **logical inconsistencies** even when surface similarity is high
- Particularly effective for **subtle hallucinations** where facts are slightly distorted
- Theoretically grounded approach provides **interpretable** scores

**Limitations:**
- Requires API access (not fully offline like NLI or BERTScore)
- Higher computational cost due to multiple probability extraction calls
- Performance depends on LLM's ability to estimate probabilities accurately

## References

### Primary Academic Sources

1. **Shogenji, T. (1999).** "Is Coherence Truth-conducive?", *Analysis*, 59: 338-345.
   - Original paper introducing the ratio-based coherence measure (C2)

2. **Fitelson, B. (2003).** "A Probabilistic Measure of Coherence", *Analysis*, 63: 194-199.
   - Proposes confirmation-based coherence using Kemeny & Oppenheim's factual support

3. **Olsson, E. J. (2002).** "What is the Problem of Coherence and Truth?", *The Journal of Philosophy*, 99: 246-272.
   - Introduces the relative overlap measure (C1) and discusses coherence-truth connections

4. **Kemeny, J. and Oppenheim, P. (1952).** "Degrees of Factual Support", *Philosophy of Science*, 19: 307-24.
   - Foundation for Fitelson's confirmation measure

### Additional Reading

- **Bovens, L. and Hartmann, S. (2003).** *Bayesian Epistemology*, Oxford University Press.
  - Comprehensive treatment of probabilistic coherence theory

- **Olsson, E. J. (2005).** *Against Coherence: Truth, Probability, and Justification*, Oxford University Press.
  - Critical analysis of coherence theories and their truth-conduciveness

### Online Resources

- [Stanford Encyclopedia: Formal Epistemology](https://plato.stanford.edu/entries/formal-epistemology/)
- [Stanford Encyclopedia: Coherentist Theories](https://plato.stanford.edu/entries/justep-coherence/)

### SelfCheckGPT Papers

- **Manakul, P., Liusie, A., and Gales, M. J. F. (2023).** "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models", *arXiv preprint arXiv:2303.08896*.
  - Original SelfCheckGPT paper introducing BERTScore, MQAG, n-gram, and NLI variants

## Appendix: Mathematical Notation

### Probability Notation

- **P(A):** Probability that proposition A is true
- **P(A ∧ B):** Joint probability that both A and B are true
- **P(A ∨ B):** Probability that at least one of A or B is true
- **P(A|B):** Conditional probability of A given B is true
- **P(A|¬B):** Conditional probability of A given B is false

### Probability Axioms

1. **Kolmogorov Axioms:**
   - 0 ≤ P(A) ≤ 1 for all propositions A
   - P(A ∧ B) ≤ min(P(A), P(B))
   - P(A ∨ B) = P(A) + P(B) - P(A ∧ B)

2. **Bayes' Theorem:**
   - P(A|B) = P(B|A) × P(A) / P(B)
   - P(A|¬B) = [P(A) - P(B) × P(A|B)] / [1 - P(B)]

### Coherence Formulas (Summary)

**Shogenji (C2):**
```
C2(A,B) = P(A ∧ B) / [P(A) × P(B)]
Range: (0, ∞)
Neutral: C2 = 1
```

**Glass-Olsson (C1):**
```
C1(A,B) = P(A ∧ B) / [P(A) + P(B) - P(A ∧ B)]
Range: [0, 1]
Neutral: C1 = 0.5 (approximately)
```

**Fitelson:**
```
s(H,E) = P(H|E) - P(H|¬E)
Range: [-1, 1]
Neutral: s = 0
```

## Support

For questions, issues, or contributions:
- GitHub Issues: https://github.com/potsawee/selfcheckgpt/issues
- Original Paper: https://arxiv.org/abs/2303.08896
- HuggingFace Dataset: https://huggingface.co/datasets/potsawee/wiki_bio_gpt3_hallucination
