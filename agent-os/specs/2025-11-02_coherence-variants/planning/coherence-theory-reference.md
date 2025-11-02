# Coherence Theory Reference for SelfCheckGPT Implementation

## Overview

This document provides the theoretical foundation for implementing three coherence-based hallucination detection variants in SelfCheckGPT. Each variant uses a different probabilistic coherence measure from formal epistemology to assess whether LLM-generated statements are mutually coherent with sampled passages from the same LLM.

**Core Hypothesis:** Hallucinated statements will show lower coherence with stochastically sampled alternative outputs because false claims lack consistent grounding, while truthful statements maintain coherence across samples.

---

## 1. Shogenji's Coherence Measure (C2)

### Citation
Shogenji, T. (1999). "Is Coherence Truth-conducive?", *Analysis*, 59: 338–345.

### Mathematical Formula

**Pairwise (2 propositions):**
```
C₂(A,B) = P(A ∧ B) / (P(A) × P(B))
```

**Generalized (n propositions):**
```
coh(A₁,...,Aₙ) = P(A₁ ∧ ... ∧ Aₙ) / [P(A₁) × ... × P(Aₙ)]
```

### Theoretical Motivation

Shogenji's measure quantifies coherence by comparing the actual joint probability of beliefs to the expected joint probability if beliefs were probabilistically independent.

**Interpretation:**
- **C₂ = 1**: Propositions are independent (neutral coherence)
- **C₂ > 1**: Positive coherence (mutual support) - propositions are more likely to be true together than independence would predict
- **C₂ < 1**: Negative coherence (conflict) - propositions are less likely to be true together
- **C₂ → ∞**: As the number of mutually supporting propositions increases, coherence grows without bound

**Intuition:** When B has no bearing on A, P(A|B) = P(A), making the ratio equal to 1. When A and B support each other, P(A|B) > P(A), yielding C₂ > 1.

### Score Range and Properties

- **Range:** (0, ∞)
- **Neutral point:** 1.0
- **Unbounded above:** Can grow arbitrarily large with strong positive coherence
- **Sensitive to set size:** Coherence increases with the number of mutually supporting propositions

### Implementation for SelfCheckGPT

**Required Probabilities (for each sentence-sample pair):**
1. P(sentence) - Individual probability of the sentence being true
2. P(sample) - Individual probability of the sampled passage being true
3. P(sentence ∧ sample) - Joint probability that both are true

**Prompt Templates:**
```
Individual: "Rate the probability that this statement is true: {statement}"
Joint: "Rate the probability that both statements are true: {sentence} AND {sample}"
```

**Calculation Steps:**
1. For each sentence S and each sampled passage Pi:
   - Extract P(S), P(Pi), P(S ∧ Pi)
   - Calculate C₂(S, Pi) = P(S ∧ Pi) / [P(S) × P(Pi)]
2. Aggregate coherence scores across all samples: mean_coherence = mean(C₂(S, P1), ..., C₂(S, Pn))
3. Normalize to [0, 1] range using min-max normalization across all sentences
4. Invert to hallucination score: hallucination_score = 1.0 - normalized_coherence

**Numerical Stability:**
- Add epsilon (1e-12) to denominators to prevent division by zero
- Clamp probabilities to [epsilon, 1.0-epsilon] range
- Handle cases where P(A ∧ B) > P(A) or P(B) (numerical errors from LLM probability estimates)

---

## 2. Glass-Olsson Relative Overlap Measure (C1)

### Citations
- Olsson, E. J. (2002). "What is the Problem of Coherence and Truth?", *The Journal of Philosophy*, 99: 246–272.
- Olsson, E. J. (2005). *Against Coherence: Truth, Probability, and Justification*, Oxford University Press.

### Mathematical Formula

**Pairwise (2 propositions):**
```
C₁(A,B) = P(A ∧ B) / P(A ∨ B)
```

**Expanded using probability axioms:**
```
C₁(A,B) = P(A ∧ B) / [P(A) + P(B) - P(A ∧ B)]
```

### Theoretical Motivation

The Glass-Olsson measure captures coherence as the proportion of "agreement" between propositions - the overlap relative to their union.

**Interpretation:**
- **C₁ = 1**: Complete agreement (propositions are logically equivalent)
- **C₁ = 0**: Complete disagreement (propositions are disjoint)
- **0 < C₁ < 1**: Partial overlap/agreement

**Intuition:** Think of Venn diagrams - coherence is the size of the intersection divided by the size of the union. High coherence means large overlap relative to total coverage.

### Score Range and Properties

- **Range:** [0, 1]
- **Bounded:** Unlike C2, always produces normalized scores
- **Maximum coherence:** Achieved when propositions are logically equivalent and consistent
- **Prior probability sensitive:** Unlike C2, considers how much "probability space" propositions occupy

### Implementation for SelfCheckGPT

**Required Probabilities (for each sentence-sample pair):**
1. P(sentence) - Individual probability of the sentence being true
2. P(sample) - Individual probability of the sampled passage being true
3. P(sentence ∧ sample) - Joint probability that both are true
4. P(sentence ∨ sample) - Derived: P(A) + P(B) - P(A ∧ B)

**Prompt Templates:**
```
Individual: "Rate the probability that this statement is true: {statement}"
Joint: "Rate the probability that both statements are true: {sentence} AND {sample}"
```

Note: P(A ∨ B) can be computed from the above probabilities, no additional prompt needed.

**Calculation Steps:**
1. For each sentence S and each sampled passage Pi:
   - Extract P(S), P(Pi), P(S ∧ Pi)
   - Calculate P(S ∨ Pi) = P(S) + P(Pi) - P(S ∧ Pi)
   - Calculate C₁(S, Pi) = P(S ∧ Pi) / P(S ∨ Pi)
2. Aggregate coherence scores across all samples: mean_coherence = mean(C₁(S, P1), ..., C₁(S, Pn))
3. Normalize to [0, 1] range (already bounded but may want consistent scaling)
4. Invert to hallucination score: hallucination_score = 1.0 - normalized_coherence

**Numerical Stability:**
- Add epsilon to denominators: P(S ∨ Pi) + epsilon
- Ensure P(S ∨ Pi) >= P(S ∧ Pi) (mathematical constraint)
- Handle edge case where both probabilities are near 0 (disjoint propositions)

---

## 3. Fitelson's Confirmation-Based Measure

### Citations
- Fitelson, B. (2003). "A Probabilistic Measure of Coherence", *Analysis*, 63: 194–199.
- Kemeny, J. and Oppenheim, P. (1952). "Degrees of Factual Support", *Philosophy of Science*, 19: 307–24.

### Mathematical Approach

Fitelson's measure is based on Kemeny and Oppenheim's factual support framework, extended to examine support relations between all subsets of a belief set.

**Factual Support Formulas (Kemeny & Oppenheim tradition):**

Several candidate formulas for measuring support s(H,E):

```
Difference measure:     s(H,E) = P(H|E) - P(H|¬E)
Simple confirmation:    s(H,E) = P(H|E) - P(H)
Normalized:            s(H,E) = [P(H|E) - P(H)] / [1 - P(H)]
Likelihood ratio:      s(H,E) = log[P(E|H) / P(E|¬H)]
```

**Fitelson's Extension:**
- Calculate support between all subset pairs (not just pairwise)
- Aggregate support values across all subsets
- Return mean support as overall coherence measure

### Theoretical Motivation

Fitelson aims to create "a quantitative, probabilistic generalization of the (deductive) logical coherence" by measuring how strongly belief subsets confirm/support each other.

**Interpretation:**
- **Positive support:** Evidence E increases probability of hypothesis H
- **Negative support:** Evidence E decreases probability of hypothesis H
- **No support:** Evidence E is irrelevant to hypothesis H

**Key feature:** Unlike C2 and C1, Fitelson's measure examines asymmetric support relationships (how well A supports B may differ from how well B supports A).

### Score Range and Properties

- **Range:** Depends on specific support formula chosen
  - Difference measure: [-1, 1]
  - Simple confirmation: [-1, 1] (approximately)
  - Normalized: [-1, 1]
- **Computational complexity:** Higher than C1/C2 due to subset enumeration
- **Maximum coherence:** When propositions are logically equivalent and consistent

### Implementation for SelfCheckGPT

**Simplified Pairwise Implementation (Practical Approach):**

Given computational constraints, implement using pairwise support (not full subset enumeration):

**Required Probabilities (for each sentence-sample pair):**
1. P(sentence) - Individual probability of the sentence being true
2. P(sample) - Individual probability of the sampled passage being true
3. P(sentence|sample) - Conditional probability of sentence given sample is true
4. P(sentence|¬sample) - Conditional probability of sentence given sample is false (optional, for difference measure)

**Prompt Templates:**
```
Individual: "Rate the probability that this statement is true: {statement}"
Conditional: "Rate the probability that statement A is true: {sentence} GIVEN that {sample} is true"
```

**Calculation Steps (using Difference Measure):**
1. For each sentence S and each sampled passage Pi:
   - Extract P(S), P(Pi), P(S|Pi)
   - Calculate P(S|¬Pi) = [P(S) - P(Pi) × P(S|Pi)] / [1 - P(Pi)]  (Bayes' theorem)
   - Calculate support: s(S, Pi) = P(S|Pi) - P(S|¬Pi)
   - Optionally calculate symmetric support: s(Pi, S) and average them
2. Aggregate support scores across all samples: mean_support = mean(s(S, P1), ..., s(S, Pn))
3. Normalize to [0, 1] range: normalized = (mean_support + 1) / 2  (mapping [-1,1] → [0,1])
4. Invert to hallucination score: hallucination_score = 1.0 - normalized

**Alternative: Simple Confirmation Measure (fewer prompts):**
```
s(S, Pi) = P(S|Pi) - P(S)
```
This requires only P(S) and P(S|Pi), avoiding the need to calculate P(S|¬Pi).

**Numerical Stability:**
- Add epsilon when calculating P(S|¬Pi) to prevent division by zero when P(Pi) ≈ 1
- Clamp conditional probabilities to [0, 1] range
- Handle cases where Bayes' theorem produces invalid results (P < 0 or P > 1) due to LLM probability errors
- If P(S|¬Pi) calculation is unstable, fall back to simple confirmation: s(S, Pi) = P(S|Pi) - P(S)

---

## Comparison of the Three Measures

| Measure | Range | Complexity | Probabilities Needed | Theoretical Focus |
|---------|-------|------------|---------------------|-------------------|
| Shogenji (C2) | (0, ∞) | Low | P(A), P(B), P(A∧B) | Independence vs support |
| Glass-Olsson (C1) | [0, 1] | Low | P(A), P(B), P(A∧B) | Relative overlap |
| Fitelson | [-1, 1] | Medium-High | P(A), P(B), P(A|B), [P(A|¬B)] | Confirmation/support |

### When to Use Each Measure

**Shogenji (C2):**
- Best for detecting strong mutual support relationships
- Sensitive to the number of samples (more samples → higher coherence for consistent statements)
- Unbounded nature may require careful normalization

**Glass-Olsson (C1):**
- Best for measuring agreement/overlap
- Naturally bounded [0,1], easier to interpret
- Good for comparing statements with different prior probabilities

**Fitelson:**
- Best for asymmetric support relationships
- Captures confirmation-theoretic intuitions
- More complex but theoretically sophisticated
- May handle nuanced cases better (e.g., one statement confirms another but not vice versa)

---

## Mapping Coherence to Hallucination Scores

All three measures assess **coherence** (higher = more coherent), but SelfCheckGPT needs **hallucination scores** (higher = more likely hallucinated).

### Inversion Strategy

**Core Assumption:** High coherence with sampled passages → Low hallucination probability

**Inversion Formula:**
```
hallucination_score = 1.0 - normalized_coherence
```

### Normalization Requirements

Before inversion, coherence scores must be normalized to [0, 1]:

**For Shogenji (C2):**
```python
# Min-max normalization across all sentence-sample pairs
normalized_C2 = (C2 - min(C2)) / (max(C2) - min(C2))
# Handle edge case: if all scores identical, return 0.5
if max(C2) == min(C2):
    normalized_C2 = 0.5
```

**For Glass-Olsson (C1):**
```python
# Already in [0,1], but may apply min-max for consistency
normalized_C1 = C1  # or apply min-max if desired
```

**For Fitelson:**
```python
# Map [-1, 1] → [0, 1]
normalized_fitelson = (support + 1) / 2
```

### Aggregation Across Samples

For each sentence, we compute coherence with multiple sampled passages. Aggregation options:

**Mean (Recommended):**
```python
mean_coherence = np.mean([C(S, P1), C(S, P2), ..., C(S, Pn)])
```

**Alternatives (for experimentation):**
- Median: More robust to outliers
- Min: Most pessimistic (lowest coherence with any sample)
- Weighted mean: Weight by sample quality/confidence

---

## Expected Coherence Score Ranges and Interpretation

### Shogenji (C2)

**Before normalization:**
- **C2 ≈ 1.0**: Neutral (independence) - may indicate weak or absent relationship
- **C2 > 2.0**: Strong positive coherence - statements mutually support each other
- **C2 < 0.5**: Negative coherence - statements conflict

**After normalization and inversion (hallucination scores):**
- **0.0-0.3**: Low hallucination risk (high coherence with samples)
- **0.3-0.7**: Medium hallucination risk (moderate coherence)
- **0.7-1.0**: High hallucination risk (low coherence with samples)

### Glass-Olsson (C1)

**Before inversion:**
- **C1 = 1.0**: Perfect agreement (likely exact match or logical equivalence)
- **C1 = 0.5**: Moderate overlap
- **C1 = 0.0**: Complete disagreement (disjoint statements)

**After inversion (hallucination scores):**
- **0.0-0.3**: Low hallucination risk (high overlap with samples)
- **0.3-0.7**: Medium hallucination risk (moderate overlap)
- **0.7-1.0**: High hallucination risk (low overlap with samples)

### Fitelson

**Before normalization:**
- **s = +1.0**: Maximum positive support (sample fully confirms sentence)
- **s = 0.0**: No support (sample is irrelevant)
- **s = -1.0**: Maximum negative support (sample contradicts sentence)

**After normalization and inversion (hallucination scores):**
- **0.0-0.3**: Low hallucination risk (strong confirmation from samples)
- **0.3-0.7**: Medium hallucination risk (weak or no confirmation)
- **0.7-1.0**: High hallucination risk (contradiction with samples)

---

## Implementation Recommendations

### 1. Start with Shogenji (Simplest)
- Requires only 3 probability extractions per sentence-sample pair
- Well-understood theoretical properties
- Easiest to debug and validate

### 2. Add Glass-Olsson (Similar Complexity)
- Uses same probabilities as Shogenji, just different formula
- Provides bounded [0,1] scores, easier to interpret
- Good for comparison with Shogenji

### 3. Implement Fitelson Last (Most Complex)
- Requires conditional probability extraction (additional prompts)
- More sophisticated but higher API costs
- Consider simplified version (simple confirmation) initially

### API Cost Optimization

**Probabilities needed per sentence (assuming k samples):**
- **Shogenji/Glass-Olsson:** 1 + k + k = 2k + 1 API calls
  - 1 for P(sentence)
  - k for P(sample_i)
  - k for P(sentence ∧ sample_i)

- **Fitelson:** 1 + k + k + k = 3k + 1 API calls
  - 1 for P(sentence)
  - k for P(sample_i)
  - k for P(sentence|sample_i)
  - k for P(sample_i|sentence) [if symmetric support desired]

**Optimization via caching:**
- Cache P(sentence) across all samples for the same sentence
- Cache P(sample_i) when evaluating multiple sentences
- Store prompt-response pairs to avoid duplicate API calls

### Validation Strategy

**Synthetic Test Cases:**
1. **Perfect coherence:** Sentence identical to samples → expect hallucination score ≈ 0.0
2. **Complete incoherence:** Sentence contradicts all samples → expect hallucination score ≈ 1.0
3. **Partial coherence:** Sentence agrees with some samples, disagrees with others → expect moderate score

**Expected Behavior:**
- Coherence scores should correlate negatively with known hallucination labels
- All three measures should show similar trends (though different magnitudes)
- Scores should be stable across runs (temperature=0.0 for deterministic LLM)

---

## References

### Primary Sources
1. Shogenji, T. (1999). "Is Coherence Truth-conducive?", *Analysis*, 59: 338–345.
2. Fitelson, B. (2003). "A Probabilistic Measure of Coherence", *Analysis*, 63: 194–199.
3. Olsson, E. J. (2002). "What is the Problem of Coherence and Truth?", *The Journal of Philosophy*, 99: 246–272.
4. Kemeny, J. and Oppenheim, P. (1952). "Degrees of Factual Support", *Philosophy of Science*, 19: 307–24.

### Encyclopedia References
- Stanford Encyclopedia of Philosophy: Formal Epistemology (https://plato.stanford.edu/entries/formal-epistemology/)
- Stanford Encyclopedia of Philosophy: Coherentist Theories of Epistemic Justification (https://plato.stanford.edu/entries/justep-coherence/)

### Additional Reading
- Bovens, L. and Hartmann, S. (2003). *Bayesian Epistemology*, Oxford University Press.
- Olsson, E. J. (2005). *Against Coherence: Truth, Probability, and Justification*, Oxford University Press.

---

## Notes on Theoretical Controversy

As the Stanford Encyclopedia notes:
> "Which measure is correct, if any, remains controversial."

This implementation treats all three measures as **exploratory hypotheses** rather than definitive solutions. The goal is to empirically evaluate which measure(s) perform best for hallucination detection on the wiki_bio_gpt3_hallucination dataset.

**Key Open Questions:**
1. Does any coherence measure outperform existing SelfCheck methods (93.42 AUC-PR baseline)?
2. Do the three measures capture different types of hallucinations?
3. How sensitive are the measures to LLM probability estimation errors?
4. Can ensemble methods combining multiple coherence measures improve performance?

These questions should guide validation experiments and future research directions.
