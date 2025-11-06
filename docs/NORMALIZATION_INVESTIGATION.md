# Normalization Strategy Investigation

## Motivation

The current coherence implementation uses **min-max normalization** for multi-sentence evaluation:

```python
normalized = (scores - min) / (max - min)
hallucination_scores = 1.0 - normalized
```

**Potential Issue**: If passages have very similar coherence scores (low variance), min-max normalization might wash out important differences, reducing discrimination between truthful and hallucinated statements.

## Hypothesis

The 8.7% performance gap between coherence methods (AUC-PR: 0.8408) and APIPrompt (AUC-PR: 0.9280) might be partially explained by suboptimal normalization.

**Specific concerns**:
1. **Low variance passages**: When all sentences in a passage have similar coherence (e.g., all around C2=1.0), min-max normalization forces some to 0.0 and others to 1.0, even if the true differences are tiny
2. **Outlier sensitivity**: A single extreme coherence value can compress the rest of the scores
3. **Passage-level normalization**: Normalizing within each passage independently loses cross-passage comparability

## Alternative Normalization Strategies

### 1. Percentile-Based Normalization (Robust)

```python
def normalize_percentile(scores, lower=0.1, upper=0.9):
    p_lower = np.percentile(scores, lower * 100)
    p_upper = np.percentile(scores, upper * 100)
    normalized = (scores - p_lower) / (p_upper - p_lower)
    return np.clip(normalized, 0.0, 1.0)
```

**Advantages**:
- Robust to outliers (clips extreme values)
- Preserves relative ordering for most scores
- Good for distributions with long tails

**Disadvantages**:
- Clips some information (scores below 10th or above 90th percentile)
- Arbitrary choice of percentile thresholds

**Best for**: Distributions with outliers or heavy tails

### 2. Z-Score + Sigmoid Normalization (Statistical)

```python
def normalize_zscore(scores):
    mean = np.mean(scores)
    std = np.std(scores)
    z_scores = (scores - mean) / std
    normalized = 1.0 / (1.0 + np.exp(-z_scores))  # Sigmoid
    return normalized
```

**Advantages**:
- Statistically principled (based on mean and variance)
- Sigmoid provides smooth mapping to [0, 1]
- Preserves information about magnitude of deviations

**Disadvantages**:
- Sensitive to mean/std estimation with few samples
- Assumes approximately normal distribution

**Best for**: Distributions close to normal, or when you want to emphasize standard deviations

### 3. Rank-Based Normalization (Ordinal)

```python
def normalize_rank(scores):
    ranks = np.argsort(np.argsort(scores))
    normalized = ranks / (len(scores) - 1)
    return normalized
```

**Advantages**:
- Only cares about ordering, not absolute values
- Completely robust to outliers
- Produces uniform distribution by design

**Disadvantages**:
- Loses magnitude information (scores of [0.1, 0.2, 0.9] same as [0.1, 0.11, 0.12])
- Can be overly aggressive in compressing differences

**Best for**: When only relative ordering matters, not absolute coherence

### 4. Softmax Normalization (Probability-like)

```python
def normalize_softmax(scores, temperature=1.0):
    exp_scores = np.exp(scores / temperature)
    softmax = exp_scores / np.sum(exp_scores)
    # Convert to cumulative for ranking effect
    sorted_indices = np.argsort(scores)
    cumulative = np.cumsum(softmax[sorted_indices])
    return cumulative
```

**Advantages**:
- Emphasizes differences (sharpens contrast)
- Temperature parameter controls degree of sharpening
- Produces probability-like distribution

**Disadvantages**:
- Can over-emphasize small differences (depending on temperature)
- Less interpretable than linear transformations

**Best for**: When you want to amplify differences and have tunable sharpness

### 5. No Normalization (Direct Mapping)

```python
def normalize_none(scores):
    # Apply exponential mapping (like our single-sentence fix)
    return np.exp(-scores)
```

**Advantages**:
- Uses raw coherence values directly
- Consistent with single-sentence scoring
- Theoretically grounded

**Disadvantages**:
- Loses cross-passage comparability
- Different passages get different score ranges

**Best for**: When coherence values are already well-calibrated

## Investigation Methodology

### Script Usage

```bash
# Test on 50 passages with Shogenji
python scripts/investigate_normalization.py --num-passages 50 --variant shogenji

# Test all strategies, save plots
python scripts/investigate_normalization.py \
    --num-passages 100 \
    --variant olsson \
    --output results/normalization_olsson.png

# Quick test (10 passages)
python scripts/investigate_normalization.py --num-passages 10 --variant shogenji
```

### Metrics Compared

For each normalization strategy, we measure:

1. **AUC-PR** (Area Under Precision-Recall): Primary metric for imbalanced data
2. **AUC-ROC** (Area Under ROC Curve): Overall discrimination ability
3. **PCC** (Pearson Correlation): Linear correlation with labels

**Goal**: Find strategy that maximizes AUC-PR (most important for hallucination detection)

### What to Look For

1. **Significant improvement** (>1% AUC-PR): Worth implementing immediately
2. **Modest improvement** (0.1-1% AUC-PR): Consider A/B testing
3. **No improvement** (<0.1%): Normalization is not the bottleneck

## Expected Outcomes

### Scenario 1: Normalization is the Problem

**Symptoms**:
- Raw coherence scores have high variance
- Min-max normalization shows poor separation
- Alternative strategy improves AUC-PR by >1%

**Action**: Implement better normalization strategy

### Scenario 2: Normalization is Not the Problem

**Symptoms**:
- Raw coherence scores have low variance (many duplicates)
- All normalization strategies perform similarly
- Improvement is <0.1% AUC-PR

**Action**: Look elsewhere for improvements:
- Prompt engineering (Phase 2)
- Better probability extraction
- Model selection (gpt-4o vs gpt-4o-mini)
- Ensemble methods

### Scenario 3: Raw Coherence is Too Coarse

**Symptoms**:
- Many duplicate raw coherence scores
- Low granularity (e.g., mostly values near 1.0)
- Normalization can't help because there's no signal to normalize

**Action**: Investigate probability extraction quality:
- Are probabilities too extreme (all 0.0 or 1.0)?
- Is the model over-confident?
- Do we need better prompts for probability calibration?

## Diagnostic Visualizations

The investigation script produces 6 plots:

1. **Metrics Comparison**: Bar chart comparing AUC-PR, AUC-ROC, PCC across strategies
2. **Score Distributions**: Histograms showing how different strategies distribute scores
3. **Raw Coherence Distribution**: Shows the distribution before any normalization
4. **Score Separation**: How well each strategy separates accurate vs inaccurate
5. **Correlation with Labels**: Scatter plot for best method
6. **Metrics Heatmap**: Color-coded comparison of all metrics

## Implementation Plan

### If Improvement Found

1. **Test on full dataset** (238 passages) to confirm improvement
2. **Test across all variants** (Shogenji, Fitelson, Olsson)
3. **Update `modeling_coherence.py`** to use new normalization
4. **Add configuration option** to allow users to choose normalization strategy
5. **Update documentation** and benchmarks

Example implementation:

```python
class SelfCheckShogenji:
    def __init__(self, model="gpt-4o-mini", normalization="percentile"):
        self.normalization_strategy = normalization
        # ...

    def predict(self, sentences, sampled_passages, verbose=False):
        # ... existing code ...

        # Apply chosen normalization strategy
        if self.normalization_strategy == "percentile":
            normalized_coherence = self._normalize_percentile(mean_coherence)
        elif self.normalization_strategy == "zscore":
            normalized_coherence = self._normalize_zscore(mean_coherence)
        # ... etc
```

### If No Improvement Found

Document findings and move on to other optimization strategies:
- Phase 2: Prompt improvements
- Better model selection
- Ensemble methods
- Hybrid approaches (coherence + baseline)

## Related Issues

- Single-sentence normalization bug (FIXED)
- Multi-sentence normalization investigation (THIS)
- Probability extraction calibration (FUTURE)
- Prompt optimization (Phase 2)

## Running the Investigation

### Quick Test (10 passages, ~2 min)

```bash
python scripts/investigate_normalization.py --num-passages 10 --variant shogenji
```

### Full Test (50 passages, ~10 min)

```bash
python scripts/investigate_normalization.py --num-passages 50 --variant shogenji --output results/norm_shog.png
python scripts/investigate_normalization.py --num-passages 50 --variant fitelson --output results/norm_fitel.png
python scripts/investigate_normalization.py --num-passages 50 --variant olsson --output results/norm_ols.png
```

### Comprehensive Test (100 passages, ~20 min)

```bash
for variant in shogenji fitelson olsson; do
    python scripts/investigate_normalization.py \
        --num-passages 100 \
        --variant $variant \
        --output results/normalization_${variant}_100passages.png
done
```

## Interpreting Results

### Example Output

```
================================================================================
RESULTS COMPARISON
================================================================================
Strategy                  AUC-PR       AUC-ROC      PCC
--------------------------------------------------------------------------------
Min-Max (current)         0.8408       0.7024       0.3343
Percentile (10-90)        0.8621       0.7238       0.3556  ← +2.13% improvement!
Z-Score + Sigmoid         0.8445       0.7089       0.3401
Rank-Based                0.8312       0.6891       0.3198
Softmax (T=1.0)           0.8389       0.7001       0.3289
None (exponential)        0.8234       0.6823       0.3012
--------------------------------------------------------------------------------
Best by AUC-PR: Percentile (10-90) (0.8621)
Best by PCC: Percentile (10-90) (0.3556)

Improvement over current min-max normalization:
  AUC-PR: 2.13% (Percentile (10-90))
  PCC: 6.37% (Percentile (10-90))
```

**Interpretation**: Percentile-based normalization shows significant improvement (+2.13% AUC-PR). This is worth implementing!

## Next Steps

1. **Run the investigation** on a representative sample (50-100 passages)
2. **Analyze results** to determine if normalization is the bottleneck
3. **Implement best strategy** if improvement >1%
4. **Re-run full benchmark** to measure impact on all 238 passages
5. **Document findings** and update codebase

## References

- Single-sentence normalization bug fix: `docs/BUGFIX_single_sentence_normalization.md`
- Coherence variants documentation: `docs/coherence_variants.md`
- Full benchmark results: `results/` directory
- Investigation script: `scripts/investigate_normalization.py`

## Date

2025-11-04
