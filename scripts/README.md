# Evaluation Scripts

Three scripts for evaluating SelfCheckGPT methods on the wiki_bio_gpt3_hallucination dataset.

## Scripts

### 1. `simple_eval.py` - Evaluate Coherence Methods
Evaluate the new coherence-based variants (Shogenji, Fitelson, Olsson).

```bash
# Quick test (10 passages)
python scripts/simple_eval.py --variant shogenji --verbose

# Test different variants
python scripts/simple_eval.py --variant fitelson --num-passages 10
python scripts/simple_eval.py --variant olsson --num-passages 10

# Full evaluation (238 passages) with gpt-4o-mini
python scripts/simple_eval.py --variant shogenji --num-passages 238 --model gpt-4o-mini

# Use different model
python scripts/simple_eval.py --variant shogenji --model gpt-3.5-turbo
```

**Options:**
- `--variant`: shogenji, fitelson, or olsson (default: shogenji)
- `--model`: OpenAI model name (default: gpt-4o-mini)
- `--num-passages`: Number to evaluate (default: 10, max: 238)
- `--verbose`: Show progress bars

### 2. `eval_baseline.py` - Evaluate Baseline Method
Evaluate the baseline SelfCheckAPIPrompt method for comparison.

```bash
# Quick test with gpt-4o-mini
python scripts/eval_baseline.py --model gpt-4o-mini --num-passages 10 --verbose

# Full evaluation (replicate baseline with gpt-3.5-turbo)
python scripts/eval_baseline.py --model gpt-3.5-turbo --num-passages 238
```

**Options:**
- `--model`: OpenAI model name (default: gpt-4o-mini)
- `--num-passages`: Number to evaluate (default: 10, max: 238)
- `--verbose`: Show progress bars

### 3. `compare_methods.py` - Side-by-Side Comparison
Compare baseline and coherence methods with the same model.

```bash
# Compare all methods on 10 passages with gpt-4o-mini
python scripts/compare_methods.py --model gpt-4o-mini --num-passages 10 --verbose

# Full comparison (238 passages) - WARNING: expensive!
python scripts/compare_methods.py --model gpt-4o-mini --num-passages 238

# Compare only coherence methods
python scripts/compare_methods.py --methods coherence-only --num-passages 10

# Compare only baseline
python scripts/compare_methods.py --methods baseline-only --num-passages 10

# Save results to JSON
python scripts/compare_methods.py --num-passages 10 --output results/comparison.json
```

**Options:**
- `--model`: OpenAI model name (default: gpt-4o-mini)
- `--num-passages`: Number to evaluate (default: 10, max: 238)
- `--methods`: all, baseline-only, coherence-only, or comma-separated (default: all)
- `--verbose`: Show progress bars
- `--output`: Save results to JSON file (optional)

## Example Workflow

```bash
# 1. Quick test of one coherence variant
python scripts/simple_eval.py --variant shogenji --num-passages 5 --verbose

# 2. Quick test of baseline
python scripts/eval_baseline.py --num-passages 5 --verbose

# 3. Full comparison on small subset
python scripts/compare_methods.py --num-passages 10 --verbose

# 4. Full evaluation (when ready)
python scripts/compare_methods.py --num-passages 238 --output results/full_comparison.json
```

## Published Baseline (for reference)

From the SelfCheckGPT paper (EMNLP 2023):
- Method: SelfCheckAPIPrompt with gpt-3.5-turbo
- Dataset: wiki_bio_gpt3_hallucination (238 passages)
- **AUC-PR: 93.42**
- **AUC-ROC: 67.09**
- **PCC: 78.32**

## Dataset

The `wiki_bio_gpt3_hallucination` dataset includes:
- 238 Wikipedia biography passages
- Sentence-level annotations (accurate/minor_inaccurate/major_inaccurate)
- 20 pre-generated stochastic samples per passage

All scripts use the pre-generated samples for consistency with the published results.
