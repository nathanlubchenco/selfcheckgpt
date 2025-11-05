# Evaluation Scripts

Unified evaluation script for all SelfCheckGPT methods with optional parallelization.

## Quick Start

```bash
# View help and examples
python scripts/evaluate.py --help

# Quick test (10 passages)
python scripts/evaluate.py --methods shogenji --num-passages 10 --workers 4 --verbose

# Just baseline
python scripts/evaluate.py --methods apiprompt --num-passages 238 --verbose

# Apples-to-apples comparison (all methods use gpt-4o-mini)
python scripts/evaluate.py --methods apiprompt,shogenji,fitelson,olsson --num-passages 238 --workers 4 --verbose

# All methods (traditional + API + coherence)
python scripts/evaluate.py --methods all --num-passages 238 --workers 4 --verbose
```

## Available Methods

### Traditional Methods (Local Models, No API Cost)
- **nli**: SelfCheckNLI using DeBERTa-v3-large
- **mqag**: SelfCheckMQAG using T5 + Longformer
- **bertscore**: SelfCheckBERTScore using RoBERTa
- **ngram**: SelfCheckNgram (unigram model)

### API-Based Methods
- **apiprompt**: SelfCheckAPIPrompt (baseline from paper)

### Coherence Methods (API-Based)
- **shogenji**: Shogenji's coherence measure
- **fitelson**: Fitelson's confirmation measure
- **olsson**: Glass-Olsson overlap measure

### Method Shortcuts
- **all**: All methods (traditional + API + coherence)
- **traditional**: Just traditional methods (nli, mqag, bertscore, ngram)
- **api-only**: Just apiprompt
- **coherence**: Just coherence methods (shogenji, fitelson, olsson)

## Common Use Cases

### 1. Reproduce Baseline Results
```bash
python scripts/evaluate.py \
  --methods apiprompt \
  --api-model gpt-4o-mini \
  --num-passages 238 \
  --verbose
```

### 2. Compare Coherence Methods to Baseline
```bash
python scripts/evaluate.py \
  --methods apiprompt,shogenji,fitelson,olsson \
  --api-model gpt-4o-mini \
  --num-passages 238 \
  --workers 4 \
  --verbose \
  --output results.json
```

### 3. Full Benchmark (All Methods)
```bash
python scripts/evaluate.py \
  --methods all \
  --num-passages 238 \
  --workers 4 \
  --verbose \
  --output full_benchmark.json
```

### 4. Traditional Methods Only (No API Cost)
```bash
python scripts/evaluate.py \
  --methods traditional \
  --num-passages 238 \
  --verbose \
  --device cuda
```

### 5. Quick Development Test
```bash
python scripts/evaluate.py \
  --methods shogenji \
  --num-passages 10 \
  --workers 4 \
  --verbose
```

## Parameters

### Required
- `--methods`: Methods to evaluate (comma-separated or shortcut)

### Optional
- `--num-passages`: Number of passages (default: 238, use 10 for quick test)
- `--api-model`: OpenAI model for API methods (default: gpt-4o-mini)
- `--workers`: Parallel workers for API methods (default: 1, recommended: 4-8)
- `--device`: Device for local models (default: auto, options: cuda/cpu)
- `--output`: Save results to JSON file (optional)
- `--verbose`: Show progress bars and detailed output

## Parallelization

API-based methods (apiprompt, shogenji, fitelson, olsson) support parallelization:

| Workers | Speedup | Time (238 passages) | Best For |
|---------|---------|---------------------|----------|
| 1 | 1x | ~39 hours | Baseline |
| 4 | ~3.2x | **~12 hours** | **Recommended** |
| 8 | ~4.5x | ~9 hours | Fast API tier |

**Note:** Traditional methods always run sequentially (GPU memory constraints).

### Example with Parallelization
```bash
# 4 workers (3.2x speedup) - recommended
python scripts/evaluate.py \
  --methods coherence \
  --num-passages 238 \
  --workers 4 \
  --verbose

# 8 workers (4.5x speedup) - requires tier 2+ API access
python scripts/evaluate.py \
  --methods coherence \
  --num-passages 238 \
  --workers 8 \
  --verbose
```

## Output Format

### Console Output
```
================================================================================
RESULTS (238 passages)
================================================================================
Method                         Type                 AUC-PR     AUC-ROC    PCC
--------------------------------------------------------------------------------
SelfCheckAPIPrompt             API (gpt-4o-mini)    0.9281     0.8573     0.5891
SelfCheckShogenji              Coherence (...)      0.9649     0.8279     0.3609
...

Published Baselines (238 passages from EMNLP 2023 paper):
  SelfCheckAPIPrompt (gpt-3.5-turbo): AUC-PR=93.42, AUC-ROC=67.09, PCC=78.32
  SelfCheckNLI (DeBERTa):              AUC-PR=92.50, AUC-ROC=N/A,   PCC=74.14

================================================================================
CACHE STATISTICS (API Cost Reduction)
================================================================================
SelfCheckShogenji:
  Cache hits: 1,580 / 3,649 (43.3%)
  Actual API calls: 2,069
  Cost savings: ~43%
```

### JSON Output (--output results.json)
```json
{
  "timestamp": "2025-11-04T17:00:00",
  "config": {
    "api_model": "gpt-4o-mini",
    "device": "cuda",
    "num_workers": 4,
    "num_passages": 238,
    "methods": ["apiprompt", "shogenji"]
  },
  "results": [
    {
      "method": "SelfCheckAPIPrompt",
      "auc_pr": 0.9281,
      "auc_roc": 0.8573,
      "pcc": 0.5891,
      "num_sentences": 1908,
      "num_accurate": 516,
      "num_inaccurate": 1392
    },
    ...
  ]
}
```

## Tips

1. **Start small**: Test with `--num-passages 10` before running full evaluation
2. **Save results**: Always use `--output` to avoid losing long-running evaluations
3. **Parallelization**: Use `--workers 4` for API methods (3.2x speedup)
4. **API costs**: Coherence methods have ~40% cache hit rate, reducing costs significantly
5. **GPU memory**: Traditional methods run sequentially to avoid GPU memory issues

## Troubleshooting

### "Rate limit exceeded"
- Reduce `--workers` (try 2 instead of 4)
- Check your OpenAI API tier at https://platform.openai.com/settings/organization/limits

### Out of GPU memory
- Use `--device cpu` for traditional methods
- Close other GPU applications

### Import errors
```bash
# Make sure you're in the venv and have installed the package
pip install -e .
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

## See Also

- Main docs: `README.md` (project root)
- Coherence theory: `docs/coherence_variants.md`
- Parallelization guide: `docs/parallelization_guide.md`
