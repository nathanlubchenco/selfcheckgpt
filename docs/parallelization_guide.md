# Parallelization Guide for SelfCheckGPT Evaluation

This guide explains how to use parallelization to significantly reduce wall-clock time for evaluations.

## Quick Start

```bash
# Standard evaluation (sequential)
python scripts/compare_methods.py --model gpt-4o-mini --num-passages 238 --verbose
# Estimated time: ~39 hours for all methods

# Parallelized evaluation (4 workers)
python scripts/compare_methods_parallel.py --model gpt-4o-mini --num-passages 238 --num-workers 4 --verbose
# Estimated time: ~10-12 hours for all methods
```

## Available Scripts

### 1. `compare_methods.py` (Original, Sequential)
- Processes one passage at a time
- Simplest, most reliable
- Best for small evaluations (<50 passages)
- Estimated time for 238 passages: ~39 hours

### 2. `compare_methods_parallel.py` (Passage-Level Parallelization)
- Processes multiple passages concurrently
- Uses ThreadPoolExecutor for API-based methods
- Configurable number of workers
- **Recommended for most use cases**

```bash
# Example: Run coherence methods with 8 workers
python scripts/compare_methods_parallel.py \
  --methods shogenji,fitelson,olsson \
  --num-passages 238 \
  --num-workers 8 \
  --verbose
```

## Speedup Estimates

### Passage-Level Parallelization

Wall-clock time reduction based on number of workers:

| Workers | Speedup | Time (238 passages, all methods) | Notes |
|---------|---------|----------------------------------|-------|
| 1 | 1x | ~39 hours | Same as sequential |
| 2 | ~1.8x | ~22 hours | Good for API rate limits |
| 4 | ~3.2x | **~12 hours** | **Recommended** |
| 8 | ~4.5x | ~9 hours | May hit rate limits |
| 16 | ~5x | ~8 hours | Diminishing returns + rate limits |

**Note:** Speedup is not linear due to:
- OpenAI API rate limits
- Cache contention
- Python GIL (for non-I/O operations)
- Network latency variance

### Per-Method Time Estimates (238 passages)

#### Sequential (1 worker)
- **SelfCheckAPIPrompt**: ~3.4 hours
- **SelfCheckShogenji**: ~9.5 hours
- **SelfCheckFitelson**: ~17.2 hours
- **SelfCheckOlsson**: ~8.7 hours
- **Total**: ~39 hours

#### Parallel (4 workers)
- **SelfCheckAPIPrompt**: ~1 hour
- **SelfCheckShogenji**: ~2.8 hours
- **SelfCheckFitelson**: ~5 hours
- **SelfCheckOlsson**: ~2.5 hours
- **Total**: **~11.3 hours**

#### Parallel (8 workers)
- **SelfCheckAPIPrompt**: ~0.6 hours
- **SelfCheckShogenji**: ~1.7 hours
- **SelfCheckFitelson**: ~3 hours
- **SelfCheckOlsson**: ~1.5 hours
- **Total**: **~6.8 hours**

## Choosing Number of Workers

### Conservative (2-4 workers)
```bash
--num-workers 4
```
- **Pros**: Reliable, respects API rate limits, 3-4x speedup
- **Cons**: Not maximum possible speed
- **Best for**: Production evaluations, API cost concerns

### Aggressive (8-16 workers)
```bash
--num-workers 8
```
- **Pros**: Maximum practical speedup (~5x)
- **Cons**: May hit rate limits, requires higher tier API access
- **Best for**: Quick experiments, tier 4+ OpenAI accounts

### API Rate Limits (OpenAI)

| Tier | RPM Limit | Recommended Workers |
|------|-----------|---------------------|
| Free | 3 | 1-2 |
| Tier 1 | 500 | 2-4 |
| Tier 2 | 5,000 | 4-8 |
| Tier 3+ | 10,000+ | 8-16 |

Check your tier: https://platform.openai.com/settings/organization/limits

## Advanced: Async API Calls

For even faster evaluation, use the async API client (experimental):

```python
# In your code
from selfcheckgpt.modeling_coherence_api_async import CoherenceAPIClientAsync

# This allows concurrent API calls within a single passage
# Can provide additional 2-3x speedup on top of passage parallelization
```

**Note:** This feature is experimental and may require code modifications.

## Best Practices

### 1. Start Small
```bash
# Test with 10 passages first
python scripts/compare_methods_parallel.py \
  --methods shogenji \
  --num-passages 10 \
  --num-workers 4 \
  --verbose
```

### 2. Save Results Incrementally
```bash
# Save results to avoid losing progress
python scripts/compare_methods_parallel.py \
  --methods shogenji \
  --num-passages 238 \
  --num-workers 4 \
  --output results_shogenji.json \
  --verbose
```

### 3. Run Methods Separately
```bash
# Run each method independently to avoid losing all progress on error
python scripts/compare_methods_parallel.py --methods shogenji --num-passages 238 --num-workers 4 --output shogenji.json
python scripts/compare_methods_parallel.py --methods fitelson --num-passages 238 --num-workers 4 --output fitelson.json
python scripts/compare_methods_parallel.py --methods olsson --num-passages 238 --num-workers 4 --output olsson.json
```

### 4. Monitor Rate Limits
Watch for "rate_limit" warnings in output. If you see many:
- Reduce `--num-workers`
- Add delays between batches
- Upgrade API tier

### 5. Cost Estimation

Before running full evaluation, estimate costs:

```python
from selfcheckgpt.modeling_coherence_api import CoherenceAPIClient

estimate = CoherenceAPIClient.estimate_api_calls(
    num_sentences=1908,  # From your baseline: 238 passages = 1908 sentences
    num_samples=5,
    num_variants=3,  # Shogenji, Fitelson, Olsson
    include_conditional=True  # Fitelson needs conditional probabilities
)
print(f"Estimated API calls: {estimate['total_calls']:,}")
print(f"Estimated cost: ${estimate['estimated_cost_usd']:.2f}")
```

## Troubleshooting

### "Rate limit exceeded" errors
- Reduce `--num-workers`
- Check your API tier limits
- Add retry logic (already included in code)

### Out of memory errors
- Reduce `--num-workers` for GPU-based methods
- Use CPU instead: `--device cpu`
- Process methods separately

### Inconsistent results
- Parallelization should not affect results due to caching
- If results differ, check for non-deterministic behavior
- Set `temperature=0.0` in API calls (already done)

### Progress bars not showing
- Use `--verbose` flag
- Check that `tqdm` is installed: `pip install tqdm`

## Example Workflows

### Full Evaluation (Recommended)
```bash
# 1. Test with small sample
python scripts/compare_methods_parallel.py --methods shogenji --num-passages 10 --num-workers 4 --verbose

# 2. Run baseline for comparison
python scripts/eval_baseline.py --model gpt-4o-mini --num-passages 238 --verbose

# 3. Run coherence methods in parallel with 4 workers
python scripts/compare_methods_parallel.py \
  --methods shogenji,fitelson,olsson \
  --num-passages 238 \
  --num-workers 4 \
  --verbose \
  --output results_coherence_parallel.json

# Estimated total time: ~3.4 hours (baseline) + 10.3 hours (coherence) = ~13.7 hours
```

### Quick Comparison (1 hour)
```bash
# Just run Shogenji (fastest coherence method)
python scripts/compare_methods_parallel.py \
  --methods apiprompt,shogenji \
  --num-passages 238 \
  --num-workers 8 \
  --verbose
```

### Budget-Conscious (minimize API costs)
```bash
# Use caching aggressively, run once and save results
python scripts/compare_methods_parallel.py \
  --methods shogenji \
  --num-passages 238 \
  --num-workers 2 \
  --output results_shogenji.json

# Check cache stats to see savings
# Expected: ~40-45% cache hit rate = 40-45% cost reduction
```

## Performance Monitoring

Track performance during execution:

```bash
# In another terminal, monitor API usage
watch -n 60 'curl -H "Authorization: Bearer $OPENAI_API_KEY" https://api.openai.com/v1/usage'

# Monitor system resources
htop  # or top on macOS
```

## Summary

**Recommended configuration for 238-passage evaluation:**

```bash
python scripts/compare_methods_parallel.py \
  --methods apiprompt,shogenji,fitelson,olsson \
  --api-model gpt-4o-mini \
  --num-passages 238 \
  --num-workers 4 \
  --verbose \
  --output results_full.json
```

**Expected results:**
- Wall-clock time: ~12 hours (vs. 39 hours sequential)
- Total API calls: ~250,000 (with ~40% cache hit rate)
- Estimated cost: $15-30 (depending on token counts)
- Speedup: **3.2x**
