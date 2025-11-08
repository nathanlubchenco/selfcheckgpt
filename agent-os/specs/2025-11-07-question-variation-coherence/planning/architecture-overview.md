# Architecture Overview: Question Variation Coherence

## High-Level Approach

This feature implements a novel application of coherence theory to hallucination prevention: using question variation consistency rather than answer variation consistency to guide model abstention.

### Core Hypothesis

**True factual knowledge is robust to question paraphrasing; hallucinated responses show inconsistency when questions are rephrased.**

If multiple paraphrased versions of a question yield incoherent/contradictory answers, the model should abstain ("I don't know") rather than risk a hallucinated response.

## System Architecture

### Repository Structure: Hybrid Approach

```
cosmos-coherence/                          selfcheckgpt/
  src/cosmos_coherence/                      selfcheckgpt/
    benchmarks/                                modeling_coherence.py       [IMPORT]
      simpleqa_coherence_cli.py  [NEW]        modeling_coherence_api.py  [IMPORT]
      implementations/                        utils_coherence.py         [IMPORT]
        simpleqa_coherence_benchmark.py [NEW]
        simpleqa_benchmark.py    [EXTEND]
        simpleqa_grader.py       [REUSE]
    llm/
      openai_client.py           [REUSE]
```

**Decision Rationale:**
- **Implementation location**: cosmos-coherence (SimpleQA infrastructure exists here)
- **Coherence logic**: Import from selfcheckgpt (coherence theory implementation exists here)
- **Separation of concerns**: Question/answer handling in cosmos-coherence, coherence calculation in selfcheckgpt

### Data Flow Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. LOAD SIMPLEQA QUESTION                                           │
│    - HuggingFace dataset: basicv8vc/SimpleQA                        │
│    - Question: "What is the capital of France?"                     │
│    - Ground truth: "Paris"                                          │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. GENERATE QUESTION VARIATIONS (NEW COMPONENT)                     │
│    - Use OpenAI API with temperature range [0.7, 1.0, 1.3]         │
│    - Prompt: "Rephrase this question: {question}"                  │
│    - Generate N variations (default: 5)                             │
│    - Example outputs:                                               │
│      * "Which city serves as France's capital?"                     │
│      * "What's the capital city of France?"                         │
│      * "France's capital is located in which city?"                 │
│      * "Name the capital of France"                                 │
│      * "What city is the French capital?"                           │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. COLLECT ANSWERS FOR EACH VARIATION (NEW COMPONENT)               │
│    - Query target model with ORIGINAL question → original_answer    │
│    - Query target model with EACH variation → variation_answers[]   │
│    - Example:                                                        │
│      Original Q: "What is the capital of France?"                   │
│      Original A: "Paris"                                            │
│      Variation 1: "Which city serves as France's capital?"          │
│      Answer 1: "Paris"                                              │
│      Variation 2: "What's the capital city of France?"              │
│      Answer 2: "Paris"                                              │
│      Variation 3: [etc.]                                            │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 4. CALCULATE COHERENCE SCORE (IMPORT FROM selfcheckgpt)             │
│    - Import: from selfcheckgpt import SelfCheckShogenji             │
│    - Adapt predict() interface:                                     │
│      coherence_score = variant.predict(                             │
│          sentences=[original_answer],                               │
│          sampled_passages=variation_answers                         │
│      )                                                               │
│    - Score interpretation:                                          │
│      * High coherence → Consistent answers → Likely true            │
│      * Low coherence → Inconsistent answers → Likely hallucination  │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 5. APPLY ABSTENTION LOGIC (NEW COMPONENT)                           │
│    - Compare coherence_score to threshold (configurable, e.g., 0.5) │
│    - Decision logic:                                                │
│      if coherence_score < threshold:                                │
│          final_answer = "I don't know"                              │
│      else:                                                           │
│          final_answer = original_answer                             │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 6. GRADE ANSWER (REUSE cosmos-coherence SimpleQAGrader)             │
│    - Use existing AI grading: CORRECT / INCORRECT / NOT_ATTEMPTED   │
│    - Map abstentions ("I don't know") → NOT_ATTEMPTED               │
│    - Grade final_answer against ground_truth                        │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 7. AGGREGATE METRICS & COMPARE TO BASELINE                          │
│    - Standard SimpleQA metrics:                                     │
│      * % correct (overall_correct)                                  │
│      * % incorrect                                                  │
│      * % not attempted                                              │
│      * Precision: correct_given_attempted                           │
│    - Run baseline (no coherence) for comparison                     │
│    - Hypothesis: Coherence ↑ precision, ↓ recall                    │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Question Variation Generator (NEW)

**File**: `cosmos-coherence/src/cosmos_coherence/benchmarks/implementations/question_variation_generator.py`

```python
class QuestionVariationGenerator:
    """Generate paraphrased versions of questions for coherence analysis."""

    def __init__(
        self,
        client: OpenAIClient,
        model: str = "gpt-4o-mini",
        temperature_range: List[float] = [0.7, 1.0, 1.3],
        num_variations: int = 5
    ):
        self.client = client
        self.model = model
        self.temperature_range = temperature_range
        self.num_variations = num_variations

    async def generate_variations(self, question: str) -> List[str]:
        """Generate N paraphrased versions of the question."""
        # Distribute variations across temperature range
        # Return list of paraphrased questions
```

**Responsibilities:**
- Generate diverse question paraphrases using temperature-based sampling
- Distribute generations across temperature range for maximum diversity
- Handle API calls with retries and rate limiting
- Validate that variations are meaningful paraphrases (not identical, not off-topic)

### 2. Coherence-Guided Evaluator (NEW)

**File**: `cosmos-coherence/src/cosmos_coherence/benchmarks/implementations/simpleqa_coherence_benchmark.py`

```python
class SimpleQACoherenceBenchmark(SimpleQABenchmark):
    """SimpleQA benchmark with coherence-guided abstention."""

    def __init__(
        self,
        coherence_variant: str = "shogenji",  # or "fitelson", "olsson"
        coherence_threshold: float = 0.5,
        num_variations: int = 5,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Import coherence from selfcheckgpt
        from selfcheckgpt import SelfCheckShogenji, SelfCheckFitelson, SelfCheckOlsson

        self.coherence = self._init_coherence_variant(coherence_variant)
        self.threshold = coherence_threshold
        self.variation_generator = QuestionVariationGenerator(...)

    async def evaluate_with_coherence(self, item: SimpleQAItem) -> Dict:
        """Evaluate question with coherence-guided abstention."""
        # 1. Generate question variations
        variations = await self.variation_generator.generate_variations(item.question)

        # 2. Get answers for original + variations
        original_answer = await self._get_answer(item.question)
        variation_answers = await self._get_answers_async(variations)

        # 3. Calculate coherence
        coherence_score = self.coherence.predict(
            sentences=[original_answer],
            sampled_passages=variation_answers
        )

        # 4. Apply threshold
        if coherence_score < self.threshold:
            final_answer = "I don't know"
        else:
            final_answer = original_answer

        # 5. Grade using existing grader
        result = await self._grader.grade_response(
            question=item.question,
            expert_answer=item.best_answer,
            submission=final_answer
        )

        # Return with metadata
        return {
            "question": item.question,
            "variations": variations,
            "original_answer": original_answer,
            "variation_answers": variation_answers,
            "coherence_score": coherence_score,
            "final_answer": final_answer,
            "grade": result.grade,
            "is_correct": result.is_correct
        }
```

**Responsibilities:**
- Orchestrate full coherence-guided evaluation pipeline
- Manage question variations and answer collection
- Apply coherence scoring via imported selfcheckgpt modules
- Implement threshold-based abstention logic
- Integrate with existing grading infrastructure

### 3. CLI Interface (NEW)

**File**: `cosmos-coherence/src/cosmos_coherence/benchmarks/simpleqa_coherence_cli.py`

```python
@app.command()
def run_with_coherence(
    model: str = typer.Option("gpt-4o-mini", help="Target model to evaluate"),
    coherence_variant: str = typer.Option("shogenji", help="Coherence measure: shogenji|fitelson|olsson"),
    num_variations: int = typer.Option(5, help="Number of question paraphrases"),
    coherence_threshold: float = typer.Option(0.5, help="Threshold for abstention (0.0-1.0)"),
    sample_size: Optional[int] = typer.Option(None, help="Number of questions (default: all 4,326)"),
    output: Optional[Path] = typer.Option(None, help="Path to save results JSON"),
    checkpoint_interval: int = typer.Option(50, help="Save progress every N questions"),
    resume_from: Optional[Path] = typer.Option(None, help="Resume from checkpoint"),
    workers: int = typer.Option(4, help="Parallel workers for API calls"),
    run_baseline: bool = typer.Option(True, help="Also run baseline (no coherence) for comparison"),
):
    """Run SimpleQA evaluation with coherence-guided abstention."""
```

**Responsibilities:**
- Expose all configuration parameters as CLI options
- Initialize coherence benchmark with user settings
- Run evaluation with progress tracking and checkpointing
- Optionally run baseline for comparison
- Display results table and save JSON output

### 4. Imported Coherence Logic (FROM selfcheckgpt)

**No new code needed - direct imports:**

```python
from selfcheckgpt import SelfCheckShogenji, SelfCheckFitelson, SelfCheckOlsson
from selfcheckgpt.modeling_coherence_api import CoherenceAPIClient
from selfcheckgpt.utils_coherence import normalize_coherence_scores
```

**Responsibilities:**
- Probability extraction via OpenAI structured output
- Coherence formula calculations (Shogenji, Fitelson, Olsson)
- Score normalization and caching

## API Call Budget Analysis

### Calls Per Question

**Without Coherence (Baseline):**
- 1 call: Answer generation
- 1 call: AI grading
- **Total: 2 calls/question**

**With Coherence (This Implementation):**
- 1 call: Original answer generation
- N calls: Question variation generation (N = num_variations, default 5)
- N calls: Variation answer generation
- ~11 calls: Coherence calculation (Shogenji/Olsson with N=5 samples: 1 + 2×5 = 11 probability extractions)
- 1 call: AI grading
- **Total: ~18 calls/question with N=5 variations**

### Cost Estimation (gpt-4o-mini)

**Full SimpleQA dataset (4,326 questions) with N=5 variations:**
- API calls: 4,326 × 18 = ~77,868 calls
- Approximate cost (gpt-4o-mini pricing): $5-10 for full evaluation
- **Cache optimization**: Coherence probability extraction uses LRU cache, reducing duplicate calls significantly

**Comparison:**
- Baseline SimpleQA: 4,326 × 2 = ~8,652 calls
- Coherence SimpleQA: ~9x more API calls than baseline
- Tradeoff: Higher cost for potentially better precision (fewer incorrect answers)

## Integration Points

### 1. selfcheckgpt Package (Import)

**What we import:**
- Coherence variant classes: `SelfCheckShogenji`, `SelfCheckFitelson`, `SelfCheckOlsson`
- API client: `CoherenceAPIClient`
- Utilities: `normalize_coherence_scores`, coherence formula functions

**How we use it:**
- Install selfcheckgpt as dependency in cosmos-coherence
- Import coherence modules directly
- Adapt `predict()` interface to question-variation use case

### 2. cosmos-coherence Infrastructure (Extend)

**What we extend:**
- `SimpleQABenchmark`: Create subclass `SimpleQACoherenceBenchmark`
- `simpleqa_cli.py`: Add new command `run-with-coherence`
- `OpenAIClient`: Reuse existing caching and async capabilities

**What we reuse:**
- Dataset loading (HuggingFace integration)
- Grading logic (`SimpleQAGrader`)
- Checkpoint/resume functionality
- Progress tracking (rich library)

### 3. OpenAI API (External)

**API usage:**
- Question variation generation (new)
- Answer generation (existing)
- Coherence probability extraction (imported from selfcheckgpt)
- AI grading (existing)

**Optimization strategies:**
- Cache API responses (both repos have caching)
- Parallel/async API calls (ThreadPoolExecutor, asyncio)
- Rate limiting to avoid 429 errors

## Configuration & Extensibility

### Configurable Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `--coherence-variant` | shogenji | Which coherence measure to use |
| `--num-variations` | 5 | Number of question paraphrases |
| `--coherence-threshold` | 0.5 | Abstention threshold (0.0-1.0) |
| `--temperature-range` | [0.7, 1.0, 1.3] | Temperatures for variation generation |
| `--model` | gpt-4o-mini | Target model to evaluate |
| `--sample-size` | 4326 | Number of questions to evaluate |
| `--workers` | 4 | Parallel workers for API calls |
| `--checkpoint-interval` | 50 | Save progress every N questions |

### Extensibility Points

**Future enhancements:**
1. **Answer selection strategy**: Instead of threshold-based abstention, select answer from highest-coherence question variation
2. **Persistent variation dataset**: Cache generated question variations for reproducibility
3. **Multi-threshold analysis**: Evaluate multiple thresholds in single run to plot precision-recall curve
4. **Coherence ensemble**: Combine all three coherence measures (Shogenji, Fitelson, Olsson) into single score
5. **Adaptive thresholding**: Learn optimal threshold based on validation set

## Testing Strategy

### Unit Tests (pytest)

**New components to test:**
- `QuestionVariationGenerator`:
  - Generates correct number of variations
  - Distributions across temperature range
  - Handles API errors gracefully

- `SimpleQACoherenceBenchmark`:
  - Coherence calculation correct
  - Threshold logic works (abstain when score < threshold)
  - Integration with grading logic

- CLI commands:
  - Parameter parsing
  - Configuration validation
  - Output format correctness

### Integration Tests

**End-to-end test:**
- Run on small sample (10 questions)
- Verify all three coherence variants work
- Confirm baseline comparison runs
- Check checkpoint/resume functionality

### Manual Validation

**Spot-check coherence behavior:**
- Pick questions where model tends to hallucinate
- Verify that coherence scores are indeed low (triggering abstention)
- Pick factual questions
- Verify that coherence scores are high (allowing answer)

## Error Handling & Resilience

### Failure Scenarios

1. **API rate limit exceeded**
   - Solution: Retry with exponential backoff (existing in OpenAIClient)

2. **Question variation generation fails**
   - Solution: Retry with different temperature; if all fail, skip question and log error

3. **Coherence calculation error (numerical instability)**
   - Solution: Coherence modules handle epsilon smoothing; catch exceptions and log

4. **Checkpoint corruption**
   - Solution: Validate checkpoint JSON on load; fall back to restart if invalid

5. **Partial evaluation completion**
   - Solution: Checkpoint every 50 questions; resume from last checkpoint

### Monitoring & Logging

**Progress tracking:**
- Rich progress bar: questions processed, current accuracy, abstention rate
- Periodic checkpoint saves with status update

**Cache statistics:**
- Display cache hit rate at end (cost savings)
- Estimate total API calls vs. cached calls

**Error logging:**
- Log failed questions to separate file
- Continue evaluation despite individual failures

## Success Criteria

### Implementation Completeness

- [ ] Question variation generator working with configurable parameters
- [ ] Coherence calculation integrated from selfcheckgpt
- [ ] Abstention logic implemented with threshold
- [ ] All three coherence variants operational (Shogenji, Fitelson, Olsson)
- [ ] CLI tool functional with all options
- [ ] Checkpoint/resume capability working
- [ ] Baseline comparison runs successfully

### Research Validation

- [ ] Coherence-guided approach reduces % incorrect answers
- [ ] Precision (correct_given_attempted) improves vs. baseline
- [ ] Abstention rate is reasonable (10-30%, not 0% or 100%)
- [ ] At least one coherence variant shows clear benefit
- [ ] Cost-benefit analysis favorable (improvement justifies 9x API calls)

### Code Quality

- [ ] Reuses existing infrastructure (minimal code duplication)
- [ ] Maintains compatibility with both repositories
- [ ] Well-documented configuration options
- [ ] Unit tests cover new components
- [ ] Reproducible results (seeded generation, checkpointing)
