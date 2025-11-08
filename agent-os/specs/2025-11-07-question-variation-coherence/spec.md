# Specification: Question Variation Coherence for SimpleQA

## Executive Summary

### What This Feature Does
This feature implements a novel approach to hallucination detection and prevention by applying coherence theory to question variations themselves. The system generates multiple paraphrased versions of a benchmark question, calculates coherence scores across those question variations (treating them as "sampled passages"), and uses the coherence score to guide answering: for YES/NO questions, coherence directly determines the answer polarity; for other questions, coherence determines whether to answer confidently or abstain ("I don't know").

### Why It's Needed
Traditional coherence-based hallucination detection (as implemented in SelfCheckGPT) requires pre-generated answer samples and evaluates coherence across those samples. This approach proved incompatible with the SimpleQA benchmark for two reasons:
1. SimpleQA provides no pre-generated answer samples (unlike wiki_bio_gpt3_hallucination)
2. SimpleQA answers are short factual responses, making coherence analysis within a single answer impractical

By shifting coherence evaluation from answers to questions, this approach sidesteps both limitations while maintaining the theoretical foundation of coherence-based hallucination detection.

### Key Innovation
Instead of measuring coherence across multiple sampled answers to a fixed question, this feature measures coherence across the question variations themselves. By treating paraphrased questions as "sampled passages" in the coherence framework, we can detect ambiguous or ill-posed questions that are likely to trigger hallucinations. The core hypothesis: **well-formed questions based on factual premises maintain high coherence across paraphrasing, while ambiguous or false-premise questions show low coherence, indicating higher hallucination risk**.

## Background & Motivation

### Previous SimpleQA Coherence Limitations
Earlier attempts to apply coherence variants (Shogenji, Fitelson, Olsson) to SimpleQA benchmark evaluation encountered fundamental blockers:

**Blocker 1: No Pre-Generated Samples**
- Traditional SelfCheckGPT approach requires multiple stochastically sampled passages
- wiki_bio_gpt3_hallucination dataset provides these samples
- SimpleQA dataset provides only: question, ground truth answer, metadata
- No mechanism to generate comparable answer samples exists in SimpleQA infrastructure

**Blocker 2: Short Answer Length**
- Traditional coherence measures evaluate multiple sentences within a passage
- SimpleQA answers are typically 1-3 words ("Paris", "1889", "Albert Einstein")
- Coherence formulas require multiple statements to compare
- Single-word answers cannot demonstrate internal coherence

### How This Approach Sidesteps Limitations
**Question-Based Coherence Strategy:**
1. Generate N paraphrased versions of the original question (e.g., "What is France's capital?" → "Which city serves as the capital of France?")
2. Treat the paraphrased questions themselves as "sampled passages" in the coherence framework
3. Compute coherence score between the original question and its variations
4. For YES/NO questions: Use coherence threshold to directly determine YES or NO answer
5. For other question types: Use coherence threshold to decide whether it's safe to answer

**Coherence Interpretation:**
- High coherence across question variations → question is well-formed and unambiguous → safe to answer
- Low coherence across question variations → question is ambiguous or ill-posed → likely to trigger hallucination

This transforms the problem from "coherence within a single short answer" to "coherence of the question itself across semantically equivalent paraphrases."

### Research Hypothesis
**Primary Hypothesis:** Questions that are well-formed and based on factual premises will maintain high coherence across paraphrasing. Questions that are ambiguous, ill-posed, or based on false premises will show low coherence when paraphrased, as the model struggles to maintain semantic consistency. For YES/NO questions, coherence score can directly predict answer polarity.

**Expected Benefits:**
- Higher precision (correct_given_attempted) at the cost of lower recall (overall_correct)
- Reduced incorrect answer rate through strategic abstention
- Empirical validation of coherence theory's applicability to question-level analysis

## Technical Approach

### High-Level Architecture

**Repository Strategy: Hybrid Approach**
- **Implementation location:** cosmos-coherence repository (existing SimpleQA infrastructure)
- **Coherence logic source:** selfcheckgpt package (import existing coherence variants)
- **Rationale:** Reuse SimpleQA CLI, dataset loading, grading, and checkpointing from cosmos-coherence while importing proven coherence calculation implementations from selfcheckgpt

**Integration Pattern:**
```python
# In cosmos-coherence new module
from selfcheckgpt import SelfCheckShogenji, SelfCheckFitelson, SelfCheckOlsson
from selfcheckgpt.modeling_coherence_api import CoherenceAPIClient
from selfcheckgpt.utils_coherence import normalize_coherence_scores

# Extend existing cosmos-coherence infrastructure
from cosmos_coherence.benchmarks.implementations.simpleqa_benchmark import SimpleQABenchmark
from cosmos_coherence.benchmarks.implementations.simpleqa_grader import SimpleQAGrader
from cosmos_coherence.llm.openai_client import OpenAIClient
```

### 7-Step Data Flow Pipeline

**Step 1: Load SimpleQA Question**
- Source: HuggingFace dataset `basicv8vc/SimpleQA` (4,326 questions)
- Extract: question text, ground truth answer, metadata
- Example: `{"question": "What is the capital of France?", "best_answer": "Paris"}`

**Step 2: Generate Question Variations**
- Use OpenAI API with configurable temperature range (default: [0.7, 1.0, 1.3])
- Prompt template: `"Rephrase the following question while preserving its exact meaning: {question}"`
- Generate N variations (configurable, default: 5)
- Distribute generations across temperature values for diversity
- Validate variations are meaningful paraphrases (not identical, not semantically different)

**Step 3: Calculate Question Coherence Score**
- Import coherence variant from selfcheckgpt (Shogenji, Fitelson, or Olsson)
- Treat question variations as "sampled passages" in coherence framework:
  ```python
  coherence_score = coherence_variant.predict(
      sentences=[original_question],     # Evaluate this question
      sampled_passages=question_variations  # Against these paraphrases
  )
  ```
- Output: Single coherence score per question (aggregated across variations)
- Interpretation: Higher score = more coherent question = well-formed and unambiguous

**Step 4: Determine Answer Strategy Based on Question Type**
- Detect if question is YES/NO type (pattern matching or metadata from dataset)
- For YES/NO questions:
  - Use coherence score to directly determine polarity
  - High coherence → YES, Low coherence → NO (or vice versa, to be determined empirically)
- For other question types:
  - Use coherence as abstention threshold
  - Proceed to Step 5 for answer generation

**Step 5: Generate Answer (If Applicable)**
- Define coherence threshold (configurable parameter, default: 0.5)
- Decision logic for non-YES/NO questions:
  ```python
  if coherence_score < threshold:
      final_answer = "I don't know"  # Abstain due to low question coherence
  else:
      # Query target model with original question
      final_answer = model.generate(original_question)
  ```
- For YES/NO questions, answer was determined in Step 4 based on coherence
- Threshold tuning: Lower threshold = more answers attempted (higher recall, lower precision)
- Threshold tuning: Higher threshold = more abstentions (lower recall, higher precision)

**Step 6: Grade Answer Using SimpleQA Grader**
- Reuse existing `SimpleQAGrader` from cosmos-coherence
- Grade `final_answer` against ground truth using OpenAI AI grading
- Possible grades: CORRECT, INCORRECT, NOT_ATTEMPTED
- Map abstention responses ("I don't know") → NOT_ATTEMPTED
- For YES/NO questions answered via coherence, grade the coherence-determined answer
- Track: is_correct (boolean), grade (string), coherence_score (float), metadata (dict)

**Step 7: Aggregate Metrics & Compare to Baseline**
- Calculate standard SimpleQA metrics:
  - % correct (overall_correct / total_questions)
  - % incorrect (overall_incorrect / total_questions)
  - % not_attempted (overall_not_attempted / total_questions)
  - Precision: correct_given_attempted = correct / (correct + incorrect)
- Run baseline evaluation (same model, no coherence, no abstention) for comparison
- Expected tradeoff: Coherence-guided approach should increase precision while decreasing recall
- For YES/NO questions: Analyze if coherence polarity correlates with correctness

### Question Variation Generation Strategy

**Diversity Mechanisms:**
1. **Temperature Range:** Use multiple temperature values (0.7, 1.0, 1.3) to generate diverse paraphrases
2. **Distribution:** Cycle through temperatures for N variations (e.g., 5 variations → use each temp at least once)
3. **Validation:** Check that variations are not identical to original (minimum edit distance threshold)

**Prompt Engineering:**
- Clear instruction to preserve semantic meaning
- Avoid adding or removing information
- Example prompt: `"Rephrase this question without changing its meaning: {question}"`

**Quality Control:**
- Filter out variations that are identical to original question
- Filter out variations that significantly alter question semantics (optional, via similarity threshold)
- Log rejected variations for debugging

### Coherence Scoring Methodology

**Variant Selection:**
All three coherence measures from selfcheckgpt will be evaluated independently:

1. **SelfCheckShogenji** - Shogenji's ratio-based independence measure
   - Formula: `C2(A,B) = P(A ∧ B) / (P(A) × P(B))`
   - Interpretation: Higher ratio = stronger mutual support
   - Use case: Detects violations of probabilistic independence

2. **SelfCheckFitelson** - Fitelson's confirmation-based support measure
   - Formula: `s(H,E) = P(H|E) - P(H|¬E)`
   - Interpretation: Positive value = evidence supports hypothesis
   - Use case: Asymmetric confirmation relationships
   - Note: Requires conditional probabilities (more API calls)

3. **SelfCheckOlsson** - Glass-Olsson relative overlap measure
   - Formula: `C1(A,B) = P(A ∧ B) / P(A ∨ B)`
   - Interpretation: Higher overlap = stronger agreement
   - Use case: Measures relative agreement between statements

**API Call Budget Per Question:**
- Question variations: N calls (default: 5)
- Coherence probability extraction on questions (Shogenji/Olsson): 1 + 2N calls (default: 11 with N=5)
- Coherence probability extraction on questions (Fitelson): 1 + 3N calls (default: 16 with N=5)
- Answer generation (only if coherence >= threshold AND non-YES/NO question): 0-1 call
- AI grading: 1 call
- **Total (Shogenji/Olsson):** ~12-13 calls per question with N=5
- **Total (Fitelson):** ~17-18 calls per question with N=5
- **Note:** YES/NO questions skip answer generation (answered via coherence), saving 1 API call

**Caching Optimization:**
- Both repositories implement LRU caching for API responses
- Coherence API client caches probability extraction results
- Question variation cache: Similar questions may yield cache hits
- Expected cache hit rate: ~20-30% due to question variation patterns
- Cache reduces effective API calls by approximately 20-25%

### Threshold-Based Abstention Mechanism

**Threshold Interpretation:**
- Coherence score range: [0.0, 1.0] after normalization
- Threshold acts as question quality/clarity cutoff
- Below threshold: Low coherence across question variations → ambiguous/ill-posed question → likely to trigger hallucination → abstain
- Above threshold: High coherence across question variations → well-formed question → safe to answer

**YES/NO Question Special Handling:**
- For YES/NO questions, coherence score directly determines answer polarity (empirically calibrated)
- High coherence → YES (or NO, to be determined through experimentation)
- Low coherence → opposite polarity
- No abstention for YES/NO questions (always provide binary answer based on coherence)

**Threshold Selection Strategy:**
- Default: 0.5 (middle ground)
- Experimentation: Evaluate multiple thresholds (0.3, 0.5, 0.7) to plot precision-recall curve
- Separate threshold tuning for YES/NO polarity mapping
- Configurable via CLI parameter for threshold tuning experiments

**Abstention Response Format:**
- Standard abstention: `"I don't know"`
- Alternative formats (configurable): `"I am not confident in my answer"`, `"Insufficient information"`
- Grader maps any abstention-like response → NOT_ATTEMPTED
- YES/NO questions never abstain (always return binary answer via coherence)

## Functional Requirements

### Command-Line Interface Specification

**Primary Command:**
```bash
cosmos-coherence simpleqa run-with-coherence \
    --model gpt-4o-mini \
    --coherence-variant shogenji \
    --num-variations 5 \
    --coherence-threshold 0.5 \
    --sample-size 100 \
    --output results.json \
    --checkpoint-interval 50 \
    --workers 4
```

**Required Parameters:**
- `--model`: Target model to evaluate (e.g., gpt-4o-mini, gpt-4-turbo)

**Optional Parameters:**
- `--coherence-variant`: Choice of coherence measure (shogenji|fitelson|olsson, default: shogenji)
- `--num-variations`: Number of question paraphrases to generate (default: 5)
- `--coherence-threshold`: Threshold for abstention decision, 0.0-1.0 (default: 0.5)
- `--temperature-range`: Comma-separated temperatures for variation generation (default: "0.7,1.0,1.3")
- `--sample-size`: Number of SimpleQA questions to evaluate (default: all 4,326)
- `--output`: Path to save results JSON (default: auto-generated timestamp)
- `--checkpoint-interval`: Save progress every N questions (default: 50, set 0 to disable)
- `--resume-from`: Path to checkpoint file to resume interrupted evaluation
- `--workers`: Number of parallel workers for API calls (default: 4)
- `--run-baseline`: Also run baseline (no coherence) for comparison (default: true)
- `--abstention-response`: Custom abstention text (default: "I don't know")
- `--grader-model`: Model for AI grading (default: gpt-4o-mini)
- `--verbose`: Show detailed output for debugging (default: false)

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | (required) | Target model to evaluate |
| `coherence_variant` | enum | "shogenji" | Coherence measure: shogenji, fitelson, olsson |
| `num_variations` | int | 5 | Number of question paraphrases per question |
| `coherence_threshold` | float | 0.5 | Abstention threshold (0.0-1.0) |
| `temperature_range` | list[float] | [0.7, 1.0, 1.3] | Temperatures for variation generation |
| `sample_size` | int | 4326 | Number of questions to evaluate (max: 4,326) |
| `output` | path | auto | JSON output file path |
| `checkpoint_interval` | int | 50 | Save progress every N questions (0=disabled) |
| `resume_from` | path | null | Checkpoint file to resume from |
| `workers` | int | 4 | Parallel workers for concurrent API calls |
| `run_baseline` | bool | true | Run baseline (no coherence) for comparison |
| `abstention_response` | string | "I don't know" | Custom text for abstentions |
| `grader_model` | string | "gpt-4o-mini" | Model for AI grading |
| `verbose` | bool | false | Show detailed debug output |

### Input/Output Specifications

**Input: SimpleQA Dataset**
- Source: HuggingFace `basicv8vc/SimpleQA`
- Format: Dataset with fields `{question, best_answer, metadata}`
- Size: 4,326 questions
- Access: Load via `datasets.load_dataset("simpleqa")`

**Output: Results JSON**
```json
{
  "config": {
    "model": "gpt-4o-mini",
    "coherence_variant": "shogenji",
    "num_variations": 5,
    "coherence_threshold": 0.5,
    "sample_size": 100,
    "timestamp": "2025-11-07T10:30:00"
  },
  "metrics": {
    "overall_correct": 0.72,
    "overall_incorrect": 0.18,
    "overall_not_attempted": 0.10,
    "correct_given_attempted": 0.80,
    "total_questions": 100,
    "correct_count": 72,
    "incorrect_count": 18,
    "not_attempted_count": 10
  },
  "baseline_metrics": {
    "overall_correct": 0.75,
    "overall_incorrect": 0.25,
    "overall_not_attempted": 0.00,
    "correct_given_attempted": 0.75
  },
  "coherence_stats": {
    "mean_coherence": 0.62,
    "median_coherence": 0.58,
    "std_coherence": 0.21,
    "abstention_rate": 0.10
  },
  "api_stats": {
    "total_calls": 1800,
    "cached_calls": 540,
    "cache_hit_rate": 0.30,
    "estimated_cost_usd": 0.15
  },
  "per_question_results": [
    {
      "question": "What is the capital of France?",
      "ground_truth": "Paris",
      "question_variations": ["Which city is France's capital?", "What city serves as France's capital?", ...],
      "coherence_score": 0.95,
      "threshold_met": true,
      "question_type": "factual",
      "final_answer": "Paris",
      "grade": "CORRECT",
      "is_correct": true
    },
    ...
  ]
}
```

**Checkpoint Format:**
```json
{
  "config": {...},
  "progress": {
    "questions_completed": 50,
    "last_index": 49,
    "timestamp": "2025-11-07T10:35:00"
  },
  "partial_results": [...],
  "partial_metrics": {...}
}
```

### Evaluation Metrics

**Standard SimpleQA Metrics:**
1. **Overall Correct** = (# correct answers) / (total questions)
   - Target: Maximize while maintaining acceptable abstention rate

2. **Overall Incorrect** = (# incorrect answers) / (total questions)
   - Target: Minimize through coherence-guided abstention

3. **Overall Not Attempted** = (# abstentions) / (total questions)
   - Target: Keep reasonable (10-30%, not 0% or 100%)

4. **Precision (Correct Given Attempted)** = (# correct) / (# correct + # incorrect)
   - Target: Higher than baseline (key success metric)

**Extended Metrics (Optional, for Analysis):**
1. **Coherence Distribution**
   - Mean, median, std dev of coherence scores
   - Histogram of scores by correctness (correct vs incorrect vs abstained)

2. **Coherence-Correctness Correlation**
   - Pearson correlation between coherence score and correctness
   - Expected: Positive correlation (higher coherence → more likely correct)

3. **Abstention Rate Analysis**
   - Abstention rate by coherence score bin
   - Expected: Higher abstention rate at lower coherence scores

4. **Threshold Sensitivity**
   - Precision-recall curve across threshold values
   - Optimal threshold identification

**Comparison Metrics (Coherence vs Baseline):**
- Δ Overall Correct (expected: negative, due to abstentions)
- Δ Overall Incorrect (expected: negative, key goal)
- Δ Precision (expected: positive, key goal)
- Δ Not Attempted (expected: positive, tradeoff)

## Implementation Details

### Repository Location
**Primary Repository:** cosmos-coherence
- Path: `/Users/nathanlubchenco/workspace/cosmos-coherence`
- New modules:
  - `src/cosmos_coherence/benchmarks/simpleqa_coherence_cli.py` (CLI commands)
  - `src/cosmos_coherence/benchmarks/implementations/simpleqa_coherence_benchmark.py` (core logic)
  - `src/cosmos_coherence/benchmarks/implementations/question_variation_generator.py` (variation generation)

**Imported Dependencies:** selfcheckgpt package
- Path: `/Users/nathanlubchenco/workspace/selfcheckgpt`
- Import strategy: Add selfcheckgpt to cosmos-coherence requirements.txt
- Imported modules:
  ```python
  from selfcheckgpt import SelfCheckShogenji, SelfCheckFitelson, SelfCheckOlsson
  from selfcheckgpt.modeling_coherence_api import CoherenceAPIClient
  from selfcheckgpt.utils_coherence import normalize_coherence_scores
  ```

### Key Components and Responsibilities

**Component 1: QuestionVariationGenerator**
- **Location:** `cosmos-coherence/src/cosmos_coherence/benchmarks/implementations/question_variation_generator.py`
- **Responsibilities:**
  - Generate N paraphrased versions of input question
  - Distribute generations across temperature range for diversity
  - Validate variations are meaningful (not identical, not semantically divergent)
  - Handle API errors with retries and fallbacks
  - Log rejected variations for debugging
- **Dependencies:** OpenAIClient (cosmos-coherence), asyncio
- **Interface:**
  ```python
  async def generate_variations(question: str) -> List[str]:
      """Generate N paraphrased questions."""
  ```

**Component 2: SimpleQACoherenceBenchmark**
- **Location:** `cosmos-coherence/src/cosmos_coherence/benchmarks/implementations/simpleqa_coherence_benchmark.py`
- **Responsibilities:**
  - Orchestrate full coherence-guided evaluation pipeline
  - Manage question variation generation and coherence scoring
  - Apply coherence scoring to question variations via imported selfcheckgpt modules
  - Implement threshold-based abstention logic and YES/NO direct answering
  - Integrate with existing SimpleQAGrader for answer evaluation
  - Track per-question metadata (variations, coherence scores, decisions)
- **Dependencies:**
  - SimpleQABenchmark (parent class)
  - QuestionVariationGenerator
  - SelfCheck* coherence variants (from selfcheckgpt)
  - SimpleQAGrader
- **Interface:**
  ```python
  async def evaluate_with_coherence(item: SimpleQAItem) -> Dict:
      """Evaluate question with coherence-guided abstention."""
  ```

**Component 3: SimpleQA Coherence CLI**
- **Location:** `cosmos-coherence/src/cosmos_coherence/benchmarks/simpleqa_coherence_cli.py`
- **Responsibilities:**
  - Expose CLI command `run-with-coherence` with all configuration parameters
  - Initialize SimpleQACoherenceBenchmark with user settings
  - Run evaluation with progress tracking (rich library)
  - Manage checkpointing and resume functionality
  - Optionally run baseline for comparison
  - Display results table and save JSON output
  - Show API usage statistics and cost estimates
- **Dependencies:** Typer, Rich, SimpleQACoherenceBenchmark
- **Interface:** CLI command with options (see Command-Line Interface Specification)

**Component 4: Imported Coherence Logic (from selfcheckgpt)**
- **Location:** selfcheckgpt package (no new code)
- **Modules:**
  - `modeling_coherence.py`: SelfCheckShogenji, SelfCheckFitelson, SelfCheckOlsson
  - `modeling_coherence_api.py`: CoherenceAPIClient with structured output
  - `utils_coherence.py`: Coherence formulas and normalization utilities
- **Responsibilities:**
  - Extract probabilities via OpenAI API with structured output
  - Calculate coherence scores using formal epistemology formulas
  - Normalize scores to [0, 1] range
  - Cache API responses for cost optimization

### Code Reuse Opportunities

**From selfcheckgpt:**
1. **Coherence Variants** (`modeling_coherence.py`)
   - Directly import SelfCheckShogenji, SelfCheckFitelson, SelfCheckOlsson classes
   - Reuse `predict()` interface with adaptation for question-variation use case
   - Pattern: `scores = variant.predict(sentences=[original_question], sampled_passages=question_variations)`

2. **Coherence API Client** (`modeling_coherence_api.py`)
   - Import CoherenceAPIClient for probability extraction
   - Reuse LRU caching mechanism (10,000 entry cache)
   - Reuse structured output schema for reliable probability parsing
   - Reuse cache statistics tracking

3. **Evaluation Script Pattern** (`scripts/evaluate.py`)
   - Follow parallel processing pattern using ThreadPoolExecutor
   - Adapt evaluate_single_passage() function for question-level evaluation
   - Reuse progress tracking with tqdm
   - Reuse metrics calculation (AUC-PR, Pearson correlation)

**From cosmos-coherence:**
1. **SimpleQA CLI Infrastructure** (`simpleqa_cli.py`)
   - Extend existing CLI structure with new `run-with-coherence` command
   - Reuse Typer app pattern and Rich console output
   - Reuse checkpoint/resume functionality pattern
   - Reuse configuration loading from YAML

2. **SimpleQA Benchmark** (`simpleqa_benchmark.py`)
   - Extend SimpleQABenchmark class (create SimpleQACoherenceBenchmark subclass)
   - Reuse HuggingFace dataset loading
   - Reuse get_prompt() method for question formatting
   - Reuse async evaluation patterns

3. **SimpleQA Grader** (`simpleqa_grader.py`)
   - Directly reuse existing SimpleQAGrader for answer evaluation
   - Reuse AI grading logic (CORRECT/INCORRECT/NOT_ATTEMPTED)
   - Reuse grading prompt templates
   - Map abstentions to NOT_ATTEMPTED grade

4. **OpenAI Client** (`llm/openai_client.py`)
   - Reuse for question variation generation
   - Reuse for answer collection
   - Reuse caching, retry logic, rate limiting
   - Reuse async/await support for concurrent API calls

### API Call Budget and Cost Considerations

**Calls Per Question (N=5 variations, Shogenji variant):**
- Question variation generation: 5 calls
- Coherence probability extraction on questions: 11 calls (1 + 2×5)
- Answer generation (conditional): 0-1 call (only if coherence >= threshold AND non-YES/NO question)
- AI grading: 1 call
- **Total: 17-18 calls per question** (17 for YES/NO, 18 for others that pass threshold)

**Full Dataset Cost Estimate (4,326 questions):**
- Average calls per question: ~17.5 (assuming mix of YES/NO and other question types)
- Total API calls: 4,326 × 17.5 = ~75,705 calls
- With 20-25% cache hit rate: ~57,000-60,000 effective calls
- Using gpt-4o-mini (estimated $0.0001 per call): ~$5.70-6.00 for full evaluation
- Using gpt-4-turbo: ~$40-45 for full evaluation

**Optimization Strategies:**
1. **LRU Caching:** Coherence API client caches probability extraction (20-30% hit rate expected)
2. **Parallel Processing:** Use ThreadPoolExecutor with configurable workers (default: 4)
3. **Checkpointing:** Save progress every 50 questions to avoid re-processing on failures
4. **Model Selection:** Default to gpt-4o-mini for cost efficiency (~10x cheaper than gpt-4)
5. **Sample Size Option:** Allow evaluation on subset for rapid experimentation
6. **YES/NO Optimization:** YES/NO questions answered via coherence skip answer generation call

**Cost Comparison:**
- Baseline SimpleQA (no coherence): 4,326 × 2 = ~8,652 calls (~$0.87 with gpt-4o-mini)
- Coherence SimpleQA: ~57,000-60,000 calls (~$5.70-6.00 with gpt-4o-mini)
- **Cost multiplier: ~6.5-7x for coherence-guided approach**
- Justification: Reduced incorrect answers (improved precision) justifies higher cost for production use cases where hallucinations are costly
- Note: Significantly more efficient than answer-based coherence (would require ~23 calls/question)

## Configuration & Extensibility

### Configurable Parameters

**Core Coherence Parameters:**
- `coherence_variant`: Select coherence measure (shogenji/fitelson/olsson)
- `coherence_threshold`: Abstention decision threshold (0.0-1.0)
- Extensibility: Easy to add new coherence variants by importing additional modules from selfcheckgpt

**Question Variation Parameters:**
- `num_variations`: Number of paraphrases per question (5/10/20)
- `temperature_range`: List of temperatures for diversity ([0.7, 1.0, 1.3])
- Extensibility: Could add alternative generation strategies (back-translation, synonym replacement)

**Evaluation Parameters:**
- `model`: Target model to evaluate (any OpenAI model)
- `sample_size`: Subset evaluation for experimentation
- `workers`: Parallel processing scale (1-16)
- Extensibility: Could extend to non-OpenAI models via adapter pattern

**Infrastructure Parameters:**
- `checkpoint_interval`: Granularity of progress saving (10-100)
- `output`: Custom result file paths and formats
- Extensibility: Could add CSV, database, or cloud storage outputs

### Future Extensions

**Extension 1: Alternative Answer Guidance Mechanisms**
- Current: Use overall coherence score for threshold-based abstention
- Future Option A: Select best question variation to answer
  - Hypothesis: Answering the highest-coherence variation (instead of original) may yield more accurate response
  - Implementation: Track coherence per variation, answer the variation with max(coherence_score) instead of original
  - Benefit: Potentially better answer quality without requiring abstention
- Future Option B: Answer-based coherence (generate answers to all variations)
  - Implementation: Generate answer for each variation, compute coherence across those answers
  - Benefit: More direct answer quality assessment (but ~6 additional API calls per question)

**Extension 2: Persistent Variation Dataset**
- Current: Generate variations on-the-fly (ephemeral)
- Future: Cache generated variations for reproducibility
  - Benefit: Exact reproducibility of evaluations
  - Benefit: Eliminate variation generation cost on repeated runs
  - Implementation: Save to HuggingFace dataset or local JSON

**Extension 3: Multi-Threshold Analysis**
- Current: Single threshold evaluation
- Future: Evaluate multiple thresholds in single run
  - Benefit: Generate precision-recall curve automatically
  - Benefit: Identify optimal threshold empirically
  - Implementation: Run evaluation once, apply multiple thresholds to cached results

**Extension 4: Coherence Ensemble**
- Current: Single coherence variant per run
- Future: Combine all three variants (Shogenji, Fitelson, Olsson) into ensemble score
  - Hypothesis: Ensemble may be more robust than individual measures
  - Implementation: Weighted average or voting mechanism
  - Benefit: Potentially higher accuracy than best single variant

**Extension 5: Adaptive Thresholding**
- Current: Fixed threshold across all questions
- Future: Learn optimal threshold from validation set
  - Implementation: Split dataset into train/val, optimize threshold on val set
  - Benefit: Data-driven threshold selection
  - Benefit: Could vary threshold by question type or difficulty

**Extension 6: Alternative Benchmarks**
- Current: SimpleQA only
- Future: Extend to other factual QA benchmarks (TruthfulQA, NaturalQuestions)
  - Benefit: Validate approach generalizability
  - Implementation: Create adapter for different dataset formats

## Evaluation & Success Criteria

### Standard SimpleQA Metrics

**Primary Metrics (Key Success Indicators):**

1. **Overall Correct Rate**
   - Formula: correct_count / total_questions
   - Baseline expectation: ~75% (model-dependent)
   - Coherence expectation: ~70% (lower due to abstentions)
   - Success criterion: Not significantly lower than baseline (within 10% relative decrease)

2. **Overall Incorrect Rate**
   - Formula: incorrect_count / total_questions
   - Baseline expectation: ~25% (model-dependent)
   - Coherence expectation: ~15-18% (key goal: reduced incorrect answers)
   - **Success criterion: Reduce incorrect rate by at least 20% relative to baseline**

3. **Overall Not Attempted Rate**
   - Formula: not_attempted_count / total_questions
   - Baseline expectation: 0% (no abstention mechanism)
   - Coherence expectation: 10-30% (strategic abstention)
   - Success criterion: Abstention rate between 10-30% (not too conservative, not too aggressive)

4. **Precision (Correct Given Attempted)**
   - Formula: correct_count / (correct_count + incorrect_count)
   - Baseline expectation: ~75% (same as overall correct when no abstentions)
   - Coherence expectation: ~80-85%
   - **Success criterion: Improve precision by at least 5 percentage points vs baseline**

**Extended Metrics (For Analysis):**

1. **Coherence Score Distribution**
   - Mean coherence score by outcome (correct/incorrect/abstained)
   - Expected pattern: correct > abstained > incorrect (mean coherence scores)
   - Validation: Coherence scores should correlate with correctness

2. **Coherence-Correctness Correlation**
   - Pearson correlation coefficient between coherence score and correctness
   - Expected: Positive correlation (r > 0.3)
   - Validation: Higher coherence scores should predict higher correctness probability

3. **Abstention Rate by Coherence Bin**
   - Abstention rate for coherence scores [0-0.3], [0.3-0.5], [0.5-0.7], [0.7-1.0]
   - Expected pattern: Higher abstention at lower coherence bins
   - Validation: Threshold mechanism working as intended

### Comparison Methodology

**Baseline Run:**
- Same model, same questions, no coherence, no abstention
- Standard SimpleQA evaluation following OpenAI reference implementation
- Collect: overall_correct, overall_incorrect (not_attempted = 0)
- Calculate: precision (same as overall_correct when no abstentions)

**Coherence Run:**
- Same model, same questions, coherence variant X, threshold Y
- Coherence-guided abstention enabled
- Collect: overall_correct, overall_incorrect, overall_not_attempted, coherence scores
- Calculate: precision (correct_given_attempted)

**Comparison Analysis:**
```
Metric                    | Baseline | Coherence | Delta   | Interpretation
--------------------------|----------|-----------|---------|-------------------
Overall Correct (%)       | 75.0     | 70.0      | -5.0    | Expected tradeoff
Overall Incorrect (%)     | 25.0     | 18.0      | -7.0    | SUCCESS: Fewer errors
Overall Not Attempted (%) | 0.0      | 12.0      | +12.0   | Expected abstention
Precision (%)             | 75.0     | 79.5      | +4.5    | SUCCESS: Higher precision
```

**Success Interpretation:**
- **Primary goal achieved** if incorrect rate decreases significantly (>20% relative reduction)
- **Acceptable tradeoff** if precision improves despite lower overall correct rate
- **Practical value** demonstrated if precision gain justifies cost increase

### What Constitutes Success

**Implementation Success Criteria:**
- [ ] CLI tool runs successfully on full SimpleQA dataset (4,326 questions)
- [ ] All three coherence variants operational (Shogenji, Fitelson, Olsson)
- [ ] Checkpointing and resume functionality working reliably
- [ ] Results comparable to baseline (same evaluation framework, same grader)
- [ ] API cost estimates accurate (within 20% of actual cost)
- [ ] Parallel processing functional (4+ workers without errors)

**Research Success Criteria:**
- [ ] **Primary:** Coherence-guided approach reduces incorrect answer rate by ≥20% vs baseline
- [ ] **Primary:** Precision (correct_given_attempted) improves by ≥5 percentage points vs baseline
- [ ] Abstention rate is reasonable (10-30%, not 0% or 100%)
- [ ] Coherence scores positively correlate with correctness (Pearson r > 0.3)
- [ ] At least one coherence variant shows clear benefit (may not be all three)

**Practical Success Criteria:**
- [ ] Cost-benefit analysis favorable: Precision improvement justifies ~8x API cost increase
- [ ] Reproducible results via checkpointing and seeded generation
- [ ] Clear documentation enables future researchers to replicate and extend

**Failure Scenarios (What Would Indicate This Approach Doesn't Work):**
- Coherence scores show no correlation with correctness (r < 0.1)
- Abstention rate near 0% (threshold never triggered) or near 100% (abstaining on everything)
- No improvement in precision despite abstentions (incorrect rate not reduced)
- All three coherence variants perform similarly poorly

## Out of Scope

### Explicitly Excluded from This Specification

1. **Dataset Creation**
   - **Not included:** Creating persistent cached dataset of question variations
   - **Rationale:** Defer until approach proves successful; on-the-fly generation sufficient for initial validation
   - **Future consideration:** If successful, create cached dataset for reproducibility and efficiency

2. **Extended Metrics Beyond Debugging Needs**
   - **Not included:** Comprehensive per-category coherence analysis
   - **Not included:** Detailed abstention rate breakdowns by question type
   - **Not included:** Confidence calibration curves
   - **Rationale:** Focus on core metrics; add extended metrics only if needed for understanding results
   - **Future consideration:** Add if debugging reveals need for more granular analysis

3. **Alternative Benchmarks**
   - **Not included:** TruthfulQA, NaturalQuestions, or other factual QA benchmarks
   - **Rationale:** Validate approach on SimpleQA first before generalizing
   - **Future consideration:** Extend to other benchmarks after SimpleQA success

4. **Alternative Answer Selection Strategies**
   - **Not included:** Selecting answer from highest-coherence question variation
   - **Not included:** Ensemble answer aggregation across variations
   - **Rationale:** Start with simpler threshold-based abstention; defer more complex strategies
   - **Future consideration:** Implement if threshold approach proves successful

5. **Multi-Model Comparison**
   - **Not included:** Simultaneous evaluation of multiple target models
   - **Rationale:** Can run separately for different models; no need for parallel multi-model support
   - **Future consideration:** Batch processing across models for efficiency

6. **Non-OpenAI Model Support**
   - **Not included:** Adapters for Anthropic, Groq, or local models
   - **Rationale:** SimpleQA baseline uses OpenAI; maintain consistency for fair comparison
   - **Future consideration:** Extend to other APIs if approach proves valuable

7. **Real-Time Inference API**
   - **Not included:** Production API for real-time coherence-guided answering
   - **Rationale:** This is research/evaluation tool, not production system
   - **Future consideration:** Separate project if research validates approach

8. **Advanced Variation Generation**
   - **Not included:** Back-translation, synonym replacement, or linguistic transformations
   - **Rationale:** LLM-based paraphrasing sufficient for initial validation
   - **Future consideration:** Explore if quality of variations becomes limiting factor

9. **Upstream Contributions**
   - **Not included:** Contributing coherence variants back to selfcheckgpt repository
   - **Not included:** Publishing research findings or papers
   - **Rationale:** Defer until experimental results validate approach
   - **Future consideration:** Share with community if results are compelling

10. **Question Filtering or Exclusion**
    - **Not included:** Excluding certain question types or categories
    - **Rationale:** Must evaluate on full dataset to maintain benchmark validity
    - **Future consideration:** Per-category analysis as extended metric if needed

## Risks & Mitigation

### Risk 1: API Cost Overruns
**Description:** Full evaluation costs significantly more than estimated (~8x baseline cost)
**Probability:** Low (cost estimation based on known API call counts)
**Impact:** Medium (budget constraints may limit experimentation)
**Mitigation Strategies:**
- Implement accurate cost estimation before running full evaluation
- Start with small sample sizes (100-500 questions) for initial experiments
- Use gpt-4o-mini as default model (10x cheaper than gpt-4)
- Leverage caching aggressively (30-50% reduction expected)
- Provide cost estimates in CLI before proceeding with evaluation
- Implement early stopping if cost exceeds threshold

### Risk 2: Poor Variation Quality
**Description:** Generated question variations are too similar (low diversity) or too different (semantic drift)
**Probability:** Medium (paraphrasing quality depends on model and prompt)
**Impact:** High (degrades coherence signal quality)
**Mitigation Strategies:**
- Use multiple temperatures (0.7, 1.0, 1.3) to ensure diversity
- Implement validation to filter identical or near-identical variations
- Log rejected variations for manual inspection and prompt tuning
- Add similarity threshold checks (edit distance, semantic similarity)
- Iterate on paraphrase prompt based on quality assessment
- Provide option to manually inspect variations in verbose mode

### Risk 3: No Coherence-Correctness Correlation
**Description:** Coherence scores fail to predict answer correctness (null hypothesis: approach doesn't work)
**Probability:** Medium (novel approach, untested on this use case)
**Impact:** High (invalidates entire research hypothesis)
**Mitigation Strategies:**
- Calculate correlation metrics early (first 100 questions) to detect failure quickly
- Implement multiple coherence variants (Shogenji, Fitelson, Olsson) to increase success probability
- Analyze failure cases: Are variations too similar? Too different? Incoherent prompts?
- Provide detailed per-question debugging output for error analysis
- Consider fallback: If correlation weak, analyze whether *inconsistency* (not coherence) predicts incorrectness
- Document negative results for research value even if approach fails

### Risk 4: Threshold Tuning Challenges
**Description:** No single threshold works well; optimal threshold varies significantly by question type
**Probability:** Medium (threshold sensitivity is unknown)
**Impact:** Medium (reduces practical applicability)
**Mitigation Strategies:**
- Make threshold highly configurable (easy to experiment)
- Implement multi-threshold evaluation in single run (generate precision-recall curve)
- Analyze threshold sensitivity by question category or difficulty
- Consider adaptive thresholding as future extension if fixed threshold insufficient
- Document optimal threshold range based on precision-recall tradeoff analysis

### Risk 5: Checkpoint Corruption or Resume Failures
**Description:** Long-running evaluations fail partway through; checkpoints fail to load
**Probability:** Low (cosmos-coherence already has working checkpoint system)
**Impact:** Medium (wasted API costs, lost progress)
**Mitigation Strategies:**
- Reuse battle-tested checkpoint implementation from cosmos-coherence
- Validate checkpoint JSON on load with graceful fallback
- Save checkpoints frequently (every 50 questions by default)
- Implement atomic writes to prevent corruption during save
- Log checkpoint save/load operations for debugging
- Test resume functionality explicitly before full evaluation

### Risk 6: Parallelization Bugs or Rate Limits
**Description:** Concurrent API calls cause rate limiting errors or data corruption
**Probability:** Low (both repos have working parallel processing)
**Impact:** Medium (evaluation failures, wasted cost)
**Mitigation Strategies:**
- Reuse proven async/parallel patterns from existing codebases
- Implement exponential backoff retry logic (existing in OpenAIClient)
- Configurable worker count (default: 4, conservative)
- Respect rate limits with token bucket or semaphore
- Add error logging for failed API calls without halting evaluation
- Test with small worker counts initially before scaling up

### Risk 7: Memory Exhaustion on Large Evaluations
**Description:** Storing per-question results for all 4,326 questions exhausts memory
**Probability:** Low (results are lightweight JSON objects)
**Impact:** Low (evaluation crashes)
**Mitigation Strategies:**
- Stream results to disk incrementally (don't accumulate in memory)
- Checkpoint system already writes partial results periodically
- Monitor memory usage during development
- Provide option to save minimal results (metrics only, not per-question details)

## References

### Planning Documents
- **Requirements:** `/Users/nathanlubchenco/workspace/selfcheckgpt/agent-os/specs/2025-11-07-question-variation-coherence/planning/requirements.md`
- **Architecture Overview:** `/Users/nathanlubchenco/workspace/selfcheckgpt/agent-os/specs/2025-11-07-question-variation-coherence/planning/architecture-overview.md`
- **Raw Idea:** `/Users/nathanlubchenco/workspace/selfcheckgpt/agent-os/specs/2025-11-07-question-variation-coherence/planning/raw-idea.md`

### Related Previous Work
- **SimpleQA Compatibility Analysis:** Initial investigation that identified coherence incompatibility with SimpleQA benchmark (blocker: no pre-generated samples, short answers)
- **Coherence Variants Implementation:** `/Users/nathanlubchenco/workspace/selfcheckgpt/agent-os/specs/2025-11-02_coherence-variants/` - Original coherence theory implementation for wiki_bio_gpt3_hallucination benchmark

### Codebase References

**selfcheckgpt repository:**
- Coherence variants: `/Users/nathanlubchenco/workspace/selfcheckgpt/selfcheckgpt/modeling_coherence.py`
- Coherence API client: `/Users/nathanlubchenco/workspace/selfcheckgpt/selfcheckgpt/modeling_coherence_api.py`
- Coherence utilities: `/Users/nathanlubchenco/workspace/selfcheckgpt/selfcheckgpt/utils_coherence.py`
- Evaluation script pattern: `/Users/nathanlubchenco/workspace/selfcheckgpt/scripts/evaluate.py`

**cosmos-coherence repository:**
- SimpleQA CLI: `/Users/nathanlubchenco/workspace/cosmos-coherence/src/cosmos_coherence/benchmarks/simpleqa_cli.py`
- SimpleQA benchmark: `/Users/nathanlubchenco/workspace/cosmos-coherence/src/cosmos_coherence/benchmarks/implementations/simpleqa_benchmark.py`
- SimpleQA grader: `/Users/nathanlubchenco/workspace/cosmos-coherence/src/cosmos_coherence/benchmarks/implementations/simpleqa_grader.py`
- OpenAI client: `/Users/nathanlubchenco/workspace/cosmos-coherence/src/cosmos_coherence/llm/openai_client.py`

### Theoretical Background
- **Coherence Theory Reference:** `/Users/nathanlubchenco/workspace/selfcheckgpt/agent-os/specs/2025-11-02_coherence-variants/planning/coherence-theory-reference.md`
- Shogenji, T. (1999). "Is Coherence Truth-conducive?", Analysis, 59: 338-345
- Fitelson, B. (2003). "A Probabilistic Measure of Coherence", Analysis, 63: 194-199
- Olsson, E. J. (2002). "What is the Problem of Coherence and Truth?", The Journal of Philosophy, 99: 246-272

### External Resources
- SimpleQA Dataset: HuggingFace `basicv8vc/SimpleQA` (4,326 questions)
- OpenAI API Documentation: https://platform.openai.com/docs/api-reference
- SimpleQA Paper: OpenAI (2024) - factual accuracy benchmark for LLMs
