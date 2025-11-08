# Spec Requirements: Question Variation Coherence for SimpleQA

## Initial Description

Explore whether coherence theory can guide hallucination prevention through question variation rather than answer variation. The core hypothesis is: if multiple paraphrased versions of a question yield incoherent/contradictory answers, this signals potential hallucination, and the model should abstain from answering ("I don't know").

This represents a fundamentally different application of coherence theory compared to previous work:
- **Previous approach (incompatible)**: Applied coherence to sampled ANSWERS for a fixed question
- **This approach (novel)**: Apply coherence to ANSWERS for VARIATIONS of the question

The approach leverages the insight that true factual knowledge should be robust to question paraphrasing, while hallucinated responses will show inconsistency when the question is rephrased.

## Requirements Discussion

### First Round Questions

**Q1: Repository Selection - Where should this be implemented?**

**Answer:** Initial intuition suggests cosmos-coherence might be easier, but seeking recommendation.

**Recommendation Provided:** Implement in cosmos-coherence but import coherence modules from selfcheckgpt. This hybrid approach leverages SimpleQA infrastructure from cosmos-coherence while reusing the coherence theory implementation from selfcheckgpt. Clean separation of concerns.

**Decision:** Implement in cosmos-coherence repository, importing coherence calculation logic from selfcheckgpt package.

---

**Q2: Answer Guidance Mechanism - How should coherence scores guide answering behavior?**

**Answer:** Start with the **threshold approach**: use coherence score threshold to decide whether to answer or abstain.

**Future consideration:** Select the "best" question variation (highest coherence) and use its answer, but note this for potential future exploration rather than initial implementation.

**Decision:** Implement threshold-based abstention logic. If coherence score falls below threshold, output "I don't know" or similar abstention response.

---

**Q3: Question Variation Generation - How to generate diverse question paraphrases?**

**Answer:** Use a **range of temperatures** (not just a single value) to generate diverse paraphrases. Make the model **configurable**, but start with **gpt-4o-mini** as the default model for cost-efficiency.

**Inspiration:** wiki_bio hallucination benchmark uses temperature=1.0 for diversity in sampled passages. Apply similar principle to question generation.

**Decision:**
- Temperature range: Configurable (e.g., 0.7, 1.0, 1.3 for diversity)
- Model: Configurable with gpt-4o-mini as default
- Generation approach: Prompt-based paraphrasing via OpenAI API

---

**Q4: Workflow & Dataset Creation - Generate on-the-fly or create persistent dataset?**

**Answer:** Generate question variations **on-the-fly** for now. If the approach proves worthwhile, extend to creating a persistent dataset later for reproducibility and efficiency.

**Decision:**
- Initial implementation: Generate variations during evaluation runtime
- Future enhancement: If successful, create cached/persistent dataset of question variations

---

**Q5: Evaluation Metrics - What metrics should be tracked?**

**Answer:** Focus on **(a) standard SimpleQA metrics** comparing coherence-guided approach vs. baseline:
- % correct
- % incorrect
- % not attempted

If **(b) extended metrics** prove useful for debugging and understanding what is or isn't working, add them as needed:
- Coherence score distributions
- Correlation between coherence and correctness
- Abstention rate analysis

**Decision:**
- Primary metrics: Standard SimpleQA metrics (correct, incorrect, not_attempted counts and percentages)
- Secondary metrics: Coherence distributions, abstention analysis (add if helpful during development)

---

**Q6: Number of Question Variations - How many paraphrases per question?**

**Answer:** Make it **configurable** with reasonable defaults like 5, 10, or 20 variations.

**Reasoning:** Similar to wiki_bio which uses ~6 sampled passages. Start with 5 as default for cost-efficiency, allow scaling up for analysis.

**Decision:** Configurable parameter (--num-variations), default=5

---

**Q7: Coherence Measures - Which coherence variant(s) to use?**

**Answer:** Use **all three coherence measures** (Shogenji, Fitelson, Olsson) to compare which works best for question-based coherence detection.

**Decision:** Evaluate all three variants independently:
- SelfCheckShogenji
- SelfCheckFitelson
- SelfCheckOlsson

---

**Q8: Target Benchmark Dataset - Which dataset to evaluate on?**

**Answer:** **SimpleQA benchmark** (4,326 questions) from HuggingFace dataset `basicv8vc/SimpleQA`.

**Reasoning:** Short factual questions where correctness is well-defined; matches the use case for coherence-based abstention.

**Decision:** Primary dataset is SimpleQA with 4,326 questions

---

**Q9: Output Format - How should results be presented?**

**Answer:** **Command-line script** similar to `scripts/evaluate.py` in selfcheckgpt, but for SimpleQA instead of wiki_bio_gpt3_hallucination.

**Decision:** CLI tool with progress tracking, checkpointing, and JSON output for results

---

**Q10: Scope Boundaries - What should NOT be included?**

**Answer:** Must answer everything the original benchmark does - no exclusions of question types or categories.

**Decision:** No filtering or exclusion of questions; evaluate on full SimpleQA dataset

### Existing Code to Reference

**Similar Features Identified:**

**From selfcheckgpt repository:**
- Path: `/Users/nathanlubchenco/workspace/selfcheckgpt/selfcheckgpt/modeling_coherence.py`
  - All three coherence variant implementations (Shogenji, Fitelson, Olsson)
  - Common `predict()` interface: `predict(sentences, sampled_passages, verbose=False)`

- Path: `/Users/nathanlubchenco/workspace/selfcheckgpt/selfcheckgpt/modeling_coherence_api.py`
  - CoherenceAPIClient with OpenAI integration
  - Structured output for probability extraction
  - LRU caching for cost optimization

- Path: `/Users/nathanlubchenco/workspace/selfcheckgpt/selfcheckgpt/utils_coherence.py`
  - Coherence formula implementations
  - Score normalization utilities

- Path: `/Users/nathanlubchenco/workspace/selfcheckgpt/scripts/evaluate.py`
  - CLI evaluation script pattern for wiki_bio dataset
  - Parallel processing for API-based methods
  - Progress tracking, checkpointing, results aggregation

**From cosmos-coherence repository:**
- Path: `/Users/nathanlubchenco/workspace/cosmos-coherence/src/cosmos_coherence/benchmarks/simpleqa_cli.py`
  - Full CLI implementation for SimpleQA benchmark
  - Checkpointing and resume functionality
  - Progress tracking with rich library
  - Cache management for OpenAI API calls

- Path: `/Users/nathanlubchenco/workspace/cosmos-coherence/src/cosmos_coherence/benchmarks/implementations/simpleqa_benchmark.py`
  - SimpleQABenchmark class with HuggingFace dataset loading
  - AI grading integration via SimpleQAGrader
  - Evaluation result structures

- Path: `/Users/nathanlubchenco/workspace/cosmos-coherence/src/cosmos_coherence/benchmarks/implementations/simpleqa_grader.py`
  - OpenAI-based grading (CORRECT/INCORRECT/NOT_ATTEMPTED)
  - Metrics calculation aligned with SimpleQA paper

- Path: `/Users/nathanlubchenco/workspace/cosmos-coherence/src/cosmos_coherence/llm/openai_client.py`
  - OpenAI client with caching, retry logic, rate limiting
  - Async/await support for concurrent API calls

**Components to potentially reuse:**
- Coherence calculation logic (direct import from selfcheckgpt)
- SimpleQA CLI infrastructure (extend existing cosmos-coherence CLI)
- OpenAI client with caching (cosmos-coherence implementation)
- Grading logic (adapt existing SimpleQAGrader)
- Dataset loading (existing SimpleQABenchmark HuggingFace integration)

**Backend logic to reference:**
- Parallel/async evaluation pattern from both repositories
- Checkpoint/resume mechanism from cosmos-coherence
- Cache statistics tracking from both repositories
- Progress bar implementation using rich library

### Follow-up Questions

**Follow-up 1: Novel Approach Confirmation**

**Question:** This is fundamentally different from previous SimpleQA coherence work (which was deemed incompatible). Previous work tried to apply coherence to ANSWERS; this applies coherence to QUESTIONS. Can you confirm this novel approach sidesteps the previous blockers (no pre-generated samples, short answers)?

**Answer:** Confirmed. This approach is novel and sidesteps previous limitations:
- Previous blocker: SimpleQA has no pre-generated answer samples (unlike wiki_bio)
- New approach: Generate question variations instead, then compare answers to those variations
- Previous blocker: Answers too short for meaningful coherence analysis
- New approach: Coherence is measured across multiple answers (one per question variation), not within a single short answer

---

**Follow-up 2: Coherence Interpretation Direction**

**Question:** For this use case, should we interpret coherence scores as: low coherence = more likely hallucination (true things are robust to paraphrasing)?

**Answer:** Confirmed. Interpretation is:
- **High coherence across question variations** = Answers are consistent despite rephrasing = Likely true knowledge = Safe to answer
- **Low coherence across question variations** = Answers contradict/vary despite similar questions = Likely hallucination/uncertainty = Abstain from answering

This inverts the typical hallucination score: coherence measures confidence, not hallucination probability.

## Visual Assets

### Files Provided:
No visual assets provided.

### Visual Insights:
N/A - No design mockups or wireframes required for this research implementation.

## Requirements Summary

### Functional Requirements

**Core Functionality:**
1. **Question Variation Generation**
   - Generate N paraphrased versions of each SimpleQA question
   - Use configurable temperature range for diversity (e.g., 0.7, 1.0, 1.3)
   - Use configurable LLM model (default: gpt-4o-mini)
   - Generated on-the-fly during evaluation

2. **Answer Collection**
   - For each question variation, query the target model for an answer
   - Collect answers as a set of "sampled passages" (reusing coherence framework terminology)
   - Handle same model for questions and answers (or allow different models)

3. **Coherence Scoring**
   - Import coherence variants from selfcheckgpt: SelfCheckShogenji, SelfCheckFitelson, SelfCheckOlsson
   - Adapt coherence.predict() interface: instead of evaluating sentences from a response, evaluate the ORIGINAL ANSWER against answers to question variations
   - Calculate coherence score indicating consistency across question variations

4. **Abstention Logic**
   - Define coherence threshold (configurable parameter)
   - If coherence score < threshold: Output abstention response ("I don't know" or similar)
   - If coherence score >= threshold: Output the original answer

5. **Evaluation & Grading**
   - Use existing SimpleQAGrader for answer correctness (CORRECT/INCORRECT/NOT_ATTEMPTED)
   - Map abstentions to NOT_ATTEMPTED grade
   - Calculate standard SimpleQA metrics: % correct, % incorrect, % not attempted

6. **Baseline Comparison**
   - Run same evaluation WITHOUT coherence-guided abstention (standard SimpleQA baseline)
   - Compare metrics: Does coherence improve precision (correct given attempted) at cost of recall (overall correct)?

**User Actions Enabled:**
- Run coherence-guided SimpleQA evaluation via CLI command
- Configure coherence variant (shogenji/fitelson/olsson)
- Configure number of question variations (5/10/20)
- Configure coherence threshold for abstention
- Configure generation model and temperature range
- Resume from checkpoint if interrupted
- Export results to JSON for analysis
- Compare results against baseline (no coherence)

**Data to be Managed:**
- SimpleQA dataset (4,326 questions) loaded from HuggingFace
- Generated question variations (ephemeral, not persisted initially)
- Model answers to question variations
- Coherence scores per question
- Evaluation results (correct/incorrect/not_attempted)
- Checkpoints for resume functionality
- API call cache for cost optimization

### Reusability Opportunities

**Components from selfcheckgpt:**
- `modeling_coherence.py`: Direct import of SelfCheckShogenji, SelfCheckFitelson, SelfCheckOlsson classes
- `modeling_coherence_api.py`: CoherenceAPIClient for probability extraction
- `utils_coherence.py`: Coherence formula utilities and normalization
- Pattern: `scripts/evaluate.py` for CLI structure and parallel processing logic

**Components from cosmos-coherence:**
- `simpleqa_cli.py`: CLI infrastructure, checkpointing, progress tracking
- `simpleqa_benchmark.py`: Dataset loading, HuggingFace integration
- `simpleqa_grader.py`: AI grading logic (CORRECT/INCORRECT/NOT_ATTEMPTED)
- `openai_client.py`: OpenAI client with caching and async support

**Backend Patterns to Follow:**
- Async/await for concurrent API calls (from both repos)
- ThreadPoolExecutor for parallel processing (selfcheckgpt pattern)
- LRU cache for API response deduplication (both repos)
- Checkpoint/resume mechanism (cosmos-coherence pattern)
- Rich library for progress bars and tables (cosmos-coherence pattern)

### Scope Boundaries

**In Scope:**

1. **Question Variation Generation**
   - Prompt-based paraphrasing via OpenAI API
   - Temperature-based diversity
   - Configurable number of variations

2. **Coherence-Based Abstention**
   - All three coherence measures (Shogenji, Fitelson, Olsson)
   - Threshold-based decision logic
   - Integration with SimpleQA grading framework

3. **Evaluation Infrastructure**
   - CLI tool similar to existing SimpleQA CLI
   - Full SimpleQA dataset evaluation (4,326 questions)
   - Checkpointing and resume capability
   - Progress tracking and cache statistics
   - JSON output for results

4. **Metrics & Analysis**
   - Standard SimpleQA metrics (% correct, % incorrect, % not attempted)
   - Baseline comparison (with vs. without coherence)
   - Optional: Coherence distribution analysis
   - Optional: Correlation analysis between coherence and correctness

5. **Hybrid Repository Architecture**
   - Implement in cosmos-coherence repository
   - Import coherence logic from selfcheckgpt package
   - Reuse existing SimpleQA infrastructure from cosmos-coherence

**Out of Scope:**

1. **Advanced Answer Selection**
   - Selecting "best" question variation (future enhancement)
   - Using answer from highest-coherence question (deferred)

2. **Persistent Question Variation Dataset**
   - Caching generated question variations (future enhancement if approach proves successful)
   - Creating reproducible dataset of paraphrases (deferred)

3. **Extended Metrics (unless needed for debugging)**
   - Per-category coherence analysis
   - Detailed abstention rate breakdowns by question type
   - Confidence calibration curves

4. **Multi-Model Comparison**
   - Evaluating multiple target models simultaneously (can be run separately)
   - Cross-model coherence analysis

5. **Upstream Contributions**
   - Contributing coherence variants back to selfcheckgpt (deferred pending results)
   - Publishing research findings (deferred pending experimental validation)

### Technical Considerations

**Integration Points:**
- **selfcheckgpt package**: Import coherence classes and utilities
- **cosmos-coherence codebase**: Extend SimpleQA CLI and benchmark implementations
- **HuggingFace datasets**: Load SimpleQA dataset (`basicv8vc/SimpleQA`)
- **OpenAI API**: Question generation, answer generation, coherence probability extraction, grading

**Existing System Constraints:**
- Must maintain compatibility with existing SimpleQA infrastructure in cosmos-coherence
- Should reuse OpenAI client caching to minimize API costs
- Must handle 4,326 questions with N variations each (API call budget considerations)
- Checkpoint system required for long-running evaluations

**Technology Preferences:**
- Python 3.x (both repositories)
- Async/await for API calls (cosmos-coherence pattern)
- Rich library for CLI output (cosmos-coherence standard)
- Typer for CLI framework (cosmos-coherence standard)
- pytest for testing (both repositories)

**Similar Code Patterns to Follow:**

**CLI Pattern (cosmos-coherence):**
```python
# Follow simpleqa_cli.py structure
@app.command()
def run_with_coherence(
    model: str = typer.Option(...),
    coherence_variant: str = typer.Option("shogenji"),
    num_variations: int = typer.Option(5),
    coherence_threshold: float = typer.Option(0.5),
    # ... other params
):
    # Implementation
```

**Coherence Calculation (selfcheckgpt):**
```python
# Import and adapt from selfcheckgpt
from selfcheckgpt import SelfCheckShogenji, SelfCheckFitelson, SelfCheckOlsson

# Adapt predict() interface:
# Instead of: scores = coherence.predict(sentences, sampled_passages)
# Use: score = coherence.predict([original_answer], variation_answers)
```

**Evaluation Loop (hybrid pattern):**
```python
# Combine cosmos-coherence async loop with selfcheckgpt coherence logic
async def evaluate_with_coherence(item: SimpleQAItem):
    # 1. Generate question variations (new)
    variations = await generate_variations(item.question)

    # 2. Get answers for each variation (new)
    variation_answers = await get_answers_async(variations)

    # 3. Calculate coherence (import from selfcheckgpt)
    coherence_score = coherence_variant.predict([original_answer], variation_answers)

    # 4. Apply threshold logic (new)
    if coherence_score < threshold:
        final_answer = "I don't know"
    else:
        final_answer = original_answer

    # 5. Grade using existing grader (cosmos-coherence)
    result = await grader.grade_response(question, ground_truth, final_answer)

    return result
```

**API Cost Optimization:**
- Reuse CoherenceAPIClient caching from selfcheckgpt
- Reuse OpenAIClient caching from cosmos-coherence
- Estimate API calls before running: (4326 questions) × (N variations) × (cost per coherence calculation)
- Provide cost estimation utility before evaluation starts

**Error Handling:**
- Handle API rate limits with retry logic (existing in both repos)
- Handle malformed question variations gracefully
- Continue evaluation on single-question failures (checkpoint pattern)
- Log errors without halting full evaluation

**Performance Considerations:**
- Parallel question variation generation (async/await)
- Concurrent answer collection for variations
- Reuse coherence caching across questions
- Checkpoint progress every N questions (default: 50)

**Configuration Parameters:**
```
--model: Target model to evaluate (default: gpt-4o-mini)
--coherence-variant: shogenji|fitelson|olsson (default: shogenji)
--num-variations: Number of question paraphrases (default: 5)
--coherence-threshold: Threshold for abstention (default: 0.5)
--temperature-range: Temperatures for variation generation (default: [0.7, 1.0, 1.3])
--sample-size: Number of SimpleQA questions to evaluate (default: all 4326)
--output: Path to save results JSON
--checkpoint-interval: Save progress every N questions (default: 50)
--resume-from: Resume from checkpoint file
--workers: Number of parallel workers (default: 4)
```

### Key Research Questions to Answer

This implementation will help answer:
1. **Does question-based coherence reduce hallucinations?** Compare % incorrect between coherence-guided and baseline.
2. **What's the precision-recall tradeoff?** Measure correct_given_attempted (precision) vs overall_correct (recall).
3. **Which coherence variant works best?** Compare Shogenji, Fitelson, Olsson on SimpleQA.
4. **What's the optimal threshold?** Experiment with different coherence thresholds for abstention.
5. **Is the approach cost-effective?** Analyze API costs vs. accuracy improvement.

### Success Metrics

**Implementation Success:**
- CLI tool runs successfully on SimpleQA dataset
- All three coherence variants operational
- Checkpointing and resume functionality working
- Results comparable to baseline (same evaluation framework)

**Research Success:**
- Coherence-guided approach reduces incorrect answers (higher precision)
- Abstention rate is reasonable (not abstaining on everything)
- At least one coherence variant shows promise
- Clear understanding of precision-recall tradeoff

**Code Quality:**
- Reuses existing infrastructure effectively
- Maintains compatibility with both repositories
- Documented configuration options
- Reproducible results via checkpointing and seeded generation
