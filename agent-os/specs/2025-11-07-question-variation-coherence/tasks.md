# Task Breakdown: Question Variation Coherence for SimpleQA

## Overview

**Feature**: Coherence-guided hallucination prevention using question variations instead of answer variations
**Total Estimated Tasks**: 47 core tasks across 7 major task groups
**Implementation Repositories**: cosmos-coherence (primary) + selfcheckgpt (imports)
**Estimated Timeline**: 2-3 weeks for full implementation and validation

## Task List

### Task Group 1: Setup & Infrastructure

**Dependencies:** None

**Description:** Set up the development environment, verify repository access, configure dependencies, and establish the hybrid repository integration pattern.

- [x] 1.0 Complete infrastructure setup
  - [x] 1.1 Verify access to both repositories
    - Clone/pull latest from cosmos-coherence repository at `/Users/nathanlubchenco/workspace/cosmos-coherence`
    - Clone/pull latest from selfcheckgpt repository at `/Users/nathanlubchenco/workspace/selfcheckgpt`
    - Verify both repositories are on correct branches
  - [x] 1.2 Configure selfcheckgpt as dependency in cosmos-coherence
    - Add selfcheckgpt to cosmos-coherence requirements.txt or pyproject.toml
    - Support both development mode (local editable install) and package mode
    - Document installation instructions for developers
  - [x] 1.3 Verify coherence module imports
    - Test import: `from selfcheckgpt import SelfCheckShogenji, SelfCheckFitelson, SelfCheckOlsson`
    - Test import: `from selfcheckgpt.modeling_coherence_api import CoherenceAPIClient`
    - Test import: `from selfcheckgpt.utils_coherence import normalize_coherence_scores`
    - Verify imports work in cosmos-coherence Python environment
  - [x] 1.4 Review existing SimpleQA infrastructure in cosmos-coherence
    - Read and understand `src/cosmos_coherence/benchmarks/simpleqa_cli.py`
    - Read and understand `src/cosmos_coherence/benchmarks/implementations/simpleqa_benchmark.py`
    - Read and understand `src/cosmos_coherence/benchmarks/implementations/simpleqa_grader.py`
    - Read and understand `src/cosmos_coherence/llm/openai_client.py`
    - Identify extension points and reusable patterns
  - [x] 1.5 Set up development environment
    - Configure OpenAI API key in environment variables
    - Verify API access and quota
    - Set up pytest for testing
    - Configure code formatter/linter to match repository standards

**Acceptance Criteria:**
- Both repositories accessible and up-to-date
- selfcheckgpt coherence modules import successfully in cosmos-coherence
- Existing SimpleQA infrastructure understood and documented
- Development environment configured with API access

**Estimated Complexity:** Simple

**Status:** COMPLETED

**Implementation Notes:**
- Repositories verified and up-to-date
  - cosmos-coherence: branch `selfcheckgpt-implementation`
  - selfcheckgpt: branch `main`
- selfcheckgpt installed in editable mode: `pip3 install --break-system-packages -e /Users/nathanlubchenco/workspace/selfcheckgpt`
- Dependencies installed: numpy, tqdm, transformers, torch, openai, groq
- Note: Python 3.14 has compatibility issues with spacy, but coherence variants don't require it
- All coherence imports verified working
- Documentation created:
  - `/Users/nathanlubchenco/workspace/cosmos-coherence/docs/SELFCHECKGPT_INTEGRATION.md`
  - `/Users/nathanlubchenco/workspace/cosmos-coherence/docs/INFRASTRUCTURE_REVIEW.md`
- OpenAI API key confirmed configured
- pytest already configured in pyproject.toml
- Code formatter (black) and linter (ruff) configured in pyproject.toml

---

### Task Group 2: Question Variation Generation

**Dependencies:** Task Group 1

**Description:** Implement the QuestionVariationGenerator class to create diverse paraphrased versions of benchmark questions using temperature-based sampling.

- [ ] 2.0 Complete question variation generator
  - [ ] 2.1 Write 2-8 focused tests for QuestionVariationGenerator
    - Test: Generates correct number of variations (num_variations parameter)
    - Test: Distributes generations across temperature range correctly
    - Test: Handles API errors gracefully (mock API failures)
    - Test: Filters out identical variations (when model returns duplicate)
    - Test: Validates paraphrases are meaningful (not semantically divergent)
    - Limit to 5-6 tests maximum, focusing on critical behaviors
  - [ ] 2.2 Create QuestionVariationGenerator class
    - File: `cosmos-coherence/src/cosmos_coherence/benchmarks/implementations/question_variation_generator.py`
    - Constructor parameters: client (OpenAIClient), model (str), temperature_range (List[float]), num_variations (int)
    - Reuse OpenAIClient from cosmos-coherence for API calls
    - Follow existing async/await patterns from cosmos-coherence
  - [ ] 2.3 Implement variation generation logic
    - Method: `async def generate_variations(self, question: str) -> List[str]`
    - Prompt template: "Rephrase the following question while preserving its exact meaning: {question}"
    - Distribute N variations across temperature range (cycle through temps)
    - Use asyncio.gather for concurrent API calls if beneficial
    - Handle API errors with retries using existing OpenAIClient retry logic
  - [ ] 2.4 Add variation validation and filtering
    - Filter out variations identical to original question (exact match)
    - Filter out near-identical variations (minimum edit distance threshold, e.g., 5 chars)
    - Optionally: Check semantic similarity to avoid drift (defer if time-consuming)
    - Log rejected variations with reason for debugging
    - Retry generation if too many variations filtered (up to 2 retry attempts)
  - [ ] 2.5 Add configuration and logging
    - Make prompt template configurable (default provided, allow override)
    - Add verbose logging mode for debugging variation quality
    - Log generation statistics: attempts, rejections, final count
    - Implement caching if beneficial (cache variations by question hash)
  - [ ] 2.6 Ensure QuestionVariationGenerator tests pass
    - Run ONLY the 5-6 tests written in 2.1
    - Verify all test cases pass
    - Manually test on 3-5 sample questions from SimpleQA to validate quality

**Acceptance Criteria:**
- QuestionVariationGenerator class implemented with async support
- Generates N variations distributed across temperature range
- Validates and filters variations effectively
- Tests pass (5-6 focused tests)
- Manual testing shows high-quality paraphrases

**Estimated Complexity:** Medium

---

### Task Group 3: Coherence Integration & Scoring

**Dependencies:** Task Group 1, Task Group 2

**Description:** Integrate coherence calculation from selfcheckgpt package and adapt it for question-variation use case (scoring coherence of question variations, not answer samples).

- [ ] 3.0 Complete coherence integration
  - [ ] 3.1 Write 2-8 focused tests for coherence integration
    - Test: SelfCheckShogenji scores question variations correctly
    - Test: SelfCheckFitelson scores question variations correctly
    - Test: SelfCheckOlsson scores question variations correctly
    - Test: Score normalization works (values in [0, 1] range)
    - Test: Handles edge case: all variations identical (coherence = 1.0)
    - Test: Handles edge case: highly diverse variations (low coherence)
    - Limit to 6-7 tests maximum
  - [ ] 3.2 Create coherence variant initialization helper
    - Method: `_init_coherence_variant(variant_name: str) -> Union[SelfCheckShogenji, SelfCheckFitelson, SelfCheckOlsson]`
    - Map string names ("shogenji", "fitelson", "olsson") to classes
    - Initialize with appropriate model (default: gpt-4o-mini)
    - Handle invalid variant names with clear error message
  - [ ] 3.3 Implement coherence scoring for question variations
    - Understand coherence API: `predict(sentences: List[str], sampled_passages: List[str])`
    - Adapt for question-variation use case:
      - `sentences` = [original_question] (single-element list)
      - `sampled_passages` = question_variations (list of paraphrases)
    - Call coherence variant's predict() method
    - Extract coherence score from result (handle array vs scalar return)
  - [ ] 3.4 Add score normalization and interpretation
    - Import normalize_coherence_scores from selfcheckgpt.utils_coherence
    - Apply normalization to ensure scores in [0, 1] range
    - Document interpretation: high score = high coherence = well-formed question
    - Add optional score inversion if needed (depends on coherence variant behavior)
  - [ ] 3.5 Handle coherence calculation errors
    - Catch numerical instability errors (division by zero, invalid probabilities)
    - Log errors with question and variation details
    - Provide fallback behavior: skip question or use default score (configurable)
    - Track error rate in metrics
  - [ ] 3.6 Ensure coherence integration tests pass
    - Run ONLY the 6-7 tests written in 3.1
    - Verify all three coherence variants work correctly
    - Manually test on sample questions to validate score quality

**Acceptance Criteria:**
- All three coherence variants (Shogenji, Fitelson, Olsson) operational
- Coherence scoring adapted for question variations
- Score normalization ensures [0, 1] range
- Tests pass (6-7 focused tests)
- Error handling robust for numerical issues

**Estimated Complexity:** Medium

---

### Task Group 4: Answer Strategy Implementation

**Dependencies:** Task Group 2, Task Group 3

**Description:** Implement threshold-based abstention logic with special handling for YES/NO questions (coherence determines polarity) vs other questions (coherence determines abstention).

- [ ] 4.0 Complete answer strategy implementation
  - [ ] 4.1 Write 2-8 focused tests for answer strategy
    - Test: YES/NO question detected correctly (pattern matching)
    - Test: High coherence YES/NO question returns binary answer (YES or NO)
    - Test: Low coherence YES/NO question returns opposite polarity
    - Test: High coherence non-YES/NO question returns model answer
    - Test: Low coherence non-YES/NO question returns abstention ("I don't know")
    - Test: Threshold boundary behavior (score exactly at threshold)
    - Limit to 6 tests maximum
  - [ ] 4.2 Implement YES/NO question detection
    - Method: `_is_yes_no_question(question: str) -> bool`
    - Pattern matching: Check for phrases like "Is", "Are", "Does", "Do", "Can", "Will", "Should"
    - Check SimpleQA metadata if available (some datasets flag question type)
    - Return boolean: True if YES/NO question, False otherwise
  - [ ] 4.3 Implement coherence-based polarity mapping for YES/NO questions
    - Method: `_coherence_to_yes_no(coherence_score: float, threshold: float) -> str`
    - Logic (to be empirically validated):
      - If coherence_score >= threshold: return "YES"
      - If coherence_score < threshold: return "NO"
    - Make polarity mapping configurable (allow YES/NO flip via parameter)
    - Document that mapping should be tuned based on experimental results
  - [ ] 4.4 Implement threshold-based abstention for other questions
    - Method: `_apply_coherence_threshold(answer: str, coherence_score: float, threshold: float) -> str`
    - Logic:
      - If coherence_score >= threshold: return answer (confident)
      - If coherence_score < threshold: return abstention_response ("I don't know")
    - Make abstention response configurable (parameter: abstention_response)
  - [ ] 4.5 Integrate answer strategy into evaluation pipeline
    - Combine YES/NO detection, coherence scoring, and threshold logic
    - Flow:
      1. Generate question variations
      2. Calculate coherence score
      3. If YES/NO question: use coherence_to_yes_no()
      4. If other question: generate answer, then apply threshold
    - Track metadata: question_type, coherence_score, threshold_met, answer_source
  - [ ] 4.6 Ensure answer strategy tests pass
    - Run ONLY the 6 tests written in 4.1
    - Verify YES/NO handling works correctly
    - Verify threshold logic works for non-YES/NO questions
    - Manually test on sample YES/NO and factual questions

**Acceptance Criteria:**
- YES/NO question detection implemented and tested
- Coherence-based polarity mapping functional for YES/NO questions
- Threshold-based abstention functional for other questions
- Tests pass (6 focused tests)
- Manual validation confirms expected behavior

**Estimated Complexity:** Medium

---

### Task Group 5: Benchmark Integration & CLI

**Dependencies:** Task Group 2, Task Group 3, Task Group 4

**Description:** Create SimpleQACoherenceBenchmark class extending SimpleQABenchmark and implement CLI command for running coherence-guided evaluations.

- [ ] 5.0 Complete benchmark integration and CLI
  - [ ] 5.1 Write 2-8 focused tests for SimpleQACoherenceBenchmark
    - Test: Benchmark initializes with coherence variant correctly
    - Test: evaluate_with_coherence() returns expected result structure
    - Test: Grading integration works (calls SimpleQAGrader correctly)
    - Test: Metadata tracking complete (variations, scores, decisions)
    - Test: Checkpoint save/load works (serialize/deserialize state)
    - Limit to 5 tests maximum
  - [ ] 5.2 Create SimpleQACoherenceBenchmark class
    - File: `cosmos-coherence/src/cosmos_coherence/benchmarks/implementations/simpleqa_coherence_benchmark.py`
    - Extend SimpleQABenchmark (inherit dataset loading, grading setup)
    - Constructor parameters: coherence_variant, coherence_threshold, num_variations, temperature_range, abstention_response
    - Initialize QuestionVariationGenerator and coherence variant
    - Reuse existing SimpleQAGrader for answer evaluation
  - [ ] 5.3 Implement evaluate_with_coherence method
    - Method: `async def evaluate_with_coherence(self, item: SimpleQAItem) -> Dict`
    - Pipeline (7 steps from spec):
      1. Load SimpleQA question from item
      2. Generate question variations using QuestionVariationGenerator
      3. Calculate coherence score using coherence variant
      4. Determine answer strategy (YES/NO or threshold-based)
      5. Generate answer (if applicable: non-YES/NO and coherence >= threshold)
      6. Grade answer using SimpleQAGrader
      7. Return result dict with metadata
    - Return structure: {question, variations, coherence_score, final_answer, grade, is_correct, metadata}
  - [ ] 5.4 Implement full dataset evaluation loop
    - Method: `async def evaluate_dataset(self, sample_size: Optional[int] = None) -> List[Dict]`
    - Load SimpleQA dataset from HuggingFace (existing pattern)
    - Limit to sample_size if provided (for testing)
    - Use asyncio/ThreadPoolExecutor for parallel processing (configurable workers)
    - Implement progress tracking with rich progress bar
    - Handle individual question failures gracefully (log and continue)
  - [ ] 5.5 Add checkpointing and resume functionality
    - Method: `save_checkpoint(results: List[Dict], checkpoint_path: Path)`
    - Method: `load_checkpoint(checkpoint_path: Path) -> Tuple[List[Dict], int]`
    - Checkpoint format: JSON with config, progress, partial_results
    - Save every N questions (configurable checkpoint_interval, default: 50)
    - On resume: load checkpoint, skip completed questions, continue from last_index
    - Validate checkpoint integrity on load (handle corruption gracefully)
  - [ ] 5.6 Implement CLI command in simpleqa_coherence_cli.py
    - File: `cosmos-coherence/src/cosmos_coherence/benchmarks/simpleqa_coherence_cli.py`
    - Command: `cosmos-coherence simpleqa run-with-coherence`
    - Use Typer framework (existing pattern from cosmos-coherence)
    - Parameters (see spec section "Command-Line Interface Specification"):
      - Required: --model
      - Optional: --coherence-variant, --num-variations, --coherence-threshold, --temperature-range, --sample-size, --output, --checkpoint-interval, --resume-from, --workers, --run-baseline, --abstention-response, --grader-model, --verbose
    - Initialize SimpleQACoherenceBenchmark with parsed parameters
    - Run evaluation with progress tracking
  - [ ] 5.7 Add baseline comparison functionality
    - If --run-baseline=true: run standard SimpleQA evaluation (no coherence)
    - Reuse existing SimpleQABenchmark for baseline
    - Collect same metrics: overall_correct, overall_incorrect, precision
    - Include baseline metrics in output JSON for comparison
  - [ ] 5.8 Implement results aggregation and output
    - Calculate metrics: overall_correct, overall_incorrect, overall_not_attempted, correct_given_attempted
    - Calculate coherence statistics: mean, median, std dev, abstention_rate
    - Track API statistics: total_calls, cached_calls, cache_hit_rate, estimated_cost
    - Output results to JSON file (format specified in spec)
    - Display summary table in console using rich library
  - [ ] 5.9 Ensure benchmark integration tests pass
    - Run ONLY the 5 tests written in 5.1
    - Verify end-to-end pipeline works on small sample (10 questions)
    - Verify checkpoint/resume works correctly
    - Test CLI command with various parameter combinations

**Acceptance Criteria:**
- SimpleQACoherenceBenchmark class functional and extends SimpleQABenchmark
- evaluate_with_coherence() implements full 7-step pipeline
- Full dataset evaluation with parallel processing works
- Checkpointing and resume functionality reliable
- CLI command functional with all parameters
- Baseline comparison runs successfully
- Results output in specified JSON format
- Tests pass (5 focused tests)

**Estimated Complexity:** Complex

---

### Task Group 6: Testing & Validation

**Dependencies:** Task Groups 1-5

**Description:** Conduct comprehensive testing including unit tests, integration tests, cost validation, and baseline comparison on a sample dataset.

- [ ] 6.0 Complete testing and validation
  - [ ] 6.1 Review and consolidate all tests from previous task groups
    - Review tests from Task 2.1 (QuestionVariationGenerator: ~5-6 tests)
    - Review tests from Task 3.1 (Coherence integration: ~6-7 tests)
    - Review tests from Task 4.1 (Answer strategy: ~6 tests)
    - Review tests from Task 5.1 (Benchmark integration: ~5 tests)
    - Total existing tests: approximately 22-24 tests
    - Ensure all tests are passing and well-organized
  - [ ] 6.2 Add integration tests for end-to-end workflows
    - Test: Full evaluation on 10-question sample (all three coherence variants)
    - Test: Checkpoint save and resume (interrupt after 5 questions, resume)
    - Test: Baseline comparison produces expected metrics structure
    - Test: Parallel processing with multiple workers (2, 4 workers)
    - Test: Error handling (API failures, malformed questions)
    - Add maximum 5 new integration tests
  - [ ] 6.3 Conduct cost validation on small sample (N=50-100 questions)
    - Run evaluation on 50 questions with num_variations=5
    - Track actual API calls made (use API client logging)
    - Compare to estimated API calls: ~18 calls/question = 900 total
    - Calculate actual cost based on gpt-4o-mini pricing
    - Measure cache hit rate (expect 20-30%)
    - Validate API call budget is accurate (within 20% of estimate)
  - [ ] 6.4 Validate coherence scoring quality on sample dataset
    - Run evaluation on 100 questions from SimpleQA
    - Manually inspect 10-20 questions with varying coherence scores
    - Verify high coherence correlates with well-formed questions
    - Verify low coherence correlates with ambiguous/complex questions
    - Check for coherence score distribution (should not be all 0s or all 1s)
  - [ ] 6.5 Baseline comparison validation
    - Run coherence-guided evaluation on 100 questions (threshold=0.5)
    - Run baseline evaluation (no coherence) on same 100 questions
    - Compare metrics:
      - Baseline: overall_correct ~75%, overall_incorrect ~25%, not_attempted ~0%
      - Coherence: expect overall_incorrect reduced, not_attempted 10-30%
    - Verify coherence improves precision even if recall decreases
    - Document comparison results
  - [ ] 6.6 Test all three coherence variants on sample
    - Run 50-question sample with SelfCheckShogenji
    - Run 50-question sample with SelfCheckFitelson
    - Run 50-question sample with SelfCheckOlsson
    - Compare coherence score distributions across variants
    - Identify if one variant performs notably better (may not be clear yet)
  - [ ] 6.7 Validate YES/NO question handling
    - Identify YES/NO questions in sample (filter from SimpleQA)
    - Run coherence evaluation on YES/NO subset
    - Verify coherence-based polarity mapping works
    - Check if coherence predicts correctness for YES/NO questions
    - Adjust polarity mapping if empirical results suggest flip needed
  - [ ] 6.8 Run full test suite
    - Run all unit tests from task groups 2-5 (~22-24 tests)
    - Run all integration tests from task 6.2 (~5 tests)
    - Total: approximately 27-29 tests
    - Ensure 100% pass rate
    - Verify test coverage is reasonable for new code

**Acceptance Criteria:**
- All unit and integration tests pass (~27-29 tests total)
- Cost validation confirms API budget is accurate
- Coherence scoring quality validated on sample
- Baseline comparison shows expected tradeoff (precision up, recall down)
- All three coherence variants functional
- YES/NO question handling validated and tuned
- No more than 5 additional tests added beyond existing 22-24

**Estimated Complexity:** Complex

---

### Task Group 7: Documentation & Finalization

**Dependencies:** Task Groups 1-6

**Description:** Create comprehensive documentation, usage examples, cost analysis, and finalize the implementation for handoff or future development.

- [ ] 7.0 Complete documentation and finalization
  - [ ] 7.1 Write README section for question-variation coherence feature
    - Location: Update cosmos-coherence README.md with new section
    - Content: Overview, use case, command examples, parameter descriptions
    - Include comparison to baseline SimpleQA evaluation
    - Reference architecture and theory from selfcheckgpt coherence docs
  - [ ] 7.2 Create usage examples and tutorials
    - Example 1: Basic usage with default parameters (sample command)
    - Example 2: Threshold tuning (run with multiple thresholds: 0.3, 0.5, 0.7)
    - Example 3: Comparing all three coherence variants
    - Example 4: Resuming from checkpoint after interruption
    - Include expected output format and interpretation guidance
  - [ ] 7.3 Document API cost analysis
    - Provide cost estimates for full dataset (4,326 questions) with different models
    - Cost breakdown: gpt-4o-mini (~$5-6), gpt-4-turbo (~$40-45)
    - Cache optimization impact (20-30% reduction)
    - Cost comparison: baseline vs coherence-guided (~9x multiplier)
    - Cost-benefit analysis: when is higher cost justified?
  - [ ] 7.4 Write API documentation for new classes
    - Document QuestionVariationGenerator: class docstring, method signatures, parameters
    - Document SimpleQACoherenceBenchmark: class docstring, methods, usage pattern
    - Document helper methods: _is_yes_no_question, _coherence_to_yes_no, _apply_coherence_threshold
    - Follow existing docstring format from cosmos-coherence (Google-style or NumPy-style)
  - [ ] 7.5 Create troubleshooting guide
    - Common issue: High abstention rate (all or most questions abstained)
      - Diagnosis: Check coherence score distribution, may need lower threshold
    - Common issue: No improvement over baseline
      - Diagnosis: Coherence may not correlate with correctness for this model
    - Common issue: API cost overruns
      - Solution: Use smaller sample size, increase cache hit rate, use cheaper model
    - Common issue: Poor variation quality (all similar or all different)
      - Solution: Adjust temperature range, modify prompt template
  - [ ] 7.6 Document experimental results and insights
    - Record baseline metrics from sample evaluation (Task 6.5)
    - Record coherence-guided metrics from sample evaluation
    - Document coherence-correctness correlation (if any)
    - Document which coherence variant performed best (if determinable)
    - Document optimal threshold range based on experiments
    - Include precision-recall tradeoff analysis
  - [ ] 7.7 Create configuration file template
    - Provide YAML or JSON config file template for common use cases
    - Include parameter explanations and recommended values
    - Example configs: conservative (high threshold), balanced, aggressive (low threshold)
  - [ ] 7.8 Final code review and cleanup
    - Remove debug print statements and commented-out code
    - Ensure consistent code formatting (run formatter)
    - Verify all docstrings are complete and accurate
    - Check for any hardcoded values that should be parameters
    - Ensure error messages are clear and actionable
  - [ ] 7.9 Prepare handoff documentation
    - Known limitations section (e.g., OpenAI-only, SimpleQA-only)
    - Future enhancements list (from spec "Future Extensions" section)
    - Dependencies and version requirements
    - Testing instructions for future developers
    - Contact information or references for questions

**Acceptance Criteria:**
- README updated with comprehensive feature documentation
- Usage examples cover common scenarios
- API cost analysis documented with estimates
- API documentation complete for all new classes
- Troubleshooting guide addresses common issues
- Experimental results documented from sample evaluation
- Configuration templates provided
- Code cleaned up and reviewed
- Handoff documentation ready

**Estimated Complexity:** Medium

---

## Execution Order & Dependencies

### Phase 1: Foundation (Week 1)
1. **Task Group 1**: Setup & Infrastructure (Days 1-2) - COMPLETED
2. **Task Group 2**: Question Variation Generation (Days 3-4)
3. **Task Group 3**: Coherence Integration & Scoring (Days 4-5)

### Phase 2: Implementation (Week 2)
4. **Task Group 4**: Answer Strategy Implementation (Days 6-7)
5. **Task Group 5**: Benchmark Integration & CLI (Days 8-11)

### Phase 3: Validation & Documentation (Week 3)
6. **Task Group 6**: Testing & Validation (Days 12-14)
7. **Task Group 7**: Documentation & Finalization (Days 15-16)

## Critical Path

The critical path for this implementation is:
1. Setup → 2. Question Variations → 3. Coherence Scoring → 4. Answer Strategy → 5. Benchmark/CLI → 6. Testing → 7. Documentation

Each task group depends on the previous one, making this a largely sequential implementation with some parallelization opportunities within each group.

## Risk Mitigation Tasks

Throughout implementation, pay special attention to these high-risk areas:

1. **API Cost Management** (Task Groups 5, 6)
   - Implement cost estimation before running full evaluations
   - Start with small sample sizes (50-100 questions) before scaling
   - Monitor actual costs vs estimates in Task 6.3

2. **Coherence Quality** (Task Groups 3, 6)
   - Validate coherence scores correlate with correctness in Task 6.4
   - If correlation is weak, approach may need fundamental rethinking
   - Have fallback plan: document negative results for research value

3. **Variation Quality** (Task Groups 2, 6)
   - Manually inspect variations early in Task 2.6
   - Iterate on prompt template if quality is poor
   - Adjust temperature range based on diversity/drift tradeoff

4. **Integration Complexity** (Task Group 5)
   - Start with simplest integration (extend SimpleQABenchmark)
   - Test incrementally (don't build full pipeline before testing pieces)
   - Use existing patterns from cosmos-coherence extensively

## Success Metrics

### Implementation Success (Must-Have)
- [ ] CLI tool runs successfully on SimpleQA dataset sample (100+ questions)
- [ ] All three coherence variants functional (Shogenji, Fitelson, Olsson)
- [ ] Checkpointing/resume works reliably
- [ ] API cost estimates accurate within 20%
- [ ] All tests pass (~27-29 total)

### Research Success (Nice-to-Have)
- [ ] Coherence-guided approach reduces incorrect answers by 20%+
- [ ] Precision improves by 5+ percentage points vs baseline
- [ ] Abstention rate reasonable (10-30%)
- [ ] At least one coherence variant shows clear benefit

### Code Quality (Must-Have)
- [ ] Reuses existing infrastructure (minimal code duplication)
- [ ] Well-documented (README, API docs, examples)
- [ ] Reproducible results (checkpointing, seeded generation)
- [ ] Passes code review (clean, readable, maintainable)

## Notes

- **Test-Driven Approach**: Each task group starts with writing 2-8 focused tests and ends with running ONLY those tests. Total test count should not exceed ~30 tests.
- **Incremental Validation**: Test each component individually before integration (Task 2.6, 3.6, 4.6, 5.9).
- **Cost Awareness**: Always run on small samples first (50-100 questions) before full dataset (4,326 questions).
- **Flexibility**: YES/NO polarity mapping (Task 4.3) and threshold values may need empirical tuning based on results.
- **Hybrid Repository Pattern**: Primary implementation in cosmos-coherence, imports from selfcheckgpt. Keep this separation clean.

## File Paths Reference

**New Files to Create (cosmos-coherence):**
- `/Users/nathanlubchenco/workspace/cosmos-coherence/src/cosmos_coherence/benchmarks/implementations/question_variation_generator.py`
- `/Users/nathanlubchenco/workspace/cosmos-coherence/src/cosmos_coherence/benchmarks/implementations/simpleqa_coherence_benchmark.py`
- `/Users/nathanlubchenco/workspace/cosmos-coherence/src/cosmos_coherence/benchmarks/simpleqa_coherence_cli.py`

**Existing Files to Extend/Modify (cosmos-coherence):**
- `/Users/nathanlubchenco/workspace/cosmos-coherence/src/cosmos_coherence/benchmarks/simpleqa_cli.py` (add new command)
- `/Users/nathanlubchenco/workspace/cosmos-coherence/README.md` (add documentation)
- `/Users/nathanlubchenco/workspace/cosmos-coherence/pyproject.toml` (selfcheckgpt dependency already configured)

**Files to Import From (selfcheckgpt):**
- `/Users/nathanlubchenco/workspace/selfcheckgpt/selfcheckgpt/modeling_coherence.py` (SelfCheckShogenji, SelfCheckFitelson, SelfCheckOlsson)
- `/Users/nathanlubchenco/workspace/selfcheckgpt/selfcheckgpt/modeling_coherence_api.py` (CoherenceAPIClient)
- `/Users/nathanlubchenco/workspace/selfcheckgpt/selfcheckgpt/utils_coherence.py` (normalize_coherence_scores)

**Existing Files to Reference (cosmos-coherence):**
- `/Users/nathanlubchenco/workspace/cosmos-coherence/src/cosmos_coherence/benchmarks/implementations/simpleqa_benchmark.py` (extend this)
- `/Users/nathanlubchenco/workspace/cosmos-coherence/src/cosmos_coherence/benchmarks/implementations/simpleqa_grader.py` (reuse this)
- `/Users/nathanlubchenco/workspace/cosmos-coherence/src/cosmos_coherence/llm/openai_client.py` (reuse this)

**Documentation Files Created (Task Group 1):**
- `/Users/nathanlubchenco/workspace/cosmos-coherence/docs/SELFCHECKGPT_INTEGRATION.md` - Installation and usage guide
- `/Users/nathanlubchenco/workspace/cosmos-coherence/docs/INFRASTRUCTURE_REVIEW.md` - Detailed infrastructure analysis and extension points
