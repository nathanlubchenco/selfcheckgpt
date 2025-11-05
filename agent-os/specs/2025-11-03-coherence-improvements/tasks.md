# Task Breakdown: SelfCheckGPT Coherence Variants Improvements

## Overview
Total Task Groups: 5
Implementation Strategy: Sequential phases with some parallel work opportunities

**Core Goal:** Improve coherence-based hallucination detection through optimized probability extraction prompts and comprehensive verification infrastructure.

**Key Priorities:**
1. Benchmark creation FIRST (blocks all other work) ✓ COMPLETED
2. Prompt improvements SECOND (highest priority after benchmark)
3. Verification THIRD (parallel with prompts) ✓ COMPLETED
4. Dataset investigation FOURTH (parallel with verification) ✓ COMPLETED
5. Integration FINAL (depends on all previous phases)

## Task List

### Phase 1: Benchmark Foundation (Critical Path - Blocks All Other Work) ✓ COMPLETED

#### Task Group 1.1: Test Case Design and Creation ✓ COMPLETED
**Dependencies:** None

- [x] 1.1.0 Complete probability extraction test suite
  - [x] 1.1.1 Design test case structure and schema
  - [x] 1.1.2 Create individual probability test cases (25 cases)
  - [x] 1.1.3 Create joint probability test cases (10 cases)
  - [x] 1.1.4 Create conditional probability test cases (5 cases)
  - [x] 1.1.5 Add domain diversity across all test cases
  - [x] 1.1.6 Create special edge case test cases (included in 40 total)
  - [x] 1.1.7 Document ground truth probabilities and rationale

**Acceptance Criteria:** ✓ ALL MET

**Implementation:** `/Users/nathanlubchenco/workspace/selfcheckgpt/selfcheckgpt/benchmark/test_cases.json`

---

#### Task Group 1.2: Evaluation Metrics Implementation ✓ COMPLETED
**Dependencies:** Task Group 1.1 (needs test case schema)

- [x] 1.2.0 Complete evaluation metrics module
  - [x] 1.2.1 Implement Brier Score calculator
  - [x] 1.2.2 Implement Expected Calibration Error (ECE) calculator
  - [x] 1.2.3 Implement Probability Coherence Compliance checker
  - [x] 1.2.4 Implement Probability Consistency Score calculator
  - [x] 1.2.5 Implement Sharpness metric calculator
  - [x] 1.2.6 Create metrics summary report generator

**Acceptance Criteria:** ✓ ALL MET

**Implementation:** `/Users/nathanlubchenco/workspace/selfcheckgpt/selfcheckgpt/benchmark/metrics.py`

---

#### Task Group 1.3: Benchmark Runner Infrastructure ✓ COMPLETED
**Dependencies:** Task Groups 1.1, 1.2

- [x] 1.3.0 Complete benchmark execution framework
  - [x] 1.3.1 Create benchmark runner main module
  - [x] 1.3.2 Implement prompt variant configuration system
  - [x] 1.3.3 Add OpenAI API integration with caching
  - [x] 1.3.4 Create comparative analysis module
  - [x] 1.3.5 Add progress tracking and logging
  - [x] 1.3.6 Implement results export functionality
  - [x] 1.3.7 Run benchmark runner end-to-end test

**Acceptance Criteria:** ✓ ALL MET

**Implementation:**
- Runner: `/Users/nathanlubchenco/workspace/selfcheckgpt/selfcheckgpt/benchmark/runner.py`
- CLI script: `/Users/nathanlubchenco/workspace/selfcheckgpt/scripts/run_benchmark.py`
- Package init: `/Users/nathanlubchenco/workspace/selfcheckgpt/selfcheckgpt/benchmark/__init__.py`

---

### Phase 2: Prompt Strategy Development (High Priority - Depends on Phase 1)

#### Task Group 2.1: Baseline Prompt Evaluation
**Dependencies:** Task Group 1.3 (needs benchmark runner)

- [ ] 2.1.0 Establish baseline performance
  - [ ] 2.1.1 Document current simple prompt templates
  - [ ] 2.1.2 Run baseline prompts through benchmark
  - [ ] 2.1.3 Analyze baseline weaknesses
  - [ ] 2.1.4 Document baseline performance report

**Acceptance Criteria:**
- Complete baseline metrics documented
- Specific weaknesses identified and categorized
- Improvement targets defined (e.g., >10% Brier score reduction)
- Baseline report saved for future comparison

---

#### Task Group 2.2: Chain of Thought (CoT) Prompt Variant
**Dependencies:** Task Group 2.1

- [ ] 2.2.0 Develop and evaluate CoT prompts
  - [ ] 2.2.1 Design CoT prompt templates
  - [ ] 2.2.2 Implement CoT variant in benchmark configuration
  - [ ] 2.2.3 Run CoT prompts through benchmark
  - [ ] 2.2.4 Analyze CoT performance

**Acceptance Criteria:**
- CoT prompts designed for all three probability types
- Benchmark execution completes successfully
- Performance comparison vs baseline documented
- Cost analysis included in evaluation

---

#### Task Group 2.3: Few-Shot Prompt Variant
**Dependencies:** Task Group 2.1

- [ ] 2.3.0 Develop and evaluate few-shot prompts
  - [ ] 2.3.1 Create few-shot example library
  - [ ] 2.3.2 Design few-shot prompt templates
  - [ ] 2.3.3 Implement few-shot variant in benchmark configuration
  - [ ] 2.3.4 Run few-shot prompts through benchmark
  - [ ] 2.3.5 Analyze few-shot performance

**Acceptance Criteria:**
- Few-shot examples selected and validated
- Few-shot prompts implemented for all probability types
- Benchmark results show impact of examples on performance
- Comparative analysis vs baseline and CoT completed

---

#### Task Group 2.4: Axiom-Aware Prompt Variant
**Dependencies:** Task Group 2.1

- [ ] 2.4.0 Develop and evaluate axiom-aware prompts
  - [ ] 2.4.1 Design axiom-aware system prompt
  - [ ] 2.4.2 Integrate axiom-aware system prompt with user prompts
  - [ ] 2.4.3 Run axiom-aware prompts through benchmark
  - [ ] 2.4.4 Analyze axiom-aware performance

**Acceptance Criteria:**
- Axiom-aware system prompt designed and validated
- Integration with user prompts maintains structured output
- Axiom violation rate significantly reduced (target: <5%)
- Comparative analysis completed

---

#### Task Group 2.5: Hybrid Prompt Variant
**Dependencies:** Task Groups 2.2, 2.3, 2.4

- [ ] 2.5.0 Develop and evaluate hybrid prompts
  - [ ] 2.5.1 Analyze results from CoT, few-shot, and axiom-aware variants
  - [ ] 2.5.2 Design hybrid prompt templates
  - [ ] 2.5.3 Implement hybrid variant in benchmark configuration
  - [ ] 2.5.4 Run hybrid prompts through benchmark
  - [ ] 2.5.5 Conduct final prompt variant comparison

**Acceptance Criteria:**
- Hybrid prompt combines best elements from previous variants
- Full comparative analysis across all 5 prompt variants completed
- Clear recommendation for production integration
- Cost-benefit analysis included

---

### Phase 3: Verification and Validation (Medium Priority - Parallel with Phase 2) ✓ COMPLETED

#### Task Group 3.1: Coherence Formula Mathematical Verification ✓ COMPLETED
**Dependencies:** Task Group 1.3 (benchmark runner exists)

- [x] 3.1.0 Validate coherence formula implementations
  - [x] 3.1.1 Create unit tests for Shogenji's ratio-based measure
    - Test formula: C2(A,B) = P(A∧B) / (P(A) × P(B))
    - Known-outcome test: Independent events should yield C2 ≈ 1
    - Known-outcome test: Positive correlation should yield C2 > 1
    - Known-outcome test: Mutually exclusive events should yield C2 ≈ 0
    - Test epsilon smoothing prevents division by zero
    - Test probability clamping to valid ranges
    - Reference: utils_coherence.py coherence_shogenji() function
  - [x] 3.1.2 Create unit tests for Fitelson's confirmation measure
    - Test formula: s(H,E) = P(H|E) - P(H|¬E)
    - Known-outcome test: Strong confirmation should yield s > 0
    - Known-outcome test: Strong disconfirmation should yield s < 0
    - Known-outcome test: Independence should yield s ≈ 0
    - Test handling of P(¬E) = 0 edge case
    - Test conditional probability consistency
    - Reference: utils_coherence.py coherence_fitelson() function
  - [x] 3.1.3 Create unit tests for Olsson's overlap measure
    - Test formula: C1(A,B) = P(A∧B) / P(A∨B)
    - Known-outcome test: Identical statements should yield C1 ≈ 1
    - Known-outcome test: Disjoint statements should yield C1 ≈ 0
    - Test P(A∨B) = P(A) + P(B) - P(A∧B) calculation
    - Test division by zero handling when P(A∨B) ≈ 0
    - Reference: utils_coherence.py coherence_olsson() function
  - [x] 3.1.4 Validate formulas against probability benchmark
    - Run benchmark test cases through each coherence formula
    - Verify no NaN or Inf values in outputs
    - Check that coherence scores fall in expected ranges
    - Identify any outliers or anomalies
  - [x] 3.1.5 Test normalize_coherence_scores() function
    - Test min-max normalization: (scores - min) / (max - min)
    - Test edge case: all scores identical → should return 0.5
    - Test edge case: NaN/Inf handling
    - Verify inversion to hallucination scores: 1.0 - normalized
    - Reference: utils_coherence.py normalize_coherence_scores()
  - [x] 3.1.6 Verify epsilon smoothing and warnings
    - Test that epsilon (1e-12) prevents division by zero
    - Verify warnings issued for physically impossible probabilities
    - Test P(A∧B) > P(A) detection and handling
    - Ensure epsilon doesn't distort results significantly

**Acceptance Criteria:** ✓ ALL MET
- ✓ All unit tests pass for three coherence formulas (30 tests total)
- ✓ Benchmark test cases produce valid coherence scores (no NaN/Inf)
- ✓ Edge cases handled correctly without crashes
- ✓ Normalization function works across all input ranges
- ✓ No mathematical errors detected in formula implementations

**Implementation:** `/Users/nathanlubchenco/workspace/selfcheckgpt/tests/test_coherence_formulas.py`
**Test Results:** 30 tests passed in 0.36s

---

#### Task Group 3.2: Integration Testing with Coherence Variants ✓ COMPLETED
**Dependencies:** Task Groups 1.3, 3.1

- [x] 3.2.0 Test end-to-end coherence variant pipelines
  - [x] 3.2.1 Create integration test for SelfCheckShogenji
    - Test full predict() interface with sample sentences and passages
    - Verify probability extraction → coherence calculation → score normalization pipeline
    - Test with various numbers of sampled passages (1, 3, 5)
    - Ensure caching works correctly across multiple sentences
    - Reference: modeling_coherence.py SelfCheckShogenji class
  - [x] 3.2.2 Create integration test for SelfCheckFitelson
    - Test predict() with conditional probability extraction
    - Verify higher API call count (1 + 3*num_samples per sentence)
    - Test handling of conditional probability extraction
    - Ensure Fitelson formula receives correct conditional probabilities
    - Reference: modeling_coherence.py SelfCheckFitelson class
  - [x] 3.2.3 Create integration test for SelfCheckOlsson
    - Test predict() with union probability calculation
    - Verify P(A∨B) = P(A) + P(B) - P(A∧B) computation
    - Test overlap measure calculation
    - Ensure results correlate with expected hallucination patterns
    - Reference: modeling_coherence.py SelfCheckOlsson class
  - [x] 3.2.4 Verify cache statistics and cost estimation
    - Test CoherenceAPIClient.get_cache_stats() returns accurate counts
    - Verify cache reduces redundant API calls
    - Test CoherenceAPIClient.estimate_api_calls() matches actual usage
    - Confirm cache persists across sentences within same session
  - [x] 3.2.5 Test error handling and edge cases
    - Test with empty sentences list
    - Test with empty sampled_passages list
    - Test with very long sentences (>1000 chars)
    - Test with special characters and unicode
    - Verify graceful degradation on API failures

**Acceptance Criteria:** ✓ ALL MET
- ✓ All three coherence variants pass integration tests (18 tests total)
- ✓ predict() interface works correctly end-to-end
- ✓ Caching mechanism reduces API calls as expected
- ✓ Cost estimation utilities are accurate
- ✓ Edge cases handled gracefully without crashes

**Implementation:** `/Users/nathanlubchenco/workspace/selfcheckgpt/tests/test_coherence_integration.py`
**Test Results:** 18 tests passed in 80.03s (includes real API calls)

**Additional Implementation:**
- Test infrastructure: `/Users/nathanlubchenco/workspace/selfcheckgpt/tests/`
- pytest configuration: `/Users/nathanlubchenco/workspace/selfcheckgpt/pytest.ini`
- Test documentation: `/Users/nathanlubchenco/workspace/selfcheckgpt/tests/README.md`

**Phase 3 Summary:**
- Total tests implemented: 48 (30 unit + 18 integration)
- All tests pass successfully
- Mathematical correctness validated for all three coherence measures
- End-to-end pipelines verified for all three coherence variants
- Edge cases, error handling, and caching mechanism tested
- API cost estimation validated

---

### Phase 4: Dataset Investigation (Medium Priority - Parallel with Phases 2-3) ✓ COMPLETED

#### Task Group 4.1: SimpleQA Dataset Analysis ✓ COMPLETED
**Dependencies:** None (independent research task)

- [x] 4.1.0 Investigate SimpleQA dataset compatibility
  - [x] 4.1.1 Research SimpleQA dataset structure
  - [x] 4.1.2 Analyze SimpleQA compatibility requirements
  - [x] 4.1.3 Assess data availability and licensing
  - [x] 4.1.4 Create compatibility assessment report

**Acceptance Criteria:** ✓ ALL MET
- ✓ Comprehensive understanding of SimpleQA dataset structure
- ✓ Clear determination of compatibility: INCOMPATIBLE
- ✓ Documentation of alternative dataset recommendations (HaluEval)
- ✓ Detailed rationale for incompatibility documented

**Implementation:** `/Users/nathanlubchenco/workspace/selfcheckgpt/agent-os/specs/2025-11-03-coherence-improvements/planning/simpleqa-compatibility-report.md`

**Key Findings:**
- **Dataset Structure:** 4,326 questions with ground truth answers only
- **Format:** Parquet/CSV with fields: metadata, problem, answer, split
- **License:** MIT (publicly available via HuggingFace)
- **Compatibility Determination:** INCOMPATIBLE

**Critical Blockers:**
1. No LLM-generated responses (only ground truth answers)
2. No multiple stochastic samples per question
3. Short-form answers (1-10 words) unsuitable for sentence-level coherence analysis
4. Fundamental mismatch with coherence approach (requires multi-sentence responses)

**Recommendation:** Maintain wiki_bio_gpt3_hallucination as sole evaluation dataset. SimpleQA's short-form factuality focus directly conflicts with coherence method requirements.

**Alternative Datasets Considered:**
- HaluEval (partial compatibility - better than SimpleQA but still requires sample generation)
- TruthfulQA (incompatible - same issues as SimpleQA)
- Custom dataset creation (future work)

---

#### Task Group 4.2: SimpleQA Integration (Conditional on 4.1 Compatibility)
**Dependencies:** Task Groups 4.1 (compatibility confirmed), 1.3 (benchmark runner exists)

**STATUS: SKIPPED** - Task Group 4.1 determined SimpleQA is INCOMPATIBLE

- [N/A] 4.2.0 Integrate SimpleQA as secondary evaluation dataset (ONLY IF COMPATIBLE)
  - [N/A] 4.2.1 Create SimpleQA data loader
  - [N/A] 4.2.2 Generate stochastic samples for SimpleQA
  - [N/A] 4.2.3 Create ground truth hallucination labels
  - [N/A] 4.2.4 Adapt evaluation script for SimpleQA
  - [N/A] 4.2.5 Run baseline coherence evaluation on SimpleQA

**Note:** This task group was conditional on Task 4.1 determining SimpleQA is compatible. Since SimpleQA is INCOMPATIBLE, this task group is not applicable.

**Phase 4 Summary:**
- SimpleQA dataset thoroughly researched and analyzed
- Compatibility assessment completed with detailed rationale
- Decision: Do NOT integrate SimpleQA (incompatible with coherence approach)
- Focus remains on wiki_bio_gpt3_hallucination as sole evaluation dataset
- Alternative datasets documented for potential future expansion

---

### Phase 5: Production Integration (Final - Depends on Phases 2-4)

#### Task Group 5.1: Production Prompt Integration
**Dependencies:** Task Group 2.5 (best prompt variant selected)

- [ ] 5.1.0 Integrate improved prompts into production coherence variants
  - [ ] 5.1.1 Update CoherenceAPIClient prompt templates
  - [ ] 5.1.2 Add backward compatibility configuration option
  - [ ] 5.1.3 Update SelfCheckShogenji with improved prompts
  - [ ] 5.1.4 Update SelfCheckFitelson with improved prompts
  - [ ] 5.1.5 Update SelfCheckOlsson with improved prompts
  - [ ] 5.1.6 Update module docstrings and documentation

**Acceptance Criteria:**
- Improved prompts integrated into all three coherence variant classes
- Backward compatibility maintained through configuration option
- predict() interface unchanged (no breaking changes)
- Module documentation updated with prompt strategy options
- Performance improvements documented

---

#### Task Group 5.2: End-to-End Validation
**Dependencies:** Task Groups 5.1, 4.1 (SimpleQA investigation complete)

- [ ] 5.2.0 Validate improved coherence variants on full datasets
  - [ ] 5.2.1 Run improved Shogenji on wiki_bio_gpt3_hallucination
  - [ ] 5.2.2 Run improved Fitelson on wiki_bio_gpt3_hallucination
  - [ ] 5.2.3 Run improved Olsson on wiki_bio_gpt3_hallucination
  - [N/A] 5.2.4 Run on SimpleQA if compatible (SKIPPED - incompatible)
  - [ ] 5.2.5 Create comprehensive improvement report
  - [ ] 5.2.6 Document known limitations and failure modes

**Acceptance Criteria:**
- All three coherence variants evaluated on wiki_bio with improved prompts
- Quantified improvement vs baseline (target: >10% Brier score reduction)
- SimpleQA evaluation skipped (determined incompatible in Phase 4)
- Comprehensive improvement report created
- Known limitations documented

---

#### Task Group 5.3: Documentation and Knowledge Transfer
**Dependencies:** Task Groups 5.1, 5.2

- [ ] 5.3.0 Create comprehensive documentation for improvements
  - [ ] 5.3.1 Update main README.md
  - [ ] 5.3.2 Update coherence_variants.md documentation
  - [ ] 5.3.3 Create probability extraction methodology guide
  - [ ] 5.3.4 Document benchmark usage and extension
  - [x] 5.3.5 Create educational demo notebook (understanding_coherence.ipynb)
  - [ ] 5.3.6 Update CHANGELOG or release notes

**Acceptance Criteria:**
- All documentation updated to reflect improvements
- Benchmark methodology thoroughly documented
- ✓ Educational demo notebook created for building coherence intuition
- Clear guidance provided for users on prompt selection
- Release notes prepared

**Implementation (Task 5.3.5):**
- Educational notebook: `/Users/nathanlubchenco/workspace/selfcheckgpt/demo/understanding_coherence.ipynb`
- Comprehensive 11-section tutorial covering:
  - Quick start and basic usage
  - Simplified math explanations with visualizations
  - Success stories from wiki_bio dataset (6-10 concrete examples)
  - Failure modes and edge cases
  - Comparative analysis vs. baselines
  - Performance visualizations (distributions, ROC curves, PR curves)
  - Score interpretation guide
  - Best practices and recommendations
  - Interactive "try it yourself" section
  - Complete documentation with examples

---

## Execution Order

**Critical Path:**
1. **Phase 1: Benchmark Foundation** (Task Groups 1.1 → 1.2 → 1.3) ✓ COMPLETED
   - Creates foundation for all subsequent work
   - Blocks Phases 2, 3, and 5

**High Priority (After Phase 1):**
2. **Phase 2: Prompt Strategy Development** (Task Groups 2.1 → 2.2/2.3/2.4 → 2.5)
   - Core improvement work
   - Task Groups 2.2, 2.3, 2.4 can run in parallel after 2.1
   - Task Group 2.5 depends on 2.2, 2.3, 2.4

**Medium Priority (Parallel Opportunities):**
3. **Phase 3: Verification** (Task Groups 3.1 → 3.2) ✓ COMPLETED
   - Can run in parallel with Phase 2 after Phase 1 completes

4. **Phase 4: Dataset Investigation** (Task Groups 4.1 → 4.2) ✓ COMPLETED
   - Task 4.1 completed (SimpleQA determined incompatible)
   - Task 4.2 skipped (conditional on 4.1 compatibility)

**Final Integration:**
5. **Phase 5: Production Integration** (Task Groups 5.1 → 5.2 → 5.3)
   - Task 5.1 depends on Phase 2 completion
   - Task 5.2 depends on 5.1 and Phase 4
   - Task 5.3 depends on all previous work

## Test Strategy

**Phase 1 Testing:** ✓ COMPLETED
- ✓ Test cases validated with proper structure and ground truth
- ✓ Metrics module implements all 5 metrics with correct formulas
- ✓ Benchmark runner supports full execution and comparative analysis

**Phase 2 Testing:**
- Run each prompt variant through full 40-test benchmark
- Compare metrics systematically
- Validate improvements are statistically significant

**Phase 3 Testing:** ✓ COMPLETED
- ✓ Unit test coherence formulas with known-outcome scenarios (30 tests)
- ✓ Integration test full predict() pipelines (18 tests)
- ✓ Verify no regressions in existing functionality
- ✓ Total: 48 tests passed (30 unit + 18 integration)

**Phase 4 Testing:** ✓ COMPLETED
- ✓ SimpleQA compatibility assessed (incompatible)
- ✓ No data loading or evaluation scripts needed (skipped)
- ✓ Comprehensive compatibility report created

**Phase 5 Testing:**
- Regression testing: ensure backward compatibility maintained
- End-to-end validation on wiki_bio dataset
- Performance benchmarking: confirm improvements hold on full dataset

**Total Test Count:**
- Phase 1: ✓ Benchmark infrastructure validated
- Phase 2: 5 prompt variants × 40 test cases = 200 benchmark executions
- Phase 3: ✓ 48 tests (30 unit + 18 integration) - ALL PASSED
- Phase 4: ✓ Compatibility analysis completed (no tests needed)
- Phase 5: End-to-end validation + regression suite

## Implementation Notes

**Technology Stack:**
- Python 3.x
- OpenAI API (gpt-4o-mini recommended for cost efficiency)
- NumPy for numerical operations
- spacy for sentence tokenization
- sklearn for metric calculations (AUC-PR, AUC-ROC)
- scipy for statistical metrics (PCC)
- tqdm for progress tracking
- pytest for testing

**Key Files Implemented:**

**Phase 1:**
- Test cases: `/Users/nathanlubchenco/workspace/selfcheckgpt/selfcheckgpt/benchmark/test_cases.json`
- Metrics module: `/Users/nathanlubchenco/workspace/selfcheckgpt/selfcheckgpt/benchmark/metrics.py`
- Benchmark runner: `/Users/nathanlubchenco/workspace/selfcheckgpt/selfcheckgpt/benchmark/runner.py`
- CLI script: `/Users/nathanlubchenco/workspace/selfcheckgpt/scripts/run_benchmark.py`
- Package init: `/Users/nathanlubchenco/workspace/selfcheckgpt/selfcheckgpt/benchmark/__init__.py`

**Phase 3:**
- Unit tests: `/Users/nathanlubchenco/workspace/selfcheckgpt/tests/test_coherence_formulas.py`
- Integration tests: `/Users/nathanlubchenco/workspace/selfcheckgpt/tests/test_coherence_integration.py`
- Test init: `/Users/nathanlubchenco/workspace/selfcheckgpt/tests/__init__.py`
- Test config: `/Users/nathanlubchenco/workspace/selfcheckgpt/pytest.ini`
- Test documentation: `/Users/nathanlubchenco/workspace/selfcheckgpt/tests/README.md`

**Phase 4:**
- Compatibility report: `/Users/nathanlubchenco/workspace/selfcheckgpt/agent-os/specs/2025-11-03-coherence-improvements/planning/simpleqa-compatibility-report.md`

**Phase 5 (Partial):**
- Educational notebook: `/Users/nathanlubchenco/workspace/selfcheckgpt/demo/understanding_coherence.ipynb`

**Key Files to Reference:**
- Coherence variants: `/Users/nathanlubchenco/workspace/selfcheckgpt/selfcheckgpt/modeling_coherence.py`
- Coherence API client: `/Users/nathanlubchenco/workspace/selfcheckgpt/selfcheckgpt/modeling_coherence_api.py`
- Coherence formulas: `/Users/nathanlubchenco/workspace/selfcheckgpt/selfcheckgpt/utils_coherence.py`
- Evaluation script: `/Users/nathanlubchenco/workspace/selfcheckgpt/scripts/evaluate.py`
- Existing demo: `/Users/nathanlubchenco/workspace/selfcheckgpt/demo/coherence_demo.ipynb`

## Phase 3 Completion Summary

**What Was Implemented:**

1. **Unit Tests for Coherence Formulas** (30 tests)
   - **Shogenji Tests (8)**: Independent events, positive correlation, mutual exclusion, epsilon smoothing, probability clamping, axiom violations, batch processing, input validation
   - **Fitelson Tests (7)**: Strong confirmation, disconfirmation, independence, P(¬E)=0 edge case, conditional consistency, score bounds, input validation
   - **Olsson Tests (7)**: Identical statements, disjoint statements, P(A∨B) calculation, division by zero, score bounds, axiom violations, batch processing
   - **Normalization Tests (8)**: Min-max normalization, identical scores, NaN handling, Inf handling, all NaN, output bounds, order preservation, single value
   - All tests pass in 0.36s with no errors

2. **Integration Tests for Coherence Variants** (18 tests)
   - **SelfCheckShogenji Tests (6)**: Basic predict() interface, varying sample counts (1, 3, 5), caching validation, full pipeline, empty inputs
   - **SelfCheckFitelson Tests (3)**: Basic predict() interface, conditional probability extraction, higher API call count verification
   - **SelfCheckOlsson Tests (3)**: Basic predict() interface, union probability calculation, overlap measure validation
   - **Cache & Cost Tests (3)**: Accurate statistics, persistence across sentences, API call estimation
   - **Error Handling Tests (3)**: Very long sentences (>1000 chars), special characters/unicode, newlines in passages
   - All tests pass in 80.03s (includes real OpenAI API calls)

3. **Test Infrastructure**
   - pytest configuration with custom markers (unit, integration, slow)
   - Warning filters for expected coherence calculation warnings
   - Comprehensive test documentation with usage examples
   - Fixtures for reusable test data
   - Conditional test skipping when API key unavailable

**How to Run Tests:**

```bash
# Run all tests
pytest

# Run only unit tests (no API calls)
pytest tests/test_coherence_formulas.py

# Run only integration tests (requires OPENAI_API_KEY)
pytest tests/test_coherence_integration.py

# Run with verbose output
pytest -v

# Run specific test
pytest tests/test_coherence_formulas.py::TestShogenjiFormula::test_independent_events_yield_c2_approx_1
```

**Test Coverage Summary:**
- Mathematical correctness: ✓ All three coherence formulas validated
- Known-outcome scenarios: ✓ Independent, correlated, exclusive events tested
- Edge cases: ✓ NaN, Inf, division by zero, axiom violations handled
- End-to-end pipelines: ✓ All three variants (Shogenji, Fitelson, Olsson) tested
- Caching mechanism: ✓ Reduces redundant API calls as expected
- Error handling: ✓ Long sentences, unicode, empty inputs handled gracefully
- Cost estimation: ✓ API call estimation accurate

## Phase 4 Completion Summary

**What Was Investigated:**

1. **SimpleQA Dataset Research**
   - Official sources: HuggingFace datasets (lighteval/SimpleQA, basicv8vc/SimpleQA, google/simpleqa-verified)
   - GitHub repository: openai/simple-evals
   - Dataset size: 4,326 examples (4,320 test + 6 few_shot)
   - Format: Parquet/CSV with metadata, problem, answer, split fields
   - License: MIT (publicly accessible)

2. **Compatibility Analysis Against Requirements**
   - **Requirement 1 (LLM Responses):** NOT MET - Only ground truth answers, no LLM-generated responses
   - **Requirement 2 (Stochastic Samples):** NOT MET - Single ground truth per question, no samples
   - **Requirement 3 (Sentence Structure):** NOT MET - Short-form answers (1-10 words)
   - **Requirement 4 (Multi-Sentence):** NOT MET - Fundamental mismatch with coherence approach

3. **Compatibility Determination**
   - **Final Assessment:** INCOMPATIBLE
   - **Critical Blockers:** 3 (missing responses, missing samples, insufficient length)
   - **Major Blockers:** 1 (single-word answers)
   - **Adaptation Cost:** $6,500+ (mostly labor) with questionable scientific value

4. **Alternative Datasets Evaluated**
   - **HaluEval:** Partial compatibility (has hallucinated responses but missing samples)
   - **TruthfulQA:** Incompatible (same issues as SimpleQA)
   - **FEVER:** Incompatible (different task - claim verification)
   - **Custom Dataset:** Recommended for future if expansion needed

**Recommendation:**
- Maintain wiki_bio_gpt3_hallucination as sole evaluation dataset
- SimpleQA's short-form factuality focus directly conflicts with coherence requirements
- Document dataset requirements for future evaluation opportunities

**Next Steps (Phase 2):**
The dataset investigation is now complete. The project can proceed with:
- **Phase 2**: Prompt optimization work (baseline evaluation, CoT, few-shot, axiom-aware, hybrid)
- All validation in Phase 5 will focus solely on wiki_bio_gpt3_hallucination
