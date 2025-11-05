# Task Breakdown: SelfCheckGPT Coherence Variants Improvements

## Overview
Total Task Groups: 5
Implementation Strategy: Sequential phases with some parallel work opportunities

**Core Goal:** Improve coherence-based hallucination detection through optimized probability extraction prompts and comprehensive verification infrastructure.

**Key Priorities:**
1. Benchmark creation FIRST (blocks all other work)
2. Prompt improvements SECOND (highest priority after benchmark)
3. Verification THIRD (parallel with prompts)
4. Dataset investigation FOURTH (parallel with verification)
5. Integration FINAL (depends on all previous phases)

## Task List

### Phase 1: Benchmark Foundation (Critical Path - Blocks All Other Work)

#### Task Group 1.1: Test Case Design and Creation
**Dependencies:** None

- [ ] 1.1.0 Complete probability extraction test suite
  - [ ] 1.1.1 Design test case structure and schema
    - Define test case format: statement text, ground truth probability, probability type (individual/joint/conditional), domain, statement type
    - Create JSON or CSV schema for storing test cases
    - Plan for version control and test case evolution
  - [ ] 1.1.2 Create individual probability test cases (25 cases)
    - Very low probability (P ∈ [0.0, 0.1]): 3 cases - clear contradictions across domains
    - Low probability (P ∈ [0.1, 0.3]): 3 cases - unlikely claims with some plausibility
    - Low-medium probability (P ∈ [0.3, 0.5]): 4 cases - ambiguous statements with uncertainty
    - Medium-high probability (P ∈ [0.5, 0.7]): 4 cases - generally accepted claims with some doubt
    - High probability (P ∈ [0.7, 0.9]): 5 cases - common sense and well-established facts
    - Very high probability (P ∈ [0.9, 1.0]): 6 cases - clear tautologies and definitional truths
  - [ ] 1.1.3 Create joint probability test cases (10 cases)
    - Independent statements: 3 cases where P(A∧B) ≈ P(A) × P(B)
    - Dependent statements: 3 cases where P(A∧B) > P(A) × P(B) (positive correlation)
    - Mutually exclusive statements: 2 cases where P(A∧B) ≈ 0
    - Partially overlapping statements: 2 cases with moderate joint probability
  - [ ] 1.1.4 Create conditional probability test cases (5 cases)
    - Strong causal relationships: 2 cases where P(A|B) >> P(A)
    - Weak/no relationships: 2 cases where P(A|B) ≈ P(A)
    - Inverse relationships: 1 case where P(A|B) < P(A)
  - [ ] 1.1.5 Add domain diversity across all test cases
    - Science/physics domain: 8 cases (gravity, thermodynamics, chemistry)
    - Geography domain: 8 cases (capital cities, terrain features, climate)
    - History domain: 8 cases (historical events, dates, figures)
    - Mathematics/logic domain: 8 cases (arithmetic facts, logical statements)
    - Common sense/everyday domain: 8 cases (daily activities, social norms)
  - [ ] 1.1.6 Create special edge case test cases (included in 40 total)
    - Axiom boundary cases: 2 cases with P=0 and P=1
    - Paraphrase consistency: 4 cases with semantically identical statements phrased differently
    - Adversarial/tricky statements: 4 cases designed to catch common LLM errors
  - [ ] 1.1.7 Document ground truth probabilities and rationale
    - For each test case, explain why the assigned probability is appropriate
    - Document expected probability relationships (e.g., P(A∧B) should be ≤ min(P(A), P(B)))
    - Create validation checklist for axiom compliance

**Acceptance Criteria:**
- 40 test cases total covering all specified ranges, types, and domains
- Each test case has documented ground truth probability with justification
- Test cases stored in structured format (JSON/CSV) for programmatic access
- Domain distribution balanced (8 cases per domain)
- Probability type distribution: ~25 individual, ~10 joint, ~5 conditional

---

#### Task Group 1.2: Evaluation Metrics Implementation
**Dependencies:** Task Group 1.1 (needs test case schema)

- [ ] 1.2.0 Complete evaluation metrics module
  - [ ] 1.2.1 Implement Brier Score calculator
    - Formula: BS = (1/N) × Σ(predicted_prob - actual_outcome)²
    - Handle binary outcomes (0 or 1) from ground truth
    - Return score in [0, 1] range where 0 is perfect calibration
    - Add unit tests with known-outcome examples
  - [ ] 1.2.2 Implement Expected Calibration Error (ECE) calculator
    - Bin probabilities into 10 ranges: [0.0-0.1], [0.1-0.2], ..., [0.9-1.0]
    - Calculate empirical frequency vs predicted probability per bin
    - Weight bins by number of samples
    - Return average absolute difference
    - Handle edge case: bins with zero samples
  - [ ] 1.2.3 Implement Probability Coherence Compliance checker
    - Verify Kolmogorov axioms: 0 ≤ P(A) ≤ 1 for all statements
    - Check joint probability consistency: P(A∧B) ≤ min(P(A), P(B))
    - Check conditional probability validity: P(A|B) = P(A∧B) / P(B) when P(B) > 0
    - Return compliance percentage and list of violations
    - Add warnings for near-violations (within epsilon tolerance)
  - [ ] 1.2.4 Implement Probability Consistency Score calculator
    - Identify semantically equivalent test cases (paraphrase pairs)
    - Calculate variance in probability estimates for equivalent statements
    - Return mean variance and maximum variance across all pairs
    - Lower scores = more consistent (better)
  - [ ] 1.2.5 Implement Sharpness metric calculator
    - Formula: (1/N) × Σ|predicted_prob - 0.5|
    - Measures decisiveness of probability predictions
    - Higher values indicate more confident (further from 0.5) predictions
    - Combine with calibration metrics for interpretation
  - [ ] 1.2.6 Create metrics summary report generator
    - Aggregate all 5 metrics into single report structure
    - Include per-metric scores and overall assessment
    - Generate human-readable summary with interpretations
    - Export to JSON and text formats

**Acceptance Criteria:**
- All 5 metrics implemented with correct mathematical formulas
- Unit tests pass for each metric with known inputs/outputs
- Metrics handle edge cases gracefully (NaN, Inf, empty inputs)
- Report generator produces interpretable output

---

#### Task Group 1.3: Benchmark Runner Infrastructure
**Dependencies:** Task Groups 1.1, 1.2

- [ ] 1.3.0 Complete benchmark execution framework
  - [ ] 1.3.1 Create benchmark runner main module
    - Load test cases from structured storage
    - Execute probability extraction for each test case
    - Collect predicted probabilities
    - Invoke all metric calculators
    - Reuse pattern from: scripts/evaluate_coherence.py
  - [ ] 1.3.2 Implement prompt variant configuration system
    - Define configuration schema for different prompt strategies
    - Support swapping prompt templates without code changes
    - Store prompt variants in separate configuration files
    - Enable command-line selection of prompt variant
  - [ ] 1.3.3 Add OpenAI API integration with caching
    - Reuse CoherenceAPIClient caching mechanism
    - Track cache hit rates during benchmark execution
    - Estimate API costs before running full benchmark
    - Implement retry logic for transient API failures
  - [ ] 1.3.4 Create comparative analysis module
    - Support running multiple prompt variants sequentially
    - Generate side-by-side metric comparisons
    - Calculate improvement percentages between variants
    - Identify which test cases show biggest differences
  - [ ] 1.3.5 Add progress tracking and logging
    - Use tqdm for progress bars (follow existing pattern)
    - Log API call counts and cache statistics
    - Track execution time per test case and total
    - Report estimated costs during execution
  - [ ] 1.3.6 Implement results export functionality
    - Save raw predictions for each prompt variant
    - Export metric summaries to JSON
    - Generate human-readable markdown reports
    - Include test case details with failures highlighted
  - [ ] 1.3.7 Run benchmark runner end-to-end test
    - Execute on small subset (5-10 test cases)
    - Verify all metrics calculate correctly
    - Confirm caching reduces API calls
    - Validate output formats are correct

**Acceptance Criteria:**
- Benchmark runner executes full 40-test suite successfully
- Comparative analysis produces interpretable side-by-side reports
- API costs tracked and estimated accurately
- Execution time under 5 minutes for full suite (with caching)
- Results exported in multiple formats (JSON, markdown)

---

### Phase 2: Prompt Strategy Development (High Priority - Depends on Phase 1)

#### Task Group 2.1: Baseline Prompt Evaluation
**Dependencies:** Task Group 1.3 (needs benchmark runner)

- [ ] 2.1.0 Establish baseline performance
  - [ ] 2.1.1 Document current simple prompt templates
    - Extract existing prompts from CoherenceAPIClient
    - Document individual_prob_template format
    - Document joint_prob_template format
    - Document conditional_prob_template format (if exists)
  - [ ] 2.1.2 Run baseline prompts through benchmark
    - Execute full 40-test-case suite with current prompts
    - Collect all metric scores
    - Identify specific failure cases
    - Document baseline Brier score, ECE, compliance rate, consistency, sharpness
  - [ ] 2.1.3 Analyze baseline weaknesses
    - Identify probability ranges with worst performance
    - Identify domains with highest error rates
    - Check for systematic axiom violations
    - Note test cases with largest prediction errors
  - [ ] 2.1.4 Document baseline performance report
    - Create comprehensive baseline report
    - Include example failures with explanations
    - Set improvement targets for new prompts
    - Establish baseline as reference point for all comparisons

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
    - Individual probability CoT: "Let's think step-by-step about the probability that this statement is true: {statement}. Consider: [reasoning steps]. What is the probability?"
    - Joint probability CoT: "Let's analyze the probability that both statements are true: {statement1} AND {statement2}. First consider each individually, then their relationship. What is P(A∧B)?"
    - Conditional probability CoT: "Let's reason about the probability of {statement1} GIVEN that {statement2} is true. How does the condition affect the probability?"
    - Include explicit reasoning steps in prompts
  - [ ] 2.2.2 Implement CoT variant in benchmark configuration
    - Create CoT prompt configuration file
    - Integrate with benchmark runner
    - Ensure structured output schema still works with longer prompts
  - [ ] 2.2.3 Run CoT prompts through benchmark
    - Execute full 40-test suite with CoT prompts
    - Track API costs (CoT prompts are longer → higher cost)
    - Collect all metric scores
    - Compare against baseline
  - [ ] 2.2.4 Analyze CoT performance
    - Calculate improvement percentages vs baseline
    - Identify which test case types benefit most from CoT
    - Check if CoT reduces axiom violations
    - Assess cost-performance tradeoff

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
    - Select 3-5 high-quality probability assessment examples
    - Cover different probability ranges (low, medium, high)
    - Include both simple and complex statements
    - Ensure examples demonstrate proper probability reasoning
  - [ ] 2.3.2 Design few-shot prompt templates
    - Individual probability few-shot: Include 3 examples before target statement
    - Joint probability few-shot: Include 2 examples of joint probability assessment
    - Conditional probability few-shot: Include 2 examples of conditional reasoning
    - Format examples clearly with input-output pairs
  - [ ] 2.3.3 Implement few-shot variant in benchmark configuration
    - Create few-shot prompt configuration file
    - Handle longer context from examples
    - Ensure examples don't bias specific test cases
  - [ ] 2.3.4 Run few-shot prompts through benchmark
    - Execute full 40-test suite
    - Monitor API costs (few-shot increases token usage)
    - Collect all metric scores
    - Compare against baseline and CoT
  - [ ] 2.3.5 Analyze few-shot performance
    - Calculate improvement percentages vs baseline
    - Test if few-shot improves consistency scores
    - Check for example bias in predictions
    - Evaluate cost-performance tradeoff vs CoT

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
    - Include Kolmogorov axioms: 0 ≤ P(A) ≤ 1, P(certain) = 1, P(impossible) = 0
    - Include joint probability constraint: P(A∧B) ≤ min(P(A), P(B))
    - Include conditional probability formula: P(A|B) = P(A∧B) / P(B)
    - Emphasize adherence to probability theory principles
    - Keep concise to avoid excessive token usage
  - [ ] 2.4.2 Integrate axiom-aware system prompt with user prompts
    - Combine system prompt with individual probability user prompts
    - Combine with joint probability user prompts
    - Combine with conditional probability user prompts
    - Test that structured output still works with system prompt
  - [ ] 2.4.3 Run axiom-aware prompts through benchmark
    - Execute full 40-test suite
    - Focus on measuring Probability Coherence Compliance improvement
    - Collect all metric scores
    - Compare against baseline, CoT, and few-shot
  - [ ] 2.4.4 Analyze axiom-aware performance
    - Calculate axiom violation reduction vs baseline
    - Check if calibration (Brier score) improves
    - Assess impact on consistency scores
    - Evaluate whether axiom education helps or confuses model

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
    - Identify strengths of each approach
    - Identify which test case types each approach handles best
    - Look for complementary benefits
    - Plan optimal combination strategy
  - [ ] 2.5.2 Design hybrid prompt templates
    - Combine axiom-aware system prompt (if effective) with user prompts
    - Add CoT reasoning steps (if they improve performance)
    - Include few-shot examples (if they boost consistency)
    - Balance comprehensiveness with token efficiency
  - [ ] 2.5.3 Implement hybrid variant in benchmark configuration
    - Create hybrid prompt configuration
    - Estimate token usage and API costs
    - Ensure structured output compatibility
  - [ ] 2.5.4 Run hybrid prompts through benchmark
    - Execute full 40-test suite
    - Compare against all previous variants
    - Track total API costs for hybrid approach
  - [ ] 2.5.5 Conduct final prompt variant comparison
    - Generate comprehensive comparison report: baseline vs CoT vs few-shot vs axiom-aware vs hybrid
    - Rank variants by each metric (Brier, ECE, compliance, consistency, sharpness)
    - Consider cost-performance tradeoffs
    - Recommend best variant for production integration

**Acceptance Criteria:**
- Hybrid prompt combines best elements from previous variants
- Full comparative analysis across all 5 prompt variants completed
- Clear recommendation for production integration
- Cost-benefit analysis included

---

### Phase 3: Verification and Validation (Medium Priority - Parallel with Phase 2)

#### Task Group 3.1: Coherence Formula Mathematical Verification
**Dependencies:** Task Group 1.3 (needs benchmark runner)

- [ ] 3.1.0 Validate coherence formula implementations
  - [ ] 3.1.1 Create unit tests for Shogenji's ratio-based measure
    - Test formula: C2(A,B) = P(A∧B) / (P(A) × P(B))
    - Known-outcome test: Independent events should yield C2 ≈ 1
    - Known-outcome test: Positive correlation should yield C2 > 1
    - Known-outcome test: Mutually exclusive events should yield C2 ≈ 0
    - Test epsilon smoothing prevents division by zero
    - Test probability clamping to valid ranges
    - Reference: utils_coherence.py coherence_shogenji() function
  - [ ] 3.1.2 Create unit tests for Fitelson's confirmation measure
    - Test formula: s(H,E) = P(H|E) - P(H|¬E)
    - Known-outcome test: Strong confirmation should yield s > 0
    - Known-outcome test: Strong disconfirmation should yield s < 0
    - Known-outcome test: Independence should yield s ≈ 0
    - Test handling of P(¬E) = 0 edge case
    - Test conditional probability consistency
    - Reference: utils_coherence.py coherence_fitelson() function
  - [ ] 3.1.3 Create unit tests for Olsson's overlap measure
    - Test formula: C1(A,B) = P(A∧B) / P(A∨B)
    - Known-outcome test: Identical statements should yield C1 ≈ 1
    - Known-outcome test: Disjoint statements should yield C1 ≈ 0
    - Test P(A∨B) = P(A) + P(B) - P(A∧B) calculation
    - Test division by zero handling when P(A∨B) ≈ 0
    - Reference: utils_coherence.py coherence_olsson() function
  - [ ] 3.1.4 Validate formulas against probability benchmark
    - Run benchmark test cases through each coherence formula
    - Verify no NaN or Inf values in outputs
    - Check that coherence scores fall in expected ranges
    - Identify any outliers or anomalies
  - [ ] 3.1.5 Test normalize_coherence_scores() function
    - Test min-max normalization: (scores - min) / (max - min)
    - Test edge case: all scores identical → should return 0.5
    - Test edge case: NaN/Inf handling
    - Verify inversion to hallucination scores: 1.0 - normalized
    - Reference: utils_coherence.py normalize_coherence_scores()
  - [ ] 3.1.6 Verify epsilon smoothing and warnings
    - Test that epsilon (1e-12) prevents division by zero
    - Verify warnings issued for physically impossible probabilities
    - Test P(A∧B) > P(A) detection and handling
    - Ensure epsilon doesn't distort results significantly

**Acceptance Criteria:**
- All unit tests pass for three coherence formulas
- Benchmark test cases produce valid coherence scores (no NaN/Inf)
- Edge cases handled correctly without crashes
- Normalization function works across all input ranges
- No mathematical errors detected in formula implementations

---

#### Task Group 3.2: Integration Testing with Coherence Variants
**Dependencies:** Task Groups 1.3, 3.1

- [ ] 3.2.0 Test end-to-end coherence variant pipelines
  - [ ] 3.2.1 Create integration test for SelfCheckShogenji
    - Test full predict() interface with sample sentences and passages
    - Verify probability extraction → coherence calculation → score normalization pipeline
    - Test with various numbers of sampled passages (1, 3, 5, 10)
    - Ensure caching works correctly across multiple sentences
    - Reference: modeling_coherence.py SelfCheckShogenji class
  - [ ] 3.2.2 Create integration test for SelfCheckFitelson
    - Test predict() with conditional probability extraction
    - Verify higher API call count (1 + 3*num_samples per sentence)
    - Test handling of P(A|¬B) extraction
    - Ensure Fitelson formula receives correct conditional probabilities
    - Reference: modeling_coherence.py SelfCheckFitelson class
  - [ ] 3.2.3 Create integration test for SelfCheckOlsson
    - Test predict() with union probability calculation
    - Verify P(A∨B) = P(A) + P(B) - P(A∧B) computation
    - Test overlap measure calculation
    - Ensure results correlate with expected hallucination patterns
    - Reference: modeling_coherence.py SelfCheckOlsson class
  - [ ] 3.2.4 Verify cache statistics and cost estimation
    - Test CoherenceAPIClient.get_cache_stats() returns accurate counts
    - Verify cache reduces redundant API calls
    - Test CoherenceAPIClient.estimate_api_calls() matches actual usage
    - Confirm cache persists across sentences within same session
  - [ ] 3.2.5 Test error handling and edge cases
    - Test with empty sentences list
    - Test with empty sampled_passages list
    - Test with very long sentences (>1000 chars)
    - Test with special characters and unicode
    - Verify graceful degradation on API failures

**Acceptance Criteria:**
- All three coherence variants pass integration tests
- predict() interface works correctly end-to-end
- Caching mechanism reduces API calls as expected
- Cost estimation utilities are accurate
- Edge cases handled gracefully without crashes

---

### Phase 4: Dataset Investigation (Medium Priority - Parallel with Phases 2-3)

#### Task Group 4.1: SimpleQA Dataset Analysis
**Dependencies:** None (independent research task)

- [ ] 4.1.0 Investigate SimpleQA dataset compatibility
  - [ ] 4.1.1 Research SimpleQA dataset structure
    - Find official SimpleQA dataset documentation or papers
    - Determine data format (JSON, CSV, HuggingFace dataset, etc.)
    - Identify schema: questions, answers, metadata fields
    - Check if dataset is publicly accessible
  - [ ] 4.1.2 Analyze SimpleQA compatibility requirements
    - Requirement 1: Does dataset include LLM-generated responses (not just ground truth)?
    - Requirement 2: Can we generate multiple stochastic samples per question?
    - Requirement 3: Are responses structured at sentence level or can be sentence-tokenized?
    - Requirement 4: Do responses have sufficient length for coherence analysis (multi-sentence)?
    - Compare against wiki_bio_gpt3_hallucination structure
  - [ ] 4.1.3 Assess data availability and licensing
    - Check dataset license (commercial use, research only, etc.)
    - Determine download mechanism and size
    - Verify dataset is maintained and up-to-date
  - [ ] 4.1.4 Create compatibility assessment report
    - Document dataset structure with examples
    - List compatibility criteria met vs unmet
    - If compatible: outline adaptation steps needed
    - If incompatible: explain specific blockers and suggest alternatives (TruthfulQA, HaluEval, FEVER)

**Acceptance Criteria:**
- Comprehensive understanding of SimpleQA dataset structure
- Clear determination of compatibility (yes/no with rationale)
- Documentation of adaptation requirements if compatible
- Alternative dataset recommendations if incompatible

---

#### Task Group 4.2: SimpleQA Integration (Conditional on 4.1 Compatibility)
**Dependencies:** Task Groups 4.1 (compatibility confirmed), 1.3 (benchmark runner exists)

- [ ] 4.2.0 Integrate SimpleQA as secondary evaluation dataset (ONLY IF COMPATIBLE)
  - [ ] 4.2.1 Create SimpleQA data loader
    - Download and cache SimpleQA dataset
    - Parse dataset into coherence-compatible format
    - Extract questions, LLM responses, ground truth answers
    - Apply sentence tokenization using spacy (follow existing pattern)
  - [ ] 4.2.2 Generate stochastic samples for SimpleQA
    - If samples don't exist: use OpenAI API to generate multiple responses per question
    - If samples exist: validate they meet quality requirements
    - Store samples for reuse to avoid regeneration costs
    - Document sample generation methodology
  - [ ] 4.2.3 Create ground truth hallucination labels
    - Compare LLM responses against SimpleQA ground truth
    - Label sentences as hallucinated (0) or non-hallucinated (1)
    - Handle partial hallucinations and ambiguous cases
    - Document labeling methodology
  - [ ] 4.2.4 Adapt evaluation script for SimpleQA
    - Extend scripts/evaluate_coherence.py to support SimpleQA
    - Implement dataset-specific loading logic
    - Maintain metric calculation compatibility (AUC-PR, PCC, AUC-ROC)
    - Follow existing evaluation pattern for consistency
  - [ ] 4.2.5 Run baseline coherence evaluation on SimpleQA
    - Execute all three coherence variants (Shogenji, Fitelson, Olsson)
    - Calculate AUC-PR, PCC, AUC-ROC metrics
    - Compare results with wiki_bio_gpt3_hallucination performance
    - Document any dataset-specific observations

**Acceptance Criteria:**
- SimpleQA dataset loaded and formatted for coherence analysis
- Stochastic samples generated or validated
- Ground truth hallucination labels created
- Evaluation script runs successfully on SimpleQA
- Baseline performance metrics documented

**Note:** This task group is conditional on Task 4.1 determining SimpleQA is compatible. If incompatible, skip this group and document alternatives.

---

### Phase 5: Production Integration (Final - Depends on Phases 2-4)

#### Task Group 5.1: Production Prompt Integration
**Dependencies:** Task Group 2.5 (best prompt variant selected)

- [ ] 5.1.0 Integrate improved prompts into production coherence variants
  - [ ] 5.1.1 Update CoherenceAPIClient prompt templates
    - Replace individual_prob_template with best-performing variant
    - Replace joint_prob_template with optimized version
    - Add conditional_prob_template improvements (for Fitelson)
    - Preserve existing structured output schema
    - Reference: modeling_coherence_api.py CoherenceAPIClient class
  - [ ] 5.1.2 Add backward compatibility configuration option
    - Create prompt_strategy parameter in CoherenceAPIClient
    - Support values: "simple" (legacy), "optimized" (new default), "cot", "few-shot", "axiom-aware", "hybrid"
    - Allow users to select prompt strategy via constructor
    - Default to best-performing strategy from benchmark
  - [ ] 5.1.3 Update SelfCheckShogenji with improved prompts
    - Ensure improved prompts work with Shogenji formula
    - Test that caching mechanism still functions
    - Verify API call counts match expectations
    - Reference: modeling_coherence.py SelfCheckShogenji class
  - [ ] 5.1.4 Update SelfCheckFitelson with improved prompts
    - Ensure conditional probability prompts improved
    - Test higher API call count handled correctly
    - Verify Fitelson confirmation measure calculations
    - Reference: modeling_coherence.py SelfCheckFitelson class
  - [ ] 5.1.5 Update SelfCheckOlsson with improved prompts
    - Ensure improved prompts work with Olsson overlap measure
    - Test union probability extraction
    - Verify coherence score calculations
    - Reference: modeling_coherence.py SelfCheckOlsson class
  - [ ] 5.1.6 Update module docstrings and documentation
    - Document prompt improvements in class docstrings
    - Add examples of different prompt strategies
    - Update coherence_variants.md with new prompt options
    - Document performance gains from benchmark

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
    - Execute full dataset evaluation with improved prompts
    - Calculate AUC-PR, PCC, AUC-ROC metrics
    - Compare against documented baseline performance
    - Quantify improvement percentages
    - Reference: scripts/evaluate_coherence.py
  - [ ] 5.2.2 Run improved Fitelson on wiki_bio_gpt3_hallucination
    - Execute with optimized conditional probability prompts
    - Track API costs (Fitelson has highest call count)
    - Calculate all metrics
    - Compare against baseline
  - [ ] 5.2.3 Run improved Olsson on wiki_bio_gpt3_hallucination
    - Execute with improved prompts
    - Calculate metrics
    - Compare against baseline
  - [ ] 5.2.4 Run on SimpleQA if compatible (conditional)
    - If Task 4.1 confirmed compatibility: execute all three variants on SimpleQA
    - Calculate metrics and compare with wiki_bio results
    - Document cross-dataset performance
    - If incompatible: skip and note in documentation
  - [ ] 5.2.5 Create comprehensive improvement report
    - Summarize metric improvements: baseline vs improved for each variant
    - Include improvement percentages for AUC-PR, PCC, AUC-ROC
    - Document which prompt strategy was selected and why
    - List known limitations and edge cases discovered
    - Provide cost analysis: API calls and estimated costs
  - [ ] 5.2.6 Document known limitations and failure modes
    - Identify test cases or scenarios where improvements are minimal
    - Document edge cases that still cause issues
    - Note any domains or statement types with persistent errors
    - Suggest future work directions

**Acceptance Criteria:**
- All three coherence variants evaluated on wiki_bio with improved prompts
- Quantified improvement vs baseline (target: >10% Brier score reduction)
- SimpleQA evaluation completed if compatible
- Comprehensive improvement report created
- Known limitations documented

---

#### Task Group 5.3: Documentation and Knowledge Transfer
**Dependencies:** Task Groups 5.1, 5.2

- [ ] 5.3.0 Create comprehensive documentation for improvements
  - [ ] 5.3.1 Update main README.md
    - Add section on prompt strategy selection
    - Document performance improvements from optimization
    - Update coherence variants examples with prompt options
    - Reference: README.md
  - [ ] 5.3.2 Update coherence_variants.md documentation
    - Document benchmark methodology and test cases
    - Explain prompt optimization process
    - Provide guidance on prompt strategy selection
    - Include metric interpretation guide
    - Reference: docs/coherence_variants.md
  - [ ] 5.3.3 Create probability extraction methodology guide
    - Explain evaluation metrics (Brier, ECE, compliance, consistency, sharpness)
    - Document how to use benchmark for future prompt testing
    - Provide best practices for probability extraction
    - Include troubleshooting guide for common issues
  - [ ] 5.3.4 Document benchmark usage and extension
    - Explain test case structure and how to add new cases
    - Document benchmark runner command-line interface
    - Provide examples of running comparative analyses
    - Guide for testing custom prompt variants
  - [ ] 5.3.5 Create demo notebook for benchmark exploration
    - Follow pattern from demo/coherence_demo.ipynb
    - Create demo/probability_benchmark_demo.ipynb
    - Include interactive examples of different prompt strategies
    - Show side-by-side comparisons with visualizations
    - Demonstrate metric calculations and interpretations
  - [ ] 5.3.6 Update CHANGELOG or release notes
    - Document new prompt optimization features
    - List performance improvements with metrics
    - Note backward compatibility maintenance
    - Mention new prompt_strategy configuration option

**Acceptance Criteria:**
- All documentation updated to reflect improvements
- Benchmark methodology thoroughly documented
- Demo notebook created for interactive exploration
- Clear guidance provided for users on prompt selection
- Release notes prepared

---

## Execution Order

**Critical Path:**
1. **Phase 1: Benchmark Foundation** (Task Groups 1.1 → 1.2 → 1.3)
   - Creates foundation for all subsequent work
   - Blocks Phases 2, 3, and 5

**High Priority (After Phase 1):**
2. **Phase 2: Prompt Strategy Development** (Task Groups 2.1 → 2.2/2.3/2.4 → 2.5)
   - Core improvement work
   - Task Groups 2.2, 2.3, 2.4 can run in parallel after 2.1
   - Task Group 2.5 depends on 2.2, 2.3, 2.4

**Medium Priority (Parallel Opportunities):**
3. **Phase 3: Verification** (Task Groups 3.1 → 3.2)
   - Can run in parallel with Phase 2 after Phase 1 completes

4. **Phase 4: Dataset Investigation** (Task Groups 4.1 → 4.2)
   - Task 4.1 can run anytime (no dependencies)
   - Task 4.2 depends on 4.1 and is conditional

**Final Integration:**
5. **Phase 5: Production Integration** (Task Groups 5.1 → 5.2 → 5.3)
   - Task 5.1 depends on Phase 2 completion
   - Task 5.2 depends on 5.1 and Phase 4
   - Task 5.3 depends on all previous work

**Recommended Parallel Execution:**
- After Phase 1 completes:
  - Start Phase 2 (Prompt Development) - highest priority
  - Start Phase 3 (Verification) - medium priority
  - Start Phase 4.1 (SimpleQA Investigation) - medium priority
- Once Phase 2 completes:
  - Start Phase 5 (Integration)

## Test Strategy

**Phase 1 Testing:**
- Validate test case ground truth probabilities manually
- Unit test each metric calculator with known inputs
- Integration test benchmark runner end-to-end

**Phase 2 Testing:**
- Run each prompt variant through full 40-test benchmark
- Compare metrics systematically
- Validate improvements are statistically significant

**Phase 3 Testing:**
- Unit test coherence formulas with known-outcome scenarios
- Integration test full predict() pipelines
- Verify no regressions in existing functionality

**Phase 4 Testing:**
- If SimpleQA compatible: validate data loading and evaluation scripts
- Test stochastic sample generation quality
- Verify metrics calculate correctly on new dataset

**Phase 5 Testing:**
- Regression testing: ensure backward compatibility maintained
- End-to-end validation on wiki_bio dataset
- Performance benchmarking: confirm improvements hold on full dataset

**Total Test Count Guidance:**
- Phase 1: ~15-20 unit tests for metrics + benchmark runner integration test
- Phase 2: 5 prompt variants × 40 test cases = 200 benchmark executions
- Phase 3: ~20-25 unit tests for formulas + 3 integration tests for variants
- Phase 4: Dataset-specific tests (if applicable)
- Phase 5: End-to-end validation + regression suite

**Note:** This project follows an iterative benchmark-driven approach rather than traditional TDD. Most validation occurs through the probability extraction benchmark rather than exhaustive unit tests.

## Implementation Notes

**Technology Stack:**
- Python 3.x
- OpenAI API (gpt-4o-mini recommended for cost efficiency)
- NumPy for numerical operations
- spacy for sentence tokenization
- sklearn for metric calculations (AUC-PR, AUC-ROC)
- scipy for statistical metrics (PCC)
- tqdm for progress tracking

**Key Files to Reference:**
- Coherence variants: `/Users/nathanlubchenco/workspace/selfcheckgpt/selfcheckgpt/modeling_coherence.py`
- Coherence API client: `/Users/nathanlubchenco/workspace/selfcheckgpt/selfcheckgpt/modeling_coherence_api.py`
- Coherence formulas: `/Users/nathanlubchenco/workspace/selfcheckgpt/selfcheckgpt/utils_coherence.py`
- Evaluation script: `/Users/nathanlubchenco/workspace/selfcheckgpt/scripts/evaluate_coherence.py`
- Existing demo: `/Users/nathanlubchenco/workspace/selfcheckgpt/demo/coherence_demo.ipynb`

**Reuse Opportunities:**
- Extend CoherenceAPIClient for new prompts (don't rebuild)
- Follow evaluate_coherence.py pattern for benchmark runner
- Reuse caching mechanism for cost efficiency
- Model demo notebook after coherence_demo.ipynb

**Cost Management:**
- Leverage existing LRU cache (10,000 max entries)
- Use CoherenceAPIClient.estimate_api_calls() before benchmarks
- Track cache hit rates during execution
- Consider using gpt-4o-mini for cost efficiency (current recommendation)

**Important Constraints:**
- Maintain backward compatibility with simple prompts
- Preserve predict() interface (no breaking changes)
- Keep caching mechanism functional with new prompts
- Ensure structured output schema works with all prompt variants
