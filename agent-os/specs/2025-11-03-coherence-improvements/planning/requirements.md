# Spec Requirements: SelfCheckGPT Coherence Variants Improvements

## Initial Description

The coherence variants for SelfCheckGPT were recently implemented, introducing three new hallucination detection methods based on formal coherence theory (Shogenji, Fitelson, Olsson). These variants use OpenAI's API to extract probabilities from statements and apply coherence formulas to detect hallucinations.

This spec focuses on improvements to make these variants more accurate, robust, and broadly applicable:

1. **Probability extraction prompts** - Current prompts are extremely basic. Explore improvements through prompt iteration and optimization, including chain of thought reasoning, few shot examples, and in-context learning about probability theory to avoid axiom violations.

2. **Probability extraction model** - Evaluate whether gpt-4.5-mini (or future models like gpt-5-mini/nano) perform better at probability extraction.

3. **Verifying implementation of coherence measures** - Add tests to confirm these are working as expected. Check for outliers and mathematical errors.

4. **Other datasets** - Currently tightly coupled to wiki_bio_gpt3_hallucination dataset. Extend to other datasets to verify coherence measures are broadly useful.

5. **Other models** - Evaluate different models (like gpt-4.5-mini) compared to baseline for coherence.

6. **General improvements** - Make the feature more performant, accessible, and explainable.

## Requirements Discussion

### First Round Questions & Answers

**Q1: For the probability extraction prompt improvements, what priority order would you like to tackle these in?**
- Option A: Start with prompt improvements (CoT, few-shot, axiom education), verify they work, then tackle model testing
- Option B: Start with model testing first to see if better models solve the problem
- Option C: Do both in parallel

**Answer:** Start with prompt improvements first (Option A). The workflow should be:
1. Improve prompts for probability extraction
2. Create verification/test suite to measure quality
3. Once prompts are solid, then test different models in the future

**Q2: For probability extraction verification, should we create a standalone benchmark with known-probability test cases?**

For example:
- Clear factual statements ("Water freezes at 0°C") → Should have high P(A)
- Clear contradictions ("The sky is green") → Should have low P(A)
- Mutually exclusive statements → P(A∧B) should be ~0
- Independent statements → P(A∧B) should ≈ P(A) × P(B)

Would you want 15-20 such test cases covering different probability scenarios?

**Answer:** YES - Create a comprehensive probability extraction benchmark with 30-50 test cases (user said "we'll probably need a few more than that"). Include:
- Clear facts
- Domain variety
- Linguistic structures
- Edge cases
- Adversarial cases
- Test BOTH individual P(A) and joint/conditional P(A∧B) probabilities

**Q3: For datasets, are you thinking of specific alternatives to wiki_bio_gpt3_hallucination?**

Some possibilities:
- TruthfulQA (adversarial questions)
- HaluEval (various hallucination types)
- SimpleQA (factuality benchmark)
- FEVER (fact verification)
- Custom domain-specific datasets

**Answer:**
- Keep wiki_bio_gpt3_hallucination as primary dataset
- Investigate SimpleQA compatibility - confirm its data structure works with the probability extraction approach
- If compatible, add SimpleQA as secondary dataset

**Q4: For coherence measure verification, should we focus on:**
- Unit tests for the mathematical formulas (ensuring Shogenji/Fitelson/Olsson compute correctly)
- Integration tests with known-outcome scenarios
- Outlier detection in existing results
- All of the above

**Answer:** Focus on creating a validation workflow through the probability extraction benchmark. The benchmark test cases will serve to verify:
- Mathematical formula correctness
- Known-outcome scenarios
- Outlier detection
The same test suite validates both prompts AND coherence formulas.

**Q5: For model evaluation (gpt-4.5-mini vs gpt-4o-mini), what metrics matter most?**
- Detection accuracy (AUC-PR on wiki_bio dataset)
- Probability calibration quality (are the probabilities well-calibrated?)
- API cost efficiency
- Inference speed
- All of the above with different weights

**Answer:** AVOID model testing for now. When they do test models in the future, prioritize:
- gpt-5-mini (future model)
- gpt-5-nano (future model)

Note: User mentions future OpenAI models that don't currently exist.

**Q6: Should we add visualization capabilities to make coherence scores more explainable?**

For example:
- Heatmaps showing sentence-level scores
- Probability comparison charts (P(A∧B) vs P(A)×P(B))
- Coherence formula breakdowns
- Confidence intervals

**Answer:** PASS for now - visualization is out of scope for this iteration.

**Q7: For prompt improvements, should we evaluate different prompting strategies independently?**

Suggested approach:
- Baseline: Current simple prompts
- CoT variant: "Let's think step by step about the probability..."
- Few-shot variant: Include 3-5 example probability assessments
- Axiom-aware variant: Include probability theory principles in system prompt
- Hybrid: Combine multiple strategies

Test each on the verification benchmark before deploying to full coherence detection.

**Answer:** YES - User confirms this workflow approach. Create independent evaluation of prompts using the verification benchmark before deploying to full hallucination detection.

**Q8: Are there specific aspects you want to exclude or defer to future work?**

**Answer:**
- Visualization: Out of scope
- Model testing: Defer to future (but note gpt-5-mini/nano as priorities when testing does happen)
- Proposition extraction: Note as future work (not in scope)

### Existing Code to Reference

**Similar Features Identified:** None provided by user. The coherence variants are new implementations without direct precedents in the codebase.

### Follow-up Questions

**Follow-up 1:** You mentioned Brier score for success metrics. What metrics would be most appropriate for evaluating probability extraction quality?

**Answer:** User is uncertain and wants suggestions for metrics to evaluate probability extraction quality. Brier score came to mind but they want a comprehensive proposal.

**Follow-up 2:** For the probability extraction benchmark, you said "we'll probably need a few more than that" regarding 15-20 test cases. Would 30-50 test cases be appropriate to cover:
- Range of probability values (low: 0.0-0.3, medium: 0.4-0.6, high: 0.7-1.0)
- Statement types (factual, ambiguous, contradictions, tautologies)
- Domain variety (science, geography, history, common sense)
- Joint vs conditional vs individual probabilities
- Axiom compliance scenarios

**Answer:** User confirmed that more test cases would be needed but didn't specify exact number. Implicit confirmation that 30-50 would be reasonable.

## Visual Assets

### Files Provided:
No visual assets provided.

### Visual Insights:
N/A - No visual assets to analyze.

## SimpleQA Dataset Compatibility Investigation

**Finding:** SimpleQA is not currently referenced anywhere in the SelfCheckGPT codebase. Based on general knowledge, SimpleQA is typically a factuality benchmark consisting of short questions and factual answers.

**Compatibility Assessment:**
- **Unknown** - No documentation exists in the codebase about SimpleQA structure
- **Action Required:** Need to investigate SimpleQA dataset structure to determine if it can be adapted for coherence-based hallucination detection
- **Compatibility Criteria:**
  - Must have LLM-generated responses (not just ground truth answers)
  - Must be able to generate multiple stochastic samples for each question
  - Must have sentence-level or statement-level structure suitable for probability extraction

**Recommendation:** Document as open question requiring investigation before implementation.

## Proposed Success Metrics for Probability Extraction Quality

Based on the user's mention of Brier score and the need to evaluate probability extraction independently, here are recommended metrics:

### 1. Brier Score (Calibration)
**What it measures:** Mean squared error between predicted probabilities and actual outcomes (0 or 1)
- Formula: BS = (1/N) × Σ(predicted_prob - actual_outcome)²
- Range: [0, 1] where 0 is perfect
- **Why appropriate:** Measures how well-calibrated the probability estimates are. Essential for coherence formulas that rely on accurate probability values.

### 2. Expected Calibration Error (ECE)
**What it measures:** Average absolute difference between predicted probabilities and empirical frequencies within probability bins
- Bins probabilities into ranges (0-0.1, 0.1-0.2, etc.)
- Compares predicted vs actual frequency in each bin
- **Why appropriate:** Detects systematic over/under-confidence in probability estimates. More interpretable than Brier score for identifying specific calibration issues.

### 3. Probability Coherence Compliance (Custom Metric)
**What it measures:** Adherence to probability axioms
- Kolmogorov axioms: 0 ≤ P(A) ≤ 1, P(certain) = 1, P(A∪B) = P(A) + P(B) for disjoint events
- Joint probability consistency: P(A∧B) ≤ min(P(A), P(B))
- Conditional probability consistency: P(A|B) = P(A∧B) / P(B) when P(B) > 0
- **Why appropriate:** Coherence formulas depend on mathematically valid probabilities. Violations lead to undefined or nonsensical coherence scores.

### 4. Probability Consistency Score (Variance)
**What it measures:** Consistency of probability estimates for semantically equivalent statements
- Test with paraphrased statements (should get similar P values)
- Test with logically equivalent formulations (should get identical P values)
- Measure variance across equivalent formulations
- **Why appropriate:** LLMs should assign similar probabilities to statements with identical meaning. High variance indicates unreliable extraction.

### 5. Sharpness (Confidence)
**What it measures:** How decisive the probability predictions are (distance from 0.5)
- Formula: (1/N) × Σ|predicted_prob - 0.5|
- Higher values = more confident predictions
- **Why appropriate:** Combined with calibration metrics, helps assess if the model is both confident AND accurate. Too much uncertainty (probabilities near 0.5) reduces coherence measure discriminability.

**Recommended Primary Metrics:**
1. **Brier Score** - Overall calibration quality
2. **ECE** - Specific calibration issues
3. **Probability Coherence Compliance** - Mathematical validity

**Recommended Secondary Metrics:**
4. **Probability Consistency Score** - Reliability
5. **Sharpness** - Confidence/discriminability

## Test Case Quantity Recommendation

**Recommended: 40 test cases**

**Rationale:**
- Covers 8 probability ranges × 5 statement types = 40 base cases
- Allows statistical significance in metric calculations
- Manageable for manual creation and validation
- Sufficient for detecting systematic biases

**Proposed Distribution:**

**By Probability Range (8 ranges):**
- Very low: P ∈ [0.0, 0.1] - 5 cases
- Low: P ∈ [0.1, 0.3] - 5 cases
- Low-medium: P ∈ [0.3, 0.5] - 5 cases
- Medium-high: P ∈ [0.5, 0.7] - 5 cases
- High: P ∈ [0.7, 0.9] - 5 cases
- Very high: P ∈ [0.9, 1.0] - 5 cases
- Joint probabilities: P(A∧B) - 5 cases
- Conditional probabilities: P(A|B) - 5 cases

**By Statement Type (mix across all):**
- Clear facts (high P): ~10 cases
- Common sense (medium-high P): ~8 cases
- Ambiguous statements (medium P): ~8 cases
- Unlikely claims (low P): ~8 cases
- Clear contradictions (very low P): ~6 cases

**By Domain (mix across all):**
- Science/physics: ~8 cases
- Geography: ~8 cases
- History: ~8 cases
- Mathematics/logic: ~8 cases
- Common sense/everyday: ~8 cases

**Probability Type Coverage:**
- Individual probabilities P(A): ~25 cases
- Joint probabilities P(A∧B): ~10 cases
- Conditional probabilities P(A|B): ~5 cases

**Special Test Cases (included in above):**
- Axiom edge cases (P=0, P=1, disjoint events): ~4 cases
- Paraphrase consistency tests: ~4 cases
- Adversarial/tricky statements: ~4 cases

## Updated Scope Summary

### IN SCOPE

**1. Probability Extraction Prompt Improvements**
- Design and test multiple prompting strategies:
  - Chain of thought (CoT) reasoning
  - Few-shot examples (3-5 exemplars)
  - Axiom-aware system prompts (probability theory education)
  - Hybrid approaches combining strategies
- Create standalone evaluation framework for prompt testing
- Implement best-performing prompt variants in coherence variants

**2. Probability Extraction Verification Benchmark**
- Create comprehensive test suite: **40 test cases**
- Coverage requirements:
  - 8 probability ranges (very low to very high, plus joint/conditional)
  - 5 statement types (facts, common sense, ambiguous, unlikely, contradictions)
  - 5 domains (science, geography, history, math, common sense)
  - Axiom compliance scenarios
  - Paraphrase consistency tests
  - Adversarial cases
- Implement evaluation metrics:
  - Primary: Brier score, ECE, Probability Coherence Compliance
  - Secondary: Probability Consistency Score, Sharpness
- Use benchmark to validate both prompts AND coherence formula implementations

**3. Coherence Measure Verification**
- Use probability benchmark to verify mathematical correctness of:
  - Shogenji's ratio-based measure
  - Fitelson's confirmation measure
  - Olsson's overlap measure
- Detect outliers and numerical instabilities
- Confirm edge case handling (axiom violations, numerical errors)

**4. Dataset Compatibility Investigation**
- Maintain wiki_bio_gpt3_hallucination as primary evaluation dataset
- **Investigate SimpleQA dataset:**
  - Analyze data structure and format
  - Assess compatibility with coherence approach (requires LLM responses + stochastic samples)
  - If compatible: integrate as secondary evaluation dataset
  - If incompatible: document why and suggest alternatives
- Document dataset requirements for future extensions

**5. Documentation & Explainability**
- Document prompt improvement methodology and results
- Create comprehensive benchmark documentation
- Explain metric choices and interpretations
- Provide guidance on probability extraction quality assessment

### OUT OF SCOPE

**1. Visualization Capabilities**
- Heatmaps, probability charts, coherence breakdowns
- Deferred to future iteration

**2. Model Testing/Comparison**
- Testing gpt-4.5-mini, gpt-5-mini, gpt-5-nano
- Comparative model evaluation
- **Note for future:** When model testing is added, prioritize gpt-5-mini and gpt-5-nano

**3. Proposition Extraction**
- Automatic extraction of propositions from long-form text
- Breaking down complex sentences into atomic claims
- Deferred to future work

### FUTURE WORK

**1. Advanced Model Evaluation**
- Compare OpenAI model generations (gpt-5-mini, gpt-5-nano as priorities)
- Cost-performance tradeoffs across models
- Model-specific prompt optimization

**2. Visualization & User Interface**
- Interactive probability exploration
- Coherence score explanations
- Debugging tools for probability extraction

**3. Proposition Extraction Pipeline**
- Sentence decomposition into atomic propositions
- Enhanced granularity for coherence detection
- Integration with existing coherence variants

**4. Additional Datasets**
- Expand beyond wiki_bio and SimpleQA
- Domain-specific benchmarks
- Multi-lingual coherence detection

**5. Advanced Coherence Measures**
- Explore additional coherence formulas from epistemology literature
- Weighted coherence measures
- Hierarchical coherence assessment

## Clear Deliverables

### Phase 1: Benchmark Creation (Foundation)
1. **Probability Extraction Test Suite** (40 test cases)
   - Structured dataset with ground truth probabilities
   - Diverse coverage (ranges, types, domains)
   - Edge cases and adversarial examples

2. **Evaluation Metrics Implementation**
   - Brier score calculator
   - Expected Calibration Error (ECE) calculator
   - Probability Coherence Compliance checker
   - Probability Consistency Score calculator
   - Sharpness metric

3. **Benchmark Runner**
   - Execute test suite against any prompt variant
   - Generate comprehensive metric reports
   - Compare multiple prompt strategies

### Phase 2: Prompt Optimization
1. **Prompt Variants Implementation**
   - Baseline (current simple prompts)
   - Chain of Thought (CoT) variant
   - Few-shot variant (3-5 examples)
   - Axiom-aware variant (probability theory in system prompt)
   - Hybrid variant (best combination)

2. **Prompt Evaluation Results**
   - Metric scores for each prompt variant
   - Comparative analysis document
   - Recommended prompt strategy

3. **Production Prompt Updates**
   - Integrate best-performing prompts into coherence variants
   - Update documentation with new prompts
   - Maintain backward compatibility option

### Phase 3: Verification & Validation
1. **Coherence Formula Verification**
   - Unit tests for Shogenji, Fitelson, Olsson formulas
   - Validation against benchmark test cases
   - Outlier detection and analysis

2. **SimpleQA Investigation Report**
   - Dataset structure analysis
   - Compatibility assessment
   - Integration plan (if compatible) OR alternative recommendations (if incompatible)

3. **Updated Documentation**
   - Probability extraction methodology guide
   - Benchmark usage instructions
   - Metric interpretation guide
   - Best practices for prompt selection

### Phase 4: Integration & Testing
1. **End-to-End Validation**
   - Run improved prompts on wiki_bio_gpt3_hallucination
   - Compare AUC-PR and PCC against baseline
   - Run on SimpleQA (if compatible)

2. **Performance Documentation**
   - Benchmark results summary
   - Improvement quantification
   - Known limitations and edge cases

3. **Release Artifacts**
   - Updated coherence variants with improved prompts
   - Probability extraction benchmark (shareable test suite)
   - Comprehensive improvement report

## Requirements Summary

### Functional Requirements

**Probability Extraction Improvements:**
- Support multiple prompting strategies (CoT, few-shot, axiom-aware, hybrid)
- Provide configurable prompt templates for coherence variants
- Maintain backward compatibility with existing simple prompts

**Verification & Benchmarking:**
- Standalone benchmark for probability extraction quality (40 test cases)
- Support for testing individual P(A), joint P(A∧B), and conditional P(A|B) probabilities
- Comprehensive metric evaluation (Brier, ECE, coherence compliance, consistency, sharpness)
- Automated comparison of prompt strategies

**Coherence Formula Validation:**
- Unit tests for mathematical correctness of all three coherence formulas
- Edge case handling verification
- Outlier detection capabilities

**Dataset Expansion:**
- Maintain compatibility with wiki_bio_gpt3_hallucination
- Investigate and potentially integrate SimpleQA dataset
- Document requirements for future dataset additions

### Technical Considerations

**Integration Points:**
- Existing coherence variants (SelfCheckShogenji, SelfCheckFitelson, SelfCheckOlsson)
- OpenAI API integration in CoherenceAPIClient
- Probability extraction prompt system
- Caching mechanism for API calls

**Existing System Constraints:**
- Must use OpenAI API for probability extraction (structured output feature)
- Must maintain caching functionality for cost efficiency
- Should preserve existing API call estimation utilities
- Must work with current sentence tokenization (spacy)

**Technology Preferences:**
- Python 3.x
- OpenAI API (current implementation)
- NumPy for numerical operations
- Existing testing framework (Jupyter notebooks → transition to pytest)

**Similar Code Patterns:**
- Existing coherence variants provide template for implementation patterns
- MQAG scoring variants show precedent for multiple strategy evaluation
- Existing demo notebooks show evaluation methodology patterns

### Reusability Opportunities

**Components that exist:**
- CoherenceAPIClient with caching (can extend for new prompts)
- utils_coherence.py formulas (verify, don't rebuild)
- demo/coherence_demo.ipynb pattern (can create benchmark_demo.ipynb)

**Backend patterns to reference:**
- API caching strategy from modeling_coherence_api.py
- Multiple scoring methodologies from MQAG (similar to multiple prompt strategies)
- Cost estimation utilities (extend for benchmark)

**Similar features to model after:**
- MQAG's multiple scoring methods (counting, bayes, bayes_with_alpha) → analogous to multiple prompt strategies
- Existing predict() interface pattern → maintain for consistency
- Demo notebook structure → create similar for benchmark

## Scope Boundaries

### What WILL Be Built

1. **Probability Extraction Benchmark System**
   - 40 carefully designed test cases
   - 5 evaluation metrics
   - Automated benchmark runner
   - Comparative analysis tools

2. **Improved Probability Extraction Prompts**
   - 4-5 prompt strategy variants
   - Independent evaluation of each
   - Production integration of best performer

3. **Verification Suite**
   - Mathematical validation of coherence formulas
   - Outlier detection
   - Edge case testing

4. **Dataset Investigation**
   - SimpleQA compatibility analysis
   - Integration if compatible
   - Documentation of dataset requirements

### What WON'T Be Built

1. **Visualization Tools**
   - No heatmaps or charts
   - No interactive probability exploration
   - No debugging UI

2. **Model Comparison System**
   - No gpt-4.5-mini vs gpt-4o-mini testing
   - No infrastructure for model evaluation
   - (But document gpt-5-mini/nano as future priorities)

3. **Proposition Extraction**
   - No automatic sentence decomposition
   - No atomic claim extraction
   - Defer to future work

### Open Questions

1. **SimpleQA Dataset Structure**
   - Need to investigate actual data format
   - Determine if it includes LLM-generated responses and stochastic samples
   - Assess if sentence-level annotation is possible

2. **Optimal Test Case Count**
   - Proposed 40 test cases - confirm this is sufficient
   - May adjust based on initial benchmark creation experience

3. **Metric Weighting**
   - Should metrics be weighted/combined into single score?
   - Or keep separate for different purposes?

4. **Prompt Strategy Combinations**
   - Which hybrid approaches should be tested?
   - CoT + few-shot? CoT + axiom-aware? All three?

5. **Backward Compatibility**
   - How to handle users who want to keep simple prompts?
   - Configuration flag? Separate classes? Version parameter?

## Implementation Priority

**Phase 1 (Critical Path):** Benchmark Creation
- Build the 40-test-case probability extraction benchmark
- Implement 5 evaluation metrics
- Create benchmark runner infrastructure
→ **Blocks all other work**

**Phase 2 (High Priority):** Prompt Improvement
- Develop 4-5 prompt strategy variants
- Evaluate each on benchmark
- Select best performer
→ **Depends on Phase 1**

**Phase 3 (Medium Priority):** Verification
- Validate coherence formula implementations
- Run benchmark against formulas
- Detect and fix any issues
→ **Can proceed in parallel with Phase 2**

**Phase 4 (Medium Priority):** Dataset Investigation
- Investigate SimpleQA structure
- Assess compatibility
- Plan integration or alternatives
→ **Can proceed in parallel with Phases 2-3**

**Phase 5 (Final):** Integration
- Integrate best prompts into production
- Run end-to-end validation
- Document results and improvements
→ **Depends on Phases 2-4**

## Success Criteria

This improvement initiative will be considered successful when:

1. **Benchmark Quality**
   - 40 high-quality test cases created with ground truth probabilities
   - 5 evaluation metrics implemented and validated
   - Benchmark runner executes reliably and produces interpretable reports

2. **Prompt Improvement**
   - At least 3 prompt strategies evaluated beyond baseline
   - Best-performing strategy shows measurable improvement on benchmark metrics
   - Improved prompts integrated into production coherence variants

3. **Mathematical Verification**
   - All three coherence formulas validated against benchmark
   - No mathematical errors detected
   - Edge cases handled correctly

4. **Dataset Expansion**
   - SimpleQA compatibility determined with documented analysis
   - If compatible: integration path defined
   - If incompatible: alternative datasets suggested

5. **Documentation Quality**
   - Comprehensive methodology documentation
   - Clear metric interpretation guides
   - Best practices for probability extraction

**Quantitative Targets:**
- Brier score improvement: >10% reduction vs baseline prompts
- ECE improvement: >15% reduction vs baseline prompts
- Probability Coherence Compliance: >95% adherence to axioms
- Benchmark execution time: <5 minutes for full suite

**Qualitative Targets:**
- Improved explainability of probability extraction process
- Increased confidence in coherence measure validity
- Foundation for future model and dataset expansion
