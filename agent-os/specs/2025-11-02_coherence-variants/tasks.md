# Task Breakdown: Coherence-Based Hallucination Detection Variants

## Overview
Total Task Groups: 5
Estimated Development Time: Medium-Large (research + implementation)
Architecture: Hybrid (OpenAI API + NumPy/SciPy for coherence formulas)

## Task List

### Research & Theoretical Foundation

#### Task Group 1: Coherence Theory Research
**Dependencies:** None
**Effort:** Medium (3-4 hours research + documentation)
**Can be parallelized:** Yes (can happen alongside other initial setup)

- [x] 1.0 Complete theoretical research foundation
  - [x] 1.1 Research and locate academic sources for coherence theories
    - Find Shogenji's coherence measure paper/source
    - Find Fitelson's confirmation measure paper/source
    - Find Olsson's support-based measure paper/source
    - Identify canonical formulas with mathematical notation
  - [x] 1.2 Download and store coherence theory sources
    - Create `/Users/nathanlubchenco/workspace/selfcheckgpt/agent-os/specs/2025-11-02_coherence-variants/planning/theory-sources/` directory
    - Download PDFs or save URLs for all three coherence theories
    - Create `theory-sources/README.md` documenting each source and its key formulas
  - [x] 1.3 Extract canonical mathematical formulas
    - Document Shogenji formula: C(A,B) = P(A ∧ B) / (P(A) × P(B))
    - Document Fitelson formula with conditional probability structure
    - Document Olsson formula with support-based structure
    - Note any formula variations or parameters mentioned in sources
  - [x] 1.4 Create theory reference document
    - Write `planning/coherence-theory-reference.md` with formulas and citations
    - Include notes on theoretical motivation for each measure
    - Document expected coherence score ranges and interpretations

**Acceptance Criteria:**
- All three coherence theory sources located and stored
- Mathematical formulas documented with proper notation
- Reference document created for implementation guidance
- No implementation conflicts with theoretical foundations

---

### Shared Infrastructure Layer

#### Task Group 2: API Client and Probability Extraction Infrastructure
**Dependencies:** None (can run in parallel with Task Group 1)
**Effort:** Medium (4-5 hours)

- [ ] 2.0 Complete shared API client infrastructure
  - [ ] 2.1 Create `selfcheckgpt/modeling_coherence_api.py` with CoherenceAPIClient base class
    - Initialize OpenAI client with api_key (defaults to OPENAI_API_KEY environment variable)
    - Follow pattern from `SelfCheckAPIPrompt` in `modeling_selfcheck_apiprompt.py`
    - Print initialization message with model name
  - [ ] 2.2 Implement deterministic completion method with structured output
    - Create `completion(prompt: str) -> float` method that returns probability value
    - Set temperature=0.0 for deterministic responses
    - Use OpenAI's structured output feature with JSON schema for reliable probability extraction
    - Define response_format with JSON schema: `{"probability": float}` with constraints [0.0, 1.0]
    - Parse JSON response and extract probability value
    - Set max_tokens=20 for concise responses
  - [ ] 2.3 Implement prompt response caching mechanism
    - Create dictionary cache keyed by (prompt_text, model_name) tuple
    - Implement cache lookup before API calls
    - Implement cache storage after successful API calls
    - Add optional LRU cache with max_size=10000 (use functools.lru_cache or manual implementation)
  - [ ] 2.4 Add retry logic and error handling
    - Implement retry decorator with exponential backoff (max 3 retries)
    - Handle rate limit errors with clear messages
    - Handle authentication failures with actionable error messages
    - Log warnings for retries and cache misses
  - [ ] 2.5 Implement probability extraction prompt templates
    - Create prompt template for individual probability: "Rate the probability that this statement is true: {statement}"
    - Create prompt template for joint probability: "Rate the probability that both statements are true: {statement1} AND {statement2}"
    - Create prompt template for conditional probability: "Rate the probability that statement A is true: {statement1} GIVEN that {statement2} is true"
    - All prompts use structured output, so no need to specify "[0.0-1.0]" in prompt text
  - [ ] 2.7 Add cache statistics and cost estimation utilities
    - Implement `get_cache_stats()` returning hit rate and size
    - Implement `estimate_api_calls(num_sentences, num_samples, num_variants)` for cost estimation
    - Log cache hit rates when verbose=True

**Acceptance Criteria:**
- CoherenceAPIClient initializes with OpenAI client
- Deterministic completions work with temperature=0.0
- Structured output (JSON schema) ensures reliable probability extraction
- Caching mechanism reduces duplicate API calls
- Retry logic handles transient failures gracefully
- All probability values are guaranteed to be valid floats in [0.0, 1.0] range

---

### Coherence Formula Implementations

#### Task Group 3: Mathematical Coherence Measures
**Dependencies:** Task Group 1 (theory research)
**Effort:** Medium (3-4 hours)

- [ ] 3.0 Complete coherence formula implementations
  - [ ] 3.1 Create `selfcheckgpt/utils_coherence.py` module
    - Import numpy and scipy as needed
    - Add module docstring explaining coherence measures
    - Include references to theory sources from Task 1.2
  - [ ] 3.2 Implement Shogenji coherence measure
    - Function signature: `coherence_shogenji(probs_individual: np.ndarray, probs_joint: np.ndarray, epsilon: float = 1e-12) -> np.ndarray`
    - Formula: C(A,B) = P(A ∧ B) / (P(A) × P(B))
    - Add epsilon smoothing to denominators to prevent division by zero
    - Input: probs_individual shape (n, 2) for [P(A), P(B)], probs_joint shape (n,) for P(A∧B)
    - Output: coherence scores shape (n,)
    - Add docstring with formula notation and references
  - [ ] 3.3 Implement Fitelson confirmation measure
    - Function signature: `coherence_fitelson(probs_individual: np.ndarray, probs_joint: np.ndarray, probs_conditional: np.ndarray, epsilon: float = 1e-12) -> np.ndarray`
    - Formula: Based on difference between P(A|B) and P(A|¬B) (verify exact formula from Task 1.3)
    - Add epsilon smoothing for numerical stability
    - Input: probs_individual, probs_joint, probs_conditional arrays
    - Output: confirmation coherence scores
    - Add docstring with formula and theoretical motivation
  - [ ] 3.4 Implement Olsson support-based measure
    - Function signature: `coherence_olsson(probs_individual: np.ndarray, probs_joint: np.ndarray, epsilon: float = 1e-12) -> np.ndarray`
    - Formula: Support-based measure analyzing mutual justification (verify from Task 1.3)
    - Add epsilon smoothing
    - Input/output matching other coherence functions
    - Add docstring with formula
  - [ ] 3.5 Add numerical stability handling
    - Implement epsilon=1e-12 smoothing consistently across all formulas
    - Clamp intermediate probability calculations to [epsilon, 1.0-epsilon] range
    - Handle edge cases: P=0, P=1, P(A∧B) > P(A) or P(B) (numerical errors)
    - Add assertions or warnings for physically impossible probabilities
  - [ ] 3.6 Add utility function for score normalization
    - Function signature: `normalize_coherence_scores(scores: np.ndarray) -> np.ndarray`
    - Implement min-max normalization to [0.0, 1.0] range
    - Handle case where all scores are identical (return 0.5)
    - Return normalized scores as numpy array

**Acceptance Criteria:**
- All three coherence formulas implemented matching theory sources
- Numerical stability with epsilon smoothing prevents division by zero
- Functions accept numpy arrays and return numpy arrays
- Score normalization handles edge cases correctly
- Docstrings include mathematical notation and references

---

### Detection Variant Implementations

#### Task Group 4: SelfCheck Coherence Variants
**Dependencies:** Task Groups 2 (API client) and 3 (formulas)
**Effort:** Large (6-8 hours for all three variants)

- [ ] 4.0 Complete all three coherence detection variants
  - [ ] 4.1 Create `selfcheckgpt/modeling_coherence.py` module
    - Import CoherenceAPIClient from modeling_coherence_api
    - Import coherence formulas from utils_coherence
    - Import numpy, tqdm for progress tracking
  - [ ] 4.2 Implement SelfCheckShogenji class
    - Class signature: `SelfCheckShogenji(model="gpt-4o-mini", api_key=None)`
    - Initialize CoherenceAPIClient in `__init__`
    - Store model and api_key as instance attributes
  - [ ] 4.3 Implement SelfCheckShogenji.predict() method
    - Method signature: `predict(sentences: List[str], sampled_passages: List[str], verbose: bool = False) -> np.ndarray`
    - For each sentence, extract P(sentence) using individual probability prompt
    - For each sentence-sample pair, extract P(sentence), P(sample), P(sentence ∧ sample) using joint probability prompt
    - Calculate Shogenji coherence for each sentence-sample pair using `coherence_shogenji()`
    - Aggregate coherence scores across samples using mean (axis=-1)
    - Normalize aggregated scores to [0.0, 1.0] using min-max normalization
    - Invert normalized scores: hallucination_score = 1.0 - normalized_coherence
    - Return sentence-level hallucination scores as np.ndarray shape (num_sentences,)
    - Add tqdm progress bar when verbose=True
  - [ ] 4.4 Implement SelfCheckFitelson class
    - Class signature matching SelfCheckShogenji
    - Initialize CoherenceAPIClient
    - Follow same initialization pattern
  - [ ] 4.5 Implement SelfCheckFitelson.predict() method
    - Method signature matching SelfCheckShogenji.predict()
    - Extract individual probabilities: P(sentence), P(sample)
    - Extract joint probability: P(sentence ∧ sample)
    - Extract conditional probability: P(sentence | sample) using conditional prompt
    - Calculate Fitelson coherence using `coherence_fitelson()`
    - Handle cases where conditional probability is unreliable (fall back to joint/individual approximation)
    - Aggregate, normalize, invert to hallucination scores
    - Return sentence-level scores with same shape as other variants
  - [ ] 4.6 Implement SelfCheckOlsson class
    - Class signature matching other variants
    - Initialize CoherenceAPIClient
  - [ ] 4.7 Implement SelfCheckOlsson.predict() method
    - Method signature matching other variants
    - Extract probabilities for support relationship assessment
    - Calculate Olsson coherence using `coherence_olsson()`
    - Aggregate, normalize, invert to hallucination scores
    - Ensure output shape and range match other variants for drop-in replacement
  - [ ] 4.8 Ensure consistent behavior across all variants
    - All variants return np.ndarray shape (num_sentences,)
    - All scores in [0.0, 1.0] range with higher = more hallucination
    - All variants show tqdm progress when verbose=True
    - All variants use same caching and retry logic via CoherenceAPIClient
    - API call patterns are efficient (minimize redundant probability extractions where possible)

**Acceptance Criteria:**
- All three coherence variants (Shogenji, Fitelson, Olsson) implemented in modeling_coherence.py
- All variants follow same interface: `predict(sentences, sampled_passages, verbose=False)`
- Coherence scores properly inverted to hallucination scores (higher = more hallucination)
- Progress bars display when verbose=True
- Output format matches existing SelfCheck variants for compatibility
- Caching reduces API costs for repeated evaluations

---

### Configuration and Utilities

#### Task Group 5: Configuration Management
**Dependencies:** Task Groups 2-4
**Effort:** Small (1-2 hours)

- [ ] 5.0 Complete configuration and constants
  - [ ] 5.1 Add CoherenceConfig class to `selfcheckgpt/utils.py`
    - Define default OpenAI model: "gpt-4o-mini"
    - Store JSON schema for structured probability output
    - Store epsilon constant: 1e-12
    - Store max_tokens for probability extraction: 20
    - Follow pattern of MQAGConfig and NLIConfig in utils.py
  - [ ] 5.2 Add probability extraction prompt templates to CoherenceConfig
    - individual_prob_template: "Rate the probability that this statement is true: {statement}"
    - joint_prob_template: "Rate the probability that both statements are true: {statement1} AND {statement2}"
    - conditional_prob_template: "Rate the probability that statement A is true: {statement1} GIVEN that {statement2} is true"
    - Store as class attributes for easy customization
    - Note: Structured output JSON schema handles format, not prompt text
  - [ ] 5.3 Add numerical constants for score normalization
    - normalization_epsilon: 1e-12 (for preventing division by zero in normalization)
    - score_bounds: (0.0, 1.0) (output hallucination score range)
    - Note: No need for default_unparseable_prob due to structured output
  - [ ] 5.4 Update module exports in `selfcheckgpt/__init__.py`
    - Add imports for SelfCheckShogenji, SelfCheckFitelson, SelfCheckOlsson
    - Add import for CoherenceAPIClient if exposing publicly
    - Maintain alphabetical ordering with existing exports
    - Update __all__ list if present

**Acceptance Criteria:**
- CoherenceConfig class added to utils.py with all defaults
- Prompt templates easily customizable via config
- Numerical constants centralized for consistency
- New coherence variants importable from selfcheckgpt package

---

### Validation & Evaluation

#### Task Group 6: Interactive Validation (No Formal Tests Required)
**Dependencies:** Task Groups 1-5 (all implementation complete)
**Effort:** Medium (3-4 hours)

- [ ] 6.0 Complete validation and evaluation infrastructure
  - [ ] 6.1 Create `demo/coherence_demo.ipynb` Jupyter notebook for interactive testing
    - Import all three coherence variants
    - Create simple test examples with known properties (high/low coherence cases)
    - Manually inspect probability extraction for sanity
    - Visualize coherence scores vs hallucination scores for sample data
    - Test with small subset of wiki_bio_gpt3_hallucination dataset (5-10 passages)
  - [ ] 6.2 Create `scripts/evaluate_coherence.py` standalone evaluation script
    - Load wiki_bio_gpt3_hallucination dataset using `datasets.load_dataset("potsawee/wiki_bio_gpt3_hallucination")['evaluation']`
    - Implement evaluation loop for all three coherence variants
    - Calculate AUC-PR using sklearn.metrics.average_precision_score
    - Calculate PCC (Pearson) using scipy.stats.pearsonr
    - Calculate AUC-ROC using sklearn.metrics.roc_auc_score
    - Support command-line arguments: --variant (shogenji/fitelson/olsson/all), --model, --num-samples
  - [ ] 6.3 Implement results output and comparison
    - Save results to JSON file: `results/coherence_evaluation_{timestamp}.json`
    - JSON structure: {variant_name: {auc_pr: float, pcc: float, auc_roc: float}}
    - Print comparison table showing all variants vs existing best (93.42 AUC-PR baseline)
    - Include cache statistics and estimated API costs in output
  - [ ] 6.4 Create `demo/coherence_evaluation.ipynb` for visualization
    - Load evaluation results JSON
    - Plot ROC curves for all three variants
    - Plot Precision-Recall curves for all three variants
    - Plot hallucination score distributions (histograms)
    - Show per-sentence analysis for interesting cases (high/low coherence disagreements)
    - Compare against existing SelfCheckAPIPrompt baseline
  - [ ] 6.5 Add cost estimation and logging
    - Log total API calls made during evaluation
    - Log cache hit rate and savings
    - Estimate total cost based on OpenAI API pricing (gpt-4o-mini pricing)
    - Print cost summary at end of evaluation
  - [ ] 6.6 Run validation experiments
    - Test each variant on small subset (10 passages) and inspect outputs manually
    - Verify coherence scores are in expected ranges
    - Verify hallucination scores correlate with ground truth labels
    - Identify any implementation issues or unexpected behaviors
    - Document findings in notebook or markdown file

**Acceptance Criteria:**
- Interactive demo notebook allows manual inspection of coherence detection
- Evaluation script computes AUC-PR, PCC, and AUC-ROC metrics
- Results saved to JSON with comparison to baseline
- Visualization notebook shows ROC/PR curves and score distributions
- Cost estimation provides transparency for API usage
- Manual validation confirms variants behave as expected on sample data

---

### Documentation & Integration

#### Task Group 7: Documentation and Package Integration
**Dependencies:** Task Group 6 (validation complete)
**Effort:** Small-Medium (2-3 hours)

- [ ] 7.0 Complete documentation and integration
  - [ ] 7.1 Update main README.md with coherence variants
    - Add section "Coherence-Based Detection" after existing variants
    - Include usage examples for SelfCheckShogenji, SelfCheckFitelson, SelfCheckOlsson
    - Document required API key (OPENAI_API_KEY environment variable)
    - Show example of probability extraction caching and structured output
  - [ ] 7.2 Create coherence-specific documentation
    - Write `docs/coherence_variants.md` with detailed explanation
    - Include mathematical formulas for each coherence measure
    - Explain theoretical motivation (why coherence measures detect hallucinations)
    - Document prompt templates and customization options
    - Include performance benchmarks from Task 6.2 evaluation
  - [ ] 7.3 Add docstrings to all new code
    - Complete docstrings for CoherenceAPIClient class and methods
    - Complete docstrings for coherence formula functions in utils_coherence.py
    - Complete docstrings for all three SelfCheck variant classes
    - Follow Google/NumPy docstring format matching existing codebase
    - Include parameter types, return types, and usage examples
  - [ ] 7.4 Update CLAUDE.md project instructions
    - Add coherence variants to architecture section
    - Document new modules: modeling_coherence_api.py, modeling_coherence.py, utils_coherence.py
    - Add coherence variants to detection strategies list
    - Include API cost management notes (caching, estimation)
    - Update module structure diagram
  - [ ] 7.5 Create minimal usage example script
    - Write `examples/coherence_example.py` showing basic usage
    - Demonstrate initialization with OpenAI
    - Show predict() call with sample sentences and passages
    - Include cache statistics output and structured output explanation
    - Make it copy-paste ready for quick start
  - [ ] 7.6 Verify package integration
    - Ensure all new modules import correctly
    - Check that coherence variants work alongside existing variants
    - Verify no breaking changes to existing code
    - Test package installation in clean environment (optional: create virtual env and pip install -e .)

**Acceptance Criteria:**
- README.md updated with coherence variant examples
- Dedicated documentation file explains theory and usage
- All code has complete docstrings
- CLAUDE.md reflects new architecture additions
- Minimal example script provides quick-start guidance
- Package integration verified (no breaking changes)

---

## Execution Order

Recommended implementation sequence:

**Phase 1: Foundation (Parallel)**
1. Task Group 1: Coherence Theory Research (can run in parallel) - COMPLETED
2. Task Group 2: API Client Infrastructure (can run in parallel)

**Phase 2: Mathematical Implementation (Sequential)**
3. Task Group 3: Coherence Formula Implementations (requires Task Group 1)

**Phase 3: Detection Variants (Sequential)**
4. Task Group 4: SelfCheck Coherence Variants (requires Task Groups 2 and 3)
5. Task Group 5: Configuration Management (requires Task Groups 2-4)

**Phase 4: Validation & Documentation (Sequential)**
6. Task Group 6: Interactive Validation (requires Task Groups 1-5)
7. Task Group 7: Documentation & Integration (requires Task Group 6)

---

## Key Implementation Notes

### Testing Philosophy for Research Project
- **No formal unit tests required** per user's testing standards for research projects
- **Validation via Jupyter notebooks** in demo/ directory for interactive testing
- **Manual inspection** of outputs on sample data to verify correctness
- **Optional tests** only if they provide experimental value (e.g., validating mathematical formulas)

### API Cost Management Strategy
- Implement caching EARLY (Task Group 2) to minimize API costs throughout development
- Use small dataset subsets (5-10 passages) during development and validation
- Reserve full 238-passage evaluation for final benchmarking
- Consider using cheaper models (gpt-4o-mini, llama3-70b) initially before testing with GPT-4

### Uncertainty Points from Spec (Revisit During Implementation)
- **Spec line 41**: "Calculate Shogenji coherence between sentence and each sample" - verify aggregation approach
- **Spec line 42**: Coherence-to-hallucination inversion formula - may need experimentation
- **Spec line 24**: Default 0.5 for unparseable probabilities - consider alternatives during validation

### Mathematical Correctness Priority
- Task Group 1 (theory research) is CRITICAL - verify formulas before implementation
- Cross-reference multiple sources for Fitelson and Olsson measures (less standard than Shogenji)
- Document any formula variations or parameters discovered in theory sources
- Validate formula implementations with known test cases (if available in papers)

### Dependencies and Blockers
- Task Group 3 BLOCKED until theory formulas verified in Task Group 1 - UNBLOCKED (Task Group 1 complete)
- Task Group 4 BLOCKED until both API client (Task Group 2) and formulas (Task Group 3) complete
- Task Group 6 BLOCKED until all variants implemented and minimally functional
- No external blockers identified (all dependencies within codebase)

### Parallelization Opportunities
- Task Groups 1 and 2 can run completely in parallel (research + API client)
- Three coherence variants (4.2-4.3, 4.4-4.5, 4.6-4.7) can be implemented in parallel after shared infrastructure complete
- Evaluation of three variants in Task Group 6 can run in parallel (independent evaluations)

---

## Success Metrics

**Implementation Complete When:**
- All three coherence variants importable and functional: `from selfcheckgpt import SelfCheckShogenji, SelfCheckFitelson, SelfCheckOlsson`
- Variants follow same interface as existing SelfCheck methods
- Interactive demo notebook validates correct behavior on sample data
- Evaluation script runs on wiki_bio_gpt3_hallucination without errors
- AUC-PR and PCC metrics computed and compared to baseline (93.42 AUC-PR target)
- Documentation allows users to understand and use coherence variants
- API costs are manageable via caching and estimation utilities

**Stretch Goals (Optional):**
- Achieve AUC-PR > 93.42 (beat current best SelfCheckAPIPrompt)
- Identify which coherence measure performs best and document theoretical insights
- Optimize prompt templates for better probability extraction
- Experiment with different aggregation strategies beyond mean
