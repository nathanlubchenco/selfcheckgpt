# Specification: SelfCheckGPT Coherence Variants Improvements

## Goal

Improve the accuracy and reliability of coherence-based hallucination detection by optimizing probability extraction through prompt engineering and establishing comprehensive verification infrastructure to validate coherence formula implementations.

## User Stories

- As a researcher using SelfCheckGPT, I want probability extraction to be accurate and calibrated so that coherence measures produce reliable hallucination scores
- As a developer, I want comprehensive verification tools so that I can validate coherence formula correctness and identify outliers or mathematical errors
- As a user of coherence variants, I want improved prompts that adhere to probability theory axioms so that extracted probabilities are mathematically consistent

## Specific Requirements

**Probability Extraction Benchmark System**
- Create standalone test suite with 40 test cases covering probability ranges, statement types, domains, and axiom compliance
- Cover 8 probability ranges: very low (0.0-0.1), low (0.1-0.3), low-medium (0.3-0.5), medium-high (0.5-0.7), high (0.7-0.9), very high (0.9-1.0), joint probabilities, conditional probabilities
- Cover 5 statement types: clear facts, common sense statements, ambiguous statements, unlikely claims, clear contradictions
- Cover 5 domains: science/physics, geography, history, mathematics/logic, common sense/everyday knowledge
- Include special test cases for axiom edge cases, paraphrase consistency, and adversarial statements
- Design test cases with ground truth probabilities or expected probability relationships

**Evaluation Metrics Implementation**
- Implement Brier Score to measure mean squared error between predicted probabilities and actual outcomes (primary calibration metric)
- Implement Expected Calibration Error (ECE) to detect systematic over/under-confidence in probability estimates across binned probability ranges
- Implement Probability Coherence Compliance checker to verify adherence to Kolmogorov axioms, joint probability consistency, and conditional probability consistency
- Implement Probability Consistency Score to measure variance in probability estimates for semantically equivalent statements
- Implement Sharpness metric to quantify decisiveness of predictions (distance from 0.5) as measure of confidence

**Prompt Strategy Variants**
- Implement baseline prompt strategy using current simple prompts as reference point
- Implement Chain of Thought (CoT) variant with step-by-step reasoning prompts for probability assessment
- Implement few-shot variant with 3-5 example probability assessments to guide model
- Implement axiom-aware variant with probability theory principles included in system prompt to reduce axiom violations
- Implement hybrid variant combining best elements from CoT, few-shot, and axiom-aware approaches
- Maintain backward compatibility with existing simple prompts through configuration option

**Coherence Formula Verification**
- Use probability benchmark to validate mathematical correctness of Shogenji, Fitelson, and Olsson formula implementations
- Test known-outcome scenarios where coherence scores should have predictable values
- Detect outliers and numerical instabilities in coherence score calculations
- Verify edge case handling for axiom violations and numerical errors
- Confirm epsilon smoothing prevents division by zero without distorting results

**Benchmark Runner Infrastructure**
- Create automated runner to execute test suite against any prompt variant
- Generate comprehensive metric reports comparing Brier Score, ECE, Probability Coherence Compliance, Consistency Score, and Sharpness
- Support comparative analysis of multiple prompt strategies side-by-side
- Provide interpretable output showing which test cases failed and why
- Track API call counts and estimated costs for benchmark execution

**SimpleQA Dataset Investigation**
- Analyze SimpleQA dataset structure to determine format and content
- Assess compatibility with coherence approach: requires LLM-generated responses, stochastic samples, and sentence-level structure
- Document specific compatibility criteria: dataset must support probability extraction at statement level
- If compatible, design integration approach for using SimpleQA as secondary evaluation dataset
- If incompatible, document reasons and suggest alternative datasets for future expansion

**Production Integration of Improved Prompts**
- Integrate best-performing prompt variant into SelfCheckShogenji, SelfCheckFitelson, and SelfCheckOlsson classes
- Update CoherenceAPIClient prompt templates with optimized versions
- Preserve existing caching mechanism for cost efficiency with new prompts
- Maintain consistent predict() interface for backward compatibility
- Document prompt improvements and performance gains in module docstrings

**End-to-End Validation**
- Run improved prompts on wiki_bio_gpt3_hallucination dataset and compare AUC-PR, PCC, and AUC-ROC against baseline
- Quantify improvement percentages for each coherence variant
- Run improved prompts on SimpleQA dataset if compatibility investigation confirms viability
- Document known limitations and edge cases discovered during validation

## Visual Design

No visual assets provided. Visualization is explicitly out of scope for this iteration.

## Existing Code to Leverage

**CoherenceAPIClient (modeling_coherence_api.py)**
- Extend existing prompt template system (individual_prob_template, joint_prob_template, conditional_prob_template) for new prompt variants
- Reuse OpenAI structured output schema for reliable probability extraction
- Leverage existing caching mechanism (LRU cache with 10,000 max entries) to minimize API costs for benchmark
- Use retry logic with exponential backoff for transient API failures
- Extend get_cache_stats() method for benchmark cost tracking

**Evaluation Script Pattern (scripts/evaluate_coherence.py)**
- Follow existing evaluation methodology: load_dataset, evaluate_variant, compute_metrics, save_results pattern
- Reuse metric computation infrastructure (AUC-PR, PCC, AUC-ROC using sklearn and scipy)
- Adopt existing cost estimation utilities for benchmark API call tracking
- Model benchmark runner after evaluate_variant() function structure with tqdm progress bars

**Coherence Formula Utilities (utils_coherence.py)**
- Validate existing coherence_shogenji(), coherence_fitelson(), coherence_olsson() implementations against benchmark test cases
- Verify epsilon smoothing, probability clamping, and axiom violation warnings work correctly
- Test normalize_coherence_scores() edge case handling (NaN/Inf, identical scores)
- Ensure existing error handling for physically impossible probabilities (P(A∧B) > P(A)) is sufficient

**MQAG Multiple Scoring Pattern (modeling_mqag.py)**
- Model prompt variant evaluation after MQAG's multiple scoring methods (counting, bayes, bayes_with_alpha)
- Adopt pattern of configurable scoring strategies that can be compared independently
- Use similar approach for selecting best-performing variant based on benchmark metrics

**Demo Notebook Structure (demo/coherence_demo.ipynb)**
- Create benchmark_demo.ipynb following existing coherence_demo.ipynb pattern
- Use Jupyter notebook for interactive benchmark exploration and visualization of metric results
- Include example test cases with explanations of expected probability values
- Show side-by-side comparison of prompt variants on same test cases

## Out of Scope

- Visualization tools including heatmaps, probability charts, coherence breakdowns, or debugging UI
- Model comparison infrastructure for testing gpt-4.5-mini, gpt-5-mini, gpt-5-nano (defer to future work, but document as priorities)
- Proposition extraction for automatic decomposition of complex sentences into atomic claims
- Expansion beyond wiki_bio_gpt3_hallucination and SimpleQA datasets in this iteration
- Advanced coherence measures beyond the three already implemented (Shogenji, Fitelson, Olsson)
- Multi-lingual coherence detection or domain-specific benchmarks
- Weighted coherence measures or hierarchical coherence assessment
- Interactive user interfaces for probability exploration
- Automated hyperparameter tuning for prompts
- Cross-model prompt optimization (prompts optimized for OpenAI models only)
