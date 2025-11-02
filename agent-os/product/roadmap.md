# Product Roadmap

1. [ ] Core Coherence Infrastructure — Create base coherence calculation framework including LLM API client wrapper for prompt-based probability estimation, probability extraction/parsing utilities, and common coherence scoring formulas (NumPy/SciPy). `S`

1a. [ ] Prompt Engineering for Probability Extraction — Design and test prompt templates for extracting probability estimates from LLMs (e.g., "Rate the probability [0.0-1.0] that this statement is true: [statement]"). Validate prompt reliability across OpenAI and Groq models. `S`

2. [ ] SelfCheckShogenji Implementation — Implement Shogenji's probabilistic coherence measure with predict() interface, calculating how much statements increase each other's probability to detect weak logical support. `M`

3. [ ] SelfCheckFitelson Implementation — Implement Fitelson's confirmation-theoretic coherence measure with predict() interface, evaluating mutual confirmation between statements for hallucination detection. `M`

4. [ ] SelfCheckOlsson Implementation — Implement Olsson's support-based coherence measure with predict() interface, analyzing statement justification relationships to complete theoretical coverage. `M`

5. [ ] Unit Tests for Coherence Variants — Create comprehensive unit tests for all three coherence variants covering edge cases, numerical stability, and interface compliance with existing SelfCheck methods. `S`

6. [ ] Benchmark Evaluation Pipeline — Develop evaluation pipeline to benchmark all three coherence variants on wiki_bio_gpt3_hallucination dataset, computing AUC-PR and PCC metrics for comparison with existing methods. `M`

7. [ ] Statistical Significance Analysis — Implement statistical significance testing (paired t-tests, bootstrap confidence intervals) to determine if coherence measures significantly outperform consistency measures. `S`

8. [ ] Performance Comparison Visualization — Create visualization notebooks displaying ROC/PR curves, coherence vs consistency score distributions, and per-method performance comparisons across all 9 variants. `S`

9. [ ] Demo Notebook: Shogenji — Create interactive Jupyter notebook demonstrating SelfCheckShogenji with example passages, hallucination scores, and interpretability visualizations showing logical support calculations. `S`

10. [ ] Demo Notebook: Fitelson — Create interactive Jupyter notebook demonstrating SelfCheckFitelson with example passages, showing confirmation relationships and how mutual support reveals hallucinations. `S`

11. [ ] Demo Notebook: Olsson — Create interactive Jupyter notebook demonstrating SelfCheckOlsson with example passages, illustrating justification analysis and support-based detection. `S`

12. [ ] Demo Notebook: Comparative Analysis — Create comprehensive notebook comparing all three coherence variants side-by-side with existing methods, showing when each approach excels. `M`

13. [ ] Research Documentation — Document coherence theory foundations, implementation details, experimental results, and theoretical insights bridging formal epistemology with hallucination detection. `M`

14. [ ] Integration Guide — Create guide for existing SelfCheckGPT users to adopt coherence variants, including installation, API usage, and migration from consistency-based methods. `S`

15. [ ] API Documentation — Generate comprehensive API documentation for all three coherence variants covering predict() signatures, scoring interpretation, and configuration options. `S`

> Notes
> - Order prioritizes establishing foundational infrastructure first, then implementing variants in parallel-ready sequence
> - Each coherence variant (items 2-4) is independently implementable after item 1 completes
> - Evaluation phase (items 6-8) depends on all three variants being implemented
> - Demo notebooks (items 9-12) can be developed in parallel once corresponding variants exist
> - Documentation phase (items 13-15) synthesizes findings after evaluation completes
> - Research dissemination (paper preparation, upstream contribution) is intentionally excluded pending experimental results
