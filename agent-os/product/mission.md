# Product Mission

## Pitch
SelfCheckGPT-Coherence is a research extension to the production SelfCheckGPT package that helps researchers and ML engineers detect LLM hallucinations by providing formal coherence theory-based detection variants that measure logical support relationships rather than surface-level consistency.

## Users

### Primary Customers
- **Academic Researchers**: PhD students, postdocs, and faculty working on LLM reliability, hallucination detection, and epistemic logic
- **ML Research Engineers**: Industry researchers developing state-of-the-art hallucination detection systems for production LLM applications

### User Personas
**Research Scientist** (PhD student, 25-35 years)
- **Role:** Doctoral researcher in NLP/AI safety
- **Context:** Investigating theoretical foundations of hallucination detection for LLMs; needs rigorous, theoretically-grounded methods
- **Pain Points:** Current hallucination detection methods lack formal theoretical grounding; difficult to understand why methods work or fail; need explainability beyond black-box metrics
- **Goals:** Publish research demonstrating that formal coherence theory provides deeper insights than surface consistency measures; contribute to theoretical understanding of LLM reliability

**ML Engineer** (Senior engineer, 28-40 years)
- **Role:** Production ML engineer at AI-first company
- **Context:** Building hallucination detection systems for customer-facing LLM applications
- **Pain Points:** Existing consistency-based methods miss subtle logical errors; need higher accuracy and interpretability for production use
- **Goals:** Achieve higher AUC-PR than current best (93.42); deploy theoretically-grounded detection that can be explained to stakeholders

## The Problem

### Surface Consistency Misses Deep Logical Flaws
Current SelfCheckGPT variants (NLI, BERTScore, MQAG, etc.) primarily measure surface-level agreement between sampled outputs. A hallucinated claim can be internally consistent across samples while still lacking logical support from the underlying information. This results in false negatives where logically unsupported statements pass consistency checks.

**Quantifiable Impact:** Current best method achieves 93.42 AUC-PR on wiki_bio_gpt3_hallucination dataset, leaving 6.58 points of potential improvement and unknown theoretical understanding of why consistency measures work.

**Our Solution:** Implement three complementary coherence measures (Shogenji, Fitelson, Olsson) that evaluate logical support relationships between statements rather than surface agreement, providing both improved accuracy and theoretical grounding in formal epistemology.

### Lack of Theoretical Foundation
Existing methods are empirically driven without grounding in formal coherence theory. Researchers cannot explain why consistency checks detect hallucinations or predict when they will fail.

**Quantifiable Impact:** No existing hallucination detection work bridges formal coherence theory with practical LLM evaluation, limiting both theoretical understanding and method innovation.

**Our Solution:** Ground detection methods in established coherence theories (Shogenji's probabilistic coherence, Fitelson's confirmation theory, Olsson's support relations), enabling principled analysis of when and why detection succeeds.

## Differentiators

### Formal Coherence Theory Foundation
Unlike existing SelfCheckGPT variants that measure surface-level consistency, we implement detection methods grounded in formal epistemology and coherence theory. This results in both higher accuracy (hypothesis: >94 AUC-PR) and theoretical interpretability.

### Logical Support vs. Surface Agreement
Where SelfCheckNLI asks "do samples contradict?" and BERTScore asks "are samples similar?", coherence measures ask "do statements logically support each other?" This deeper semantic analysis captures hallucinations that pass consistency checks.

### Research-Driven Extension Architecture
Unlike building a new package from scratch, we extend the mature SelfCheckGPT infrastructure (EMNLP 2023, production-ready). This provides immediate benchmarking against 6 existing methods, established evaluation datasets, and potential for upstream contribution.

## Technical Approach

### Hybrid Architecture: LLM APIs + Coherence Formulas

**Probability Estimation via Prompt-Based Assessment (Decision: 2025-11-02)**
- Use OpenAI API with structured output (JSON schema) to extract probability estimates
- Example prompts: "Rate the probability this statement is true: [statement]" or "How likely is it that both statements are true?"
- Structured output ensures reliable probability extraction without text parsing complexity
- Leverages existing `SelfCheckAPIPrompt` infrastructure pattern for API integration
- **Simplified approach (Updated 2025-11-02):** OpenAI only (no Groq) to leverage structured output support

**Coherence Calculation Pipeline**
1. **Input:** Sentences to evaluate + sampled passages (standard SelfCheck interface)
2. **LLM API Calls:** Query OpenAI with structured output to assess individual statement probabilities (P(A), P(B)) and joint probabilities (P(A ∧ B))
3. **Coherence Formula:** Feed probability estimates into NumPy/SciPy implementations of Shogenji/Fitelson/Olsson formulas
4. **Output:** Sentence-level hallucination scores (higher = more likely hallucination)

**Why This Architecture?**
- **Reliability:** Structured output (JSON schema) ensures consistent, parseable probability values
- **Simplicity:** Single API provider (OpenAI) reduces complexity and maintenance burden
- **Theoretical Grounding:** Coherence formulas remain mathematically pure while leveraging state-of-the-art LLMs for probability estimation
- **Future Extensibility:** Can add other providers later if they support structured output

## Key Features

### Core Features
- **SelfCheckShogenji:** Probabilistic coherence measure based on Shogenji's theory that evaluates how much statements increase the probability of each other, detecting hallucinations through weak logical support relationships. Uses LLM-based probability estimation fed into coherence formula.
- **SelfCheckFitelson:** Confirmation-theoretic coherence measure implementing Fitelson's framework for assessing mutual confirmation between statements, providing complementary perspective on logical relationships. Combines LLM probability assessment with confirmation theory calculations.
- **SelfCheckOlsson:** Coherence measure based on Olsson's support theory that analyzes how statements justify each other, completing the theoretical coverage of major coherence approaches. Integrates LLM support assessments with formal support-based coherence metrics.

### Evaluation Features
- **Direct Benchmark Comparison:** Evaluate all three coherence variants on the same wiki_bio_gpt3_hallucination dataset (238 annotated passages) used by existing methods, enabling direct AUC-PR and PCC comparisons
- **Statistical Significance Testing:** Rigorous statistical analysis to determine if coherence measures significantly outperform consistency measures on hallucination detection
- **Performance Analysis Dashboard:** Jupyter notebooks visualizing per-sentence hallucination scores, ROC/PR curves, and coherence vs. consistency score distributions

### Integration Features
- **Common Interface:** All coherence variants implement the same `predict(sentences, sampled_passages)` interface as existing SelfCheckGPT methods, enabling drop-in replacement
- **Existing Infrastructure:** Leverage established sentence tokenization (spacy), device management (PyTorch), and model loading patterns from SelfCheckGPT
- **Demo Notebooks:** Interactive Jupyter notebooks demonstrating each coherence variant with example outputs and interpretability visualizations

## Success Criteria

1. **Implementation:** All three coherence variants (Shogenji, Fitelson, Olsson) implemented with common interface
2. **Accuracy:** At least one coherence variant achieves higher AUC-PR than current best (93.42) on wiki_bio_gpt3_hallucination
3. **Validation:** Statistical significance testing confirms coherence > consistency for hallucination detection
4. **Interpretability:** Demonstrate through examples how coherence measures catch hallucinations missed by consistency checks
5. **Research Impact:** Publishable findings bridging formal coherence theory and practical LLM hallucination detection
