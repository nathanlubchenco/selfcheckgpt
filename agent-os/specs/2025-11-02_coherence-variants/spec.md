# Specification: Coherence-Based Hallucination Detection Variants

## Goal
Extend SelfCheckGPT with three formal coherence theory-based hallucination detection variants (Shogenji, Fitelson, Olsson) that measure logical support relationships between statements rather than surface consistency, aiming to outperform the current best method (93.42 AUC-PR) through theoretically-grounded probabilistic coherence measures implemented via hybrid LLM API + NumPy architecture.

## User Stories
- As a research scientist investigating LLM reliability, I want to use theoretically-grounded coherence measures to detect hallucinations so that I can understand why detection succeeds or fails beyond empirical black-box metrics
- As an ML engineer building production hallucination detection, I want coherence variants that achieve higher accuracy than existing methods so that I can catch subtle logical errors missed by surface consistency checks

## Specific Requirements

**Shared API Client Infrastructure**
- Create `CoherenceAPIClient` base class in `selfcheckgpt/modeling_coherence_api.py` that wraps OpenAI and Groq clients similar to `SelfCheckAPIPrompt`
- Support client_type parameter ("openai" or "groq") with automatic client initialization from API keys
- Implement temperature=0.0 deterministic completion method for consistency across probability extraction calls
- Include prompt response caching mechanism using dictionary keyed by (prompt_text, model_name) to minimize API costs during evaluation
- Handle API errors gracefully with retry logic and clear error messages for rate limits or authentication failures

**Probability Extraction via Prompting**
- Implement prompt templates for extracting individual statement probabilities using pattern "Rate the probability [0.0-1.0] that this statement is true: [statement]"
- Create prompt templates for joint probability estimation asking "Rate the probability [0.0-1.0] that both statements are true: [statement1] AND [statement2]"
- Create prompt templates for conditional probability estimation asking "Rate the probability [0.0-1.0] that statements A is true: [statement1] GIVEN that [statement2] is true"
- Parse LLM text responses to extract numeric probability values, handling variations like "0.7", "70%", "probability is 0.7"
- Default to 0.5 probability for unparseable responses with warning logging (strongly consider other options here)
- Max tokens set to 20 to keep responses concise and minimize costs

**Coherence Formula Implementations**
- Create `selfcheckgpt/utils_coherence.py` with three separate utility functions implementing coherence measures
- Find initial sources for all the formal coherence theories, download and store them in the repo for easy access.
- Implement `coherence_shogenji(probs_individual, probs_joint)` calculating Shogenji's measure: C(A,B) = P(A ∧ B) / (P(A) × P(B))
- Implement `coherence_fitelson(probs_individual, probs_joint, probs_conditional)` for Fitelson's confirmation measure based on difference between P(A|B) and P(A|¬B)
- Implement `coherence_olsson(probs_individual, probs_joint)` computing Olsson's support-based measure analyzing mutual justification
- Add numerical stability handling with epsilon=1e-12 smoothing to prevent division by zero
- Return coherence scores as NumPy arrays for batch processing

**SelfCheckShogenji Detection Variant**
- Implement `SelfCheckShogenji` class in `selfcheckgpt/modeling_coherence.py` following existing SelfCheck class patterns
- Initialize with client_type, model, and api_key parameters matching `SelfCheckAPIPrompt` signature
- Implement `predict(sentences, sampled_passages, verbose=False)` returning sentence-level hallucination scores
- For each sentence, extract probabilities from LLM for the sentence alone and jointly with each sampled passage
- Calculate Shogenji coherence between sentence and each sample using `coherence_shogenji()` formula (i'm not sure about this either, lets discuss further)
- Aggregate coherence scores across samples using mean, then invert via (1.0 - normalized_coherence) to convert high coherence to low hallucination score (revisit this for further discussion)
- Include tqdm progress bar when verbose=True

**SelfCheckFitelson Detection Variant**
- Implement `SelfCheckFitelson` class in `selfcheckgpt/modeling_coherence.py` with same initialization signature as SelfCheckShogenji
- Implement `predict(sentences, sampled_passages, verbose=False)` interface
- Extract individual probabilities P(sentence) and P(sample), conditional probabilities P(sentence|sample) via additional prompts
- Calculate Fitelson confirmation coherence using `coherence_fitelson()` measuring mutual confirmation strength
- Apply mean aggregation across samples then invert to hallucination score (1.0 - normalized_coherence)
- Handle cases where conditional probability estimation may be unreliable by falling back to joint probability approximations

**SelfCheckOlsson Detection Variant**
- Implement `SelfCheckOlsson` class in `selfcheckgpt/modeling_coherence.py` matching interface of other variants
- Implement `predict(sentences, sampled_passages, verbose=False)` with same signature
- Use prompt-based probability extraction to assess support relationships between sentence and samples
- Calculate Olsson support-based coherence via `coherence_olsson()` analyzing justification strength
- Mean-aggregate coherence scores then invert to produce hallucination scores in [0.0, 1.0] range
- Ensure consistent behavior across all three variants for drop-in replacement capability

**Configuration and Constants**
- Add `CoherenceConfig` class to `selfcheckgpt/utils.py` with default model configurations
- Include default probability extraction prompt templates as class attributes
- Define default OpenAI model as "gpt-4o-mini" and default Groq model as "llama3-70b-8192"
- Store numerical stability constants (epsilon, score normalization bounds)

**Hallucination Score Mapping**
- Normalize coherence scores to [0.0, 1.0] range before inversion using min-max normalization across all sentence-sample pairs
- Apply simple inversion (1.0 - normalized_coherence) to convert high coherence (low hallucination) to low score
- Ensure output scores are directly comparable to existing SelfCheck variants (higher score = higher hallucination probability)
- Return NumPy arrays matching shape of existing `predict()` outputs (sentence-level scores)

**Unit Testing with Mocked APIs**
- Create `tests/test_coherence_variants.py` using pytest framework
- Mock OpenAI and Groq API responses using unittest.mock to avoid API costs during testing
- Test each coherence variant's predict() method with fixed mock responses to verify score calculation
- Test probability extraction parsing with various response formats (numeric, percentage, text)
- Test numerical stability with edge cases (zero probabilities, near-zero probabilities)
- Test caching mechanism ensures duplicate prompts reuse cached responses
- Test error handling for API failures, unparseable responses, and invalid inputs

**Evaluation Dataset Access**
- Create `scripts/evaluate_coherence.py` standalone script loading wiki_bio_gpt3_hallucination via datasets library
- Use `load_dataset("potsawee/wiki_bio_gpt3_hallucination")['evaluation']` to access 238 annotated passages
- Calculate AUC-PR and PCC metrics for each coherence variant using sklearn.metrics
- Output results to JSON file with per-variant scores and comparison to existing methods
- Create `demo/coherence_evaluation.ipynb` Jupyter notebook for interactive exploration with visualizations
- Include ROC curves, PR curves, score distributions, and per-sentence analysis in notebook

**API Cost Management**
- Implement LRU cache with max size 10000 for prompt responses in `CoherenceAPIClient`
- Log cache hit rates to help users understand cost savings during evaluation
- Provide cost estimation utility function calculating expected API calls given number of sentences and samples
- Recommend batching evaluation runs to maximize cache utilization across multiple coherence variants

## Out of Scope
- Ensemble methods combining multiple coherence measures are excluded from initial implementation
- Fine-tuning custom probability estimation models instead of using API-based prompting
- Extending coherence measures beyond the three specified variants (Shogenji, Fitelson, Olsson)
- Implementing set-based coherence aggregation beyond simple mean of pairwise comparisons
- Supporting logprobs-based probability extraction (prompt-based approach is sufficient)
- Integration with non-OpenAI/Groq API providers in initial version
- Real-time streaming evaluation or production deployment optimization
- GUI or web interface for coherence detection
- Automatic hyperparameter tuning for prompt templates or coherence formula parameters
- Multi-language support beyond English text evaluation
