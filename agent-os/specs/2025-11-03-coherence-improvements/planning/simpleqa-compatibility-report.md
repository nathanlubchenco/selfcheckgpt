# SimpleQA Dataset Compatibility Assessment Report

**Date:** 2025-11-04
**Author:** Claude Code
**Spec:** 2025-11-03-coherence-improvements
**Task:** Task Group 4.1 - SimpleQA Dataset Analysis

---

## Executive Summary

**Compatibility Determination: INCOMPATIBLE**

SimpleQA is **not suitable** for SelfCheckGPT coherence-based hallucination detection in its current form. The dataset lacks critical requirements for the coherence approach, specifically:

1. No pre-generated LLM responses (only questions and ground truth answers)
2. No multiple stochastic samples per question
3. No sentence-level hallucination annotations
4. Short-form factual answers unsuitable for coherence analysis across multiple statements

**Recommendation:** Do NOT integrate SimpleQA as secondary evaluation dataset. Consider alternative datasets (HaluEval, TruthfulQA with custom generation, or custom dataset creation) if expansion beyond wiki_bio_gpt3_hallucination is required.

---

## 1. Dataset Structure and Format

### 1.1 Official Dataset Information

**Source:** HuggingFace Datasets
**Primary Versions:**
- `lighteval/SimpleQA` - 4,326 rows (official benchmark version)
- `basicv8vc/SimpleQA` - 4,326 rows (mirror)
- `google/simpleqa-verified` - 1,000 rows (verified subset)

**Format Options:**
- Parquet (1.07 MB)
- CSV (2.01 MB)
- Compatible with: HuggingFace datasets library, pandas, Polars

### 1.2 Schema Structure

The SimpleQA dataset contains four primary fields:

| Field | Type | Description |
|-------|------|-------------|
| `metadata` | string | Structured information including topic, answer_type, and source URLs |
| `problem` | string | The factual question to be answered |
| `answer` | string | The ground truth answer |
| `split` | string | Dataset partition ("test" or "few_shot") |

**Example Entry:**
```json
{
  "metadata": "{\"topic\": \"science\", \"answer_type\": \"date\", \"sources\": [\"https://...\", \"https://...\"]}",
  "problem": "When was the first human genome sequenced?",
  "answer": "2003",
  "split": "test"
}
```

### 1.3 Dataset Size and Content

- **Total Rows:** 4,326 examples
- **Split Distribution:** 4,320 test rows, 6 few_shot examples
- **Content Scope:** Diverse topics (science, politics, sports, art, history, geography)
- **Answer Types:** Dates, numbers, person names, place names, short factual statements
- **Answer Length:** Short-form (typically 1-10 words)

### 1.4 Data Availability and Licensing

**Availability:** Publicly accessible via HuggingFace Datasets
**License:** MIT License (part of OpenAI's simple-evals repository)
**Access:** No authentication required
**Maintenance:** Active (last update: Jan 28, 2025)

---

## 2. Compatibility Requirements Analysis

### 2.1 Requirement 1: LLM-Generated Responses

**Status: NOT MET**

**Requirement:** Dataset must include LLM-generated responses (not just ground truth answers) to evaluate for hallucinations.

**SimpleQA Structure:**
- Contains ONLY questions (`problem`) and ground truth answers (`answer`)
- Does NOT include any LLM-generated responses
- Designed for factuality benchmarking where models generate responses at evaluation time

**Impact:** CRITICAL BLOCKER - Without pre-generated LLM responses, we cannot perform coherence analysis. The coherence approach requires comparing an LLM's response against its own stochastic samples to detect inconsistencies.

**Workaround Feasibility:** Could generate responses using an LLM, but this would require:
- Significant infrastructure setup
- High API costs (4,326 questions × N samples × API calls)
- Additional engineering effort outside project scope
- Quality control for generated responses

### 2.2 Requirement 2: Multiple Stochastic Samples

**Status: NOT MET**

**Requirement:** Must be able to generate or have access to multiple stochastic samples per question for consistency analysis.

**SimpleQA Structure:**
- No stochastic samples provided
- Single ground truth answer per question
- Evaluation methodology uses grader model for single response assessment

**Comparison to wiki_bio_gpt3_hallucination:**
```python
# wiki_bio structure (COMPATIBLE)
{
  'gpt3_sentences': ['Sentence 1.', 'Sentence 2.', ...],
  'gpt3_text_samples': ['Sample 1...', 'Sample 2...', ...],  # Multiple samples!
  'annotation': ['accurate', 'inaccurate', ...]
}

# SimpleQA structure (INCOMPATIBLE)
{
  'problem': 'What is the capital of France?',
  'answer': 'Paris',  # Single ground truth only
  'metadata': '{...}',
  'split': 'test'
}
```

**Impact:** CRITICAL BLOCKER - The entire coherence approach depends on comparing multiple stochastic samples (typically 3-5) to detect inconsistencies via probability extraction. Without these samples, coherence measures (Shogenji, Fitelson, Olsson) cannot be computed.

**Workaround Feasibility:** Would require generating N stochastic samples per question:
- API Cost Estimate: 4,326 questions × 5 samples × OpenAI API = 21,630+ API calls just for generation
- Additional probability extraction API calls: ~11-16 per sentence × avg sentences per response
- Total cost would be substantial (thousands of dollars)

### 2.3 Requirement 3: Sentence-Level Structure

**Status: NOT MET**

**Requirement:** Responses must be structured at sentence level or can be sentence-tokenized for granular hallucination detection.

**SimpleQA Structure:**
- Answers are extremely short (1-10 words typical)
- Most answers are single tokens or short phrases: "Paris", "2003", "Albert Einstein"
- Not multi-sentence responses suitable for sentence-level analysis

**Example Answers:**
- "Paris" (single word)
- "2003" (single token)
- "The Mona Lisa was painted by Leonardo da Vinci" (single sentence at best)

**Comparison to wiki_bio Requirements:**
- wiki_bio: Multi-sentence biographical passages (5-10+ sentences)
- SimpleQA: Short-form factual answers (1-10 words)

**Impact:** MAJOR BLOCKER - Coherence analysis requires multiple statements/sentences to assess consistency relationships. Short answers like "Paris" or "2003" cannot be decomposed into multiple propositions for coherence measurement.

**Workaround Feasibility:** Very limited. Even if we generate long-form responses, the dataset is designed for short factuality evaluation, making it a poor fit for sentence-level analysis.

### 2.4 Requirement 4: Multi-Sentence Response Length

**Status: NOT MET**

**Requirement:** Responses must have sufficient length (multi-sentence) for coherence analysis across multiple statements.

**SimpleQA Design Philosophy:**
- Explicitly focused on "short-form factuality"
- Questions designed for concise answers
- Evaluation measures correctness of brief factual claims

**Average Response Characteristics:**
- Length: 1-10 words (not 5-10 sentences like wiki_bio)
- Structure: Single fact or entity name
- Complexity: Atomic claims, not complex narratives

**Impact:** FUNDAMENTAL MISMATCH - Coherence measures (Shogenji, Fitelson, Olsson) assess relationships between multiple statements. With only single-fact answers, there are no statement pairs to compare.

**Coherence Formula Requirements:**
```
Shogenji: C2(A,B) = P(A∧B) / (P(A) × P(B))  # Requires TWO statements A and B
Fitelson: s(H,E) = P(H|E) - P(H|¬E)          # Requires TWO statements H and E
Olsson: C1(A,B) = P(A∧B) / P(A∨B)            # Requires TWO statements A and B
```

All three formulas require at least two statements to compute coherence. SimpleQA's single-word answers cannot provide this.

---

## 3. Comparison with wiki_bio_gpt3_hallucination

### 3.1 Structure Comparison Table

| Feature | wiki_bio_gpt3_hallucination | SimpleQA | Compatible? |
|---------|----------------------------|----------|-------------|
| **LLM-Generated Responses** | Yes (`gpt3_sentences`) | No (only ground truth) | NO |
| **Multiple Stochastic Samples** | Yes (`gpt3_text_samples`, 5+ samples) | No | NO |
| **Sentence-Level Annotations** | Yes (`annotation` per sentence) | No | NO |
| **Multi-Sentence Responses** | Yes (5-10+ sentences typical) | No (1-10 words typical) | NO |
| **Hallucination Labels** | Yes (sentence-level: 'accurate'/'inaccurate') | Yes (but response-level only) | PARTIAL |
| **Dataset Size** | 238 passages, ~2,000 sentences | 4,326 questions | N/A |
| **Domain** | Biographical passages | General factual QA | N/A |
| **Response Length** | 50-200+ words | 1-10 words | NO |

### 3.2 Evaluation Methodology Comparison

**wiki_bio Coherence Evaluation:**
```python
# Load dataset with pre-generated samples
passage = dataset[idx]
sentences = passage['gpt3_sentences']  # Multi-sentence
sampled_passages = passage['gpt3_text_samples']  # 5+ samples
annotations = passage['annotation']  # Sentence-level labels

# Compute coherence scores
scores = selfcheck_shogenji.predict(
    sentences=sentences,
    sampled_passages=sampled_passages
)
```

**SimpleQA Evaluation (Current):**
```python
# Load dataset with only questions
question = dataset[idx]
problem = question['problem']
ground_truth = question['answer']  # Single word/phrase

# Generate response (NOT in dataset)
response = llm.generate(problem)  # Would need to add this

# NO stochastic samples available
# Cannot compute coherence without samples
```

**Key Insight:** SimpleQA's evaluation methodology is fundamentally different - it uses a grader model to compare a single generated response against ground truth for correctness. Coherence methods require self-consistency analysis across multiple stochastic samples, which SimpleQA does not support.

---

## 4. Alternative Datasets for Future Expansion

Given SimpleQA's incompatibility, here are alternative datasets for consideration:

### 4.1 HaluEval (PARTIALLY COMPATIBLE with modifications)

**Source:** `pminervini/HaluEval` on HuggingFace
**Size:** 64,507 rows across 7 subsets

**Structure:**
- **Dialogue:** Knowledge, dialogue history, correct + hallucinated responses
- **QA:** Knowledge-based Q&A with right + hallucinated answers
- **Summarization:** Documents with correct + hallucinated summaries

**Pros:**
- Contains LLM-generated hallucinated responses (satisfies Requirement 1)
- Longer-form responses in dialogue and summarization subsets (satisfies Requirement 4)
- Clear hallucination labels provided

**Cons:**
- Does NOT include multiple stochastic samples per example (fails Requirement 2)
- Would still require generating additional samples for coherence analysis
- Binary comparison (correct vs hallucinated) rather than consistency analysis

**Adaptation Effort:** MEDIUM
- Use provided hallucinated responses as base
- Generate 3-5 additional stochastic samples per example
- Apply sentence tokenization to longer responses
- Would still require API costs but less than SimpleQA (already has one response)

**Compatibility Assessment:** PARTIAL - Better than SimpleQA but still requires sample generation

### 4.2 TruthfulQA (INCOMPATIBLE, similar issues to SimpleQA)

**Source:** `truthful_qa` on HuggingFace

**Structure:**
- Adversarial questions designed to elicit false beliefs
- Multiple-choice and generation formats
- Ground truth labels only

**Pros:**
- Focuses on hallucination/truthfulness
- Diverse question types

**Cons:**
- No LLM-generated responses provided
- No multiple stochastic samples
- Would require full generation pipeline (same issues as SimpleQA)

**Compatibility Assessment:** INCOMPATIBLE - Same blockers as SimpleQA

### 4.3 FEVER (Fact Extraction and VERification) - INCOMPATIBLE

**Structure:**
- Claims paired with Wikipedia evidence
- Verification labels (Supported/Refuted/NotEnoughInfo)
- Focus on fact-checking, not hallucination detection

**Compatibility Assessment:** INCOMPATIBLE - Different task (claim verification vs hallucination detection)

### 4.4 Custom Dataset Generation Approach (RECOMMENDED if expansion needed)

**Concept:** Generate a custom evaluation dataset modeled after wiki_bio but in different domains

**Approach:**
1. Select diverse prompts (e.g., explain X concept, describe Y event)
2. Generate responses using target LLM (e.g., GPT-4, Claude)
3. Generate 5-10 stochastic samples per prompt
4. Manually annotate sentence-level hallucinations
5. Format as HuggingFace dataset

**Pros:**
- Full control over requirements (satisfies all 4 criteria)
- Can target specific domains or use cases
- Can use same evaluation methodology as wiki_bio

**Cons:**
- Significant annotation effort (human labeling required)
- High API costs for generation
- Time-intensive

**Estimated Effort:** HIGH (2-4 weeks for 100-200 examples)

---

## 5. Detailed Compatibility Criteria Assessment

### Summary Table

| Criteria | Requirement | SimpleQA | Status | Blocker Severity |
|----------|-------------|----------|--------|------------------|
| **1. LLM Responses** | Must include LLM-generated responses | Only ground truth answers | NOT MET | CRITICAL |
| **2. Stochastic Samples** | Must have 3-5+ samples per example | Single ground truth only | NOT MET | CRITICAL |
| **3. Sentence Structure** | Must support sentence tokenization | 1-10 word answers | NOT MET | MAJOR |
| **4. Multi-Sentence** | Must have 5-10+ sentences | Single-word answers typical | NOT MET | FUNDAMENTAL |
| **5. Hallucination Labels** | Must have hallucination annotations | Has correctness labels | PARTIAL | Minor (could adapt) |
| **6. Public Access** | Must be publicly available | Yes (MIT license) | MET | N/A |
| **7. Dataset Size** | Should have 100+ examples | 4,326 examples | MET | N/A |

**Critical Blockers:** 3 (Criteria 1, 2, 4)
**Major Blockers:** 1 (Criteria 3)
**Overall Assessment:** INCOMPATIBLE

---

## 6. Cost-Benefit Analysis of Adaptation

### 6.1 If We Attempted to Adapt SimpleQA

**Required Steps:**
1. Generate LLM responses for 4,326 questions
2. Generate 5 stochastic samples per question (21,630 total generations)
3. Sentence-tokenize responses (many will be single words - low value)
4. Extract probabilities for coherence analysis (~11 API calls per sentence)
5. Manually annotate sentence-level hallucinations

**Cost Estimates:**

**Generation Costs (GPT-4o-mini at $0.150/$0.600 per 1M tokens):**
- Main responses: 4,326 questions × ~50 tokens = 216,300 tokens
- Stochastic samples: 21,630 × ~50 tokens = 1,081,500 tokens
- Total generation: ~1.3M tokens input + ~1.3M tokens output = ~$1.17

**Probability Extraction Costs:**
- Assume avg 3 sentences per response (optimistic for short answers)
- 4,326 × 3 sentences × 11 API calls = 142,758 API calls
- ~50 tokens per call = 7.14M tokens
- Cost: ~$1.07 input + ~$4.28 output = ~$5.35

**Total API Cost Estimate:** ~$6.50-10 (rough estimate)

**Manual Annotation Effort:**
- 4,326 responses × 3 sentences = ~13,000 sentences
- Annotation rate: ~100 sentences/hour (with guidelines)
- Total time: ~130 hours of human annotation
- At $50/hour: ~$6,500 labor cost

**Total Adaptation Cost:** ~$6,510+ (mostly labor)

### 6.2 Value Proposition

**Benefits:**
- Large dataset size (4,326 examples vs wiki_bio's 238)
- Diverse topics and answer types
- Well-maintained and publicly available

**Drawbacks:**
- Fundamental mismatch with coherence approach (short answers)
- Most coherence analysis would be on single-sentence "responses"
- Low discriminative power (hard to assess coherence on "Paris")
- Massive adaptation effort for questionable scientific value
- Would be testing a use case (short QA) that coherence variants aren't designed for

**Conclusion:** The cost-benefit ratio is VERY POOR. SimpleQA's short-form nature makes it unsuitable for coherence analysis even with adaptation.

---

## 7. Recommendations

### 7.1 Primary Recommendation: DO NOT INTEGRATE SimpleQA

**Rationale:**
1. **Fundamental mismatch:** Coherence analysis requires multi-sentence responses with relationships between statements. SimpleQA's short-form answers (1-10 words) cannot provide this.
2. **Critical blockers:** Missing LLM responses, missing stochastic samples, insufficient response length.
3. **Poor cost-benefit:** Adaptation would cost $6,500+ for questionable scientific value.
4. **Better alternatives exist:** HaluEval provides longer-form responses and would be easier to adapt.

### 7.2 Alternative Paths Forward

**Option A: Maintain wiki_bio as Sole Evaluation Dataset (RECOMMENDED)**
- **Pros:** Already integrated, well-suited for coherence analysis, proven methodology
- **Cons:** Limited domain (biographical only), smaller size (238 passages)
- **Effort:** None - continue current approach
- **Recommendation:** STRONGLY RECOMMENDED for current project scope

**Option B: Adapt HaluEval for Future Expansion**
- **Pros:** Longer responses, includes hallucinations, 64k examples
- **Cons:** Still missing stochastic samples (would need to generate)
- **Effort:** Medium (2-3 weeks)
- **Recommendation:** Consider for future work (out of scope for current project)

**Option C: Create Custom Evaluation Dataset**
- **Pros:** Perfect fit for requirements, full control over domains
- **Cons:** High effort, expensive annotation
- **Effort:** High (4-6 weeks)
- **Recommendation:** Long-term goal if coherence variants prove highly effective

**Option D: Wait for Better Datasets**
- **Pros:** Hallucination detection datasets evolving rapidly
- **Cons:** Uncertain timeline
- **Recommendation:** Monitor dataset releases but don't block current work

### 7.3 Documentation for Future Dataset Requirements

For any future dataset to be compatible with coherence-based hallucination detection, it MUST have:

**Critical Requirements (Non-Negotiable):**
1. LLM-generated responses (not just ground truth)
2. Multiple stochastic samples per prompt (3-5 minimum)
3. Multi-sentence responses (5-10+ sentences ideal)
4. Sentence-level structure or tokenization capability

**Desirable Requirements:**
5. Sentence-level hallucination annotations
6. Diverse domains and response types
7. Public availability with permissive license
8. Sufficient size (100+ examples minimum)

**Anti-Patterns to Avoid:**
- Short-form QA datasets (like SimpleQA, TruthfulQA)
- Single-response datasets without samples
- Binary classification datasets without sentence-level granularity

---

## 8. Impact on Project Scope

### 8.1 Implications for Current Project

**Phase 4 (Dataset Investigation) Status:**
- **Task 4.1 (SimpleQA Analysis):** COMPLETE - determination is INCOMPATIBLE
- **Task 4.2 (SimpleQA Integration):** SKIPPED - conditional on Task 4.1, not applicable

**Phase 5 (End-to-End Validation) Implications:**
- Task 5.2.4 "Run on SimpleQA if compatible" → SKIPPED
- All validation will focus solely on wiki_bio_gpt3_hallucination dataset
- No changes to evaluation methodology required

### 8.2 Scope Adjustments

**No scope expansion needed:**
- Maintain wiki_bio as sole evaluation dataset (as originally planned)
- Focus resources on prompt optimization (Phase 2) rather than dataset adaptation
- Document dataset requirements for future work

**Documentation Updates:**
- Update spec.md and requirements.md to reflect SimpleQA incompatibility
- Add this compatibility report as reference for future dataset evaluation
- Document criteria for compatible datasets

---

## 9. Conclusion

SimpleQA is **fundamentally incompatible** with SelfCheckGPT's coherence-based hallucination detection approach. The dataset's short-form factuality focus (1-10 word answers) directly conflicts with the coherence method's requirement for multi-sentence responses with statement relationships.

**Key Blockers:**
1. No LLM-generated responses (only ground truth)
2. No multiple stochastic samples
3. No sentence-level structure
4. Single-word/phrase answers unsuitable for coherence analysis

**Recommendation:** Maintain wiki_bio_gpt3_hallucination as the sole evaluation dataset for this project. The dataset is well-suited for coherence analysis, already integrated, and provides a proven evaluation methodology.

**Future Work:** If dataset expansion is desired, HaluEval would be a better candidate than SimpleQA, though it would still require generating stochastic samples. Alternatively, creating a custom evaluation dataset modeled after wiki_bio but in different domains would provide the best fit for coherence-based detection.

---

## Appendices

### Appendix A: References

**SimpleQA Dataset:**
- HuggingFace: `lighteval/SimpleQA`, `basicv8vc/SimpleQA`
- GitHub: https://github.com/openai/simple-evals
- License: MIT

**wiki_bio_gpt3_hallucination Dataset:**
- HuggingFace: `potsawee/wiki_bio_gpt3_hallucination`
- Used in: SelfCheckGPT paper (EMNLP 2023)

**Alternative Datasets:**
- HaluEval: `pminervini/HaluEval` (64,507 examples)
- TruthfulQA: `truthful_qa` (817 questions)

### Appendix B: Technical Details

**Coherence Formula Requirements:**
```python
# All three coherence measures require TWO statements (A and B)

# Shogenji's coherence measure
def coherence_shogenji(P_A, P_B, P_A_and_B):
    """Requires: P(A), P(B), P(A∧B)"""
    return P_A_and_B / (P_A * P_B)

# Fitelson's confirmation measure
def coherence_fitelson(P_H, P_E, P_H_given_E, P_H_given_not_E):
    """Requires: P(H|E) and P(H|¬E)"""
    return P_H_given_E - P_H_given_not_E

# Olsson's overlap measure
def coherence_olsson(P_A, P_B, P_A_and_B):
    """Requires: P(A), P(B), P(A∧B), P(A∨B)"""
    P_A_or_B = P_A + P_B - P_A_and_B
    return P_A_and_B / P_A_or_B
```

**SimpleQA Example:**
```json
{
  "problem": "What is the capital of France?",
  "answer": "Paris"
}
```
This single-word answer cannot provide statements A and B for coherence measurement.

### Appendix C: Contact and Next Steps

**For questions about this assessment:**
- Reference: Task Group 4.1 in `/Users/nathanlubchenco/workspace/selfcheckgpt/agent-os/specs/2025-11-03-coherence-improvements/tasks.md`
- Related: Phase 2 (Prompt Optimization) can proceed without dataset expansion

**Next Actions:**
1. Mark Task Group 4.1 as complete in tasks.md
2. Skip Task Group 4.2 (SimpleQA Integration)
3. Proceed with Phase 2 (Prompt Optimization) using wiki_bio dataset
4. Document dataset requirements for future expansion opportunities

---

**End of Report**
