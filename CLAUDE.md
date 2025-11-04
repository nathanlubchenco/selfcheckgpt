# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SelfCheckGPT is a Python package for zero-resource black-box hallucination detection in LLM-generated text. It implements multiple detection strategies that use an LLM's own sampled outputs to identify inconsistencies without requiring external knowledge bases. The project is research-backed (EMNLP 2023) and production-ready.

**Core Concept**: Compare an LLM's generated text against multiple stochastically sampled outputs from the same LLM to detect hallucinations through inconsistency scoring.

## Development Commands

### Installation
```bash
pip install -e .  # Install in development mode
pip install selfcheckgpt  # Install from PyPI
```

### Testing
This project uses Jupyter notebooks for testing and demonstration rather than traditional unit tests:
```bash
# Run demo notebooks to test functionality
jupyter notebook demo/SelfCheck_demo1.ipynb
jupyter notebook demo/MQAG_demo1.ipynb
jupyter notebook demo/hallucination_detection.ipynb
jupyter notebook demo/coherence_demo.ipynb  # NEW: Coherence variants demo
```

### Package Building
```bash
python setup.py sdist bdist_wheel  # Build distribution packages
```

## Architecture

### Detection Strategies (Strategy Pattern)

The codebase implements nine hallucination detection variants across four modules, all sharing a common `predict()` interface:

#### Traditional Variants (modeling_selfcheck.py, modeling_selfcheck_apiprompt.py)

1. **SelfCheckNLI** (Recommended) - Uses DeBERTa-v3-large for natural language inference
   - Returns P(contradiction) as hallucination score [0.0, 1.0]
   - Best performing traditional method in paper evaluations

2. **SelfCheckMQAG** - Multiple-choice Question Answering & Generation
   - Generates questions from sentences using T5
   - Answers with Longformer (handles 4096 token context)
   - Three scoring methods: counting, vanilla Bayes, Bayes with alpha
   - Returns scores in [0.0, 1.0]

3. **SelfCheckBERTScore** - Semantic similarity using RoBERTa embeddings
   - Fast and efficient
   - Returns 1.0 - normalized BERTScore

4. **SelfCheckNgram** - Language model probability-based
   - Supports unigram (n=1), bigram (n=2), etc.
   - Returns unbounded negative log probabilities
   - Lightweight, interpretable

5. **SelfCheckLLMPrompt** - Open-source LLM prompting (Llama2, Mistral, etc.)
   - Zero-shot entailment assessment via prompting
   - Customizable prompt templates
   - Outputs converted: Yes→0.0, No→1.0, N/A→0.5

6. **SelfCheckAPIPrompt** - API-based LLM prompting (OpenAI, Groq)
   - Same approach as LLMPrompt but uses external APIs
   - Supports "openai" and "groq" client types
   - State-of-the-art performance with GPT-3.5/4 (93.42 AUC-PR)

#### Coherence-Based Variants (modeling_coherence.py) - NEW

7. **SelfCheckShogenji** - Shogenji's ratio-based independence measure
   - Formula: C2(A,B) = P(A ∧ B) / (P(A) × P(B))
   - Detects hallucinations through probabilistic independence violations
   - Returns scores in [0.0, 1.0] where higher = more hallucination

8. **SelfCheckFitelson** - Fitelson's confirmation-based support measure
   - Formula: s(H,E) = P(H|E) - P(H|¬E)
   - Measures asymmetric confirmation relationships
   - Requires conditional probability extraction (more API calls)
   - Returns scores in [0.0, 1.0]

9. **SelfCheckOlsson** - Glass-Olsson relative overlap measure
   - Formula: C1(A,B) = P(A ∧ B) / P(A ∨ B)
   - Measures agreement/overlap between statements
   - Returns scores in [0.0, 1.0]

**Coherence Variants Key Features:**
- Based on formal coherence theory from epistemology
- Uses OpenAI API with structured output for probability extraction
- Includes prompt-response caching to minimize API costs
- Provides cache statistics and cost estimation utilities
- Theoretically grounded approach to hallucination detection

### Common Interface

All detection methods implement:
```python
def predict(sentences: List[str], sampled_passages: List[str], **kwargs) -> np.ndarray
```

**Input**:
- `sentences`: List of sentences to evaluate (from LLM's response)
- `sampled_passages`: List of alternative outputs from same LLM (for consistency check)
- `passage`: Optional original context (for MQAG and n-gram methods)

**Output**: NumPy array of sentence-level scores where higher values indicate higher hallucination probability

### Key Design Patterns

1. **Lazy Initialization**: Models (especially in MQAG and LLMPrompt) are loaded on-demand to minimize memory usage
2. **Configuration Classes**: Pre-configured model URLs stored in `utils.py` (MQAGConfig, NLIConfig, LLMPromptConfig, CoherenceConfig)
3. **Modular Scoring**: MQAG implements three different scoring methodologies (counting, Bayes variants)
4. **API Caching**: Coherence variants implement LRU caching for prompt-response pairs to reduce API costs

## Module Structure

```
selfcheckgpt/
├── modeling_selfcheck.py          # Main detection variants (470 LOC)
│   ├── SelfCheckNLI               # NLI-based (recommended)
│   ├── SelfCheckMQAG              # QA-based
│   ├── SelfCheckBERTScore         # Semantic similarity
│   ├── SelfCheckNgram             # Language model
│   └── SelfCheckLLMPrompt         # Open-source LLM prompting
├── modeling_selfcheck_apiprompt.py  # API-based prompting (102 LOC)
│   └── SelfCheckAPIPrompt         # OpenAI/Groq integration
├── modeling_coherence.py          # Coherence-based variants (421 LOC) - NEW
│   ├── SelfCheckShogenji          # Shogenji's coherence measure
│   ├── SelfCheckFitelson          # Fitelson's confirmation measure
│   └── SelfCheckOlsson            # Glass-Olsson overlap measure
├── modeling_coherence_api.py      # Coherence API client (317 LOC) - NEW
│   └── CoherenceAPIClient         # OpenAI client with caching and structured output
├── modeling_mqag.py               # MQAG implementation (368 LOC)
│   └── MQAG                       # Standalone QA framework
├── modeling_ngram.py              # N-gram models (150 LOC)
│   ├── UnigramModel
│   └── NgramModel
├── utils.py                       # Configurations and utilities (122 LOC)
├── utils_coherence.py             # Coherence formulas (342 LOC) - NEW
│   ├── coherence_shogenji()       # Shogenji's ratio-based measure
│   ├── coherence_fitelson()       # Fitelson's confirmation measure
│   ├── coherence_olsson()         # Glass-Olsson overlap measure
│   └── normalize_coherence_scores() # Min-max normalization utility
└── version.py                     # Package version
```

## Important Implementation Details

### Sentence Tokenization
The codebase uses **spacy** for sentence splitting. Ensure consistency:
```python
import spacy
nlp = spacy.load("en_core_web_sm")
sentences = [sent.text.strip() for sent in nlp(passage).sents]
```

### Model Loading
Pre-trained models are hosted on HuggingFace under user `potsawee`:
- T5 models for question generation: `potsawee/t5-large-generation-squad-QuestionAnswer`
- Longformer for QA: `potsawee/longformer-large-4096-answering-race`
- DeBERTa for NLI: `potsawee/deberta-v3-large-mnli`

### Device Management
Traditional methods accept a `device` parameter for GPU acceleration:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
selfcheck_nli = SelfCheckNLI(device=device)
```

### Error Handling Behavior
- Invalid QA outputs are padded with duplicates
- Unknown LLM prompt responses (not Yes/No) are mapped to 0.5
- Answerability threshold (AT) filters low-quality generated questions
- Coherence variants clamp probabilities to valid range and handle numerical instabilities

## Coherence Variants Implementation Details

### Probability Extraction via OpenAI API

Coherence variants use OpenAI's structured output feature for reliable probability extraction:

**Prompt Templates:**
```python
individual_prob_template = "Rate the probability that this statement is true: {statement}"
joint_prob_template = "Rate the probability that both statements are true: {statement1} AND {statement2}"
conditional_prob_template = "Rate the probability that statement A is true: {statement1} GIVEN that {statement2} is true"
```

**Structured Output Schema:**
```json
{
  "type": "json_schema",
  "json_schema": {
    "name": "probability_response",
    "strict": true,
    "schema": {
      "type": "object",
      "properties": {
        "probability": {
          "type": "number",
          "description": "A probability value between 0.0 and 1.0"
        }
      },
      "required": ["probability"],
      "additionalProperties": false
    }
  }
}
```

### API Cost Management

**Caching Strategy:**
- LRU cache with max size 10,000 entries
- Cache keyed by (prompt_text, model_name) tuple
- Cache persists across sentences and variants within same session
- Cache statistics available via `client.get_cache_stats()`

**Cost Estimation:**
```python
from selfcheckgpt.modeling_coherence_api import CoherenceAPIClient

estimate = CoherenceAPIClient.estimate_api_calls(
    num_sentences=50,
    num_samples=5,
    num_variants=1,
    include_conditional=False  # Set True for Fitelson
)
```

**API Calls per Sentence:**
- Shogenji/Olsson: 1 + 2*num_samples (e.g., 11 calls for 5 samples)
- Fitelson: 1 + 3*num_samples (e.g., 16 calls for 5 samples)

**Recommended Model:** `gpt-4o-mini` for cost-efficiency with good performance

### Coherence Formula Implementations

**Numerical Stability:**
- Epsilon smoothing (1e-12) prevents division by zero
- Probabilities clamped to [epsilon, 1.0-epsilon] range
- Handles edge cases: P(A∧B) > P(A) or P(B) (numerical errors)
- Warnings issued for physically impossible probabilities

**Score Normalization:**
```python
# Min-max normalization to [0, 1] range
normalized = (scores - min) / (max - min)
# Handle edge case: all scores identical → return 0.5
# Invert to hallucination scores: 1.0 - normalized
```

## Dataset

The project includes the `wiki_bio_gpt3_hallucination` dataset:
- 238 annotated passages with sentence-level hallucination labels
- Accessible via HuggingFace: `load_dataset("potsawee/wiki_bio_gpt3_hallucination")`
- Used for evaluating detection methods
- Contains GPT-3 generated text compared against Wikipedia ground truth

## API Integration

### OpenAI Integration
```python
from selfcheckgpt.modeling_selfcheck_apiprompt import SelfCheckAPIPrompt
selfcheck = SelfCheckAPIPrompt(client_type="openai", model="gpt-3.5-turbo")
```

### Groq Integration
```python
selfcheck = SelfCheckAPIPrompt(client_type="groq", model="llama3-70b-8192", api_key="your-key")
```

### Coherence Variants (OpenAI Only)
```python
import os
os.environ['OPENAI_API_KEY'] = 'your-api-key-here'

from selfcheckgpt.modeling_coherence import SelfCheckShogenji, SelfCheckFitelson, SelfCheckOlsson

selfcheck_shogenji = SelfCheckShogenji(model="gpt-4o-mini")
selfcheck_fitelson = SelfCheckFitelson(model="gpt-4o-mini")
selfcheck_olsson = SelfCheckOlsson(model="gpt-4o-mini")
```

## Performance Benchmarks

Best results on wiki_bio_gpt3_hallucination dataset:

### Traditional Methods
- **SelfCheck-Prompt (gpt-3.5-turbo)**: 93.42 AUC-PR, 78.32 PCC
- **SelfCheck-NLI**: 92.50 AUC-PR, 74.14 PCC
- **SelfCheck-Prompt (Llama2-13B)**: 91.91 AUC-PR, 75.44 PCC

### Coherence Methods (To Be Evaluated)
- **SelfCheck-Shogenji (gpt-4o-mini)**: TBD
- **SelfCheck-Fitelson (gpt-4o-mini)**: TBD
- **SelfCheck-Olsson (gpt-4o-mini)**: TBD

Metrics: AUC-PR (Area Under Precision-Recall), PCC (Pearson Correlation Coefficient)

## Dependencies

Core requirements (from setup.py):
- `transformers>=4.35` - HuggingFace models
- `torch>=1.12` - PyTorch backend
- `bert_score` - Semantic similarity
- `spacy` - Sentence tokenization
- `nltk` - N-gram generation
- `openai` - OpenAI API (required for coherence variants)
- `groq` - Groq API

## Code References

When working with specific detection methods, refer to:
- Main SelfCheck variants: `selfcheckgpt/modeling_selfcheck.py:1-470`
- API prompting: `selfcheckgpt/modeling_selfcheck_apiprompt.py:1-102`
- Coherence variants: `selfcheckgpt/modeling_coherence.py:1-421`
- Coherence API client: `selfcheckgpt/modeling_coherence_api.py:1-317`
- Coherence formulas: `selfcheckgpt/utils_coherence.py:1-342`
- MQAG implementation: `selfcheckgpt/modeling_mqag.py:1-368`
- Configuration constants: `selfcheckgpt/utils.py:1-122`

## Documentation

- Main README: `README.md` - Installation, basic usage, all variants
- Coherence documentation: `docs/coherence_variants.md` - Detailed theory and usage for coherence variants
- Demo notebooks: `demo/` - Interactive examples and evaluations
- Theory reference: `agent-os/specs/2025-11-02_coherence-variants/planning/coherence-theory-reference.md`

## Development Notes

- No formal unit test suite - testing done via notebooks in `demo/`
- Use `torch.manual_seed()` before `predict()` calls for reproducibility
- The package uses `torch.no_grad()` for all inference operations
- Print statements are used for logging (no formal logging framework)
- Coherence variants use `tqdm` for progress bars when `verbose=True`
- API costs for coherence variants can be estimated before evaluation using `CoherenceAPIClient.estimate_api_calls()`
