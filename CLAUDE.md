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
```

### Package Building
```bash
python setup.py sdist bdist_wheel  # Build distribution packages
```

## Architecture

### Detection Strategies (Strategy Pattern)

The codebase implements six hallucination detection variants in `selfcheckgpt/modeling_selfcheck.py` and `selfcheckgpt/modeling_selfcheck_apiprompt.py`, all sharing a common `predict()` interface:

1. **SelfCheckNLI** (Recommended) - Uses DeBERTa-v3-large for natural language inference
   - Returns P(contradiction) as hallucination score [0.0, 1.0]
   - Best performing method in paper evaluations

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
   - State-of-the-art performance with GPT-3.5/4

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
2. **Configuration Classes**: Pre-configured model URLs stored in `utils.py` (MQAGConfig, NLIConfig, LLMPromptConfig)
3. **Modular Scoring**: MQAG implements three different scoring methodologies (counting, Bayes variants)

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
├── modeling_mqag.py               # MQAG implementation (368 LOC)
│   └── MQAG                       # Standalone QA framework
├── modeling_ngram.py              # N-gram models (150 LOC)
│   ├── UnigramModel
│   └── NgramModel
├── utils.py                       # Configurations and utilities (122 LOC)
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
All methods accept a `device` parameter for GPU acceleration:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
selfcheck_nli = SelfCheckNLI(device=device)
```

### Error Handling Behavior
- Invalid QA outputs are padded with duplicates
- Unknown LLM prompt responses (not Yes/No) are mapped to 0.5
- Answerability threshold (AT) filters low-quality generated questions

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

## Performance Benchmarks

Best results on wiki_bio_gpt3_hallucination dataset:
- **SelfCheck-Prompt (gpt-3.5-turbo)**: 93.42 AUC-PR, 78.32 PCC
- **SelfCheck-NLI**: 92.50 AUC-PR, 74.14 PCC
- **SelfCheck-Prompt (Llama2-13B)**: 91.91 AUC-PR, 75.44 PCC

Metrics: AUC-PR (Area Under Precision-Recall), PCC (Pearson Correlation Coefficient)

## Dependencies

Core requirements (from setup.py):
- `transformers>=4.35` - HuggingFace models
- `torch>=1.12` - PyTorch backend
- `bert_score` - Semantic similarity
- `spacy` - Sentence tokenization
- `nltk` - N-gram generation
- `openai` - OpenAI API
- `groq` - Groq API

## Code References

When working with specific detection methods, refer to:
- Main SelfCheck variants: `selfcheckgpt/modeling_selfcheck.py:1-470`
- API prompting: `selfcheckgpt/modeling_selfcheck_apiprompt.py:1-102`
- MQAG implementation: `selfcheckgpt/modeling_mqag.py:1-368`
- Configuration constants: `selfcheckgpt/utils.py:1-122`

## Development Notes

- No formal unit test suite - testing done via notebooks in `demo/`
- Use `torch.manual_seed()` before `predict()` calls for reproducibility
- The package uses `torch.no_grad()` for all inference operations
- Print statements are used for logging (no formal logging framework)
