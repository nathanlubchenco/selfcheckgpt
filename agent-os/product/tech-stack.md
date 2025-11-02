# Tech Stack - SelfCheckGPT Coherence Extension

## Core Framework
- **Language:** Python 3.x
- **Package Manager:** pip
- **Distribution:** setuptools (setup.py-based)
- **Project Type:** Research Python package extension

## Coherence Computation Stack (NEW for this extension)

### Hybrid Architecture: LLM APIs + Mathematical Coherence

**LLM Integration (Probability Estimation)**
- **OpenAI API:** openai library for GPT-3.5-turbo, GPT-4, GPT-4o-mini access
- **Structured Output:** JSON schema for reliable probability extraction (OpenAI structured output feature)
- **Prompt Engineering:** Custom prompts to extract probability estimates (e.g., "Rate the probability this statement is true: [statement]")
- **API Client Management:** Leverage existing `SelfCheckAPIPrompt` infrastructure pattern
- **Decision (2025-11-02):** OpenAI only (no Groq) to leverage structured output support and reduce complexity

**Coherence Calculation (Mathematical Formulas)**
- **Numerical Computing:** NumPy (array operations, probability calculations)
- **Scientific Computing:** SciPy (statistical functions, optimization for coherence measures)
- **Statistical Analysis:** scikit-learn (evaluation metrics, statistical testing)
- **Coherence Theory Implementation:** Pure Python + NumPy (CPU-only, no GPU required)

**Decision Rationale (2025-11-02):**
- Structured output (JSON schema) is more reliable than text parsing for probability extraction
- OpenAI-only approach reduces complexity vs. supporting multiple providers
- Separates concerns: LLM APIs handle probability estimation, NumPy/SciPy handles coherence formulas
- Can add other providers later if they support similar structured output features

## Existing SelfCheckGPT Infrastructure (Leveraged)
- **Tokenization:** spacy (sentence splitting, consistent with existing variants)
- **NLP Utilities:** nltk (text preprocessing if needed)
- **Array Computing:** numpy (shared with coherence calculations)
- **Metrics:** scikit-learn (AUC-PR, PCC for benchmarking)

## Development & Testing
- **Interactive Testing:** Jupyter notebooks (.ipynb)
- **Demo Notebooks:** Jupyter notebooks in demo/ directory
- **Unit Tests:** pytest for coherence variant testing (NEW for this extension)
- **Visualization:** matplotlib, seaborn (performance comparison plots)
- **Linting:** None configured (follows existing project standards)
- **CI/CD:** None (research project)

## Data & Evaluation
- **Benchmark Dataset:** Hugging Face datasets (wiki_bio_gpt3_hallucination)
- **Evaluation Metrics:** AUC-PR, Pearson Correlation Coefficient (PCC)
- **Statistical Testing:** scipy.stats (significance testing)

## NOT Used in Coherence Extension
- **No PyTorch:** Coherence calculations are CPU-based mathematical operations (though LLMs are accessed via APIs)
- **No GPU:** No device parameter needed for coherence variants (API-based LLMs handle their own compute)
- **No Transformers:** No local pre-trained language models loaded
- **No bert_score:** Coherence measures replace semantic similarity

## Key Architectural Differences from Base SelfCheckGPT

### Existing Variants (GPU-based)
- SelfCheckNLI, MQAG, BERTScore, LLMPrompt all use PyTorch + transformers
- Require GPU acceleration for inference speed
- Load large pre-trained models (DeBERTa, T5, Longformer)

### Coherence Variants (Hybrid: API + CPU-based)
- SelfCheckShogenji, Fitelson, Olsson use OpenAI API + NumPy + SciPy
- **Step 1:** API calls to OpenAI for probability estimation (structured output with JSON schema)
- **Step 2:** CPU-only mathematical computations for coherence formulas
- No local model loading - APIs handle LLM inference remotely
- Lightweight local execution (only mathematical formulas run locally)

## Dependencies to Add

New requirements for coherence extension (to be added to setup.py):
```python
COHERENCE_REQUIREMENTS = [
    "scipy>=1.7",  # Statistical functions, optimization
    "pytest>=7.0",  # Unit testing framework
    "matplotlib>=3.5",  # Visualization
    "seaborn>=0.12",  # Statistical visualization
]
```

Existing dependencies already available (from base SelfCheckGPT):
- numpy (array operations)
- scikit-learn (evaluation metrics)
- spacy (sentence tokenization)
- **openai** (OpenAI API - already in base requirements)

## Development Environment

Minimal setup for coherence work:
```bash
pip install -e .  # Install base SelfCheckGPT (includes openai)
pip install scipy pytest matplotlib seaborn  # Add coherence dependencies
python -m spacy download en_core_web_sm  # Sentence tokenization
export OPENAI_API_KEY="your-key"  # Set API key for OpenAI
```

No GPU, Docker, or complex infrastructure required. Only API keys needed for LLM access.
