## Tech Stack - SelfCheckGPT Research Project

### Core Framework
- **Language:** Python 3.x
- **Package Manager:** pip
- **Distribution:** setuptools (setup.py-based)
- **Project Type:** Research Python package

### Machine Learning & NLP
- **Deep Learning:** PyTorch (torch >= 1.12)
- **Transformers:** Hugging Face transformers (>= 4.35)
- **Tokenizers:** sentencepiece (>= 0.2.0)
- **NLP:** spacy, nltk
- **Semantic Similarity:** bert_score
- **Metrics:** scikit-learn, numpy

### LLM API Clients
- **OpenAI:** openai Python client
- **Groq:** groq Python client

### Development & Testing
- **Interactive Testing:** Jupyter notebooks (.ipynb)
- **Demos:** Jupyter notebooks in demo/ directory
- **Unit Tests:** None (research project)
- **Linting:** None configured
- **CI/CD:** None

### Data & Models
- **Datasets:** Hugging Face datasets hub
- **Models:** Hugging Face model hub
- **Pre-trained Models:**
  - T5 (question generation)
  - Longformer (QA)
  - DeBERTa-v3 (NLI)
  - RoBERTa (BERTScore)
  - Llama2, Mistral (prompting)

### Not Used
- No database
- No API framework
- No frontend
- No deployment infrastructure
- No migrations
