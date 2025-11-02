# Spec Initialization: Coherence Theory-Based Hallucination Detection

**Date:** 2025-11-02

**Spec Name:** coherence-variants

**Initial Description:**

Implement three formal coherence theory-based hallucination detection variants for SelfCheckGPT:

1. **SelfCheckShogenji** - Probabilistic coherence measure
2. **SelfCheckFitelson** - Confirmation-theoretic coherence measure
3. **SelfCheckOlsson** - Support-based coherence measure

**Context:**

This extends the existing SelfCheckGPT package (EMNLP 2023) which currently has 6 hallucination detection methods. The new coherence variants will:

- Use a hybrid architecture: OpenAI/Groq APIs for probability estimation + NumPy/SciPy for coherence formulas
- Implement the same `predict(sentences, sampled_passages)` interface as existing methods
- Be evaluated on the wiki_bio_gpt3_hallucination dataset (238 annotated passages)
- Aim to outperform current best (93.42 AUC-PR) by measuring logical support relationships rather than surface consistency

**Technical Approach (Decided):**
- Prompt-based probability extraction from LLMs (most general, not restricted to logprobs)
- Leverage existing `SelfCheckAPIPrompt` infrastructure
- CPU-only coherence calculations (no GPU needed)

**Product Documentation:**
- Mission: agent-os/product/mission.md
- Roadmap: agent-os/product/roadmap.md (15 items across 3 phases)
- Tech stack: agent-os/product/tech-stack.md

**Status:** Initialized - Ready for requirements gathering
