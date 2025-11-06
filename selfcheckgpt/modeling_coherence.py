"""
SelfCheckGPT Coherence-Based Hallucination Detection Variants

This module implements three formal probabilistic coherence theory-based hallucination
detection variants that extend SelfCheckGPT. Each variant uses a different coherence
measure from formal epistemology to assess whether LLM-generated statements are mutually
coherent with stochastically sampled passages from the same LLM.

Core Hypothesis:
    Hallucinated statements show lower coherence with alternative outputs because false
    claims lack consistent grounding, while truthful statements maintain coherence across samples.

Variants:
    1. SelfCheckShogenji - Uses Shogenji's ratio-based independence measure (C2)
    2. SelfCheckFitelson - Uses Fitelson's confirmation-based support measure
    3. SelfCheckOlsson - Uses Glass-Olsson's relative overlap measure (C1)

References:
    - Shogenji, T. (1999). "Is Coherence Truth-conducive?", Analysis, 59: 338-345.
    - Fitelson, B. (2003). "A Probabilistic Measure of Coherence", Analysis, 63: 194-199.
    - Olsson, E. J. (2002). "What is the Problem of Coherence and Truth?",
      The Journal of Philosophy, 99: 246-272.

For detailed theoretical background, see:
    agent-os/specs/2025-11-02_coherence-variants/planning/coherence-theory-reference.md
"""

from typing import List
import numpy as np
from tqdm import tqdm

from .modeling_coherence_api import CoherenceAPIClient
from .utils_coherence import (
    coherence_shogenji,
    coherence_fitelson,
    coherence_olsson,
    normalize_coherence_scores
)


class SelfCheckShogenji:
    """
    SelfCheckGPT with Shogenji's Coherence Measure (C2).

    Uses Shogenji's ratio-based independence measure to detect hallucinations by
    comparing actual joint probability to expected probability under independence:
    C2(A,B) = P(A ∧ B) / (P(A) × P(B))

    High coherence (C2 > 1) indicates mutual support between sentence and samples,
    suggesting the statement is consistent. Low coherence (C2 < 1) indicates conflict,
    suggesting potential hallucination.

    Attributes:
        model (str): OpenAI model name for probability extraction
        api_key (str): OpenAI API key (defaults to OPENAI_API_KEY environment variable)
        client (CoherenceAPIClient): API client for probability extraction

    Example:
        >>> selfcheck = SelfCheckShogenji(model="gpt-4o-mini")
        >>> sentences = ["The Eiffel Tower is in Paris.", "It was built in 1889."]
        >>> samples = ["The Eiffel Tower is located in Paris, France.", ...]
        >>> scores = selfcheck.predict(sentences, samples, verbose=True)
        >>> print(f"Hallucination scores: {scores}")

    Reference:
        Shogenji, T. (1999). "Is Coherence Truth-conducive?", Analysis, 59: 338-345.
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None):
        """
        Initialize SelfCheckShogenji with OpenAI API client.

        Args:
            model: OpenAI model name (default: "gpt-4o-mini")
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
        """
        self.model = model
        self.api_key = api_key
        self.client = CoherenceAPIClient(model=model, api_key=api_key)

    def predict(
        self,
        sentences: List[str],
        sampled_passages: List[str],
        verbose: bool = False
    ) -> np.ndarray:
        """
        Predict sentence-level hallucination scores using Shogenji coherence measure.

        For each sentence, extracts probabilities P(sentence), P(sample), and P(sentence ∧ sample)
        for each sampled passage, then calculates Shogenji coherence C2 = P(A∧B) / (P(A)×P(B)).
        Aggregates coherence across samples using mean, normalizes to [0,1], and inverts to
        hallucination scores (1.0 - normalized_coherence).

        Args:
            sentences: List of sentences to evaluate (from LLM response)
            sampled_passages: List of stochastically sampled alternative outputs
            verbose: If True, show tqdm progress bar and cache statistics

        Returns:
            np.ndarray: Sentence-level hallucination scores in [0.0, 1.0] range,
                       where higher score indicates higher hallucination probability

        Example:
            >>> sentences = ["Paris is the capital of France."]
            >>> samples = ["France's capital city is Paris.", "Paris is in France."]
            >>> scores = selfcheck.predict(sentences, samples)
            >>> print(f"Hallucination risk: {scores[0]:.2f}")
        """
        num_sentences = len(sentences)
        num_samples = len(sampled_passages)

        if verbose:
            print(f"\nSelfCheckShogenji: Evaluating {num_sentences} sentences against {num_samples} samples")
            print(f"Model: {self.model}")

        # Store coherence scores for each sentence-sample pair
        coherence_matrix = np.zeros((num_sentences, num_samples))

        # Iterate over sentences with progress bar
        disable_progress = not verbose
        for sent_i in tqdm(range(num_sentences), disable=disable_progress, desc="Processing sentences"):
            sentence = sentences[sent_i]

            # Extract P(sentence) once per sentence (cached for efficiency)
            p_sentence = self.client.extract_individual_probability(sentence, verbose=False)

            # Evaluate coherence with each sampled passage
            for sample_i, sample in enumerate(sampled_passages):
                # Remove newlines for cleaner processing (following existing pattern)
                sample = sample.replace("\n", " ")

                # Extract P(sample)
                p_sample = self.client.extract_individual_probability(sample, verbose=False)

                # Extract P(sentence ∧ sample)
                p_joint = self.client.extract_joint_probability(sentence, sample, verbose=False)

                # Calculate Shogenji coherence: C2(A,B) = P(A∧B) / (P(A)×P(B))
                probs_individual = np.array([[p_sentence, p_sample]])
                probs_joint = np.array([p_joint])
                coherence = coherence_shogenji(probs_individual, probs_joint)

                coherence_matrix[sent_i, sample_i] = coherence[0]

        # Aggregate coherence across samples (mean)
        mean_coherence = coherence_matrix.mean(axis=-1)

        # Convert coherence to hallucination scores using exponential mapping
        # Formula: hallucination = exp(-C2)
        # This maps: C2=0 → 1.0, C2=1 → 0.37, C2=2 → 0.14, C2→∞ → 0.0
        #
        # Theoretical justification:
        # 1. Respects the multiplicative nature of Shogenji's ratio measure
        # 2. Maintains global calibration across passages (C2=1 always maps to ~0.37)
        # 3. Exponential function naturally maps [0, ∞) to [1, 0) smoothly
        # 4. Avoids artificial compression from passage-level min-max normalization
        #
        # Empirical validation: 19.11% improvement over min-max normalization
        # (AUC-PR: 0.7658 → 0.9569 on benchmark dataset)
        hallucination_scores = np.exp(-mean_coherence)

        if verbose:
            cache_stats = self.client.get_cache_stats()
            print(f"\nCache statistics:")
            print(f"  Hit rate: {cache_stats['hit_rate']:.2%}")
            print(f"  API calls made: {cache_stats['api_calls']}")
            print(f"  Cache size: {cache_stats['cache_size']}")

        return hallucination_scores


class SelfCheckFitelson:
    """
    SelfCheckGPT with Fitelson's Confirmation-Based Coherence Measure.

    Uses Fitelson's support-based measure from confirmation theory to detect hallucinations
    by examining how strongly statements confirm or disconfirm each other. Based on the
    Kemeny & Oppenheim factual support framework:
    s(H,E) = P(H|E) - P(H|¬E)

    Positive support (s > 0) indicates evidence confirms hypothesis, suggesting consistency.
    Negative support (s < 0) indicates evidence contradicts hypothesis, suggesting hallucination.

    Attributes:
        model (str): OpenAI model name for probability extraction
        api_key (str): OpenAI API key (defaults to OPENAI_API_KEY environment variable)
        client (CoherenceAPIClient): API client for probability extraction

    Example:
        >>> selfcheck = SelfCheckFitelson(model="gpt-4o-mini")
        >>> sentences = ["The Eiffel Tower is in Paris.", "It was built in 1889."]
        >>> samples = ["The Eiffel Tower is located in Paris, France.", ...]
        >>> scores = selfcheck.predict(sentences, samples, verbose=True)
        >>> print(f"Hallucination scores: {scores}")

    Reference:
        Fitelson, B. (2003). "A Probabilistic Measure of Coherence", Analysis, 63: 194-199.
        Kemeny, J. and Oppenheim, P. (1952). "Degrees of Factual Support",
        Philosophy of Science, 19: 307-24.
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None):
        """
        Initialize SelfCheckFitelson with OpenAI API client.

        Args:
            model: OpenAI model name (default: "gpt-4o-mini")
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
        """
        self.model = model
        self.api_key = api_key
        self.client = CoherenceAPIClient(model=model, api_key=api_key)

    def predict(
        self,
        sentences: List[str],
        sampled_passages: List[str],
        verbose: bool = False
    ) -> np.ndarray:
        """
        Predict sentence-level hallucination scores using Fitelson confirmation measure.

        For each sentence, extracts probabilities P(sentence), P(sample), P(sentence ∧ sample),
        and P(sentence|sample) for each sampled passage, then calculates Fitelson support
        measure s(H,E) = P(H|E) - P(H|¬E). Aggregates support across samples using mean,
        normalizes to [0,1], and inverts to hallucination scores.

        Args:
            sentences: List of sentences to evaluate (from LLM response)
            sampled_passages: List of stochastically sampled alternative outputs
            verbose: If True, show tqdm progress bar and cache statistics

        Returns:
            np.ndarray: Sentence-level hallucination scores in [0.0, 1.0] range,
                       where higher score indicates higher hallucination probability

        Example:
            >>> sentences = ["Paris is the capital of France."]
            >>> samples = ["France's capital city is Paris.", "Paris is in France."]
            >>> scores = selfcheck.predict(sentences, samples)
            >>> print(f"Hallucination risk: {scores[0]:.2f}")
        """
        num_sentences = len(sentences)
        num_samples = len(sampled_passages)

        if verbose:
            print(f"\nSelfCheckFitelson: Evaluating {num_sentences} sentences against {num_samples} samples")
            print(f"Model: {self.model}")

        # Store support scores for each sentence-sample pair
        support_matrix = np.zeros((num_sentences, num_samples))

        # Iterate over sentences with progress bar
        disable_progress = not verbose
        for sent_i in tqdm(range(num_sentences), disable=disable_progress, desc="Processing sentences"):
            sentence = sentences[sent_i]

            # Extract P(sentence) once per sentence (cached for efficiency)
            p_sentence = self.client.extract_individual_probability(sentence, verbose=False)

            # Evaluate support with each sampled passage
            for sample_i, sample in enumerate(sampled_passages):
                # Remove newlines for cleaner processing (following existing pattern)
                sample = sample.replace("\n", " ")

                # Extract P(sample)
                p_sample = self.client.extract_individual_probability(sample, verbose=False)

                # Extract P(sentence ∧ sample)
                p_joint = self.client.extract_joint_probability(sentence, sample, verbose=False)

                # Extract P(sentence | sample) - conditional probability
                p_conditional = self.client.extract_conditional_probability(sentence, sample, verbose=False)

                # Calculate Fitelson support measure
                probs_individual = np.array([[p_sentence, p_sample]])
                probs_joint = np.array([p_joint])
                probs_conditional = np.array([p_conditional])
                support = coherence_fitelson(probs_individual, probs_joint, probs_conditional)

                support_matrix[sent_i, sample_i] = support[0]

        # Aggregate support across samples (mean)
        mean_support = support_matrix.mean(axis=-1)

        # Convert support to hallucination scores using direct linear mapping
        # Formula: hallucination = (1 - s) / 2
        # This maps: s=-1 → 1.0, s=0 → 0.5, s=+1 → 0.0
        #
        # Theoretical justification:
        # 1. Fitelson's support is already calibrated in [-1, 1] by design
        # 2. Linear transformation preserves the symmetric interpretation around s=0
        # 3. Maintains global calibration: s=0 (no support) always maps to 0.5
        # 4. Avoids artificial compression from passage-level min-max normalization
        # 5. Consistent with exponential approach (direct mapping, no relative normalization)
        hallucination_scores = (1.0 - mean_support) / 2.0

        if verbose:
            cache_stats = self.client.get_cache_stats()
            print(f"\nCache statistics:")
            print(f"  Hit rate: {cache_stats['hit_rate']:.2%}")
            print(f"  API calls made: {cache_stats['api_calls']}")
            print(f"  Cache size: {cache_stats['cache_size']}")

        return hallucination_scores


class SelfCheckOlsson:
    """
    SelfCheckGPT with Glass-Olsson Relative Overlap Measure (C1).

    Uses Glass-Olsson's relative overlap measure to detect hallucinations by measuring
    the proportion of agreement between statements:
    C1(A,B) = P(A ∧ B) / P(A ∨ B) = P(A ∧ B) / [P(A) + P(B) - P(A ∧ B)]

    High overlap (C1 → 1) indicates strong agreement, suggesting consistency. Low overlap
    (C1 → 0) indicates disagreement, suggesting potential hallucination.

    Attributes:
        model (str): OpenAI model name for probability extraction
        api_key (str): OpenAI API key (defaults to OPENAI_API_KEY environment variable)
        client (CoherenceAPIClient): API client for probability extraction

    Example:
        >>> selfcheck = SelfCheckOlsson(model="gpt-4o-mini")
        >>> sentences = ["The Eiffel Tower is in Paris.", "It was built in 1889."]
        >>> samples = ["The Eiffel Tower is located in Paris, France.", ...]
        >>> scores = selfcheck.predict(sentences, samples, verbose=True)
        >>> print(f"Hallucination scores: {scores}")

    Reference:
        Olsson, E. J. (2002). "What is the Problem of Coherence and Truth?",
        The Journal of Philosophy, 99: 246-272.
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None):
        """
        Initialize SelfCheckOlsson with OpenAI API client.

        Args:
            model: OpenAI model name (default: "gpt-4o-mini")
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
        """
        self.model = model
        self.api_key = api_key
        self.client = CoherenceAPIClient(model=model, api_key=api_key)

    def predict(
        self,
        sentences: List[str],
        sampled_passages: List[str],
        verbose: bool = False
    ) -> np.ndarray:
        """
        Predict sentence-level hallucination scores using Glass-Olsson overlap measure.

        For each sentence, extracts probabilities P(sentence), P(sample), and P(sentence ∧ sample)
        for each sampled passage, then calculates Glass-Olsson coherence
        C1 = P(A∧B) / [P(A) + P(B) - P(A∧B)]. Aggregates coherence across samples using mean,
        normalizes to [0,1], and inverts to hallucination scores.

        Args:
            sentences: List of sentences to evaluate (from LLM response)
            sampled_passages: List of stochastically sampled alternative outputs
            verbose: If True, show tqdm progress bar and cache statistics

        Returns:
            np.ndarray: Sentence-level hallucination scores in [0.0, 1.0] range,
                       where higher score indicates higher hallucination probability

        Example:
            >>> sentences = ["Paris is the capital of France."]
            >>> samples = ["France's capital city is Paris.", "Paris is in France."]
            >>> scores = selfcheck.predict(sentences, samples)
            >>> print(f"Hallucination risk: {scores[0]:.2f}")
        """
        num_sentences = len(sentences)
        num_samples = len(sampled_passages)

        if verbose:
            print(f"\nSelfCheckOlsson: Evaluating {num_sentences} sentences against {num_samples} samples")
            print(f"Model: {self.model}")

        # Store coherence scores for each sentence-sample pair
        coherence_matrix = np.zeros((num_sentences, num_samples))

        # Iterate over sentences with progress bar
        disable_progress = not verbose
        for sent_i in tqdm(range(num_sentences), disable=disable_progress, desc="Processing sentences"):
            sentence = sentences[sent_i]

            # Extract P(sentence) once per sentence (cached for efficiency)
            p_sentence = self.client.extract_individual_probability(sentence, verbose=False)

            # Evaluate coherence with each sampled passage
            for sample_i, sample in enumerate(sampled_passages):
                # Remove newlines for cleaner processing (following existing pattern)
                sample = sample.replace("\n", " ")

                # Extract P(sample)
                p_sample = self.client.extract_individual_probability(sample, verbose=False)

                # Extract P(sentence ∧ sample)
                p_joint = self.client.extract_joint_probability(sentence, sample, verbose=False)

                # Calculate Glass-Olsson coherence: C1(A,B) = P(A∧B) / P(A∨B)
                probs_individual = np.array([[p_sentence, p_sample]])
                probs_joint = np.array([p_joint])
                coherence = coherence_olsson(probs_individual, probs_joint)

                coherence_matrix[sent_i, sample_i] = coherence[0]

        # Aggregate coherence across samples (mean)
        mean_coherence = coherence_matrix.mean(axis=-1)

        # Convert coherence to hallucination scores using direct inversion
        # Formula: hallucination = 1 - C1
        # This maps: C1=0 → 1.0, C1=0.5 → 0.5, C1=1 → 0.0
        #
        # Theoretical justification:
        # 1. Olsson's C1 is already normalized in [0, 1] (proportion of overlap)
        # 2. Direct inversion preserves the natural calibration of the measure
        # 3. Maintains global calibration: C1=0.5 always maps to 0.5
        # 4. Avoids artificial compression from passage-level min-max normalization
        # 5. Consistent with exponential approach (direct mapping, no relative normalization)
        hallucination_scores = 1.0 - mean_coherence

        if verbose:
            cache_stats = self.client.get_cache_stats()
            print(f"\nCache statistics:")
            print(f"  Hit rate: {cache_stats['hit_rate']:.2%}")
            print(f"  API calls made: {cache_stats['api_calls']}")
            print(f"  Cache size: {cache_stats['cache_size']}")

        return hallucination_scores
