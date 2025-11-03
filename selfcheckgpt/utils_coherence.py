"""
Coherence Measures for Hallucination Detection

This module implements three formal probabilistic coherence measures from epistemology
for use in SelfCheckGPT hallucination detection:

1. Shogenji's Coherence Measure (C2) - Ratio-based independence measure
2. Glass-Olsson Relative Overlap Measure (C1) - Proportion of agreement
3. Fitelson's Confirmation-Based Measure - Support-based confirmation

References:
- Shogenji, T. (1999). "Is Coherence Truth-conducive?", Analysis, 59: 338-345.
- Olsson, E. J. (2002). "What is the Problem of Coherence and Truth?",
  The Journal of Philosophy, 99: 246-272.
- Fitelson, B. (2003). "A Probabilistic Measure of Coherence", Analysis, 63: 194-199.
- Kemeny, J. and Oppenheim, P. (1952). "Degrees of Factual Support",
  Philosophy of Science, 19: 307-24.

For detailed theoretical background, see:
agent-os/specs/2025-11-02_coherence-variants/planning/coherence-theory-reference.md
"""

import numpy as np
import warnings


def coherence_shogenji(probs_individual, probs_joint, epsilon=1e-12):
    """
    Compute Shogenji's coherence measure (C2).

    Formula: C2(A,B) = P(A ∧ B) / (P(A) × P(B))

    This measure quantifies coherence by comparing the actual joint probability
    of beliefs to the expected joint probability if beliefs were probabilistically
    independent.

    Interpretation:
    - C2 = 1: Propositions are independent (neutral coherence)
    - C2 > 1: Positive coherence (mutual support)
    - C2 < 1: Negative coherence (conflict)

    Args:
        probs_individual (np.ndarray): Shape (n, 2) containing [P(A), P(B)]
            for each of n proposition pairs
        probs_joint (np.ndarray): Shape (n,) containing P(A ∧ B) for each pair
        epsilon (float): Smoothing constant to prevent division by zero (default: 1e-12)

    Returns:
        np.ndarray: Shape (n,) containing coherence scores in range (0, ∞)

    Example:
        >>> probs_ind = np.array([[0.7, 0.6], [0.3, 0.4]])
        >>> probs_joint = np.array([0.5, 0.1])
        >>> coherence_shogenji(probs_ind, probs_joint)
        array([1.19047619, 0.83333333])

    Reference:
        Shogenji, T. (1999). "Is Coherence Truth-conducive?", Analysis, 59: 338-345.
    """
    probs_individual = np.asarray(probs_individual)
    probs_joint = np.asarray(probs_joint)

    if probs_individual.ndim != 2 or probs_individual.shape[1] != 2:
        raise ValueError("probs_individual must have shape (n, 2)")
    if probs_joint.ndim != 1:
        raise ValueError("probs_joint must have shape (n,)")
    if len(probs_individual) != len(probs_joint):
        raise ValueError("probs_individual and probs_joint must have same length")

    # Clamp probabilities to valid range [epsilon, 1.0-epsilon]
    probs_individual = np.clip(probs_individual, epsilon, 1.0 - epsilon)
    probs_joint = np.clip(probs_joint, epsilon, 1.0 - epsilon)

    # Extract P(A) and P(B)
    p_a = probs_individual[:, 0]
    p_b = probs_individual[:, 1]

    # Check for physically impossible probabilities: P(A∧B) > P(A) or P(A∧B) > P(B)
    violations = (probs_joint > p_a) | (probs_joint > p_b)
    if np.any(violations):
        warnings.warn(
            f"Found {violations.sum()} cases where P(A∧B) > P(A) or P(A∧B) > P(B). "
            "This violates probability axioms. Clamping P(A∧B) to min(P(A), P(B)).",
            RuntimeWarning
        )
        probs_joint = np.minimum(probs_joint, np.minimum(p_a, p_b))

    # Calculate C2 = P(A∧B) / (P(A) × P(B))
    denominator = p_a * p_b + epsilon
    coherence = probs_joint / denominator

    return coherence


def coherence_olsson(probs_individual, probs_joint, epsilon=1e-12):
    """
    Compute Glass-Olsson relative overlap measure (C1).

    Formula: C1(A,B) = P(A ∧ B) / P(A ∨ B)
             = P(A ∧ B) / [P(A) + P(B) - P(A ∧ B)]

    This measure captures coherence as the proportion of agreement between
    propositions - the overlap relative to their union.

    Interpretation:
    - C1 = 1: Complete agreement (propositions are logically equivalent)
    - C1 = 0: Complete disagreement (propositions are disjoint)
    - 0 < C1 < 1: Partial overlap/agreement

    Args:
        probs_individual (np.ndarray): Shape (n, 2) containing [P(A), P(B)]
            for each of n proposition pairs
        probs_joint (np.ndarray): Shape (n,) containing P(A ∧ B) for each pair
        epsilon (float): Smoothing constant to prevent division by zero (default: 1e-12)

    Returns:
        np.ndarray: Shape (n,) containing coherence scores in range [0, 1]

    Example:
        >>> probs_ind = np.array([[0.7, 0.6], [0.3, 0.4]])
        >>> probs_joint = np.array([0.5, 0.1])
        >>> coherence_olsson(probs_ind, probs_joint)
        array([0.625, 0.16666667])

    Reference:
        Olsson, E. J. (2002). "What is the Problem of Coherence and Truth?",
        The Journal of Philosophy, 99: 246-272.
    """
    probs_individual = np.asarray(probs_individual)
    probs_joint = np.asarray(probs_joint)

    if probs_individual.ndim != 2 or probs_individual.shape[1] != 2:
        raise ValueError("probs_individual must have shape (n, 2)")
    if probs_joint.ndim != 1:
        raise ValueError("probs_joint must have shape (n,)")
    if len(probs_individual) != len(probs_joint):
        raise ValueError("probs_individual and probs_joint must have same length")

    # Clamp probabilities to valid range [epsilon, 1.0-epsilon]
    probs_individual = np.clip(probs_individual, epsilon, 1.0 - epsilon)
    probs_joint = np.clip(probs_joint, epsilon, 1.0 - epsilon)

    # Extract P(A) and P(B)
    p_a = probs_individual[:, 0]
    p_b = probs_individual[:, 1]

    # Check for physically impossible probabilities: P(A∧B) > P(A) or P(A∧B) > P(B)
    violations = (probs_joint > p_a) | (probs_joint > p_b)
    if np.any(violations):
        warnings.warn(
            f"Found {violations.sum()} cases where P(A∧B) > P(A) or P(A∧B) > P(B). "
            "This violates probability axioms. Clamping P(A∧B) to min(P(A), P(B)).",
            RuntimeWarning
        )
        probs_joint = np.minimum(probs_joint, np.minimum(p_a, p_b))

    # Calculate P(A ∨ B) = P(A) + P(B) - P(A ∧ B)
    p_union = p_a + p_b - probs_joint

    # Ensure P(A ∨ B) >= P(A ∧ B) (mathematical constraint)
    # This should always be true after the above calculation, but check for numerical errors
    violations_union = p_union < probs_joint - epsilon
    if np.any(violations_union):
        warnings.warn(
            f"Found {violations_union.sum()} cases where P(A∨B) < P(A∧B). "
            "This violates probability axioms. Setting P(A∨B) = P(A∧B).",
            RuntimeWarning
        )
        p_union = np.maximum(p_union, probs_joint)

    # Calculate C1 = P(A∧B) / P(A∨B)
    denominator = p_union + epsilon
    coherence = probs_joint / denominator

    # Clamp to [0, 1] range (should already be in range, but ensure numerical stability)
    coherence = np.clip(coherence, 0.0, 1.0)

    return coherence


def coherence_fitelson(probs_individual, probs_joint, probs_conditional, epsilon=1e-12):
    """
    Compute Fitelson's confirmation-based coherence measure.

    This implementation uses the difference measure from the Kemeny & Oppenheim
    factual support tradition:

    Formula: s(H,E) = P(H|E) - P(H|¬E)

    Where:
    - P(H|¬E) is computed via Bayes' theorem: P(H|¬E) = [P(H) - P(E) × P(H|E)] / [1 - P(E)]

    If P(H|¬E) calculation is unstable, falls back to simple confirmation:
    s(H,E) = P(H|E) - P(H)

    This measure examines support relations, quantifying how strongly propositions
    confirm or disconfirm each other.

    Interpretation:
    - s = +1: Maximum positive support (E fully confirms H)
    - s = 0: No support (E is irrelevant to H)
    - s = -1: Maximum negative support (E contradicts H)

    Args:
        probs_individual (np.ndarray): Shape (n, 2) containing [P(H), P(E)]
            for each of n proposition pairs (hypothesis, evidence)
        probs_joint (np.ndarray): Shape (n,) containing P(H ∧ E) for each pair
        probs_conditional (np.ndarray): Shape (n,) containing P(H|E) for each pair
        epsilon (float): Smoothing constant to prevent division by zero (default: 1e-12)

    Returns:
        np.ndarray: Shape (n,) containing support scores in range [-1, 1]

    Example:
        >>> probs_ind = np.array([[0.5, 0.6], [0.3, 0.4]])
        >>> probs_joint = np.array([0.4, 0.15])
        >>> probs_cond = np.array([0.667, 0.375])  # P(H|E)
        >>> coherence_fitelson(probs_ind, probs_joint, probs_cond)
        array([0.417, 0.125])

    Reference:
        Fitelson, B. (2003). "A Probabilistic Measure of Coherence", Analysis, 63: 194-199.
        Kemeny, J. and Oppenheim, P. (1952). "Degrees of Factual Support",
        Philosophy of Science, 19: 307-24.
    """
    probs_individual = np.asarray(probs_individual)
    probs_joint = np.asarray(probs_joint)
    probs_conditional = np.asarray(probs_conditional)

    if probs_individual.ndim != 2 or probs_individual.shape[1] != 2:
        raise ValueError("probs_individual must have shape (n, 2)")
    if probs_joint.ndim != 1:
        raise ValueError("probs_joint must have shape (n,)")
    if probs_conditional.ndim != 1:
        raise ValueError("probs_conditional must have shape (n,)")
    if not (len(probs_individual) == len(probs_joint) == len(probs_conditional)):
        raise ValueError("All probability arrays must have same length")

    # Clamp probabilities to valid range [epsilon, 1.0-epsilon]
    probs_individual = np.clip(probs_individual, epsilon, 1.0 - epsilon)
    probs_joint = np.clip(probs_joint, epsilon, 1.0 - epsilon)
    probs_conditional = np.clip(probs_conditional, epsilon, 1.0 - epsilon)

    # Extract P(H) and P(E)
    p_h = probs_individual[:, 0]
    p_e = probs_individual[:, 1]
    p_h_given_e = probs_conditional

    # Initialize support scores array
    support = np.zeros(len(probs_individual))

    # Try to compute P(H|¬E) using Bayes' theorem: P(H|¬E) = [P(H) - P(E) × P(H|E)] / [1 - P(E)]
    # This can be unstable when P(E) ≈ 1 or when numerator goes negative

    numerator = p_h - p_e * p_h_given_e
    denominator = 1.0 - p_e + epsilon

    # Check for unstable cases
    unstable = (denominator < 10 * epsilon) | (numerator < -epsilon)

    if np.any(unstable):
        # Use simple confirmation measure for unstable cases: s(H,E) = P(H|E) - P(H)
        support[unstable] = p_h_given_e[unstable] - p_h[unstable]

        if np.any(~unstable):
            # Use full difference measure for stable cases
            p_h_given_not_e = numerator[~unstable] / denominator[~unstable]
            # Clamp to [0, 1]
            p_h_given_not_e = np.clip(p_h_given_not_e, 0.0, 1.0)
            # Calculate difference measure: s(H,E) = P(H|E) - P(H|¬E)
            support[~unstable] = p_h_given_e[~unstable] - p_h_given_not_e
    else:
        # All cases are stable, use full difference measure
        p_h_given_not_e = numerator / denominator
        # Clamp to [0, 1]
        p_h_given_not_e = np.clip(p_h_given_not_e, 0.0, 1.0)
        # Calculate difference measure: s(H,E) = P(H|E) - P(H|¬E)
        support = p_h_given_e - p_h_given_not_e

    # Clamp final support scores to [-1, 1] range
    support = np.clip(support, -1.0, 1.0)

    return support


def normalize_coherence_scores(scores):
    """
    Normalize coherence scores to [0.0, 1.0] range using min-max normalization.

    This utility function is used to normalize coherence scores before inverting
    them to hallucination scores. Min-max normalization ensures all scores are
    in a consistent [0, 1] range.

    Formula: normalized = (score - min) / (max - min)

    Edge cases:
    - If all scores are identical (max == min), returns 0.5 for all scores
    - If scores contain NaN or Inf, they are handled gracefully

    Args:
        scores (np.ndarray): Array of coherence scores to normalize

    Returns:
        np.ndarray: Normalized scores in range [0.0, 1.0]

    Example:
        >>> scores = np.array([0.5, 1.0, 1.5, 2.0])
        >>> normalize_coherence_scores(scores)
        array([0.0, 0.33333333, 0.66666667, 1.0])

        >>> identical = np.array([1.5, 1.5, 1.5])
        >>> normalize_coherence_scores(identical)
        array([0.5, 0.5, 0.5])
    """
    scores = np.asarray(scores)

    # Handle NaN and Inf
    valid_mask = np.isfinite(scores)
    if not np.any(valid_mask):
        # All scores are NaN or Inf, return 0.5
        return np.full_like(scores, 0.5, dtype=np.float64)

    # Get min and max of valid scores
    min_score = np.min(scores[valid_mask])
    max_score = np.max(scores[valid_mask])

    # Check if all valid scores are identical
    if np.abs(max_score - min_score) < 1e-12:
        # All scores are identical, return 0.5
        return np.full_like(scores, 0.5, dtype=np.float64)

    # Apply min-max normalization
    normalized = (scores - min_score) / (max_score - min_score)

    # Replace NaN/Inf values with 0.5
    normalized[~valid_mask] = 0.5

    # Clamp to [0, 1] to handle any numerical errors
    normalized = np.clip(normalized, 0.0, 1.0)

    return normalized
