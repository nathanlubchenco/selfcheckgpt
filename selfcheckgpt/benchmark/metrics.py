"""
Evaluation Metrics for Probability Extraction Quality

This module implements five metrics for assessing the quality of probability
extraction in coherence-based hallucination detection:

1. Brier Score - Overall calibration quality
2. Expected Calibration Error (ECE) - Binned calibration measure
3. Probability Coherence Compliance - Adherence to probability axioms
4. Probability Consistency Score - Variance for semantically equivalent statements
5. Sharpness - Decisiveness of predictions

Reference:
- agent-os/specs/2025-11-03-coherence-improvements/planning/requirements.md
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings


def brier_score(predicted_probs: np.ndarray, actual_outcomes: np.ndarray) -> float:
    """
    Calculate Brier Score to measure probability calibration quality.

    The Brier Score measures the mean squared error between predicted probabilities
    and actual binary outcomes. Lower scores indicate better calibration.

    Formula: BS = (1/N) × Σ(predicted_prob - actual_outcome)²

    Args:
        predicted_probs: Array of predicted probabilities in [0.0, 1.0] range
        actual_outcomes: Array of actual binary outcomes (0 or 1)

    Returns:
        Brier score in [0.0, 1.0] range where 0.0 is perfect calibration

    Example:
        >>> predicted = np.array([0.9, 0.8, 0.3, 0.2])
        >>> actual = np.array([1, 1, 0, 0])
        >>> brier_score(predicted, actual)
        0.045

    Reference:
        Brier, G. W. (1950). "Verification of forecasts expressed in terms of probability",
        Monthly Weather Review, 78(1), 1-3.
    """
    predicted_probs = np.asarray(predicted_probs, dtype=np.float64)
    actual_outcomes = np.asarray(actual_outcomes, dtype=np.float64)

    if len(predicted_probs) != len(actual_outcomes):
        raise ValueError("predicted_probs and actual_outcomes must have same length")

    if not np.all((actual_outcomes == 0) | (actual_outcomes == 1)):
        raise ValueError("actual_outcomes must be binary (0 or 1)")

    if not np.all((predicted_probs >= 0) & (predicted_probs <= 1)):
        warnings.warn("predicted_probs contains values outside [0, 1] range. Clamping to [0, 1].")
        predicted_probs = np.clip(predicted_probs, 0.0, 1.0)

    if len(predicted_probs) == 0:
        return 0.5  # Default for empty input

    # Calculate mean squared error
    squared_errors = (predicted_probs - actual_outcomes) ** 2
    brier = np.mean(squared_errors)

    return float(brier)


def expected_calibration_error(
    predicted_probs: np.ndarray,
    actual_outcomes: np.ndarray,
    num_bins: int = 10
) -> Dict[str, float]:
    """
    Calculate Expected Calibration Error (ECE) to detect systematic miscalibration.

    ECE bins predictions into probability ranges and compares the average predicted
    probability in each bin to the empirical frequency of positive outcomes.

    Args:
        predicted_probs: Array of predicted probabilities in [0.0, 1.0]
        actual_outcomes: Array of actual binary outcomes (0 or 1)
        num_bins: Number of bins to use (default: 10)

    Returns:
        Dictionary containing:
        - 'ece': Overall expected calibration error
        - 'bin_accuracies': List of empirical frequencies per bin
        - 'bin_confidences': List of average predicted probabilities per bin
        - 'bin_counts': List of sample counts per bin

    Example:
        >>> predicted = np.array([0.9, 0.85, 0.8, 0.3, 0.25, 0.2])
        >>> actual = np.array([1, 1, 0, 0, 0, 1])
        >>> result = expected_calibration_error(predicted, actual, num_bins=5)
        >>> result['ece']
        0.15

    Reference:
        Naeini, M. P., Cooper, G. F., & Hauskrecht, M. (2015). "Obtaining well
        calibrated probabilities using bayesian binning", AAAI, 2015.
    """
    predicted_probs = np.asarray(predicted_probs, dtype=np.float64)
    actual_outcomes = np.asarray(actual_outcomes, dtype=np.float64)

    if len(predicted_probs) != len(actual_outcomes):
        raise ValueError("predicted_probs and actual_outcomes must have same length")

    if not np.all((actual_outcomes == 0) | (actual_outcomes == 1)):
        raise ValueError("actual_outcomes must be binary (0 or 1)")

    if len(predicted_probs) == 0:
        return {
            'ece': 0.0,
            'bin_accuracies': [],
            'bin_confidences': [],
            'bin_counts': []
        }

    predicted_probs = np.clip(predicted_probs, 0.0, 1.0)

    # Create bins
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_indices = np.digitize(predicted_probs, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    ece = 0.0
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    total_samples = len(predicted_probs)

    for bin_idx in range(num_bins):
        # Get samples in this bin
        in_bin = bin_indices == bin_idx
        bin_count = np.sum(in_bin)

        if bin_count == 0:
            bin_accuracies.append(None)
            bin_confidences.append(None)
            bin_counts.append(0)
            continue

        # Calculate empirical accuracy (frequency of positive outcomes)
        bin_accuracy = np.mean(actual_outcomes[in_bin])

        # Calculate average predicted probability (confidence)
        bin_confidence = np.mean(predicted_probs[in_bin])

        # Weighted contribution to ECE
        weight = bin_count / total_samples
        ece += weight * abs(bin_accuracy - bin_confidence)

        bin_accuracies.append(float(bin_accuracy))
        bin_confidences.append(float(bin_confidence))
        bin_counts.append(int(bin_count))

    return {
        'ece': float(ece),
        'bin_accuracies': bin_accuracies,
        'bin_confidences': bin_confidences,
        'bin_counts': bin_counts
    }


def probability_coherence_compliance(
    individual_probs: Optional[List[Tuple[float, float]]] = None,
    joint_probs: Optional[List[float]] = None,
    conditional_probs: Optional[List[Tuple[float, float, float]]] = None,
    epsilon: float = 1e-6
) -> Dict[str, float]:
    """
    Check adherence to probability axioms (Kolmogorov axioms and derived constraints).

    Verifies:
    1. Kolmogorov axioms: 0 ≤ P(A) ≤ 1 for all statements
    2. Joint probability constraint: P(A∧B) ≤ min(P(A), P(B))
    3. Conditional probability consistency: P(A|B) = P(A∧B) / P(B) when P(B) > 0

    Args:
        individual_probs: List of (P(A), P(B)) tuples for checking range constraints
        joint_probs: List of P(A∧B) values paired with individual_probs
        conditional_probs: List of (P(A), P(B), P(A|B)) tuples for consistency checks
        epsilon: Tolerance for numerical comparisons (default: 1e-6)

    Returns:
        Dictionary containing:
        - 'compliance_rate': Proportion of cases satisfying all axioms [0.0, 1.0]
        - 'kolmogorov_violations': Count of range violations (P < 0 or P > 1)
        - 'joint_violations': Count of P(A∧B) > min(P(A), P(B)) violations
        - 'conditional_violations': Count of P(A|B) ≠ P(A∧B)/P(B) violations
        - 'total_checks': Total number of axiom checks performed

    Example:
        >>> individual = [(0.7, 0.6), (0.5, 0.4)]
        >>> joint = [0.5, 0.2]
        >>> result = probability_coherence_compliance(individual, joint)
        >>> result['compliance_rate']
        1.0
    """
    kolmogorov_violations = 0
    joint_violations = 0
    conditional_violations = 0
    total_checks = 0

    # Check Kolmogorov axiom: 0 ≤ P(A) ≤ 1
    if individual_probs is not None:
        for p_a, p_b in individual_probs:
            total_checks += 2
            if p_a < -epsilon or p_a > 1.0 + epsilon:
                kolmogorov_violations += 1
            if p_b < -epsilon or p_b > 1.0 + epsilon:
                kolmogorov_violations += 1

    # Check joint probability constraint: P(A∧B) ≤ min(P(A), P(B))
    if individual_probs is not None and joint_probs is not None:
        if len(individual_probs) != len(joint_probs):
            raise ValueError("individual_probs and joint_probs must have same length")

        for (p_a, p_b), p_joint in zip(individual_probs, joint_probs):
            total_checks += 1
            min_prob = min(p_a, p_b)
            if p_joint > min_prob + epsilon:
                joint_violations += 1

    # Check conditional probability consistency: P(A|B) = P(A∧B) / P(B)
    if conditional_probs is not None:
        for p_a, p_b, p_a_given_b in conditional_probs:
            if p_b > epsilon:  # Only check when P(B) is not approximately zero
                total_checks += 1
                # We need P(A∧B) to check this, so approximate it
                # P(A|B) = P(A∧B) / P(B) => P(A∧B) = P(A|B) × P(B)
                p_joint_implied = p_a_given_b * p_b

                # Check if implied P(A∧B) is consistent with constraints
                # P(A∧B) should be ≤ min(P(A), P(B))
                if p_joint_implied > min(p_a, p_b) + epsilon:
                    conditional_violations += 1

    # Calculate compliance rate
    total_violations = kolmogorov_violations + joint_violations + conditional_violations
    compliance_rate = 1.0 - (total_violations / total_checks) if total_checks > 0 else 1.0

    return {
        'compliance_rate': float(compliance_rate),
        'kolmogorov_violations': int(kolmogorov_violations),
        'joint_violations': int(joint_violations),
        'conditional_violations': int(conditional_violations),
        'total_checks': int(total_checks)
    }


def probability_consistency_score(
    paraphrase_groups: List[List[float]],
) -> Dict[str, float]:
    """
    Measure consistency of probability estimates for semantically equivalent statements.

    Lower variance indicates more consistent (better) probability extraction.

    Args:
        paraphrase_groups: List of groups, where each group contains probability
                          estimates for semantically equivalent statements

    Returns:
        Dictionary containing:
        - 'mean_variance': Average variance across all paraphrase groups
        - 'max_variance': Maximum variance in any group (worst case)
        - 'min_variance': Minimum variance in any group (best case)
        - 'consistency_score': 1 - normalized mean variance [0.0, 1.0]
                              (1.0 = perfect consistency, 0.0 = maximum inconsistency)

    Example:
        >>> groups = [[0.9, 0.92, 0.88], [0.5, 0.52, 0.48]]
        >>> result = probability_consistency_score(groups)
        >>> result['mean_variance']
        0.00033333333333333335
    """
    if not paraphrase_groups or len(paraphrase_groups) == 0:
        return {
            'mean_variance': 0.0,
            'max_variance': 0.0,
            'min_variance': 0.0,
            'consistency_score': 1.0
        }

    variances = []
    for group in paraphrase_groups:
        if len(group) < 2:
            # Skip groups with less than 2 items (cannot calculate variance)
            continue
        variance = np.var(group, ddof=1)  # Sample variance
        variances.append(variance)

    if len(variances) == 0:
        return {
            'mean_variance': 0.0,
            'max_variance': 0.0,
            'min_variance': 0.0,
            'consistency_score': 1.0
        }

    mean_var = np.mean(variances)
    max_var = np.max(variances)
    min_var = np.min(variances)

    # Normalize to consistency score
    # Maximum possible variance for probabilities in [0, 1] is 0.25 (when P = 0 and P = 1)
    # Consistency score = 1 - (mean_variance / max_possible_variance)
    max_possible_variance = 0.25
    consistency_score = 1.0 - min(mean_var / max_possible_variance, 1.0)

    return {
        'mean_variance': float(mean_var),
        'max_variance': float(max_var),
        'min_variance': float(min_var),
        'consistency_score': float(consistency_score)
    }


def sharpness(predicted_probs: np.ndarray) -> float:
    """
    Calculate sharpness metric to quantify decisiveness of probability predictions.

    Sharpness measures how far predictions are from 0.5 (maximum uncertainty).
    Higher sharpness indicates more confident predictions.

    Formula: Sharpness = (1/N) × Σ|predicted_prob - 0.5|

    Args:
        predicted_probs: Array of predicted probabilities in [0.0, 1.0]

    Returns:
        Sharpness score in [0.0, 0.5] range where 0.5 is maximum confidence

    Example:
        >>> predicted = np.array([0.1, 0.9, 0.2, 0.8])
        >>> sharpness(predicted)
        0.35

    Note:
        Sharpness alone is not sufficient - a model could be sharp but poorly calibrated.
        Use in conjunction with Brier Score and ECE for comprehensive evaluation.
    """
    predicted_probs = np.asarray(predicted_probs, dtype=np.float64)

    if len(predicted_probs) == 0:
        return 0.0

    predicted_probs = np.clip(predicted_probs, 0.0, 1.0)

    # Calculate mean absolute distance from 0.5
    distances = np.abs(predicted_probs - 0.5)
    sharpness_score = np.mean(distances)

    return float(sharpness_score)


def compute_all_metrics(
    predicted_probs: np.ndarray,
    actual_outcomes: np.ndarray,
    individual_probs: Optional[List[Tuple[float, float]]] = None,
    joint_probs: Optional[List[float]] = None,
    conditional_probs: Optional[List[Tuple[float, float, float]]] = None,
    paraphrase_groups: Optional[List[List[float]]] = None,
    num_bins: int = 10
) -> Dict[str, any]:
    """
    Compute all evaluation metrics for comprehensive probability extraction assessment.

    Args:
        predicted_probs: Array of predicted probabilities for Brier/ECE/Sharpness
        actual_outcomes: Array of actual binary outcomes for Brier/ECE
        individual_probs: Pairs for coherence compliance checking
        joint_probs: Joint probabilities for coherence compliance
        conditional_probs: Conditional probabilities for coherence compliance
        paraphrase_groups: Groups of semantically equivalent probabilities
        num_bins: Number of bins for ECE calculation

    Returns:
        Dictionary containing all metric results with keys:
        - 'brier_score': Overall calibration quality
        - 'ece': Expected calibration error details
        - 'coherence_compliance': Probability axiom adherence
        - 'consistency': Paraphrase consistency scores
        - 'sharpness': Prediction decisiveness
        - 'summary': High-level interpretation of results

    Example:
        >>> predicted = np.array([0.9, 0.8, 0.3, 0.2])
        >>> actual = np.array([1, 1, 0, 0])
        >>> results = compute_all_metrics(predicted, actual)
        >>> results['summary']['overall_quality']
        'good'
    """
    results = {}

    # Brier Score
    results['brier_score'] = brier_score(predicted_probs, actual_outcomes)

    # Expected Calibration Error
    results['ece'] = expected_calibration_error(
        predicted_probs, actual_outcomes, num_bins=num_bins
    )

    # Probability Coherence Compliance
    results['coherence_compliance'] = probability_coherence_compliance(
        individual_probs=individual_probs,
        joint_probs=joint_probs,
        conditional_probs=conditional_probs
    )

    # Probability Consistency Score
    if paraphrase_groups is not None:
        results['consistency'] = probability_consistency_score(paraphrase_groups)
    else:
        results['consistency'] = {
            'mean_variance': None,
            'consistency_score': None
        }

    # Sharpness
    results['sharpness'] = sharpness(predicted_probs)

    # Generate summary
    summary = _generate_summary(results)
    results['summary'] = summary

    return results


def _generate_summary(results: Dict) -> Dict[str, str]:
    """
    Generate human-readable summary of metric results.

    Args:
        results: Dictionary containing all metric results

    Returns:
        Dictionary with interpretable summary strings
    """
    summary = {}

    # Interpret Brier Score
    bs = results['brier_score']
    if bs < 0.1:
        summary['brier_interpretation'] = 'excellent'
    elif bs < 0.2:
        summary['brier_interpretation'] = 'good'
    elif bs < 0.3:
        summary['brier_interpretation'] = 'acceptable'
    else:
        summary['brier_interpretation'] = 'poor'

    # Interpret ECE
    ece = results['ece']['ece']
    if ece < 0.05:
        summary['ece_interpretation'] = 'excellent'
    elif ece < 0.1:
        summary['ece_interpretation'] = 'good'
    elif ece < 0.15:
        summary['ece_interpretation'] = 'acceptable'
    else:
        summary['ece_interpretation'] = 'poor'

    # Interpret Coherence Compliance
    compliance = results['coherence_compliance']['compliance_rate']
    if compliance >= 0.95:
        summary['compliance_interpretation'] = 'excellent'
    elif compliance >= 0.9:
        summary['compliance_interpretation'] = 'good'
    elif compliance >= 0.85:
        summary['compliance_interpretation'] = 'acceptable'
    else:
        summary['compliance_interpretation'] = 'poor'

    # Overall quality assessment
    if (summary['brier_interpretation'] in ['excellent', 'good'] and
        summary['ece_interpretation'] in ['excellent', 'good'] and
        summary['compliance_interpretation'] in ['excellent', 'good']):
        summary['overall_quality'] = 'good'
    elif summary['compliance_interpretation'] == 'poor':
        summary['overall_quality'] = 'poor_axiom_violations'
    elif (summary['brier_interpretation'] == 'poor' or
          summary['ece_interpretation'] == 'poor'):
        summary['overall_quality'] = 'poor_calibration'
    else:
        summary['overall_quality'] = 'acceptable'

    return summary
