"""
Unit tests for coherence formula implementations.

Tests validate mathematical correctness of Shogenji, Fitelson, and Olsson
coherence measures against known-outcome scenarios and edge cases.
"""

import numpy as np
import pytest
import warnings

from selfcheckgpt.utils_coherence import (
    coherence_shogenji,
    coherence_fitelson,
    coherence_olsson,
    normalize_coherence_scores
)


class TestShogenjiFormula:
    """Test Shogenji's ratio-based coherence measure: C2(A,B) = P(A∧B) / (P(A) × P(B))"""

    def test_independent_events_yield_c2_approx_1(self):
        """Independent events should yield C2 ≈ 1"""
        # For independent events: P(A∧B) = P(A) × P(B), so C2 = 1
        probs_individual = np.array([[0.6, 0.4], [0.7, 0.3], [0.5, 0.5]])
        # Calculate expected joint probabilities for independence
        probs_joint = np.array([0.6 * 0.4, 0.7 * 0.3, 0.5 * 0.5])

        coherence = coherence_shogenji(probs_individual, probs_joint)

        # Should all be very close to 1.0 (within epsilon)
        assert np.allclose(coherence, 1.0, rtol=1e-10), \
            f"Expected C2≈1.0 for independent events, got {coherence}"

    def test_positive_correlation_yields_c2_greater_than_1(self):
        """Positively correlated events should yield C2 > 1"""
        # P(A∧B) > P(A) × P(B) indicates positive correlation
        probs_individual = np.array([[0.6, 0.4], [0.8, 0.7]])
        # Set joint probability higher than independence would predict
        probs_joint = np.array([0.35, 0.65])  # 0.35 > 0.24, 0.65 > 0.56

        coherence = coherence_shogenji(probs_individual, probs_joint)

        # Should all be greater than 1.0
        assert np.all(coherence > 1.0), \
            f"Expected C2>1.0 for positive correlation, got {coherence}"

    def test_mutually_exclusive_yields_c2_approx_0(self):
        """Mutually exclusive events should yield C2 ≈ 0"""
        # For mutually exclusive events: P(A∧B) = 0, so C2 ≈ 0
        probs_individual = np.array([[0.5, 0.5], [0.3, 0.7]])
        probs_joint = np.array([0.0, 0.0])

        coherence = coherence_shogenji(probs_individual, probs_joint)

        # Should be very close to 0 (epsilon prevents exact 0)
        assert np.all(coherence < 0.01), \
            f"Expected C2≈0 for mutually exclusive events, got {coherence}"

    def test_epsilon_smoothing_prevents_division_by_zero(self):
        """Epsilon smoothing should prevent division by zero"""
        # Test with very small probabilities that could cause division by zero
        probs_individual = np.array([[0.0, 0.0], [1e-15, 1e-15]])
        probs_joint = np.array([0.0, 1e-20])

        # Should not raise exception
        coherence = coherence_shogenji(probs_individual, probs_joint)

        # Should return finite values
        assert np.all(np.isfinite(coherence)), \
            f"Expected finite values with epsilon smoothing, got {coherence}"

    def test_probability_clamping_to_valid_ranges(self):
        """Probabilities should be clamped to valid [epsilon, 1.0-epsilon] range"""
        # Test with out-of-range probabilities
        probs_individual = np.array([[-0.1, 1.5], [0.0, 1.0]])
        probs_joint = np.array([0.5, 0.5])

        # Should not raise exception, values should be clamped
        coherence = coherence_shogenji(probs_individual, probs_joint)

        # Should return finite, reasonable values
        assert np.all(np.isfinite(coherence)), \
            f"Expected finite values after clamping, got {coherence}"
        assert np.all(coherence >= 0), \
            f"Expected non-negative coherence, got {coherence}"

    def test_axiom_violation_warning_p_joint_greater_than_p_a(self):
        """Should warn when P(A∧B) > P(A) (axiom violation)"""
        # Create impossible probability: P(A∧B) > P(A)
        probs_individual = np.array([[0.3, 0.5]])
        probs_joint = np.array([0.6])  # 0.6 > 0.3 violates axiom

        with pytest.warns(RuntimeWarning, match="P\\(A∧B\\) > P\\(A\\)"):
            coherence = coherence_shogenji(probs_individual, probs_joint)

        # Should clamp and return finite value
        assert np.isfinite(coherence[0]), \
            f"Expected finite value after axiom violation, got {coherence}"

    def test_batch_processing(self):
        """Test processing multiple probability pairs at once"""
        # Process 5 pairs simultaneously
        probs_individual = np.array([
            [0.6, 0.4],
            [0.7, 0.3],
            [0.5, 0.5],
            [0.8, 0.2],
            [0.9, 0.1]
        ])
        probs_joint = np.array([0.24, 0.21, 0.25, 0.16, 0.09])

        coherence = coherence_shogenji(probs_individual, probs_joint)

        # Should return array of same length
        assert len(coherence) == 5, f"Expected 5 coherence scores, got {len(coherence)}"
        assert np.all(np.isfinite(coherence)), "All coherence scores should be finite"

    def test_invalid_input_shapes_raise_error(self):
        """Invalid input shapes should raise ValueError"""
        # Test wrong shape for probs_individual
        with pytest.raises(ValueError, match="must have shape \\(n, 2\\)"):
            probs_individual = np.array([0.5, 0.5])  # 1D instead of 2D
            probs_joint = np.array([0.25])
            coherence_shogenji(probs_individual, probs_joint)

        # Test mismatched lengths
        with pytest.raises(ValueError, match="must have same length"):
            probs_individual = np.array([[0.5, 0.5], [0.6, 0.4]])
            probs_joint = np.array([0.25])  # Length 1 vs 2
            coherence_shogenji(probs_individual, probs_joint)


class TestFitelsonFormula:
    """Test Fitelson's confirmation measure: s(H,E) = P(H|E) - P(H|¬E)"""

    def test_strong_confirmation_yields_positive_support(self):
        """Strong confirmation should yield s > 0"""
        # E strongly confirms H: P(H|E) >> P(H|¬E)
        probs_individual = np.array([[0.5, 0.6]])  # P(H), P(E)
        probs_joint = np.array([0.5])  # P(H∧E)
        probs_conditional = np.array([0.833])  # P(H|E) = 0.833
        # P(H|¬E) = [P(H) - P(E)×P(H|E)] / [1 - P(E)] = [0.5 - 0.6×0.833] / 0.4 ≈ 0.0

        support = coherence_fitelson(probs_individual, probs_joint, probs_conditional)

        # Should be positive
        assert support[0] > 0, f"Expected positive support, got {support[0]}"

    def test_disconfirmation_yields_negative_support(self):
        """Strong disconfirmation should yield s < 0"""
        # E disconfirms H: P(H|E) < P(H|¬E)
        probs_individual = np.array([[0.5, 0.6]])  # P(H), P(E)
        probs_joint = np.array([0.1])  # P(H∧E) - low joint probability
        probs_conditional = np.array([0.167])  # P(H|E) = 0.167
        # P(H|¬E) = [0.5 - 0.6×0.167] / 0.4 = 1.0

        support = coherence_fitelson(probs_individual, probs_joint, probs_conditional)

        # Should be negative
        assert support[0] < 0, f"Expected negative support, got {support[0]}"

    def test_independence_yields_support_approx_0(self):
        """Independence should yield s ≈ 0"""
        # E is independent of H: P(H|E) ≈ P(H|¬E) ≈ P(H)
        probs_individual = np.array([[0.5, 0.6]])  # P(H), P(E)
        probs_joint = np.array([0.3])  # P(H∧E) = P(H)×P(E)
        probs_conditional = np.array([0.5])  # P(H|E) = P(H)
        # P(H|¬E) should also equal 0.5

        support = coherence_fitelson(probs_individual, probs_joint, probs_conditional)

        # Should be close to 0
        assert abs(support[0]) < 0.1, f"Expected s≈0 for independence, got {support[0]}"

    def test_handling_p_not_e_equals_0_edge_case(self):
        """Should handle P(¬E) = 0 edge case (P(E) = 1)"""
        # When P(E) = 1, P(¬E) = 0, so P(H|¬E) is undefined
        # Should fall back to simple confirmation: s(H,E) = P(H|E) - P(H)
        probs_individual = np.array([[0.5, 1.0]])  # P(H), P(E)=1.0
        probs_joint = np.array([0.5])  # P(H∧E)
        probs_conditional = np.array([0.8])  # P(H|E)

        support = coherence_fitelson(probs_individual, probs_joint, probs_conditional)

        # Should use fallback: s = P(H|E) - P(H) = 0.8 - 0.5 = 0.3
        expected_support = 0.8 - 0.5
        assert np.isclose(support[0], expected_support, atol=0.05), \
            f"Expected fallback support {expected_support}, got {support[0]}"

    def test_conditional_probability_consistency(self):
        """Test that formulas are mathematically consistent"""
        # Use Bayes' theorem: P(H|E) = P(H∧E) / P(E)
        p_h = 0.6
        p_e = 0.5
        p_joint = 0.3  # P(H∧E)
        p_h_given_e = p_joint / p_e  # 0.3 / 0.5 = 0.6

        probs_individual = np.array([[p_h, p_e]])
        probs_joint = np.array([p_joint])
        probs_conditional = np.array([p_h_given_e])

        support = coherence_fitelson(probs_individual, probs_joint, probs_conditional)

        # Should return finite value without errors
        assert np.isfinite(support[0]), \
            f"Expected finite support with consistent probabilities, got {support}"

    def test_support_scores_bounded_to_range(self):
        """Support scores should be clamped to [-1, 1] range"""
        # Test with extreme probabilities
        probs_individual = np.array([[0.01, 0.99], [0.99, 0.01]])
        probs_joint = np.array([0.01, 0.01])
        probs_conditional = np.array([0.99, 0.99])

        support = coherence_fitelson(probs_individual, probs_joint, probs_conditional)

        # All scores should be in [-1, 1]
        assert np.all(support >= -1.0) and np.all(support <= 1.0), \
            f"Expected support in [-1, 1], got {support}"

    def test_invalid_input_shapes_raise_error(self):
        """Invalid input shapes should raise ValueError"""
        # Test wrong shape for probs_conditional
        with pytest.raises(ValueError, match="must have shape \\(n,\\)"):
            probs_individual = np.array([[0.5, 0.5]])
            probs_joint = np.array([0.25])
            probs_conditional = np.array([[0.5]])  # 2D instead of 1D
            coherence_fitelson(probs_individual, probs_joint, probs_conditional)

        # Test mismatched lengths
        with pytest.raises(ValueError, match="must have same length"):
            probs_individual = np.array([[0.5, 0.5]])
            probs_joint = np.array([0.25])
            probs_conditional = np.array([0.5, 0.6])  # Length 2 vs 1
            coherence_fitelson(probs_individual, probs_joint, probs_conditional)


class TestOlssonFormula:
    """Test Olsson's overlap measure: C1(A,B) = P(A∧B) / P(A∨B)"""

    def test_identical_statements_yield_c1_approx_1(self):
        """Identical statements should yield C1 ≈ 1"""
        # For identical statements: P(A∧B) = P(A∨B) = P(A) = P(B), so C1 = 1
        probs_individual = np.array([[0.7, 0.7], [0.5, 0.5]])
        probs_joint = np.array([0.7, 0.5])  # P(A∧B) = P(A) = P(B)

        coherence = coherence_olsson(probs_individual, probs_joint)

        # Should be very close to 1.0
        assert np.allclose(coherence, 1.0, rtol=1e-10), \
            f"Expected C1≈1.0 for identical statements, got {coherence}"

    def test_disjoint_statements_yield_c1_approx_0(self):
        """Disjoint statements should yield C1 ≈ 0"""
        # For disjoint statements: P(A∧B) = 0, so C1 = 0
        probs_individual = np.array([[0.5, 0.5], [0.3, 0.7]])
        probs_joint = np.array([0.0, 0.0])

        coherence = coherence_olsson(probs_individual, probs_joint)

        # Should be very close to 0
        assert np.all(coherence < 0.01), \
            f"Expected C1≈0 for disjoint statements, got {coherence}"

    def test_p_union_calculation(self):
        """Test P(A∨B) = P(A) + P(B) - P(A∧B) calculation"""
        # Known probabilities
        p_a = 0.6
        p_b = 0.4
        p_joint = 0.24  # Independent case
        p_union = p_a + p_b - p_joint  # 0.6 + 0.4 - 0.24 = 0.76
        expected_c1 = p_joint / p_union  # 0.24 / 0.76 ≈ 0.316

        probs_individual = np.array([[p_a, p_b]])
        probs_joint = np.array([p_joint])

        coherence = coherence_olsson(probs_individual, probs_joint)

        # Should match expected calculation
        assert np.isclose(coherence[0], expected_c1, rtol=1e-10), \
            f"Expected C1={expected_c1}, got {coherence[0]}"

    def test_division_by_zero_handling_when_p_union_approx_0(self):
        """Should handle P(A∨B) ≈ 0 without division by zero"""
        # Edge case: both probabilities very small
        probs_individual = np.array([[0.0, 0.0]])
        probs_joint = np.array([0.0])

        # Should not raise exception
        coherence = coherence_olsson(probs_individual, probs_joint)

        # Should return finite value (epsilon prevents division by zero)
        assert np.isfinite(coherence[0]), \
            f"Expected finite value with epsilon smoothing, got {coherence}"

    def test_coherence_bounded_to_0_1_range(self):
        """Coherence scores should be in [0, 1] range"""
        # Test with various probability combinations
        probs_individual = np.array([
            [0.1, 0.9],
            [0.5, 0.5],
            [0.9, 0.1],
            [0.3, 0.7]
        ])
        probs_joint = np.array([0.05, 0.25, 0.05, 0.21])

        coherence = coherence_olsson(probs_individual, probs_joint)

        # All scores should be in [0, 1]
        assert np.all(coherence >= 0.0) and np.all(coherence <= 1.0), \
            f"Expected coherence in [0, 1], got {coherence}"

    def test_axiom_violation_p_union_less_than_p_joint(self):
        """Should warn when P(A∨B) < P(A∧B) (axiom violation)"""
        # This shouldn't happen mathematically, but test numerical error handling
        # Create scenario that might trigger numerical issues
        probs_individual = np.array([[0.5, 0.5]])
        # Set joint probability artificially high to test clamping
        probs_joint = np.array([0.6])  # > min(P(A), P(B)) = 0.5

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            coherence = coherence_olsson(probs_individual, probs_joint)

            # Should issue warning about axiom violation
            assert len(w) > 0, "Expected warning for axiom violation"

        # Should still return valid value after clamping
        assert 0.0 <= coherence[0] <= 1.0, \
            f"Expected coherence in [0, 1] after clamping, got {coherence}"

    def test_batch_processing(self):
        """Test processing multiple probability pairs at once"""
        # Process 5 pairs simultaneously
        probs_individual = np.array([
            [0.6, 0.4],
            [0.7, 0.3],
            [0.5, 0.5],
            [0.8, 0.2],
            [0.9, 0.1]
        ])
        probs_joint = np.array([0.24, 0.21, 0.25, 0.16, 0.09])

        coherence = coherence_olsson(probs_individual, probs_joint)

        # Should return array of same length
        assert len(coherence) == 5, f"Expected 5 coherence scores, got {len(coherence)}"
        assert np.all(np.isfinite(coherence)), "All coherence scores should be finite"
        assert np.all(coherence >= 0.0) and np.all(coherence <= 1.0), \
            "All coherence scores should be in [0, 1]"


class TestNormalizeCoherenceScores:
    """Test normalize_coherence_scores() utility function"""

    def test_min_max_normalization(self):
        """Test basic min-max normalization: (score - min) / (max - min)"""
        scores = np.array([0.5, 1.0, 1.5, 2.0])

        normalized = normalize_coherence_scores(scores)

        # Should be normalized to [0, 1]
        expected = np.array([0.0, 0.33333333, 0.66666667, 1.0])
        assert np.allclose(normalized, expected, rtol=1e-6), \
            f"Expected {expected}, got {normalized}"

    def test_all_identical_scores_return_half(self):
        """All identical scores should return 0.5"""
        scores = np.array([1.5, 1.5, 1.5, 1.5])

        normalized = normalize_coherence_scores(scores)

        # Should all be 0.5
        assert np.allclose(normalized, 0.5), \
            f"Expected all 0.5 for identical scores, got {normalized}"

    def test_nan_handling(self):
        """NaN values should be replaced with 0.5"""
        scores = np.array([0.5, np.nan, 1.5, 2.0])

        normalized = normalize_coherence_scores(scores)

        # NaN should become 0.5
        assert normalized[1] == 0.5, f"Expected NaN→0.5, got {normalized[1]}"
        # Other values should be normalized
        assert np.isfinite(normalized).all(), "All values should be finite"

    def test_inf_handling(self):
        """Inf values should be replaced with 0.5"""
        scores = np.array([0.5, 1.0, np.inf, 2.0])

        normalized = normalize_coherence_scores(scores)

        # Inf should become 0.5
        assert normalized[2] == 0.5, f"Expected Inf→0.5, got {normalized[2]}"
        # Other values should be normalized
        assert np.isfinite(normalized).all(), "All values should be finite"

    def test_all_nan_returns_half(self):
        """All NaN scores should return all 0.5"""
        scores = np.array([np.nan, np.nan, np.nan])

        normalized = normalize_coherence_scores(scores)

        # Should all be 0.5
        assert np.allclose(normalized, 0.5), \
            f"Expected all 0.5 for all NaN, got {normalized}"

    def test_output_bounded_to_0_1(self):
        """Normalized scores should always be in [0, 1] range"""
        # Test with various input ranges
        test_cases = [
            np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
            np.array([0.001, 0.002, 0.003]),
            np.array([100, 200, 300, 400])
        ]

        for scores in test_cases:
            normalized = normalize_coherence_scores(scores)

            assert np.all(normalized >= 0.0) and np.all(normalized <= 1.0), \
                f"Expected all values in [0, 1], got {normalized} for input {scores}"

    def test_preserves_order(self):
        """Normalization should preserve relative ordering"""
        scores = np.array([3.0, 1.0, 2.0, 5.0, 4.0])

        normalized = normalize_coherence_scores(scores)

        # Check that ordering is preserved
        assert normalized[3] > normalized[4], "5.0 should map to higher than 4.0"
        assert normalized[4] > normalized[0], "4.0 should map to higher than 3.0"
        assert normalized[0] > normalized[2], "3.0 should map to higher than 2.0"
        assert normalized[2] > normalized[1], "2.0 should map to higher than 1.0"

    def test_single_value(self):
        """Single value should return 0.5"""
        scores = np.array([42.0])

        normalized = normalize_coherence_scores(scores)

        # Single value has no range, should be 0.5
        assert normalized[0] == 0.5, f"Expected 0.5 for single value, got {normalized[0]}"


if __name__ == "__main__":
    # Allow running tests directly with python
    pytest.main([__file__, "-v"])
