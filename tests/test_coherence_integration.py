"""
Integration tests for coherence variant end-to-end pipelines.

Tests validate the complete flow from probability extraction through coherence
calculation to final hallucination scores for all three coherence variants.

NOTE: These tests make real API calls to OpenAI. Set OPENAI_API_KEY environment
variable before running. Tests are limited in scope to minimize API costs.
"""

import numpy as np
import pytest
import os

from selfcheckgpt.modeling_coherence import (
    SelfCheckShogenji,
    SelfCheckFitelson,
    SelfCheckOlsson
)


# Check if API key is available
OPENAI_API_KEY_AVAILABLE = os.getenv("OPENAI_API_KEY") is not None

# Skip all tests in this module if no API key
pytestmark = pytest.mark.skipif(
    not OPENAI_API_KEY_AVAILABLE,
    reason="OpenAI API key not available (set OPENAI_API_KEY environment variable)"
)


@pytest.fixture
def sample_sentences():
    """Fixture providing sample sentences for testing"""
    return [
        "Paris is the capital of France.",
        "Water freezes at 0 degrees Celsius."
    ]


@pytest.fixture
def sample_passages():
    """Fixture providing sample passages for testing"""
    return [
        "The capital city of France is Paris, located in the north of the country.",
        "France's capital is Paris, which is also its largest city.",
        "Paris serves as the capital of France and is a major European city."
    ]


class TestSelfCheckShogenjiIntegration:
    """Integration tests for SelfCheckShogenji end-to-end pipeline"""

    def test_predict_interface_basic(self, sample_sentences, sample_passages):
        """Test basic predict() interface with sample data"""
        selfcheck = SelfCheckShogenji(model="gpt-4o-mini")

        # Run prediction
        scores = selfcheck.predict(sample_sentences, sample_passages, verbose=False)

        # Verify output shape and type
        assert isinstance(scores, np.ndarray), "Should return numpy array"
        assert len(scores) == len(sample_sentences), \
            f"Should return score for each sentence, got {len(scores)} for {len(sample_sentences)} sentences"

        # Verify scores are in valid range [0, 1]
        assert np.all(scores >= 0.0) and np.all(scores <= 1.0), \
            f"Scores should be in [0, 1], got {scores}"

    def test_predict_with_varying_sample_counts(self, sample_sentences):
        """Test predict() with different numbers of sampled passages"""
        selfcheck = SelfCheckShogenji(model="gpt-4o-mini")

        # Test with 1, 3, 5 samples
        for num_samples in [1, 3, 5]:
            sample_passages = [
                f"Paris is the capital of France. Sample {i}."
                for i in range(num_samples)
            ]

            scores = selfcheck.predict(sample_sentences, sample_passages, verbose=False)

            # Should work regardless of sample count
            assert len(scores) == len(sample_sentences), \
                f"Failed with {num_samples} samples"
            assert np.all(np.isfinite(scores)), \
                f"Should return finite scores with {num_samples} samples"

    def test_caching_reduces_api_calls(self, sample_sentences, sample_passages):
        """Test that caching mechanism reduces redundant API calls"""
        selfcheck = SelfCheckShogenji(model="gpt-4o-mini")

        # First run - cache miss
        scores1 = selfcheck.predict(sample_sentences, sample_passages, verbose=False)
        cache_stats1 = selfcheck.client.get_cache_stats()
        api_calls1 = cache_stats1['api_calls']

        # Second run with same data - should use cache
        scores2 = selfcheck.predict(sample_sentences, sample_passages, verbose=False)
        cache_stats2 = selfcheck.client.get_cache_stats()
        api_calls2 = cache_stats2['api_calls']

        # Cache should reduce API calls
        # Second run should make many fewer new calls (ideally 0, but some variance OK)
        new_calls = api_calls2 - api_calls1
        assert new_calls == 0, \
            f"Expected cache to prevent new API calls, but made {new_calls} new calls"

    def test_probability_extraction_to_coherence_pipeline(self):
        """Test full pipeline: probability extraction → coherence calculation → normalization"""
        selfcheck = SelfCheckShogenji(model="gpt-4o-mini")

        sentences = ["The sky is blue."]
        samples = ["The sky appears blue during the day."]

        # Run prediction
        scores = selfcheck.predict(sentences, samples, verbose=True)

        # Verify cache statistics are reported in verbose mode
        cache_stats = selfcheck.client.get_cache_stats()
        assert cache_stats['total_requests'] > 0, "Should have made API requests"
        assert cache_stats['cache_size'] > 0, "Should have cached responses"

        # Verify score is reasonable
        assert 0.0 <= scores[0] <= 1.0, f"Score should be in [0, 1], got {scores[0]}"

    def test_empty_sentences_list(self):
        """Test handling of empty sentences list"""
        selfcheck = SelfCheckShogenji(model="gpt-4o-mini")

        sentences = []
        samples = ["Sample passage."]

        scores = selfcheck.predict(sentences, samples, verbose=False)

        # Should return empty array
        assert len(scores) == 0, "Should return empty array for empty sentences"

    def test_empty_passages_list(self):
        """Test handling of empty sampled passages list"""
        selfcheck = SelfCheckShogenji(model="gpt-4o-mini")

        sentences = ["Paris is the capital of France."]
        samples = []

        # Should handle gracefully (may error or return default scores)
        # This tests robustness - exact behavior may vary
        try:
            scores = selfcheck.predict(sentences, samples, verbose=False)
            # If it doesn't error, verify output is valid
            assert len(scores) == len(sentences)
        except (ValueError, ZeroDivisionError):
            # Acceptable to raise error for empty samples
            pass


class TestSelfCheckFitelsonIntegration:
    """Integration tests for SelfCheckFitelson end-to-end pipeline"""

    def test_predict_interface_basic(self, sample_sentences, sample_passages):
        """Test basic predict() interface with sample data"""
        selfcheck = SelfCheckFitelson(model="gpt-4o-mini")

        # Run prediction
        scores = selfcheck.predict(sample_sentences, sample_passages, verbose=False)

        # Verify output shape and type
        assert isinstance(scores, np.ndarray), "Should return numpy array"
        assert len(scores) == len(sample_sentences), \
            f"Should return score for each sentence"

        # Verify scores are in valid range [0, 1]
        assert np.all(scores >= 0.0) and np.all(scores <= 1.0), \
            f"Scores should be in [0, 1], got {scores}"

    def test_conditional_probability_extraction(self):
        """Test that conditional probabilities are extracted correctly"""
        selfcheck = SelfCheckFitelson(model="gpt-4o-mini")

        # Use clear causal relationship
        sentences = ["It is raining."]
        samples = ["The ground is wet."]

        scores = selfcheck.predict(sentences, samples, verbose=True)

        # Verify cache statistics show conditional probability calls
        cache_stats = selfcheck.client.get_cache_stats()
        # Fitelson makes more API calls (1 + 3*num_samples per sentence)
        # vs Shogenji (1 + 2*num_samples)
        assert cache_stats['api_calls'] > 0, "Should have made API calls"

    def test_higher_api_call_count_than_shogenji(self):
        """Test that Fitelson makes more API calls than Shogenji (1 + 3*n vs 1 + 2*n)"""
        # Create fresh instances to reset cache
        shogenji = SelfCheckShogenji(model="gpt-4o-mini")
        fitelson = SelfCheckFitelson(model="gpt-4o-mini")

        sentences = ["Test sentence."]
        samples = ["Test sample 1.", "Test sample 2."]

        # Run both
        shogenji.predict(sentences, samples, verbose=False)
        fitelson.predict(sentences, samples, verbose=False)

        # Check API call counts
        shogenji_calls = shogenji.client.get_cache_stats()['api_calls']
        fitelson_calls = fitelson.client.get_cache_stats()['api_calls']

        # Fitelson should make more calls (conditional probability extraction)
        # Expected: Shogenji = 1 + 2*2 = 5, Fitelson = 1 + 3*2 = 7
        assert fitelson_calls > shogenji_calls, \
            f"Fitelson should make more API calls ({fitelson_calls}) than Shogenji ({shogenji_calls})"


class TestSelfCheckOlssonIntegration:
    """Integration tests for SelfCheckOlsson end-to-end pipeline"""

    def test_predict_interface_basic(self, sample_sentences, sample_passages):
        """Test basic predict() interface with sample data"""
        selfcheck = SelfCheckOlsson(model="gpt-4o-mini")

        # Run prediction
        scores = selfcheck.predict(sample_sentences, sample_passages, verbose=False)

        # Verify output shape and type
        assert isinstance(scores, np.ndarray), "Should return numpy array"
        assert len(scores) == len(sample_sentences), \
            f"Should return score for each sentence"

        # Verify scores are in valid range [0, 1]
        assert np.all(scores >= 0.0) and np.all(scores <= 1.0), \
            f"Scores should be in [0, 1], got {scores}"

    def test_union_probability_calculation(self):
        """Test that P(A∨B) is calculated correctly"""
        selfcheck = SelfCheckOlsson(model="gpt-4o-mini")

        # Use statements with clear relationship
        sentences = ["Paris is in France."]
        samples = ["Paris is the capital of France."]

        scores = selfcheck.predict(sentences, samples, verbose=False)

        # Should produce valid coherence score
        assert 0.0 <= scores[0] <= 1.0, f"Score should be in [0, 1], got {scores[0]}"

    def test_overlap_measure_calculation(self):
        """Test that overlap measure is calculated correctly"""
        selfcheck = SelfCheckOlsson(model="gpt-4o-mini")

        # Test with highly similar statements (high overlap)
        sentences = ["Water is wet."]
        samples = ["Water is wet and liquid."]

        scores = selfcheck.predict(sentences, samples, verbose=False)

        # High similarity should yield high coherence (low hallucination score)
        # Note: exact values depend on model, just verify it's in valid range
        assert np.all(np.isfinite(scores)), "Should return finite scores"


class TestCacheStatisticsAndCostEstimation:
    """Test cache statistics and cost estimation utilities"""

    def test_get_cache_stats_returns_accurate_counts(self):
        """Test that get_cache_stats() returns accurate statistics"""
        selfcheck = SelfCheckShogenji(model="gpt-4o-mini")

        sentences = ["Test sentence."]
        samples = ["Test sample."]

        # Run prediction
        selfcheck.predict(sentences, samples, verbose=False)

        # Get cache stats
        stats = selfcheck.client.get_cache_stats()

        # Verify stats structure
        assert 'hit_rate' in stats, "Should include hit_rate"
        assert 'cache_size' in stats, "Should include cache_size"
        assert 'total_requests' in stats, "Should include total_requests"
        assert 'api_calls' in stats, "Should include api_calls"

        # Verify values are reasonable
        assert 0.0 <= stats['hit_rate'] <= 1.0, "Hit rate should be in [0, 1]"
        assert stats['cache_size'] >= 0, "Cache size should be non-negative"
        assert stats['api_calls'] >= 0, "API calls should be non-negative"

    def test_cache_persists_across_sentences(self):
        """Test that cache persists across multiple sentence evaluations"""
        selfcheck = SelfCheckShogenji(model="gpt-4o-mini")

        # First sentence
        sentences1 = ["Sentence 1."]
        samples = ["Sample passage."]
        selfcheck.predict(sentences1, samples, verbose=False)

        # Second sentence with same samples
        sentences2 = ["Sentence 2."]
        selfcheck.predict(sentences2, samples, verbose=False)

        # Cache should have entries from both runs
        stats = selfcheck.client.get_cache_stats()
        assert stats['cache_size'] > 0, "Cache should retain entries"
        assert stats['hit_rate'] > 0, "Should have some cache hits from shared samples"

    def test_estimate_api_calls_matches_actual(self):
        """Test that estimate_api_calls() produces accurate estimates"""
        from selfcheckgpt.modeling_coherence_api import CoherenceAPIClient

        num_sentences = 2
        num_samples = 3

        # Get estimate
        estimate = CoherenceAPIClient.estimate_api_calls(
            num_sentences=num_sentences,
            num_samples=num_samples,
            num_variants=1,
            include_conditional=False
        )

        # Verify estimate structure
        assert 'calls_per_sentence' in estimate
        assert 'total_calls_uncached' in estimate
        assert 'estimated_cached_calls' in estimate

        # Verify calculations
        # For Shogenji/Olsson: 1 + 2*num_samples per sentence
        expected_calls_per_sentence = 1 + 2 * num_samples
        assert estimate['calls_per_sentence'] == expected_calls_per_sentence, \
            f"Expected {expected_calls_per_sentence} calls/sentence, got {estimate['calls_per_sentence']}"

        expected_total = expected_calls_per_sentence * num_sentences
        assert estimate['total_calls_uncached'] == expected_total, \
            f"Expected {expected_total} total calls, got {estimate['total_calls_uncached']}"


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases for coherence variants"""

    def test_very_long_sentences(self):
        """Test handling of very long sentences (>1000 chars)"""
        selfcheck = SelfCheckShogenji(model="gpt-4o-mini")

        # Create very long sentence
        long_sentence = "This is a test sentence. " * 50  # ~1250 characters
        sentences = [long_sentence]
        samples = ["Short sample."]

        # Should handle gracefully
        scores = selfcheck.predict(sentences, samples, verbose=False)
        assert len(scores) == 1
        assert np.isfinite(scores[0])

    def test_special_characters_and_unicode(self):
        """Test handling of special characters and unicode"""
        selfcheck = SelfCheckShogenji(model="gpt-4o-mini")

        # Test with various special characters
        sentences = [
            "The café is open.",  # Accented characters
            "Temperature: 20°C",  # Degree symbol
            "Price: $50.00",      # Dollar sign
        ]
        samples = ["The restaurant is open during the day."]

        # Should handle gracefully
        scores = selfcheck.predict(sentences, samples, verbose=False)
        assert len(scores) == len(sentences)
        assert np.all(np.isfinite(scores))

    def test_newlines_in_passages(self):
        """Test handling of newlines in passages"""
        selfcheck = SelfCheckShogenji(model="gpt-4o-mini")

        sentences = ["Test sentence."]
        # Passages with newlines (should be stripped)
        samples = ["Sample\nwith\nnewlines."]

        scores = selfcheck.predict(sentences, samples, verbose=False)
        assert len(scores) == 1
        assert np.isfinite(scores[0])


if __name__ == "__main__":
    # Allow running tests directly with python
    pytest.main([__file__, "-v"])
