from openai import OpenAI
from typing import Dict, Tuple, Optional
import json
import time
import os
from functools import wraps


class CoherenceAPIClient:
    """
    API client for extracting probability values from OpenAI models using structured output.
    Used by coherence-based hallucination detection variants (Shogenji, Fitelson, Olsson).

    Features:
    - Temperature=0.0 for deterministic probability extraction
    - OpenAI structured output (JSON schema) for reliable parsing
    - Hybrid prompt strategy (axiom-aware + chain-of-thought) for improved calibration
    - Prompt-response caching to minimize API costs
    - Retry logic with exponential backoff for transient failures
    - Cache statistics and cost estimation utilities

    The hybrid prompt strategy (default) combines:
    - Chain-of-thought reasoning for better probability assessment
    - Axiom awareness (probability theory constraints) for compliance
    - Structured guidance on evidence evaluation and uncertainty

    Example:
        >>> client = CoherenceAPIClient(model="gpt-4o-mini")
        >>> prob = client.extract_individual_probability("The sky is blue.")
        >>> print(f"P(statement) = {prob}")
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        max_cache_size: int = 10000,
    ):
        """
        Initialize CoherenceAPIClient with OpenAI connection.

        Args:
            model: OpenAI model name (default: "gpt-4o-mini")
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
            max_cache_size: Maximum number of cached prompt-response pairs (default: 10000)
        """
        # Initialize OpenAI client
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            # Use default environment variable OPENAI_API_KEY
            self.client = OpenAI()

        self.model = model
        self.max_cache_size = max_cache_size

        # Initialize cache: keyed by (prompt_text, model_name)
        self._cache: Dict[Tuple[str, str], float] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # Define JSON schema for structured output
        self._probability_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "probability_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "probability": {
                            "type": "number",
                            "description": "A probability value between 0.0 and 1.0"
                        }
                    },
                    "required": ["probability"],
                    "additionalProperties": False
                }
            }
        }

        # Prompt templates for probability extraction (using hybrid strategy)
        # Hybrid strategy combines axiom awareness with chain-of-thought reasoning
        # for improved calibration and coherence compliance
        self.individual_prob_template = (
            "Evaluate the probability that this statement is true: {statement}\n\n"
            "Think carefully about:\n"
            "1. Available evidence and common knowledge\n"
            "2. Uncertainty and exceptions\n"
            "3. Logical consistency\n\n"
            "Remember: Use 0.0 for impossible, 1.0 for certain, and values in between for uncertain claims. "
            "What is the probability?"
        )
        self.joint_prob_template = (
            "Evaluate the probability that BOTH statements are simultaneously true:\n"
            "Statement 1: {statement1}\n"
            "Statement 2: {statement2}\n\n"
            "Consider:\n"
            "1. Are these statements independent or related?\n"
            "2. Does one imply or contradict the other?\n"
            "3. P(A∧B) must be ≤ min(P(A), P(B))\n\n"
            "What is the joint probability?"
        )
        self.conditional_prob_template = (
            "Evaluate the conditional probability P(A|B):\n"
            "A: {statement1}\n"
            "B (given condition): {statement2}\n\n"
            "Think about:\n"
            "1. How does knowing B affects the likelihood of A?\n"
            "2. Is there a causal or logical relationship?\n"
            "3. P(A|B) = P(A∧B) / P(B)\n\n"
            "What is P(A|B)?"
        )

        print(f"Initiate OpenAI client for coherence detection... model = {model}")

    def _retry_with_backoff(max_retries: int = 3):
        """
        Decorator for implementing retry logic with exponential backoff.

        Args:
            max_retries: Maximum number of retry attempts
        """
        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                for attempt in range(max_retries):
                    try:
                        return func(self, *args, **kwargs)
                    except Exception as e:
                        # Handle rate limit errors
                        if "rate_limit" in str(e).lower():
                            if attempt < max_retries - 1:
                                wait_time = (2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                                print(f"Warning: Rate limit hit. Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                                time.sleep(wait_time)
                                continue
                            else:
                                raise ValueError(
                                    "Rate limit exceeded. Please check your OpenAI API quota and try again later."
                                ) from e

                        # Handle authentication errors
                        if "authentication" in str(e).lower() or "api_key" in str(e).lower():
                            raise ValueError(
                                "Authentication failed. Please set OPENAI_API_KEY environment variable or provide api_key parameter."
                            ) from e

                        # Handle other errors
                        if attempt < max_retries - 1:
                            wait_time = (2 ** attempt)
                            print(f"Warning: API error occurred. Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                            time.sleep(wait_time)
                            continue
                        else:
                            raise

                return None  # Should not reach here
            return wrapper
        return decorator

    @_retry_with_backoff(max_retries=3)
    def completion(self, prompt: str) -> float:
        """
        Get deterministic completion with structured output for probability extraction.

        Uses temperature=0.0 for deterministic responses and OpenAI's structured output
        feature with JSON schema to ensure reliable probability extraction.

        Args:
            prompt: Prompt text requesting probability estimate

        Returns:
            Probability value in [0.0, 1.0] range

        Raises:
            ValueError: If authentication fails or rate limit exceeded
        """
        # Check cache first
        cache_key = (prompt, self.model)
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]

        self._cache_misses += 1

        # Make API call with structured output
        # GPT-5 and o1 models have different parameter requirements
        uses_completion_tokens = self.model.startswith(('gpt-5', 'o1'))
        # GPT-5 models don't support temperature=0.0, only default (1.0)
        supports_zero_temp = not self.model.startswith('gpt-5')

        completion_params = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "response_format": self._probability_schema
        }

        # Only set temperature if model supports it
        if supports_zero_temp:
            completion_params["temperature"] = 0.0  # Deterministic responses

        # Use appropriate token parameter
        if uses_completion_tokens:
            completion_params["max_completion_tokens"] = 20
        else:
            completion_params["max_tokens"] = 20  # Concise responses

        chat_completion = self.client.chat.completions.create(**completion_params)

        # Parse JSON response
        response_text = chat_completion.choices[0].message.content
        response_json = json.loads(response_text)
        probability = float(response_json["probability"])

        # Clamp to valid range [0.0, 1.0]
        probability = max(0.0, min(1.0, probability))

        # Store in cache (with LRU eviction if needed)
        if len(self._cache) >= self.max_cache_size:
            # Simple LRU: remove oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[cache_key] = probability

        return probability

    def extract_individual_probability(self, statement: str, verbose: bool = False) -> float:
        """
        Extract individual probability P(statement) using prompt template.

        Args:
            statement: Statement to evaluate
            verbose: If True, log cache misses

        Returns:
            P(statement) in [0.0, 1.0] range
        """
        prompt = self.individual_prob_template.format(statement=statement)
        if verbose and (prompt, self.model) not in self._cache:
            print(f"Cache miss: Extracting P(statement)")
        return self.completion(prompt)

    def extract_joint_probability(
        self,
        statement1: str,
        statement2: str,
        verbose: bool = False
    ) -> float:
        """
        Extract joint probability P(statement1 AND statement2) using prompt template.

        Args:
            statement1: First statement
            statement2: Second statement
            verbose: If True, log cache misses

        Returns:
            P(statement1 AND statement2) in [0.0, 1.0] range
        """
        prompt = self.joint_prob_template.format(
            statement1=statement1,
            statement2=statement2
        )
        if verbose and (prompt, self.model) not in self._cache:
            print(f"Cache miss: Extracting P(statement1 AND statement2)")
        return self.completion(prompt)

    def extract_conditional_probability(
        self,
        statement1: str,
        statement2: str,
        verbose: bool = False
    ) -> float:
        """
        Extract conditional probability P(statement1 | statement2) using prompt template.

        Args:
            statement1: Statement to evaluate (hypothesis)
            statement2: Condition (evidence)
            verbose: If True, log cache misses

        Returns:
            P(statement1 | statement2) in [0.0, 1.0] range
        """
        prompt = self.conditional_prob_template.format(
            statement1=statement1,
            statement2=statement2
        )
        if verbose and (prompt, self.model) not in self._cache:
            print(f"Cache miss: Extracting P(statement1 | statement2)")
        return self.completion(prompt)

    def get_cache_stats(self) -> Dict[str, float]:
        """
        Get cache statistics for cost estimation.

        Returns:
            Dictionary with cache statistics:
            - 'hit_rate': Proportion of requests served from cache
            - 'cache_size': Current number of cached entries
            - 'total_requests': Total number of requests made
            - 'api_calls': Number of actual API calls made
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            'hit_rate': hit_rate,
            'cache_size': len(self._cache),
            'total_requests': total_requests,
            'api_calls': self._cache_misses
        }

    @staticmethod
    def estimate_api_calls(
        num_sentences: int,
        num_samples: int,
        num_variants: int = 1,
        include_conditional: bool = False
    ) -> Dict[str, int]:
        """
        Estimate number of API calls needed for evaluation.

        Args:
            num_sentences: Number of sentences to evaluate
            num_samples: Number of sampled passages per sentence
            num_variants: Number of coherence variants to evaluate (default: 1)
            include_conditional: If True, include conditional probability calls (for Fitelson)

        Returns:
            Dictionary with cost estimates:
            - 'calls_per_sentence': API calls needed per sentence
            - 'total_calls_uncached': Total API calls without caching
            - 'estimated_cached_calls': Estimated API calls with typical caching
        """
        # For Shogenji and Olsson:
        # - 1 call for P(sentence)
        # - num_samples calls for P(sample_i)
        # - num_samples calls for P(sentence AND sample_i)
        # Total: 1 + 2*num_samples per sentence

        # For Fitelson (with conditional):
        # - Additional num_samples calls for P(sentence | sample_i)
        # Total: 1 + 3*num_samples per sentence

        if include_conditional:
            calls_per_sentence = 1 + 3 * num_samples
        else:
            calls_per_sentence = 1 + 2 * num_samples

        total_calls_uncached = calls_per_sentence * num_sentences * num_variants

        # Estimate with caching: assume 30-50% cache hit rate for P(sample) across sentences
        # Conservative estimate: 70% of uncached calls
        estimated_cached_calls = int(total_calls_uncached * 0.7)

        return {
            'calls_per_sentence': calls_per_sentence,
            'total_calls_uncached': total_calls_uncached,
            'estimated_cached_calls': estimated_cached_calls
        }
