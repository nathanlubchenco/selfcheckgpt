"""
Async version of CoherenceAPIClient for faster parallel API calls.
Uses asyncio and OpenAI's async client for concurrent probability extraction.
"""
from openai import AsyncOpenAI
from typing import Dict, Tuple, Optional, List
import asyncio
import os
from functools import wraps


class CoherenceAPIClientAsync:
    """
    Async API client for extracting probability values from OpenAI models.

    This async version allows concurrent API calls, significantly reducing wall-clock time
    when processing multiple sentences or probability queries in parallel.

    Features:
    - Concurrent API calls using asyncio
    - Temperature=0.0 for deterministic probability extraction
    - OpenAI structured output (JSON schema) for reliable parsing
    - Prompt-response caching to minimize API costs
    - Retry logic with exponential backoff for transient failures
    - Thread-safe cache for parallel operations

    Example:
        >>> import asyncio
        >>> client = CoherenceAPIClientAsync(model="gpt-4o-mini")
        >>> prob = asyncio.run(client.extract_individual_probability("The sky is blue."))
        >>> print(f"P(statement) = {prob}")
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        max_cache_size: int = 10000,
        max_concurrent: int = 10,
    ):
        """
        Initialize CoherenceAPIClientAsync with OpenAI connection.

        Args:
            model: OpenAI model name (default: "gpt-4o-mini")
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
            max_cache_size: Maximum number of cached prompt-response pairs (default: 10000)
            max_concurrent: Maximum number of concurrent API calls (default: 10)
        """
        # Initialize AsyncOpenAI client
        if api_key:
            self.client = AsyncOpenAI(api_key=api_key)
        else:
            # Use default environment variable OPENAI_API_KEY
            self.client = AsyncOpenAI()

        self.model = model
        self.max_cache_size = max_cache_size
        self.max_concurrent = max_concurrent

        # Semaphore to limit concurrent API calls
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Initialize cache: keyed by (prompt_text, model_name)
        self._cache: Dict[Tuple[str, str], float] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_lock = asyncio.Lock()

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

        # Prompt templates for probability extraction
        self.individual_prob_template = "Rate the probability that this statement is true: {statement}"
        self.joint_prob_template = "Rate the probability that both statements are true: {statement1} AND {statement2}"
        self.conditional_prob_template = "Rate the probability that statement A is true: {statement1} GIVEN that {statement2} is true"

        print(f"Initiate async OpenAI client for coherence detection... model = {model}, max_concurrent = {max_concurrent}")

    async def _get_probability_from_api(self, prompt: str, max_retries: int = 3) -> float:
        """
        Make API call with retry logic and semaphore for rate limiting.

        Args:
            prompt: The prompt to send to the API
            max_retries: Maximum number of retry attempts

        Returns:
            Extracted probability value
        """
        async with self._semaphore:  # Limit concurrent API calls
            for attempt in range(max_retries):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a probability estimator. Respond only with a JSON object containing a probability value between 0.0 and 1.0."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        response_format=self._probability_schema,
                        temperature=0.0,  # Deterministic for caching
                    )

                    # Parse structured output
                    import json
                    result = json.loads(response.choices[0].message.content)
                    probability = float(result["probability"])

                    # Clamp to valid range [0.0, 1.0]
                    probability = max(0.0, min(1.0, probability))

                    return probability

                except Exception as e:
                    # Handle rate limit errors with exponential backoff
                    if "rate_limit" in str(e).lower() or "429" in str(e):
                        if attempt < max_retries - 1:
                            wait_time = (2 ** attempt)  # Exponential backoff
                            print(f"Rate limit hit, waiting {wait_time}s before retry {attempt+1}/{max_retries}")
                            await asyncio.sleep(wait_time)
                            continue

                    # Re-raise on final attempt or non-rate-limit errors
                    if attempt == max_retries - 1:
                        raise RuntimeError(f"Failed to extract probability after {max_retries} attempts: {e}")

                    # Wait before retry for other errors
                    await asyncio.sleep(1)

    async def _extract_probability(self, prompt: str) -> float:
        """
        Extract probability with caching (async version).

        Args:
            prompt: The prompt to send to the API

        Returns:
            Extracted probability value (from cache or API)
        """
        cache_key = (prompt, self.model)

        # Check cache (with lock for thread safety)
        async with self._cache_lock:
            if cache_key in self._cache:
                self._cache_hits += 1
                return self._cache[cache_key]

            self._cache_misses += 1

        # Make API call (outside of lock to allow concurrent API calls)
        probability = await self._get_probability_from_api(prompt)

        # Store in cache (with lock)
        async with self._cache_lock:
            # LRU eviction if cache is full
            if len(self._cache) >= self.max_cache_size:
                # Remove oldest entry (simple FIFO, not true LRU)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

            self._cache[cache_key] = probability

        return probability

    async def extract_individual_probability(self, statement: str) -> float:
        """
        Extract P(statement) asynchronously.

        Args:
            statement: The statement to evaluate

        Returns:
            Probability value [0.0, 1.0]
        """
        prompt = self.individual_prob_template.format(statement=statement)
        return await self._extract_probability(prompt)

    async def extract_joint_probability(self, statement1: str, statement2: str) -> float:
        """
        Extract P(statement1 AND statement2) asynchronously.

        Args:
            statement1: First statement
            statement2: Second statement

        Returns:
            Joint probability value [0.0, 1.0]
        """
        prompt = self.joint_prob_template.format(statement1=statement1, statement2=statement2)
        return await self._extract_probability(prompt)

    async def extract_conditional_probability(self, statement1: str, statement2: str) -> float:
        """
        Extract P(statement1 | statement2) asynchronously.

        Args:
            statement1: Statement A (hypothesis)
            statement2: Statement B (condition)

        Returns:
            Conditional probability value [0.0, 1.0]
        """
        prompt = self.conditional_prob_template.format(statement1=statement1, statement2=statement2)
        return await self._extract_probability(prompt)

    async def extract_probabilities_batch(
        self,
        statements: List[str],
        probability_type: str = "individual"
    ) -> List[float]:
        """
        Extract probabilities for multiple statements concurrently.

        Args:
            statements: List of statements (or tuples for joint/conditional)
            probability_type: "individual", "joint", or "conditional"

        Returns:
            List of probability values
        """
        if probability_type == "individual":
            tasks = [self.extract_individual_probability(stmt) for stmt in statements]
        elif probability_type == "joint":
            tasks = [self.extract_joint_probability(s1, s2) for s1, s2 in statements]
        elif probability_type == "conditional":
            tasks = [self.extract_conditional_probability(s1, s2) for s1, s2 in statements]
        else:
            raise ValueError(f"Unknown probability_type: {probability_type}")

        return await asyncio.gather(*tasks)

    def get_cache_stats(self) -> Dict[str, any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache hit rate, size, and API call counts
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "api_calls": self._cache_misses  # Each miss = 1 API call
        }

    @staticmethod
    def estimate_api_calls(
        num_sentences: int,
        num_samples: int,
        num_variants: int = 1,
        include_conditional: bool = False,
    ) -> Dict[str, int]:
        """
        Estimate number of API calls for coherence evaluation.

        Args:
            num_sentences: Number of sentences to evaluate
            num_samples: Number of sampled passages per sentence
            num_variants: Number of coherence variants to run (default: 1)
            include_conditional: Whether to include conditional probabilities (Fitelson)

        Returns:
            Dictionary with estimated API calls per variant and total
        """
        # Per sentence, per sample: P(sent), P(sample), P(sent AND sample)
        calls_per_pair = 3
        if include_conditional:
            # Fitelson also needs P(sent | sample) and P(sent | NOT sample)
            calls_per_pair = 4  # Actually needs more complex calculation

        # Total pairs per sentence
        pairs_per_sentence = num_samples

        # Per sentence: 1 for P(sent) + num_samples * calls_per_pair
        calls_per_sentence = 1 + (pairs_per_sentence * 2)  # P(sent), then for each sample: P(sample), P(sent AND sample)

        total_per_variant = num_sentences * calls_per_sentence
        total_all_variants = total_per_variant * num_variants

        return {
            "calls_per_sentence": calls_per_sentence,
            "calls_per_variant": total_per_variant,
            "total_calls": total_all_variants,
            "estimated_cost_usd": total_all_variants * 0.00001  # Rough estimate for gpt-4o-mini
        }
