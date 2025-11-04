# Raw Idea: SelfCheckGPT Coherence Variants Improvements

## Improvements to Address:

1. **Probability extraction prompts** - The current prompts are extremely basic. Explore improvements through prompt iteration and optimization. Consider:
   - Chain of thought reasoning
   - Few shot examples
   - Additional in-context learning about probability theory (to avoid axiom violations)
   - Independent evaluation of prompts for probability extraction

2. **Probability extraction model** - Is gpt-4.5-mini better at this? Set up evaluation to test different models.

3. **Verifying implementation of coherence measures** - Add tests to confirm these are working as expected. Check for outliers and mathematical errors.

4. **Other datasets** - Currently tightly coupled to wiki_bio_gpt3_hallucination dataset. Extend to other datasets to verify coherence measures are broadly useful.

5. **Other models** - Evaluate different models (like gpt-4.5-mini) compared to baseline for coherence.

6. **General improvements** - Make the feature more performant, accessible, and explainable.
