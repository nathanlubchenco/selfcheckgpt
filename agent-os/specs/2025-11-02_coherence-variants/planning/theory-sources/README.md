# Coherence Theory Sources

This directory contains references and documentation for the three formal coherence measures used in the SelfCheckGPT coherence-based hallucination detection variants.

## Primary Sources

### 1. Shogenji's Coherence Measure (C2)

**Primary Reference:**
- Shogenji, T., 1999, "Is Coherence Truth-conducive?," *Analysis*, 59: 338–345.

**Key Formula:**
```
C₂(A,B) = P(A ∧ B) / (P(A) × P(B))
```

**Generalized Formula (n propositions):**
```
coh(A₁,...,Aₙ) = P(A₁ ∧ ... ∧ Aₙ) / [P(A₁) × ... × P(Aₙ)]
```

**Theoretical Motivation:**
- Measures how interconnected beliefs are by comparing the joint probability of all beliefs to what their probability would be if independent
- When B has no bearing on A, P(A|B)=P(A), and this ratio equals 1, which is the neutral point
- Values > 1 indicate positive coherence (mutual support)
- Values < 1 indicate negative coherence (conflict)
- Sensitive to the number of reports: as n approaches infinity, so does the degree of coherence (for positively coherent sets)

**Source Location:**
- Stanford Encyclopedia of Philosophy - Formal Epistemology: https://plato.stanford.edu/entries/formal-epistemology/
- Stanford Encyclopedia of Philosophy - Coherentist Theories of Justification: https://plato.stanford.edu/entries/justep-coherence/

---

### 2. Glass-Olsson Relative Overlap Measure (C1)

**Primary Reference:**
- Olsson, E. J., 2002, "What is the Problem of Coherence and Truth?," *The Journal of Philosophy*, 99: 246–272.
- Olsson, E. J., 2005, *Against Coherence: Truth, Probability, and Justification*, Oxford University Press.

**Key Formula:**
```
C₁(A,B) = P(A ∧ B) / P(A ∨ B)
```

**Theoretical Motivation:**
- Captures the proportion of agreement between propositions
- Defined as relative overlap between belief sets
- Yields values between 0 and 1, where 1 represents complete agreement
- Value of 0 indicates complete disagreement (disjoint beliefs)
- Maximum coherence (value = 1) occurs when propositions are logically equivalent and consistent
- Unlike C2, takes into account prior probability considerations

**Note on Naming:**
This measure is referred to as the "Glass-Olsson measure" in the literature, acknowledging contributions from both David Glass and Erik J. Olsson in its development and analysis.

**Source Location:**
- Stanford Encyclopedia of Philosophy - Coherentist Theories of Justification: https://plato.stanford.edu/entries/justep-coherence/

---

### 3. Fitelson's Confirmation-Based Measure

**Primary Reference:**
- Fitelson, B., 2003, "A Probabilistic Measure of Coherence," *Analysis*, 63: 194–199.

**Foundational Reference:**
- Kemeny, J. and Oppenheim, P., 1952, "Degrees of Factual Support," *Philosophy of Science*, 19: 307–24.

**Theoretical Approach:**
Fitelson's measure is based on Kemeny and Oppenheim's (1952) measure of factual support, extended to examine support relations between all subsets of a belief set.

**Key Characteristics:**
- Draws from confirmation theory to create "a quantitative, probabilistic generalization of the (deductive) logical coherence"
- Examines support relations holding between ALL subsets in the set E (not just individual elements)
- Overall coherence degree calculated as the mean support among the subsets of E
- Maximum coherence occurs when propositions are logically equivalent and consistent
- Aligns with C1 (relative overlap measure) but differs from C2 regarding prior probability considerations

**Formula Structure:**
While the exact formula is complex and involves multiple subset calculations, the general approach is:
1. For each subset pair in the belief set, calculate factual support
2. Aggregate support values across all subset pairs
3. Return mean support as overall coherence measure

**Common Factual Support Formulas (from Kemeny & Oppenheim tradition):**
Several measures of factual support exist in the literature. Fitelson's specific implementation may use one or more of these:

```
s(H,E) = P(H|E) - P(H|¬E)    [Difference measure]
s(H,E) = P(H|E) - P(H)         [Simple confirmation]
s(H,E) = [P(H|E) - P(H)] / [1 - P(H)]  [Normalized]
```

The exact formula used by Fitelson (2003) would require accessing the original paper.

**Critical Context:**
- Siebel (2004) identified alleged counterexamples to Fitelson's measure
- Meijs (2006) offered criticisms and proposed amendments
- Olsson (2022) critiques "subset measures" generally, highlighting issues with:
  - Epistemic access challenges
  - Computational complexity
  - Probability assessment difficulties
  - Excessive support requirements

**Source Location:**
- Stanford Encyclopedia of Philosophy - Coherentist Theories of Justification: https://plato.stanford.edu/entries/justep-coherence/

---

## Additional Context

### Status of Coherence Measures

As noted in the Stanford Encyclopedia of Philosophy:
> "Which measure is correct, if any, remains controversial."

The three measures represent different philosophical approaches to quantifying coherence:
- **Shogenji (C2)**: Focuses on probabilistic independence and mutual support
- **Glass-Olsson (C1)**: Emphasizes relative overlap and agreement
- **Fitelson**: Incorporates confirmation-theoretic support relations

### Content Determination Thesis

All three measures adhere to what scholars call the "Content Determination Thesis" - they define coherence "solely in terms of the probability of...propositions (and their Boolean combinations) and standard arithmetical operations."

### Implementation Considerations

For the SelfCheckGPT implementation:
1. **Shogenji's C2** is the most straightforward to implement (simple ratio formula)
2. **Glass-Olsson's C1** requires calculating P(A ∨ B), which can be derived: P(A ∨ B) = P(A) + P(B) - P(A ∧ B)
3. **Fitelson's measure** is the most complex, requiring subset enumeration and support calculations

All three measures require extracting probabilities via LLM prompting:
- Individual probabilities: P(A), P(B)
- Joint probability: P(A ∧ B)
- Conditional probabilities (for support measures): P(A|B), P(A|¬B)

---

## References

### Primary Papers
- Shogenji, T. (1999). "Is Coherence Truth-conducive?", *Analysis*, 59: 338–345.
- Fitelson, B. (2003). "A Probabilistic Measure of Coherence", *Analysis*, 63: 194–199.
- Olsson, E. J. (2002). "What is the Problem of Coherence and Truth?", *The Journal of Philosophy*, 99: 246–272.

### Additional References
- Kemeny, J. and Oppenheim, P. (1952). "Degrees of Factual Support", *Philosophy of Science*, 19: 307–24.
- Bovens, L. and Hartmann, S. (2003). *Bayesian Epistemology*, Oxford University Press.
- Olsson, E. J. (2005). *Against Coherence: Truth, Probability, and Justification*, Oxford University Press.
- Olsson, E. J. (2022). Updates on coherence theory and related topics (referenced in SEP).

### Encyclopedia Entries
- Stanford Encyclopedia of Philosophy: Formal Epistemology
  - URL: https://plato.stanford.edu/entries/formal-epistemology/
- Stanford Encyclopedia of Philosophy: Coherentist Theories of Epistemic Justification
  - URL: https://plato.stanford.edu/entries/justep-coherence/

---

## Access Notes

The primary papers are published in academic journals that typically require institutional access:
- *Analysis* (Oxford Academic)
- *The Journal of Philosophy* (Philosophy Documentation Center)
- *Philosophy of Science* (Cambridge University Press)

For implementation purposes, the formulas and theoretical motivations documented in this README, sourced from the Stanford Encyclopedia of Philosophy (open access), provide sufficient foundation for the SelfCheckGPT coherence variants.
