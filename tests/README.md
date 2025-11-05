# SelfCheckGPT Test Suite

This directory contains tests for SelfCheckGPT coherence variants, validating mathematical correctness and end-to-end functionality.

## Test Structure

### Unit Tests (`test_coherence_formulas.py`)
Tests the mathematical implementations of coherence formulas:
- **Shogenji's measure**: C2(A,B) = P(A∧B) / (P(A) × P(B))
- **Fitelson's measure**: s(H,E) = P(H|E) - P(H|¬E)
- **Olsson's measure**: C1(A,B) = P(A∧B) / P(A∨B)
- **Normalization utilities**: Min-max normalization and edge case handling

**Coverage:**
- Known-outcome scenarios (independent events, positive correlation, etc.)
- Edge cases (division by zero, NaN/Inf handling, axiom violations)
- Input validation (shape checking, type validation)
- Batch processing capabilities

### Integration Tests (`test_coherence_integration.py`)
Tests end-to-end coherence variant pipelines:
- **SelfCheckShogenji**: Full predict() interface with probability extraction
- **SelfCheckFitelson**: Conditional probability extraction and higher API call counts
- **SelfCheckOlsson**: Union probability calculation and overlap measure
- **Cache statistics**: Caching mechanism reduces redundant API calls
- **Error handling**: Edge cases like empty lists, long sentences, special characters

**Note:** Integration tests make real API calls to OpenAI and require `OPENAI_API_KEY` environment variable.

## Running Tests

### Run All Tests
```bash
# From project root
pytest

# Or with verbose output
pytest -v
```

### Run Specific Test Files
```bash
# Unit tests only (no API calls)
pytest tests/test_coherence_formulas.py

# Integration tests (requires API key)
pytest tests/test_coherence_integration.py
```

### Run Specific Test Classes or Methods
```bash
# Run all Shogenji formula tests
pytest tests/test_coherence_formulas.py::TestShogenjiFormula

# Run specific test method
pytest tests/test_coherence_formulas.py::TestShogenjiFormula::test_independent_events_yield_c2_approx_1
```

### Run Tests with Coverage
```bash
# Install coverage tool
pip install pytest-cov

# Run with coverage report
pytest --cov=selfcheckgpt --cov-report=html
```

## Environment Setup

### Prerequisites
```bash
# Install test dependencies
pip install pytest

# For integration tests, set OpenAI API key
export OPENAI_API_KEY='your-api-key-here'
```

### Virtual Environment
```bash
# Activate virtual environment
source venv/bin/activate

# Install in development mode
pip install -e .

# Run tests
pytest
```

## Test Organization

### Unit Tests (30 tests)
- **Shogenji Formula** (8 tests)
  - Independent events yield C2≈1
  - Positive correlation yields C2>1
  - Mutually exclusive yields C2≈0
  - Epsilon smoothing prevents division by zero
  - Probability clamping to valid ranges
  - Axiom violation warnings
  - Batch processing
  - Invalid input shape errors

- **Fitelson Formula** (7 tests)
  - Strong confirmation yields positive support
  - Disconfirmation yields negative support
  - Independence yields s≈0
  - P(¬E)=0 edge case handling
  - Conditional probability consistency
  - Support scores bounded to [-1,1]
  - Invalid input shape errors

- **Olsson Formula** (7 tests)
  - Identical statements yield C1≈1
  - Disjoint statements yield C1≈0
  - P(A∨B) calculation correctness
  - Division by zero handling
  - Coherence bounded to [0,1]
  - Axiom violation handling
  - Batch processing

- **Normalization Utilities** (8 tests)
  - Min-max normalization
  - All identical scores return 0.5
  - NaN handling
  - Inf handling
  - All NaN returns 0.5
  - Output bounded to [0,1]
  - Order preservation
  - Single value handling

### Integration Tests (18 tests)
- **SelfCheckShogenji** (6 tests)
  - Basic predict() interface
  - Varying sample counts (1, 3, 5)
  - Caching reduces API calls
  - Full pipeline validation
  - Empty sentences/passages lists

- **SelfCheckFitelson** (3 tests)
  - Basic predict() interface
  - Conditional probability extraction
  - Higher API call count vs Shogenji

- **SelfCheckOlsson** (3 tests)
  - Basic predict() interface
  - Union probability calculation
  - Overlap measure calculation

- **Cache & Cost Estimation** (3 tests)
  - Accurate cache statistics
  - Cache persistence across sentences
  - API call estimation accuracy

- **Error Handling** (3 tests)
  - Very long sentences (>1000 chars)
  - Special characters and unicode
  - Newlines in passages

## Test Results

### Latest Test Run
```
Unit Tests: 30 passed (0.36s)
Integration Tests: 18 passed (79.97s)
Total: 48 tests passed
```

### Known Warnings
- RuntimeWarning from utils_coherence.py when P(A∧B) > P(A) (expected, axiom violation handling)
- RuntimeWarning for mean of empty slice when testing empty passages (expected edge case)

## Continuous Integration

Tests are designed to be run in CI/CD pipelines:
- **Unit tests**: Fast (<1s), no external dependencies, suitable for pre-commit hooks
- **Integration tests**: Slower (~80s), require API key, suitable for post-merge validation

## Contributing

When adding new coherence variants or formulas:
1. Add unit tests in `test_coherence_formulas.py` for mathematical validation
2. Add integration tests in `test_coherence_integration.py` for end-to-end validation
3. Follow existing test naming conventions: `test_<behavior>_<expected_outcome>`
4. Use descriptive assertion messages for debugging
5. Ensure tests are independent (no shared state between tests)

## Debugging Failed Tests

### View Full Traceback
```bash
pytest --tb=long
```

### Stop on First Failure
```bash
pytest -x
```

### Run Last Failed Tests
```bash
pytest --lf
```

### Print Output During Tests
```bash
pytest -s
```

### Enable Warnings
```bash
pytest -W all
```

## References

- Project testing standards: `agent-os/standards/testing/test-writing.md`
- Coherence theory documentation: `docs/coherence_variants.md`
- Benchmark tests: `selfcheckgpt/benchmark/` (separate probability extraction validation)
