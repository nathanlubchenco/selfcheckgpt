## Input validation for Python packages

- **Type Checking**: Use Python type hints where helpful
- **Fail Fast**: Check inputs early and raise clear exceptions
- **Specific Error Messages**: Raise descriptive exceptions (e.g., `ValueError("Expected temperature in [0.0, 2.0], got 3.5")`)
- **Pragmatic Validation**: Only validate inputs that could cause errors
- **Document Constraints**: Use docstrings to document valid input ranges

### Not Applicable
- No client-side/server-side distinction (Python package)
- No web security concerns (SQL injection, XSS)
- Pragmatic over defensive
