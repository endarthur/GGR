# Contributing to GGR

Thank you for your interest in contributing to GGR! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/ggr.git
cd ggr
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install in development mode:**
```bash
pip install -e ".[dev]"
```

4. **Run tests:**
```bash
pytest
```

## Development Workflow

### Test-Driven Development

GGR follows TDD principles:

1. **Write tests first** - Start by writing tests that define expected behavior
2. **Run tests** - Confirm they fail (red)
3. **Implement** - Write minimum code to pass tests (green)
4. **Refactor** - Improve code while keeping tests green
5. **Repeat** - Continue for next feature

### Code Style

- Follow PEP 8
- Use Black for formatting: `black src/ggr tests/`
- Use Ruff for linting: `ruff check src/ggr tests/`
- Use type hints: `mypy src/ggr/`

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=ggr --cov-report=html

# Specific test file
pytest tests/test_api.py

# Specific test
pytest tests/test_api.py::TestVariogramAPI::test_create_from_params
```

## Key Design Principles

1. **JupyterLite First** - All code must work in Pyodide
   - Pure Python + NumPy/SciPy/pandas only
   - No C extensions beyond what Pyodide provides

2. **Type Flexibility** - Accept both dicts and objects
   - Users shouldn't need to learn new classes
   - But provide classes for power users

3. **Clean APIs** - Follow Python best practices
   - Functional and class-based options
   - Sensible defaults
   - Clear error messages

4. **Well-Tested** - Comprehensive test coverage
   - API contract tests
   - GSLIB validation tests
   - Edge case tests

## Adding New Features

### 1. Start with Tests

Create tests in appropriate file:
- `test_api.py` - API behavior and contracts
- `test_gslib_validation.py` - Numerical correctness
- `test_edge_cases.py` - Error handling

### 2. Implement Feature

Follow existing patterns:
- Add to appropriate module
- Use type hints
- Add docstrings (NumPy style)

### 3. Update Documentation

- Add examples to docstrings
- Update README if needed
- Create notebook example if appropriate

## GSLIB Validation

All kriging implementations must be validated against GSLIB:

1. **Get reference data:**
   - Use standard GSLIB datasets (cluster.dat, etc.)
   - Run GSLIB programs (kt3d, kb2d, sgsim)
   - Save outputs

2. **Create validation test:**
```python
def test_vs_gslib_kt3d():
    result = ggr.krige(gslib_data, variogram, grid)
    assert_allclose(result, gslib_output, rtol=0.01)
```

3. **Document differences:**
   - Small numerical differences are acceptable (<1%)
   - Document any algorithmic differences

## Pull Request Process

1. **Create a branch:**
```bash
git checkout -b feature/your-feature-name
```

2. **Make changes:**
   - Write tests
   - Implement feature
   - Ensure all tests pass
   - Format and lint code

3. **Commit:**
```bash
git add .
git commit -m "Add feature: your feature description"
```

4. **Push and create PR:**
```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## Code Review Checklist

- [ ] Tests added/updated
- [ ] Tests pass
- [ ] Code formatted (Black)
- [ ] Linting passes (Ruff)
- [ ] Type hints added (mypy)
- [ ] Docstrings added/updated
- [ ] README updated if needed
- [ ] Works in JupyterLite

## Questions?

Open an issue on GitHub or start a discussion!

---

**Gamma Gamma Revolution** 💚
