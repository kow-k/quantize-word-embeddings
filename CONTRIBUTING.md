# Contributing to Adaptive Embedding Quantization

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)

---

## Code of Conduct

This project adheres to a code of professional and respectful behavior:

- **Be respectful**: Value diverse perspectives and experiences
- **Be collaborative**: Work together toward common goals
- **Be constructive**: Provide helpful feedback
- **Be professional**: Maintain a welcoming environment

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Familiarity with NumPy and basic NLP concepts

### Areas for Contribution

We welcome contributions in several areas:

1. **Core Functionality**
   - New quantization methods
   - Performance optimizations
   - Support for additional embedding formats

2. **Evaluation**
   - Additional benchmark datasets
   - New evaluation metrics
   - Cross-linguistic validation

3. **Visualization**
   - New plot types
   - Interactive visualizations
   - Dashboard interfaces

4. **Documentation**
   - Tutorial notebooks
   - Additional examples
   - Translations

5. **Testing**
   - Unit tests
   - Integration tests
   - Performance benchmarks

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/YOUR-USERNAME/embedding-quantization.git
cd embedding-quantization

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL-OWNER/embedding-quantization.git
```

### 2. Create Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### 3. Install in Editable Mode

```bash
pip install -e .
```

### 4. Verify Installation

```bash
# Run tests
python -m pytest tests/

# Try example
python examples/basic_quantization.py
```

---

## How to Contribute

### Reporting Issues

Before creating an issue:

1. **Search existing issues** to avoid duplicates
2. **Use issue templates** when available
3. **Provide details**: Python version, OS, error messages, minimal reproducible example

**Good issue example:**

```markdown
**Bug**: Quantization fails on embeddings with NaN values

**Environment:**
- Python 3.9.5
- NumPy 1.21.0
- OS: Ubuntu 20.04

**Minimal example:**
\```python
from adaptive_quantization import AdaptiveQuantizer
import numpy as np

embeddings = np.random.randn(1000, 100)
embeddings[0, 0] = np.nan  # Introduce NaN

quantizer = AdaptiveQuantizer(base_k=32)
quantized = quantizer.quantize(embeddings)  # Raises error
\```

**Expected:** Should handle NaN gracefully or provide clear error
**Actual:** ValueError with unclear message
```

### Suggesting Features

Feature requests should include:

1. **Use case**: What problem does this solve?
2. **Proposed solution**: How should it work?
3. **Alternatives considered**: What other approaches did you think about?
4. **Priority**: Nice-to-have vs critical

**Good feature request example:**

```markdown
**Feature**: Support for binary Word2Vec format

**Use case:** Many users have embeddings in binary .bin format (original Word2Vec format). 
Currently they must convert to .vec format first.

**Proposed solution:**
- Add `load_binary_embeddings()` function
- Use gensim.KeyedVectors.load_word2vec_format(binary=True) under the hood
- Make compatible with existing quantization pipeline

**Alternatives:**
1. Require users to convert manually (current situation)
2. Add format auto-detection in load_embeddings()

**Priority:** Medium - improves user experience but workaround exists
```

---

## Coding Standards

### Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with these specifics:

- **Line length**: 100 characters (slightly more flexible than PEP 8's 79)
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Prefer single quotes for strings, double for docstrings
- **Naming**:
  - Classes: `PascalCase`
  - Functions/methods: `snake_case`
  - Constants: `UPPER_SNAKE_CASE`
  - Private: prefix with `_underscore`

### Code Quality Tools

```bash
# Format code
black adaptive_quantization.py

# Check style
flake8 adaptive_quantization.py

# Type checking
mypy adaptive_quantization.py

# Sort imports
isort adaptive_quantization.py
```

### Docstring Format

Use Google-style docstrings:

```python
def quantize_dimension(values: np.ndarray, k: int) -> np.ndarray:
    """
    Quantize a single dimension to k levels.
    
    Args:
        values: 1D array of dimension values
        k: Number of quantization levels
        
    Returns:
        Quantized values with k unique values
        
    Raises:
        ValueError: If k < 2 or values is empty
        
    Example:
        >>> values = np.array([1.0, 2.5, 3.7, 4.2])
        >>> quantized = quantize_dimension(values, k=2)
        >>> len(np.unique(quantized))
        2
    """
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    if len(values) == 0:
        raise ValueError("values cannot be empty")
    
    # Implementation
    ...
```

### Type Hints

Use type hints for function signatures:

```python
from typing import List, Dict, Tuple, Optional
import numpy as np

def analyze_embeddings(
    embeddings: np.ndarray,
    base_k: int = 32,
    verbose: bool = True
) -> List[DimensionProfile]:
    """Analyze embedding dimensions."""
    ...
```

---

## Testing

### Writing Tests

Tests use `pytest`. Place tests in `tests/` directory:

```python
# tests/test_quantization.py
import numpy as np
import pytest
from adaptive_quantization import AdaptiveQuantizer

class TestAdaptiveQuantizer:
    
    def test_basic_quantization(self):
        """Test basic quantization functionality."""
        embeddings = np.random.randn(100, 50)
        quantizer = AdaptiveQuantizer(base_k=32, verbose=False)
        quantized = quantizer.quantize(embeddings)
        
        assert quantized.shape == embeddings.shape
        assert np.all(np.isfinite(quantized))
    
    def test_k_values(self):
        """Test that k values are reasonable."""
        embeddings = np.random.randn(100, 50)
        quantizer = AdaptiveQuantizer(base_k=32, verbose=False)
        quantizer.analyze_dimensions(embeddings)
        
        for profile in quantizer.dimension_profiles:
            assert profile.optimal_k >= 16
            assert profile.optimal_k <= 64
    
    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        quantizer = AdaptiveQuantizer(base_k=32)
        
        with pytest.raises(ValueError):
            quantizer.quantize(np.array([]))  # Empty array
        
        with pytest.raises(ValueError):
            quantizer.quantize(np.array([1, 2, 3]))  # 1D array
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_quantization.py

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test
pytest tests/test_quantization.py::TestAdaptiveQuantizer::test_basic_quantization
```

### Test Coverage

Aim for:
- **Core functionality**: >90% coverage
- **Utilities**: >80% coverage
- **Visualization**: >70% coverage

```bash
# Generate coverage report
pytest --cov=. --cov-report=term-missing
```

---

## Documentation

### Docstrings

All public functions, classes, and methods must have docstrings:

```python
class AdaptiveQuantizer:
    """
    Adaptive quantization engine for word embeddings.
    
    Analyzes each dimension's distribution and applies optimal
    quantization method per dimension. Supports multiple strategies
    including uniform, Lebesgue, k-means, and robust quantization.
    
    Attributes:
        base_k: Base quantization level (default: 32, recommended: 2^5)
        use_lebesgue: Whether to use equi-depth for skewed dimensions
        verbose: Whether to print progress messages
        dimension_profiles: List of analyzed dimension profiles
        
    Example:
        >>> quantizer = AdaptiveQuantizer(base_k=32)
        >>> quantized = quantizer.quantize(embeddings)
        >>> quantizer.print_summary()
    """
```

### README Updates

When adding features, update:

1. **README.md**: Add to Features section
2. **API_REFERENCE.md**: Add detailed API documentation
3. **EXAMPLES.md**: Add usage examples
4. **CHANGELOG.md**: Document changes

### Code Comments

Use comments for:
- **Complex algorithms**: Explain the approach
- **Non-obvious choices**: Why this specific implementation
- **TODOs**: Mark areas for future improvement

```python
# Use Shapiro-Wilk test for normality
# Sample to 5000 points to avoid slowness on large embeddings
sample_size = min(5000, len(values))
_, p_value = stats.shapiro(np.random.choice(values, sample_size))

# TODO: Consider Kolmogorov-Smirnov as alternative for very large samples
```

---

## Pull Request Process

### Before Submitting

Checklist:

- [ ] Tests pass: `pytest`
- [ ] Code formatted: `black .`
- [ ] Style checked: `flake8`
- [ ] Types checked: `mypy`
- [ ] Docstrings added/updated
- [ ] Examples work
- [ ] CHANGELOG.md updated

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe testing done:
- Unit tests added/updated
- Manual testing performed
- Edge cases considered

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-reviewed code
- [ ] Commented complex code
- [ ] Updated documentation
- [ ] Added tests
- [ ] Tests pass
- [ ] No new warnings

## Related Issues
Fixes #123
Related to #456
```

### Review Process

1. **Automated checks** must pass (tests, linting)
2. **Code review** by maintainer
3. **Discussion** if needed
4. **Approval** and merge

### Commit Messages

Use clear, descriptive commit messages:

```bash
# Good
git commit -m "Add Walsh function k=32 recommendation to README"
git commit -m "Fix quantization failure on empty dimensions"
git commit -m "Optimize memory usage in analyze_dimensions()"

# Bad
git commit -m "Fix bug"
git commit -m "Update"
git commit -m "WIP"
```

Format:
```
<type>: <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Tests
- `chore`: Maintenance

Example:
```
feat: Add support for binary Word2Vec format

Implements load_binary_embeddings() function using gensim to read
.bin format files. Maintains compatibility with existing pipeline
by converting to standard numpy array format internally.

Resolves #42
```

---

## Development Workflow

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

Branch naming:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation
- `refactor/` - Code improvements
- `test/` - Test additions

### 2. Make Changes

```bash
# Edit files
# Add tests
# Update docs
```

### 3. Test Locally

```bash
# Run tests
pytest

# Check style
flake8 .
black --check .

# Type check
mypy adaptive_quantization.py
```

### 4. Commit

```bash
git add .
git commit -m "feat: Add new quantization method"
```

### 5. Push and PR

```bash
git push origin feature/your-feature-name
# Create pull request on GitHub
```

### 6. Address Review Feedback

```bash
# Make requested changes
git add .
git commit -m "Address review comments"
git push origin feature/your-feature-name
```

---

## Specific Contribution Areas

### Adding Quantization Methods

To add a new quantization method:

1. Add method to `AdaptiveQuantizer` class:

```python
def _your_method_quantize(self, values: np.ndarray, k: int) -> np.ndarray:
    """
    Your quantization method.
    
    Args:
        values: Input values to quantize
        k: Number of quantization levels
        
    Returns:
        Quantized values
    """
    # Implementation
    ...
```

2. Add classification logic in `DimensionClassifier.classify_dimension()`:

```python
elif your_condition:
    dist_type = "your_type"
    optimal_k = base_k * multiplier
    method = "your_method"
```

3. Add case in `quantize_dimension()`:

```python
elif method == "your_method":
    return self._your_method_quantize(values, k)
```

4. Add tests, documentation, examples

### Adding Evaluation Metrics

To add a new evaluation benchmark:

1. Add method to `EmbeddingEvaluator`:

```python
def evaluate_your_benchmark(self, embeddings, word_to_idx):
    """Evaluate on your benchmark."""
    # Load data
    # Compute scores
    # Return correlation
    ...
```

2. Update `QuantizationEvaluator.compare_methods()`
3. Add to documentation
4. Include sample data or download instructions

### Adding Visualizations

To add a new visualization:

1. Add method to `QuantizationVisualizer`:

```python
def plot_your_visualization(self, data, save_path=None):
    """Create your visualization."""
    fig, ax = plt.subplots(figsize=self.figsize_base)
    
    # Create plot
    ...
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
```

2. Add to EXAMPLES.md
3. Include in test suite

---

## Questions?

- **GitHub Issues**: For bug reports and feature requests
- **Email**: kow.k@ks.kyorin-u.ac.jp for research collaboration
- **Pull Requests**: For code contributions

Thank you for contributing to making embedding quantization better! 🎉
