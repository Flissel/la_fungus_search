# EmbeddingGemma Test Suite

This directory contains comprehensive tests for the EmbeddingGemma project, including unit tests, integration tests, end-to-end tests, performance tests, and stress tests.

## Test Structure

```
tests/
├── README.md                    # This file
├── conftest.py                  # Pytest configuration and fixtures
├── test_integration.py          # Integration tests for MCPMRetriever
├── test_e2e.py                  # End-to-end tests for the complete application
├── test_performance.py          # Performance and stress tests
├── mcmp/                        # MCMP module tests
│   ├── test_comprehensive.py    # Comprehensive MCMP tests
│   ├── test_embeddings.py       # Embedding tests
│   ├── test_indexing.py         # Indexing tests
│   ├── test_pca.py             # PCA tests
│   ├── test_simulation.py      # Simulation tests
│   └── test_visualize.py       # Visualization tests
├── rag/                        # RAG module tests
│   ├── test_comprehensive.py    # Comprehensive RAG tests
│   ├── test_chunking.py        # Chunking tests
│   ├── test_config.py          # Configuration tests
│   ├── test_embeddings.py      # RAG embedding tests
│   ├── test_errors.py          # Error handling tests
│   ├── test_generation.py      # Generation tests
│   ├── test_indexer.py         # Indexer tests
│   ├── test_search.py          # Search tests
│   └── test_vectorstore.py     # Vectorstore tests
└── ui/                         # UI module tests
    ├── test_comprehensive.py    # Comprehensive UI tests
    └── [other UI component tests]
```

## Test Categories

### Unit Tests
- **Location**: `tests/mcmp/`, `tests/rag/`, `tests/ui/`
- **Purpose**: Test individual functions and classes in isolation
- **Speed**: Fast execution
- **Coverage**: High coverage of individual components

### Integration Tests
- **Location**: `tests/test_integration.py`
- **Purpose**: Test how components work together
- **Speed**: Medium execution time
- **Coverage**: Component interactions

### End-to-End Tests
- **Location**: `tests/test_e2e.py`
- **Purpose**: Test complete workflows and user scenarios
- **Speed**: Slower execution
- **Coverage**: Full application workflows

### Performance Tests
- **Location**: `tests/test_performance.py`
- **Purpose**: Test performance characteristics and scalability
- **Speed**: Variable execution time
- **Coverage**: Performance benchmarks and memory usage

## Running Tests

### Using the Test Runner Script

The easiest way to run tests is using the provided `run-tests.py` script:

```bash
# Run fast unit tests only
python run-tests.py fast

# Run integration tests
python run-tests.py integration

# Run performance tests
python run-tests.py performance

# Run all tests
python run-tests.py all

# Run tests with coverage
python run-tests.py coverage

# Run linting checks
python run-tests.py lint

# Run security scan
python run-tests.py security

# Simulate full CI pipeline
python run-tests.py ci

# Run specific test file
python run-tests.py specific --test-path tests/test_integration.py
```

### Using pytest Directly

You can also run tests directly with pytest:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_integration.py

# Run with coverage
pytest --cov=src/embeddinggemma

# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only performance tests
pytest -m performance

# Run tests in parallel (requires pytest-xdist)
pytest -n auto

# Run with verbose output
pytest -v
```

## CI/CD Integration

The project includes a comprehensive GitHub Actions workflow (`.github/workflows/test-suite.yml`) that runs:

- **Unit Tests**: Across multiple Python versions (3.10, 3.11, 3.12)
- **Integration Tests**: End-to-end application workflows
- **Performance Tests**: Performance benchmarks and memory leak detection
- **Cross-Platform Tests**: Compatibility across Windows, macOS, and Linux
- **Linting and Type Checking**: Code quality checks
- **Security Scanning**: Dependency vulnerability scanning
- **Coverage Reporting**: Test coverage analysis

## Test Markers

Tests are marked with categories for selective execution:

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only performance tests
pytest -m performance

# Run only stress tests
pytest -m stress

# Run only end-to-end tests
pytest -m e2e

# Skip slow tests
pytest -m "not slow"
```

## Configuration

Test configuration is managed through:

- **`pytest.ini`**: Pytest configuration and markers
- **`pyproject.toml`**: Project metadata and tool configurations
- **`conftest.py`**: Shared fixtures and setup

## Writing Tests

### Guidelines for New Tests

1. **Follow naming conventions**: `test_*.py` for files, `test_*` for functions
2. **Use descriptive test names**: Explain what the test verifies
3. **Keep tests focused**: Test one specific behavior per test function
4. **Use appropriate fixtures**: Leverage existing fixtures in `conftest.py`
5. **Mock external dependencies**: Use `unittest.mock` for external services
6. **Handle edge cases**: Test error conditions and boundary values
7. **Document complex tests**: Add comments explaining test logic

### Test Structure

```python
def test_descriptive_name():
    """Test that does something specific."""
    # Arrange
    setup_test_data()

    # Act
    result = function_under_test()

    # Assert
    assert result == expected_value
```

## Performance Testing

Performance tests include:

- **Benchmarks**: Measure execution time for critical operations
- **Memory Tests**: Monitor memory usage and detect leaks
- **Scalability Tests**: Test performance with increasing data sizes
- **Stress Tests**: Test system behavior under extreme conditions

## Coverage Goals

The test suite aims for:

- **Unit Tests**: >90% coverage of individual modules
- **Integration Tests**: Coverage of key component interactions
- **End-to-End Tests**: Coverage of critical user workflows
- **Performance Tests**: Baseline performance metrics

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `PYTHONPATH` includes the `src` directory
2. **Missing Dependencies**: Install development dependencies with `pip install -e ".[dev]"`
3. **Memory Issues**: Some performance tests may require more RAM
4. **Timeout Issues**: Increase timeout for slow tests in `pytest.ini`

### Debug Mode

For debugging failing tests:

```bash
# Run with detailed output
pytest -v -s

# Run with debugger
pytest --pdb

# Run specific test with output capture disabled
pytest tests/test_file.py::TestClass::test_method -s
```

## Contributing

When adding new tests:

1. Follow the existing test structure and naming conventions
2. Add appropriate test markers for categorization
3. Ensure tests are isolated and don't depend on external services
4. Update this README if adding new test categories
5. Run the full test suite before submitting changes

## Metrics

The test suite tracks:

- **Execution Time**: Total time for test runs
- **Coverage**: Code coverage percentages
- **Flaky Tests**: Tests that occasionally fail
- **Performance Baselines**: Performance regression detection

Run `python run-tests.py ci` to see a summary of all test categories.