# Contributing to ETH/FDUSD Advanced Trading Bot

Thank you for your interest in contributing to the ETH/FDUSD Advanced Trading Bot! This project aims to provide a sophisticated, production-ready algorithmic trading system with proprietary mathematical models for identifying market bottoms and tops.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Environment Setup](#development-environment-setup)
4. [Contributing Guidelines](#contributing-guidelines)
5. [Pull Request Process](#pull-request-process)
6. [Issue Reporting](#issue-reporting)
7. [Coding Standards](#coding-standards)
8. [Testing Requirements](#testing-requirements)
9. [Documentation Standards](#documentation-standards)
10. [Security Considerations](#security-considerations)

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow. Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to ensure a welcoming environment for everyone.

### Our Pledge

We are committed to making participation in this project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- Python 3.11 or higher
- Git 2.25 or newer
- Basic understanding of algorithmic trading concepts
- Familiarity with mathematical modeling and machine learning
- Experience with financial markets and risk management

### Areas for Contribution

We welcome contributions in the following areas:

1. **Mathematical Models**: Improvements to CCI and DDM algorithms
2. **Machine Learning**: Enhanced ML models and feature engineering
3. **Risk Management**: Advanced risk control mechanisms
4. **Performance Optimization**: Code optimization and efficiency improvements
5. **Documentation**: Technical documentation and user guides
6. **Testing**: Unit tests, integration tests, and backtesting frameworks
7. **Security**: Security audits and vulnerability assessments
8. **User Interface**: Web interfaces and monitoring dashboards

## Development Environment Setup

### 1. Fork and Clone the Repository

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/your-username/eth-fdusd-trading-bot.git
cd eth-fdusd-trading-bot

# Add the original repository as upstream
git remote add upstream https://github.com/original-repo/eth-fdusd-trading-bot.git
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### 3. Configure Development Settings

```bash
# Copy environment template
cp .env.example .env.dev

# Edit development configuration
nano .env.dev
```

Set up your development environment with testnet credentials:
```bash
BINANCE_API_KEY=your_testnet_api_key
BINANCE_SECRET_KEY=your_testnet_secret_key
BINANCE_TESTNET=true
DEBUG_MODE=true
```

### 4. Verify Installation

```bash
# Run tests to verify setup
python -m pytest tests/

# Run configuration validation
python src/main.py --validate

# Run linting
flake8 src/
black --check src/
```

## Contributing Guidelines

### Types of Contributions

#### Bug Reports
- Use the bug report template
- Include detailed reproduction steps
- Provide system information and logs
- Test on the latest version before reporting

#### Feature Requests
- Use the feature request template
- Explain the use case and benefits
- Consider backward compatibility
- Discuss implementation approach

#### Code Contributions
- Follow the coding standards
- Include comprehensive tests
- Update documentation
- Ensure backward compatibility

#### Documentation Improvements
- Fix typos and grammatical errors
- Improve clarity and completeness
- Add examples and use cases
- Update outdated information

### Contribution Workflow

1. **Check Existing Issues**: Search for existing issues or discussions related to your contribution.

2. **Create an Issue**: For significant changes, create an issue to discuss the approach before implementing.

3. **Create a Branch**: Create a feature branch from the main branch.
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Changes**: Implement your changes following the coding standards.

5. **Test Thoroughly**: Ensure all tests pass and add new tests for your changes.

6. **Update Documentation**: Update relevant documentation and docstrings.

7. **Commit Changes**: Use clear, descriptive commit messages.
   ```bash
   git commit -m "feat: add enhanced CCI calculation with volume weighting"
   ```

8. **Push and Create PR**: Push your branch and create a pull request.

## Pull Request Process

### Before Submitting

- [ ] All tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] Branch is up to date with main

### PR Requirements

1. **Clear Description**: Explain what changes you made and why.

2. **Issue Reference**: Link to related issues using keywords like "Fixes #123".

3. **Testing Evidence**: Include test results and performance benchmarks.

4. **Breaking Changes**: Clearly document any breaking changes.

5. **Screenshots**: Include screenshots for UI changes.

### Review Process

1. **Automated Checks**: All CI/CD checks must pass.

2. **Code Review**: At least one maintainer must review and approve.

3. **Testing**: Changes are tested in multiple environments.

4. **Documentation Review**: Documentation changes are reviewed for accuracy.

5. **Security Review**: Security-sensitive changes undergo additional review.

### Merge Criteria

- All review comments are addressed
- All automated tests pass
- Documentation is complete and accurate
- No merge conflicts exist
- Maintainer approval is obtained

## Issue Reporting

### Bug Reports

Use the bug report template and include:

```markdown
**Bug Description**
A clear description of the bug.

**Reproduction Steps**
1. Step one
2. Step two
3. Step three

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python Version: [e.g., 3.11.0]
- Bot Version: [e.g., 2.1.0]
- Exchange: [e.g., Binance Testnet]

**Logs**
```
Include relevant log excerpts
```

**Additional Context**
Any other relevant information.
```

### Feature Requests

Use the feature request template:

```markdown
**Feature Summary**
Brief description of the feature.

**Problem Statement**
What problem does this solve?

**Proposed Solution**
How should this be implemented?

**Alternatives Considered**
Other approaches you considered.

**Additional Context**
Any other relevant information.
```

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line Length**: 88 characters (Black formatter default)
- **Imports**: Use absolute imports, group by standard/third-party/local
- **Docstrings**: Use Google-style docstrings
- **Type Hints**: Required for all public functions and methods

### Code Formatting

We use automated formatting tools:

```bash
# Format code with Black
black src/ tests/

# Sort imports with isort
isort src/ tests/

# Check style with flake8
flake8 src/ tests/
```

### Naming Conventions

- **Classes**: PascalCase (e.g., `TradingEngine`)
- **Functions/Variables**: snake_case (e.g., `calculate_cci`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_POSITION_SIZE`)
- **Private Methods**: Leading underscore (e.g., `_internal_method`)

### Documentation Standards

#### Docstring Format

```python
def calculate_cci(self, market_data: pd.DataFrame) -> float:
    """
    Calculate the Capitulation Confluence Index.
    
    The CCI combines multiple technical indicators to identify potential
    market bottoms with high accuracy. This proprietary algorithm analyzes
    oversold momentum, volume capitulation, and support level confluence.
    
    Args:
        market_data: DataFrame containing OHLCV data with minimum 100 periods.
                    Required columns: ['open', 'high', 'low', 'close', 'volume']
    
    Returns:
        CCI value between 0 and 100, where higher values indicate stronger
        capitulation signals.
    
    Raises:
        ValueError: If market_data has insufficient data or missing columns.
        
    Example:
        >>> data = get_market_data('ETHFDUSD', '15m', 200)
        >>> cci_value = model.calculate_cci(data)
        >>> print(f"CCI: {cci_value:.2f}")
        CCI: 23.45
    """
```

#### Code Comments

- Use comments sparingly for complex logic
- Explain "why" not "what"
- Keep comments up to date with code changes

```python
# Calculate volume-weighted price to account for liquidity variations
# This helps identify genuine capitulation vs. low-volume noise
vwap = (market_data['close'] * market_data['volume']).sum() / market_data['volume'].sum()
```

## Testing Requirements

### Test Coverage

- Minimum 80% code coverage required
- Critical components (trading logic, risk management) require 95% coverage
- All public methods must have tests

### Test Types

#### Unit Tests
```python
def test_cci_calculation_with_valid_data(self):
    """Test CCI calculation with valid market data."""
    # Arrange
    market_data = create_sample_market_data(periods=100)
    model = MathematicalModels()
    
    # Act
    cci_value = model.calculate_cci(market_data)
    
    # Assert
    self.assertIsInstance(cci_value, float)
    self.assertGreaterEqual(cci_value, 0)
    self.assertLessEqual(cci_value, 100)
```

#### Integration Tests
```python
async def test_trading_engine_complete_cycle(self):
    """Test complete trading cycle from signal to execution."""
    # Test end-to-end trading workflow
    pass
```

#### Performance Tests
```python
def test_cci_calculation_performance(self):
    """Test CCI calculation performance with large datasets."""
    # Benchmark calculation speed
    pass
```

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=src --cov-report=html

# Run specific test file
python -m pytest tests/unit_tests/test_mathematical_models.py

# Run performance tests
python -m pytest tests/performance/ -m performance
```

## Security Considerations

### Security Guidelines

1. **Never Commit Secrets**: API keys, passwords, or sensitive data
2. **Input Validation**: Validate all external inputs
3. **Error Handling**: Don't expose sensitive information in errors
4. **Dependencies**: Keep dependencies updated and audit regularly
5. **Code Review**: Security-sensitive code requires additional review

### Reporting Security Issues

**DO NOT** create public issues for security vulnerabilities. Instead:

1. Email security@yourdomain.com with details
2. Include "SECURITY" in the subject line
3. Provide detailed reproduction steps
4. Allow reasonable time for response before disclosure

### Security Checklist

- [ ] No hardcoded credentials
- [ ] Input validation implemented
- [ ] Error messages don't leak information
- [ ] Dependencies are up to date
- [ ] Code follows security best practices

## Development Tools

### Recommended IDE Setup

**VS Code Extensions:**
- Python
- Pylance
- Black Formatter
- GitLens
- Python Docstring Generator

**PyCharm Configuration:**
- Enable Black formatter
- Configure flake8 as external tool
- Set up pytest as test runner

### Git Hooks

Pre-commit hooks automatically run checks:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8
  - repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
      - id: isort
```

### Debugging

Use the built-in debugging capabilities:

```python
# Enable debug mode
export DEBUG_MODE=true

# Use logging for debugging
import logging
logger = logging.getLogger(__name__)
logger.debug("Debug information here")

# Use pdb for interactive debugging
import pdb; pdb.set_trace()
```

## Release Process

### Version Numbering

We use Semantic Versioning (SemVer):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] Version number bumped
- [ ] Changelog updated
- [ ] Security review completed
- [ ] Performance benchmarks run

## Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Discord**: Real-time chat and community support
- **Email**: security@yourdomain.com for security issues

### Documentation

- **Installation Guide**: [docs/installation.md](docs/installation.md)
- **API Documentation**: [docs/api_reference.md](docs/api_reference.md)
- **Trading Strategy**: [docs/trading_strategy.md](docs/trading_strategy.md)
- **Risk Management**: [docs/risk_management.md](docs/risk_management.md)

### Support

For questions about contributing:

1. Check existing documentation
2. Search GitHub issues and discussions
3. Ask in the Discord community
4. Create a GitHub discussion for complex questions

## Recognition

Contributors are recognized in several ways:

- **Contributors List**: Listed in README.md
- **Release Notes**: Mentioned in release announcements
- **Hall of Fame**: Special recognition for significant contributions
- **Swag**: Stickers and merchandise for active contributors

## License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project (MIT License).

---

Thank you for contributing to the ETH/FDUSD Advanced Trading Bot! Your contributions help make algorithmic trading more accessible and effective for the community.

