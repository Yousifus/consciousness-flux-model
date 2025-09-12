# Contributing to Consciousness Flux Model

Thank you for your interest in contributing to the Consciousness Flux Model! This document provides guidelines for contributing to the project.

## Getting Started

### Prerequisites
- Python 3.10 or higher
- Git
- Basic understanding of mathematical modeling and consciousness research

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/consciousness-flux-model.git
   cd consciousness-flux-model
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in development mode
   ```

4. **Run tests to ensure everything works**
   ```bash
   pytest tests/ -v
   ```

## How to Contribute

### Types of Contributions

1. **Bug Reports**: Report issues with clear reproduction steps
2. **Feature Requests**: Suggest new functionality or improvements
3. **Code Contributions**: Fix bugs, add features, improve documentation
4. **Research Contributions**: Improve the mathematical model, add new priors
5. **Documentation**: Improve README, add examples, clarify concepts

### Development Workflow

1. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   pytest tests/ -v
   python -m src.consciousness_flux_model_v1 --priors PHYSICALIST
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```

5. **Push and create a Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

### Code Style Guidelines

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and reasonably sized
- Add type hints where appropriate

### Testing Guidelines

- Write tests for new functionality
- Ensure all existing tests pass
- Aim for good test coverage
- Test edge cases and error conditions

### Mathematical Model Contributions

When contributing to the mathematical model:

- Clearly document the mathematical reasoning
- Provide references to relevant literature
- Test with different philosophical priors
- Ensure numerical stability and reasonable parameter ranges

### Documentation Contributions

- Use clear, concise language
- Provide examples where helpful
- Keep documentation up-to-date with code changes
- Use proper markdown formatting

## Issue Guidelines

### Before Submitting an Issue

1. Check if the issue already exists
2. Try the latest version
3. Provide clear reproduction steps
4. Include system information (OS, Python version)

### Issue Templates

Use the provided issue templates for:
- Bug reports
- Feature requests
- Research discussions

## Pull Request Guidelines

### Before Submitting a PR

1. Ensure all tests pass
2. Update documentation if needed
3. Follow the commit message format
4. Keep PRs focused and reasonably sized

### PR Review Process

1. Automated checks must pass (CI/CD)
2. At least one maintainer review required
3. Address all review comments
4. Merge after approval

## Research and Philosophy

This project explores consciousness from multiple philosophical perspectives:

- **Physicalist**: Materialist approach to consciousness
- **IIT**: Integrated Information Theory
- **Panpsychist**: Consciousness as fundamental property

When contributing research:

- Respect different philosophical viewpoints
- Provide clear mathematical formulations
- Acknowledge limitations and assumptions
- Cite relevant literature

## Community Guidelines

- Be respectful and constructive
- Help others learn and grow
- Focus on the science and mathematics
- Welcome diverse perspectives on consciousness

## Questions?

Feel free to:
- Open an issue for questions
- Join discussions in the Issues section
- Contact maintainers directly

Thank you for contributing to consciousness research! 🌌
