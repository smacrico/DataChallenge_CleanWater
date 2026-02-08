# Contributing to Water Quality SA Predictor

Thank you for your interest in contributing to this project!

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/water-quality-sa-predictor.git
   cd water-quality-sa-predictor
   ```

2. **Create a virtual environment:**
   ```bash
   make venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   make install
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your Snowflake credentials
   ```

## Code Standards

- **Python Version:** 3.11+
- **Style:** PEP 8 compliant, enforced by `black`, `isort`, and `ruff`
- **Type Hints:** Required for all function signatures
- **Docstrings:** Google-style docstrings for all public functions/classes
- **Testing:** All new features must include tests with ≥80% coverage

## Workflow

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes and commit:**
   ```bash
   git add .
   git commit -m "feat: add feature description"
   ```
   Use conventional commit prefixes: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`

3. **Run tests and linting:**
   ```bash
   make lint
   make test
   ```

4. **Push and create a Pull Request:**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Pre-commit hooks** will automatically run on commit. Fix any issues before pushing.

## Pull Request Guidelines

- Ensure CI passes (linting, tests, smoke tests)
- Include clear description of changes
- Reference related issues
- Update documentation if needed
- Add tests for new functionality

## Code Review Process

- At least one approval required
- All CI checks must pass
- Maintainers will review within 2-3 business days

## Reporting Issues

- Use the GitHub issue tracker
- Include clear reproduction steps
- Provide environment details (Python version, OS, dependencies)
- Check for existing issues before creating new ones

## Questions?

Open a discussion in GitHub Discussions or reach out to the maintainers.

Thank you for contributing! 🚀
