# Contributing to Context Store

Thank you for your interest in contributing to Context Store! This guide will help you get started.

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Node.js 18 or higher
- Docker and Docker Compose
- Git

### Local Development

1. **Clone the repository**

   ```bash
   git clone https://github.com/credentum/context-store.git
   cd context-store
   ```

2. **Set up Python environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements-dev.txt
   ```

3. **Set up Node.js environment**

   ```bash
   npm install
   ```

4. **Install pre-commit hooks**

   ```bash
   pre-commit install
   ```

5. **Start development services**

   ```bash
   docker-compose up -d qdrant neo4j redis
   ```

6. **Copy environment configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

## Development Workflow

### Making Changes

1. **Create a new branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**

   - Follow the coding standards below
   - Add tests for new functionality
   - Update documentation as needed

3. **Run tests locally**

   ```bash
   # Python tests
   pytest --cov=src

   # TypeScript tests
   npm test

   # Integration tests
   pytest tests/integration/
   ```

4. **Run code quality checks**

   ```bash
   # This will run automatically with pre-commit hooks
   pre-commit run --all-files
   ```

5. **Commit your changes**

   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

6. **Push and create a pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

## Coding Standards

### Python

- **Style**: Follow PEP 8, enforced by Black (line length: 100)
- **Imports**: Organized with isort
- **Type hints**: Required for all functions and methods
- **Docstrings**: Google style for all public functions and classes
- **Testing**: pytest with minimum 80% coverage

Example:

```python
from typing import Dict, List, Optional

def process_context(
    content: Dict[str, Any],
    context_type: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Process context data for storage.

    Args:
        content: The context content to process
        context_type: Type of context (design, decision, etc.)
        metadata: Optional metadata dictionary

    Returns:
        Processed context data

    Raises:
        ValidationError: If content validation fails
    """
    # Implementation here
    pass
```

### TypeScript

- **Style**: Prettier formatting (2 spaces, single quotes)
- **Linting**: ESLint with TypeScript rules
- **Types**: Strict TypeScript configuration
- **Testing**: Jest for unit tests

Example:

```typescript
interface ContextData {
  id: string;
  type: ContextType;
  content: Record<string, unknown>;
  metadata?: Record<string, unknown>;
}

export async function storeContext(data: ContextData): Promise<string> {
  // Implementation here
}
```

### Commit Messages

Use [Conventional Commits](https://conventionalcommits.org/):

- `feat: add new feature`
- `fix: bug fix`
- `docs: documentation changes`
- `style: formatting changes`
- `refactor: code refactoring`
- `test: add or update tests`
- `chore: maintenance tasks`

## Testing

### Python Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_storage.py

# Run integration tests
pytest tests/integration/
```

### TypeScript Tests

```bash
# Run all tests
npm test

# Run with coverage
npm run test:coverage

# Run in watch mode
npm run test:watch
```

### Integration Tests

Integration tests require running services:

```bash
# Start services
docker-compose up -d

# Run integration tests
pytest tests/integration/

# Clean up
docker-compose down
```

## Documentation

### Code Documentation

- All public functions and classes must have docstrings
- Use Google style docstrings for Python
- Use JSDoc comments for TypeScript
- Include examples in docstrings when helpful

### API Documentation

- Update API documentation when adding new endpoints
- Include request/response examples
- Document error responses

### User Documentation

- Update README.md for user-facing changes
- Add deployment guides for new features
- Update configuration documentation

## Pull Request Process

1. **Fill out the PR template completely**
2. **Ensure all tests pass**
3. **Verify code coverage meets requirements**
4. **Update documentation as needed**
5. **Request review from maintainers**

### PR Checklist

- [ ] Tests added for new functionality
- [ ] Documentation updated
- [ ] Code follows style guidelines
- [ ] All CI checks pass
- [ ] Breaking changes documented

## Release Process

1. **Update version numbers**

   - `pyproject.toml`
   - `package.json`
   - `src/__init__.py`

2. **Update CHANGELOG.md**

   - Move unreleased changes to new version
   - Add release date

3. **Create release tag**

   ```bash
   git tag -a v0.2.0 -m "Release v0.2.0"
   git push origin v0.2.0
   ```

4. **GitHub Actions will automatically**
   - Build and test the release
   - Create GitHub release
   - Publish to package registries

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/credentum/context-store/issues)
- **Discussions**: [GitHub Discussions](https://github.com/credentum/context-store/discussions)
- **Discord**: [Join our Discord server](https://discord.gg/credentum)

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## License

By contributing to Context Store, you agree that your contributions will be licensed under the MIT License.
