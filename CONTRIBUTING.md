# Contributing to duralang

Thanks for contributing.

## Development Setup

1. Create and activate a virtual environment.
2. Install dependencies from `pyproject.toml`.
3. Start a local Temporal server:

```bash
temporal server start-dev
```

## Testing

Run the full test suite before opening a PR.

```bash
pytest -q
```

## Pull Requests

- Keep PRs focused and small.
- Add or update tests for behavior changes.
- Update documentation for user-facing changes.
- Use clear commit messages.

## Reporting Issues

Please include:

- Expected behavior
- Actual behavior
- Minimal reproduction
- Environment details (OS, Python version, provider)
