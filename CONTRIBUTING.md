# Contributing to MCE

Thank you for considering contributing to the Model Context Engine! Every contribution helps make AI agents more token-efficient.

## How to Contribute

### Reporting Bugs

1. Check [existing issues](https://github.com/DexopT/MCE/issues) to avoid duplicates.
2. Open a new issue with:
   - Clear title describing the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - Your Python version and OS

### Suggesting Features

Open an issue with the `enhancement` label. Describe:
- What problem the feature solves
- How you envision it working
- Any alternative approaches you've considered

### Submitting Code

1. **Fork** the repository
2. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** following the code style below
4. **Add tests** for new functionality
5. **Run the test suite**:
   ```bash
   cd mce-core
   python -m pytest tests/ -v
   ```
6. **Commit** with a clear message:
   ```bash
   git commit -m "feat: add support for XYZ"
   ```
7. **Push** and open a **Pull Request**

## Code Style

- **Python 3.11+** with type hints throughout
- Use `from __future__ import annotations` in all files
- Follow the existing module structure (`schemas/`, `utils/`, `models/`, `engine/`, `core/`)
- Include docstrings for all public classes and functions
- Use the MCE logger (`utils.logger.get_logger`) instead of `print()`

## Commit Convention

Use [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix | Purpose |
|--------|---------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation only |
| `refactor:` | Code restructuring |
| `test:` | Adding or updating tests |
| `chore:` | Build, CI, or tooling changes |

## Development Setup

```bash
git clone https://github.com/DexopT/MCE.git
cd MCE/mce-core
pip install -r requirements.txt
python -m pytest tests/ -v     # verify everything works
python main.py                 # start the proxy
```

## Questions?

Open an issue or reach out to [Yılmaz Karaağaç](mailto:dexopt1@gmail.com).
