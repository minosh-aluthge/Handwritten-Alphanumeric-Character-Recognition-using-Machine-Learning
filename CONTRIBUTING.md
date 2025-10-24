# CONTRIBUTING.md

## 🤝 Contributing to AlphaNum Character Recognition

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## 📋 Code of Conduct

- Be respectful and constructive in all interactions
- Provide constructive feedback
- Help others learn and grow
- Report unacceptable behavior to maintainers

## 🐛 Reporting Issues

### Before Creating an Issue
- Check existing issues to avoid duplicates
- Update to the latest version
- Verify you're following the documentation

### Creating an Issue
Include:
1. **Clear title** describing the problem
2. **Description** of the issue with context
3. **Steps to reproduce** (if applicable)
4. **Expected vs actual behavior**
5. **Environment** (OS, Python version, GPU/CPU)
6. **Error messages** and stack traces
7. **Screenshots** (if applicable)

### Issue Template
```
## Description
Brief description of the issue

## Steps to Reproduce
1. ...
2. ...
3. ...

## Expected Behavior
What should happen

## Actual Behavior
What actually happened

## Environment
- OS: [e.g., Windows 11, Ubuntu 22.04]
- Python: 3.x
- Framework: TensorFlow / PyTorch
- GPU: Yes/No (NVIDIA, AMD, etc.)

## Error Output
```
[paste error message here]
```
```

## 💡 Suggesting Enhancements

### Before Suggesting
- Check if already suggested
- Explain why it's needed
- Consider the scope

### Enhancement Template
```
## Description
What should be improved or added

## Motivation
Why this enhancement is important

## Proposed Solution
Describe your idea

## Alternative Solutions
Other approaches you considered

## Additional Context
Any other information
```

## 🛠️ Development Setup

### 1. Fork and Clone
```bash
# Fork on GitHub, then
git clone https://github.com/yourusername/AlphaNum-Character-Recognition.git
cd AlphaNum-Character-Recognition
git remote add upstream https://github.com/original/AlphaNum-Character-Recognition.git
```

### 2. Create a Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix-name
```

**Branch naming conventions:**
- `feature/description` - New feature
- `fix/description` - Bug fix
- `docs/description` - Documentation
- `refactor/description` - Code refactoring
- `test/description` - Test additions

### 3. Set Up Development Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development tools
```

### 4. Make Changes
- Follow coding standards (see below)
- Create small, focused commits
- Write descriptive commit messages
- Update tests
- Update documentation

### 5. Test Your Changes
```bash
# Run tests
pytest tests/

# For specific test
pytest tests/test_models.py::test_lenet5

# With coverage
pytest --cov=. tests/
```

### 6. Commit and Push
```bash
git add .
git commit -m "feat: add new model comparison feature"
git push origin feature/your-feature-name
```

### 7. Create Pull Request
- Go to GitHub and create a PR
- Link related issues
- Describe changes clearly
- Wait for review

## 📝 Coding Standards

### Python Style
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use type hints where applicable
- Maximum line length: 100 characters
- Use meaningful variable names

### Example:
```python
from typing import Tuple, List
import numpy as np

def process_images(images: np.ndarray, 
                  size: Tuple[int, int] = (48, 48)) -> np.ndarray:
    """
    Process images to standard size.
    
    Args:
        images: Input images array
        size: Target size (height, width)
        
    Returns:
        Processed images array
    """
    processed = []
    for img in images:
        resized = cv2.resize(img, size)
        processed.append(resized)
    return np.array(processed)
```

### Notebooks
- Clear cell organization
- Markdown explanations
- Reproducible code
- No output cells with errors
- Clean saved notebooks (no extraneous outputs)

### Documentation
- Docstrings for functions/classes
- README sections for features
- Comments for complex logic
- Type hints in code

## 📋 Pull Request Process

1. **Update documentation** - README, MODELS.md, etc.
2. **Add tests** - Cover new functionality
3. **Update CHANGELOG** - Describe changes
4. **Pass CI/CD** - Ensure tests pass
5. **Request review** - Usually 1-2 reviewers
6. **Address feedback** - Update PR based on comments
7. **Squash commits** (optional) - Keep history clean
8. **Merge** - After approval

### PR Title Format
```
[type]: Brief description

type: feat|fix|docs|style|refactor|test|chore
```

Examples:
- `feat: add ResNet-50 model`
- `fix: correct confusion matrix calculation`
- `docs: update installation guide`
- `test: add model evaluation tests`

## 🧪 Testing

### Writing Tests
```python
# tests/test_models.py
import unittest
import numpy as np
from models import LeNet5

class TestLeNet5(unittest.TestCase):
    def setUp(self):
        self.model = LeNet5()
        self.sample_input = np.random.rand(1, 48, 48, 1)
    
    def test_model_loading(self):
        """Test model loads without error"""
        self.assertIsNotNone(self.model)
    
    def test_prediction_shape(self):
        """Test output shape is correct"""
        output = self.model.predict(self.sample_input)
        self.assertEqual(output.shape, (1, 62))
    
    def test_accuracy_above_threshold(self):
        """Test model accuracy meets minimum"""
        accuracy = self.model.evaluate(test_data)
        self.assertGreater(accuracy, 0.80)

if __name__ == '__main__':
    unittest.main()
```

### Running Tests
```bash
# All tests
pytest

# Specific file
pytest tests/test_models.py

# With verbose output
pytest -v

# With coverage
pytest --cov=. --cov-report=html
```

## 📚 Documentation

### For New Features
1. Update relevant `.md` file
2. Add docstrings to code
3. Update README if user-facing
4. Add notebook examples
5. Update MODELS.md for architecture changes

### Documentation Template
```markdown
## New Feature Name

### Description
What does it do?

### Usage
```python
code example
```

### Parameters
- **param1**: Description (type)
- **param2**: Description (type, default)

### Returns
Description of return value

### Example
```python
more detailed example
```

### Notes
Any important information
```

## 🚀 Areas for Contribution

### High Priority
- [ ] Additional model architectures (EfficientNet, etc.)
- [ ] Data augmentation improvements
- [ ] Performance optimization
- [ ] Cross-platform web app compatibility
- [ ] Better GPU support

### Medium Priority
- [ ] More comprehensive tests
- [ ] Docker containerization
- [ ] API documentation (OpenAPI/Swagger)
- [ ] Model interpretability tools
- [ ] Transfer learning examples

### Good First Issues
- [ ] Documentation improvements
- [ ] Bug fixes (marked as "good-first-issue")
- [ ] Adding comments to code
- [ ] Creating unit tests
- [ ] Example scripts

## 📊 Commit Message Guidelines

**Format:**
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Type:** feat, fix, docs, style, refactor, test, chore
**Scope:** models, notebooks, web, docs, ci, etc.
**Subject:** Imperative mood, lowercase, no period

**Example:**
```
feat(models): add EfficientNet implementation

- Implement EfficientNet-B0 architecture
- Add transfer learning from ImageNet weights
- Achieve 94% accuracy on test set
- Update MODELS.md with benchmarks

Fixes #123
```

## 🔍 Code Review

### For Reviewers
- Be constructive and respectful
- Suggest improvements, not demands
- Ask questions to understand intent
- Test the changes locally
- Approve when satisfied

### For Contributors
- Don't take feedback personally
- Ask clarifying questions
- Make requested changes
- Re-request review after updates
- Thank reviewers

## 📦 Release Process

### Version Numbering
Follow [Semantic Versioning](https://semver.org/):
- `MAJOR.MINOR.PATCH` (e.g., 1.2.3)
- Increment MAJOR for breaking changes
- Increment MINOR for new features
- Increment PATCH for bug fixes

### Before Release
- Update version in code
- Update CHANGELOG.md
- Update README if needed
- Tag release in git
- Create GitHub release with notes

## ✨ Recognition

Contributors will be recognized in:
- README.md contributors section
- GitHub contributors page
- Release notes

## 📞 Questions?

- Create a **Discussion** on GitHub
- Check existing documentation
- Search closed issues
- Ask in community channels

## 🙏 Thank You!

Your contributions make this project better for everyone. We appreciate:
- Code contributions
- Bug reports
- Feature suggestions
- Documentation improvements
- Testing and feedback

---

**Happy Contributing!** 🎉

For more information, see [README.md](README.md) and [SETUP.md](SETUP.md).

**Last Updated**: October 2024
