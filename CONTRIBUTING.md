# Contributing to Multi-Source Research Aggregator

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/multi-source-research-aggregator.git`
3. Create a feature branch: `git checkout -b feature/your-feature-name`

## Development Setup

### Prerequisites

- Python 3.8+
- Poetry (recommended) or pip

### Installation

```bash
# Using Poetry (recommended)
poetry install

# Using pip
pip install -r requirements.txt
```

### Environment Setup

```bash
cp .env.example .env
# Edit .env and add your API keys
```

## Running the Application

```bash
# Using Poetry
poetry run python research_aggregator.py

# Using pip
python research_aggregator.py
```

## Testing

Before submitting a pull request:

1. Test the application with different queries
2. Ensure all sources are working correctly
3. Verify API key functionality (optional but recommended)

## Adding New Data Sources

To add a new research data source:

1. Create a new `fetch_your_source()` async function
2. Add harmonization logic in `harmonise_result()`
3. Update the `aggregate()` function to include your source
4. Test thoroughly with various queries

## Submitting Changes

1. Ensure your code follows the style guidelines
2. Test your changes thoroughly
3. Update documentation if needed
4. Submit a pull request with a clear description

## Issues

When reporting issues, please include:

- Python version
- Operating system
- Full error message
- Steps to reproduce
- Expected vs actual behavior

Thank you for contributing!
