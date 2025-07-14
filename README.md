# Multi-Source Research Aggregator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/dependency%20management-poetry-blue.svg)](https://python-poetry.org/)

A standalone Python application that aggregates research papers and documents from multiple academic and health-related sources, featuring semantic search capabilities powered by sentence transformers and FAISS.

**Developed by [SolaceAI](https://github.com/solaceai-evidence)**

## Features

- **Multiple Data Sources:**

  - Semantic Scholar (academic papers)
  - OpenAlex (open academic works)
  - ReliefWeb (humanitarian reports)
  - WHO IRIS (World Health Organization documents)

- **Semantic Search:** Uses sentence transformers and FAISS for semantic similarity search
- **Data Harmonization:** Normalizes results from different sources into a consistent format
- **Vector Storage:** Builds and persists vector embeddings for fast retrieval

## Installation

### Using Poetry (Recommended)

1. Clone or download this project
2. Install Poetry if you haven't already: `curl -sSL https://install.python-poetry.org | python3 -`
3. Install dependencies:

```bash
poetry install
```

4. Set up your API keys (optional but recommended):

```bash
cp .env.example .env
# Edit .env and add your Semantic Scholar API key
```

### Using pip

1. Clone or download this project
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

### API Keys

While the script works without API keys, using them provides higher rate limits and better reliability:

- **Semantic Scholar API Key**: Get one from [Semantic Scholar API](https://www.semanticscholar.org/product/api#Partner-Form)
  - Copy `.env.example` to `.env`
  - Add your API key: `SEMANTIC_SCHOLAR_API_KEY=your_key_here`

## Usage

### Using Poetry

```bash
poetry run python research_aggregator.py
```

### Using pip

```bash
python research_aggregator.py
```

### Basic Usage

Run the script with the default query:

```bash
python research_aggregator.py
```

### Custom Queries

Edit the `query_term` variable in the `main()` function to search for different topics:

```python
query_term = "your custom search query here"
```

### Example Queries

- "cholera outbreaks floods Ethiopia"
- "malaria vaccine effectiveness in Africa"
- "COVID-19 mental health impacts"
- "climate change health risks"

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to:

- Report issues
- Submit feature requests
- Add new data sources
- Improve documentation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Semantic Scholar](https://www.semanticscholar.org/) for academic paper search
- [OpenAlex](https://openalex.org/) for open academic works
- [ReliefWeb](https://reliefweb.int/) for humanitarian reports
- [WHO IRIS](https://iris.who.int/) for World Health Organization documents
- [Sentence Transformers](https://www.sbert.net/) for semantic embeddings
- [FAISS](https://faiss.ai/) for efficient similarity search

## Organization

This project is maintained by [SolaceAI Evidence](https://github.com/solaceai-evidence), an organization focused on developing AI tools for evidence-based research and decision making.

## Output

The script will:

1. Fetch results from all configured sources
2. Harmonize the data format
3. Build a vector store for semantic search
4. Display the top 5 most relevant results with:
   - Source name
   - Title
   - URL
   - Authors (first 3)
   - Publication year
   - Missing abstract warnings

## Requirements

- Python 3.7+
- Internet connection for API calls
- Dependencies listed in `requirements.txt`

## Notes

- The script creates a `vector_store.pkl` file to persist embeddings
- First run will download the sentence transformer model (~90MB)
- Some sources may have rate limits or require API keys for heavy usage
