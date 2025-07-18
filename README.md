# Multi-Source Research Aggregator

A standalone Python script that aggregates research papers and documents from multiple academic and health-related sources.

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

1. Clone or download this project
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

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
