import asyncio
import os
import pickle
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional

import httpx
import numpy as np
import faiss
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Configuration
class Config:
    APP_NAME = "MultiSourceResearchAggregator"
    SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    OPENALEX_URL = "https://api.openalex.org/works"
    RELIEFWEB_URL = "https://api.reliefweb.int/v1/reports"
    WHO_IRIS_OAI_URL = "https://iris.who.int/oai/request"
    PERSISTENCE_PATH = "vector_store.pkl"
    SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    DEFAULT_LIMIT = 5


# Initialize model with error handling
def initialize_model() -> SentenceTransformer:
    """Initialize the sentence transformer model with error handling."""
    print("Loading sentence transformer model...")
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("This might be due to network issues or first-time download.")
        print("Please check your internet connection and try again.")
        exit(1)


model = initialize_model()


# Utility functions
def reconstruct_abstract(inverted_index: Dict[str, List[int]]) -> Optional[str]:
    """Reconstruct abstract from OpenAlex inverted index format."""
    if not inverted_index:
        return None
    position_map = {
        pos: word for word, positions in inverted_index.items() for pos in positions
    }
    return " ".join(position_map[i] for i in sorted(position_map))


def clean_html(raw_html: str) -> Optional[str]:
    """Clean HTML content and extract text."""
    return (
        BeautifulSoup(raw_html, "html.parser").get_text(separator=" ").strip()
        if raw_html
        else None
    )


def flag_missing_abstract(doc: Dict) -> Dict:
    """Flag documents with missing abstracts."""
    if not doc.get("abstract"):
        doc["missing_abstract"] = True
    return doc


# HTTP Client helper
async def make_request(
    url: str,
    method: str = "GET",
    params: Dict = None,
    json_data: Dict = None,
    headers: Dict = None,
    follow_redirects: bool = False,
) -> Dict:
    """Generic HTTP request handler."""
    async with httpx.AsyncClient(follow_redirects=follow_redirects) as client:
        if method.upper() == "POST":
            response = await client.post(url, json=json_data, headers=headers or {})
        else:
            response = await client.get(url, params=params, headers=headers or {})
        response.raise_for_status()
        return response


# Data fetchers
async def fetch_semantic_scholar(
    query: str, limit: int = Config.DEFAULT_LIMIT
) -> List[Dict]:
    """Fetch data from Semantic Scholar API."""
    print("Fetching from Semantic Scholar...")

    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,authors,year,url",
    }
    headers = {}

    if Config.SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = Config.SEMANTIC_SCHOLAR_API_KEY
        print("  → Using API key")
    else:
        print("  → No API key, using rate-limited access")

    response = await make_request(
        Config.SEMANTIC_SCHOLAR_URL, params=params, headers=headers
    )
    data = response.json().get("data", [])
    print(f"  ✓ Completed ({len(data)} results)")
    return data


async def fetch_openalex(query: str, limit: int = Config.DEFAULT_LIMIT) -> List[Dict]:
    """Fetch data from OpenAlex API."""
    print("Fetching from OpenAlex...")

    params = {"search": query, "per-page": limit}
    response = await make_request(Config.OPENALEX_URL, params=params)
    data = response.json().get("results", [])
    print(f"  ✓ Completed ({len(data)} results)")
    return data


async def fetch_reliefweb(query: str, limit: int = Config.DEFAULT_LIMIT) -> List[Dict]:
    """Fetch data from ReliefWeb API."""
    print("Fetching from ReliefWeb...")

    payload = {
        "query": {"value": query},
        "limit": limit,
        "fields": {"include": ["title", "url", "body-html", "source", "date"]},
    }
    response = await make_request(
        Config.RELIEFWEB_URL, method="POST", json_data=payload
    )
    data = response.json().get("data", [])
    print(f"  ✓ Completed ({len(data)} results)")
    return data


async def fetch_who_iris(query: str, limit: int = Config.DEFAULT_LIMIT) -> List[Dict]:
    """Fetch data from WHO IRIS OAI-PMH endpoint."""
    print("Fetching from WHO IRIS...")

    params = {
        "verb": "ListRecords",
        "metadataPrefix": "oai_dc",
        "set": "default",
        "q": query,
    }

    response = await make_request(
        Config.WHO_IRIS_OAI_URL, params=params, follow_redirects=True
    )
    xml_data = response.text

    # Parse XML
    root = ET.fromstring(xml_data)
    ns = {
        "oai": "http://www.openarchives.org/OAI/2.0/",
        "dc": "http://purl.org/dc/elements/1.1/",
    }

    records = []
    for record in root.findall(".//oai:record", ns):
        if metadata := record.find(".//oai:metadata", ns):
            if dc := metadata.find("dc:dc", ns):
                records.append(
                    {
                        "title": dc.findtext("dc:title", default="", namespaces=ns),
                        "abstract": dc.findtext(
                            "dc:description", default="", namespaces=ns
                        ),
                        "authors": (
                            [creator]
                            if (
                                creator := dc.findtext(
                                    "dc:creator", default="", namespaces=ns
                                )
                            )
                            else []
                        ),
                        "year": dc.findtext("dc:date", default="", namespaces=ns),
                        "url": dc.findtext("dc:identifier", default="", namespaces=ns),
                        "source": "WHO IRIS",
                    }
                )

    result = records[:limit]
    print(f"  ✓ Completed ({len(result)} results)")
    return result


# Data harmonization
def harmonise_result(source: str, item: Dict) -> Dict:
    """Convert results from different sources to a unified format."""
    harmonizers = {
        "semantic_scholar": lambda x: {
            "title": x.get("title"),
            "abstract": x.get("abstract"),
            "authors": [a.get("name") for a in x.get("authors", [])],
            "year": x.get("year"),
            "url": x.get("url"),
            "source": "Semantic Scholar",
        },
        "openalex": lambda x: {
            "title": x.get("title"),
            "abstract": reconstruct_abstract(x.get("abstract_inverted_index")),
            "authors": [
                a.get("author", {}).get("display_name")
                for a in x.get("authorships", [])
            ],
            "year": x.get("publication_year"),
            "url": x.get("id"),
            "source": "OpenAlex",
        },
        "reliefweb": lambda x: {
            "title": x.get("fields", {}).get("title"),
            "abstract": clean_html(x.get("fields", {}).get("body-html")),
            "authors": _extract_reliefweb_author(x.get("fields", {})),
            "year": x.get("fields", {}).get("date"),
            "url": x.get("fields", {}).get("url"),
            "source": "ReliefWeb",
        },
        "who_iris": lambda x: x,  # Already harmonized
    }

    doc = harmonizers.get(source, lambda x: {})(item)
    return flag_missing_abstract(doc)


def _extract_reliefweb_author(fields: Dict) -> List[str]:
    """Helper to extract author name from ReliefWeb source field."""
    source_field = fields.get("source", [])
    if isinstance(source_field, list) and source_field:
        author_name = (
            source_field[0].get("name", "")
            if isinstance(source_field[0], dict)
            else str(source_field[0])
        )
    elif isinstance(source_field, dict):
        author_name = source_field.get("name", "")
    else:
        author_name = ""
    return [author_name] if author_name else []


# Vector store operations
def build_vector_store(documents: List[Dict]) -> Dict:
    """Build FAISS vector store from documents."""
    if not documents:
        return {"index": None, "documents": []}

    corpus = [
        f"{doc.get('title', '')}. {doc.get('abstract') or ''}" for doc in documents
    ]
    embeddings = model.encode(corpus, convert_to_tensor=False)

    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings[0].shape[0])
    index.add(np.array(embeddings, dtype=np.float32))

    # Save to disk
    store_data = {"index": index, "documents": documents}
    with open(Config.PERSISTENCE_PATH, "wb") as f:
        pickle.dump(store_data, f)

    return store_data


def semantic_search(query: str, store: Dict, top_k: int = 5) -> List[Dict]:
    """Perform semantic search on the vector store."""
    if store["index"] is None:
        return []

    query_vec = model.encode([query], convert_to_tensor=False)[0].astype(np.float32)
    distances, indices = store["index"].search(np.array([query_vec]), top_k)
    return [store["documents"][i] for i in indices[0]]


# Main aggregation function
async def aggregate(query: str, limit: int = Config.DEFAULT_LIMIT) -> List[Dict]:
    """Aggregate research from all sources."""
    print(f"Searching for: {query}")
    print("Fetching from multiple sources...")

    # Fetch from all sources concurrently
    sources_data = await asyncio.gather(
        fetch_semantic_scholar(query, limit),
        fetch_openalex(query, limit),
        fetch_reliefweb(query, limit),
        fetch_who_iris(query, limit),
        return_exceptions=True,
    )

    # Process results and handle any exceptions
    ss_data, oa_data, rw_data, who_data = sources_data
    source_names = ["Semantic Scholar", "OpenAlex", "ReliefWeb", "WHO IRIS"]

    print(f"\nResults per source:")
    results = []
    for i, (name, data, source_key) in enumerate(
        zip(
            source_names,
            sources_data,
            ["semantic_scholar", "openalex", "reliefweb", "who_iris"],
        )
    ):
        if isinstance(data, Exception):
            print(f"  {name}: Error - {data}")
            data = []
        else:
            print(f"  {name}: {len(data)} results")

        results.extend([harmonise_result(source_key, item) for item in data])

    print(f"\nTotal aggregated results: {len(results)}")
    return results


# Main application
def display_results(results: List[Dict], query: str, top_k: int = 5) -> None:
    """Display search results in a formatted manner."""
    print(f"\nTop {top_k} most relevant results for: '{query}'")
    print("-" * 60)

    for i, result in enumerate(results[:top_k], 1):
        print(f"\n{i}. {result['source']} | {result['title']}")
        print(f"   URL: {result['url']}")

        if authors := result.get("authors"):
            author_list = ", ".join(filter(None, authors[:3]))
            if author_list:
                print(f"   Authors: {author_list}")

        if year := result.get("year"):
            print(f"   Year: {year}")

        if result.get("missing_abstract"):
            print("   ⚠️  Missing or incomplete abstract")


async def main(query: str = "malaria vaccine effectiveness in Africa") -> None:
    """Main application entry point."""
    print("Multi-Source Research Aggregator")
    print("=" * 40)

    # Aggregate results from all sources
    all_results = await aggregate(query)

    if not all_results:
        print("No results found.")
        return

    # Build vector store and perform semantic search
    print("\nBuilding vector store for semantic search...")
    store = build_vector_store(all_results)

    if store["index"] is None:
        print("No documents to search.")
        return

    top_hits = semantic_search(query, store, top_k=5)
    display_results(top_hits, query)


def run_with_error_handling(func, *args, **kwargs) -> None:
    """Run a function with comprehensive error handling."""
    try:
        if asyncio.iscoroutinefunction(func):
            asyncio.run(func(*args, **kwargs))
        else:
            func(*args, **kwargs)
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback

        traceback.print_exc()


def cli_main() -> None:
    """Entry point for Poetry script command."""
    print("Starting Multi-Source Research Aggregator (CLI)...")
    run_with_error_handling(main)


if __name__ == "__main__":
    print("Starting Multi-Source Research Aggregator...")
    run_with_error_handling(main)
