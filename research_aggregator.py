import asyncio
import httpx
import xml.etree.ElementTree as ET
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util
import torch
import faiss
import numpy as np
import pickle
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Constants
APP_NAME = "MultiSourceResearchAggregator"
SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
OPENALEX_URL = "https://api.openalex.org/works"
RELIEFWEB_URL = "https://api.reliefweb.int/v1/reports"
WHO_IRIS_OAI_URL = "https://iris.who.int/oai/request"
PERSISTENCE_PATH = "vector_store.pkl"

# Model Initialization TODO: replace by crossencoder with task-specific prompts
model = SentenceTransformer("all-MiniLM-L6-v2")


# Utility functions to reconstruct abstracts and clean HTML (OpenAlex inverted index abstracts)
def reconstruct_abstract(inverted_index: Dict[str, List[int]]) -> str:
    if not inverted_index:
        return None
    position_map = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            position_map[pos] = word
    return " ".join(position_map[i] for i in sorted(position_map))


def clean_html(raw_html: str) -> str:
    if not raw_html:
        return None
    return BeautifulSoup(raw_html, "html.parser").get_text(separator=" ").strip()


def flag_missing_abstract(doc: Dict) -> Dict:
    if not doc.get("abstract"):
        doc["missing_abstract"] = True
    return doc


def debug_raw_data(data, source_name, limit=1):
    """Debug function to show raw API response structure"""
    print(f"\n--- DEBUG: {source_name} raw data structure ---")
    if not data:
        print("No data received")
        return

    for i, item in enumerate(data[:limit]):
        print(f"Item {i+1} keys: {list(item.keys())}")

        # Show abstract-related fields
        if source_name == "Semantic Scholar":
            print(
                f"  abstract: {item.get('abstract', 'MISSING')[:100] if item.get('abstract') else 'MISSING'}..."
            )
        elif source_name == "OpenAlex":
            abstract_idx = item.get("abstract_inverted_index")
            print(
                f"  abstract_inverted_index: {'Present' if abstract_idx else 'MISSING'}"
            )
            if abstract_idx:
                reconstructed = reconstruct_abstract(abstract_idx)
                print(
                    f"  reconstructed preview: {reconstructed[:100] if reconstructed else 'FAILED'}..."
                )
        elif source_name == "ReliefWeb":
            fields = item.get("fields", {})
            body_html = fields.get("body-html")
            print(f"  body-html: {'Present' if body_html else 'MISSING'}")
            if body_html:
                cleaned = clean_html(body_html)
                print(f"  cleaned preview: {cleaned[:100] if cleaned else 'FAILED'}...")
        elif source_name == "WHO IRIS":
            print(
                f"  abstract (description): {item.get('abstract', 'MISSING')[:100] if item.get('abstract') else 'MISSING'}..."
            )
    print("--- END DEBUG ---\n")


# Scholar search engines fetchers
async def fetch_semantic_scholar(query: str, limit: int = 5) -> List[Dict]:
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,authors,year,url",
    }

    headers = {}
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    # Retry logic for rate limiting
    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                r = await client.get(
                    SEMANTIC_SCHOLAR_URL, params=params, headers=headers
                )
                r.raise_for_status()
                return r.json().get("data", [])
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:  # Rate limit
                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # Exponential backoff
                    print(
                        f"Rate limited by Semantic Scholar. Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    print(
                        "Semantic Scholar rate limit exceeded. Returning empty results."
                    )
                    return []
            else:
                print(f"Semantic Scholar API error: {e}")
                return []
        except Exception as e:
            print(f"Semantic Scholar fetch error: {e}")
            return []

    return []


async def fetch_openalex(query: str, limit: int = 5) -> List[Dict]:
    params = {"search": query, "per-page": limit}
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            r = await client.get(OPENALEX_URL, params=params)
            r.raise_for_status()
            return r.json().get("results", [])
    except Exception as e:
        print(f"OpenAlex fetch error: {e}")
        return []


async def fetch_reliefweb(query: str, limit: int = 5) -> List[Dict]:
    payload = {
        "query": {"value": query},
        "limit": limit,
        "fields": {"include": ["title", "url", "body-html", "source", "date"]},
    }
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            r = await client.post(RELIEFWEB_URL, json=payload)
            r.raise_for_status()
            return r.json().get("data", [])
    except Exception as e:
        print(f"ReliefWeb fetch error: {e}")
        return []


async def fetch_who_iris(query: str, limit: int = 5) -> List[Dict]:
    """
    WHO IRIS OAI-PMH endpoint doesn't support search parameters.
    We fetch recent records and filter them locally by title/description/subject.
    """
    params = {
        "verb": "ListRecords",
        "metadataPrefix": "oai_dc",
    }

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
            r = await client.get(WHO_IRIS_OAI_URL, params=params)
            r.raise_for_status()
            xml_data = r.text

        root = ET.fromstring(xml_data)

        # Check for OAI-PMH errors
        error_elements = root.findall(".//{http://www.openarchives.org/OAI/2.0/}error")
        if error_elements:
            return []

        ns = {
            "oai": "http://www.openarchives.org/OAI/2.0/",
            "dc": "http://purl.org/dc/elements/1.1/",
        }

        # Get all records and filter locally
        all_records = root.findall(".//oai:record", ns)

        # Filter records by query terms (case-insensitive)
        query_terms = [
            term.lower() for term in query.split() if len(term) > 2
        ]  # Skip short words
        matched_records = []

        for record in all_records:
            metadata = record.find(".//oai:metadata", ns)
            if metadata is not None:
                dc = metadata.find("dc:dc", ns)
                if dc is not None:
                    title = dc.findtext("dc:title", default="", namespaces=ns)
                    description = dc.findtext(
                        "dc:description", default="", namespaces=ns
                    )
                    subject = dc.findtext("dc:subject", default="", namespaces=ns)

                    # Combine title, description, and subject for matching
                    searchable_text = f"{title} {description} {subject}".lower()

                    # Check if any query terms match (need at least one meaningful term)
                    if any(
                        term in searchable_text for term in query_terms if len(term) > 3
                    ):
                        creator = dc.findtext("dc:creator", default="", namespaces=ns)
                        date = dc.findtext("dc:date", default="", namespaces=ns)
                        identifier = dc.findtext(
                            "dc:identifier", default="", namespaces=ns
                        )

                        matched_records.append(
                            {
                                "title": title,
                                "abstract": description,
                                "authors": [creator] if creator else [],
                                "year": date,
                                "url": identifier,
                                "source": "WHO IRIS",
                            }
                        )

                        if len(matched_records) >= limit:
                            break

        return matched_records

    except Exception as e:
        print(f"WHO IRIS fetch error: {e}")
        return []


# Harmonise results from different sources
def harmonise_result(source: str, item: Dict) -> Dict:
    if source == "semantic_scholar":
        doc = {
            "title": item.get("title"),
            "abstract": item.get("abstract"),
            "authors": [a.get("name") for a in item.get("authors", [])],
            "year": item.get("year"),
            "url": item.get("url"),
            "source": "Semantic Scholar",
        }
    elif source == "openalex":
        doc = {
            "title": item.get("title"),
            "abstract": reconstruct_abstract(item.get("abstract_inverted_index")),
            "authors": [
                a.get("author", {}).get("display_name")
                for a in item.get("authorships", [])
            ],
            "year": item.get("publication_year"),
            "url": item.get("id"),
            "source": "OpenAlex",
        }
    elif source == "reliefweb":
        fields = item.get("fields", {})
        # Handle source field which can be a list or dict
        source_field = fields.get("source", [])
        if isinstance(source_field, list) and source_field:
            author_name = source_field[0].get("name", "") if source_field[0] else ""
        elif isinstance(source_field, dict):
            author_name = source_field.get("name", "")
        else:
            author_name = ""

        doc = {
            "title": fields.get("title"),
            "abstract": clean_html(fields.get("body-html")),
            "authors": [author_name] if author_name else [],
            "year": fields.get("date"),
            "url": fields.get("url"),
            "source": "ReliefWeb",
        }
    elif source == "who_iris":
        doc = item  # Already harmonised
    else:
        doc = {}

    return flag_missing_abstract(doc)


# ----- Semantic Search Engine -----
def build_vector_store(documents: List[Dict]) -> Dict:
    if not documents:
        return {"index": None, "documents": []}  # No documents to index

    # Build a vector store from the documents
    corpus = [
        doc.get("title", "") + ". " + (doc.get("abstract") or "") for doc in documents
    ]
    embeddings = model.encode(corpus, convert_to_tensor=False)
    dim = embeddings[0].shape[0]
    # Ensure embeddings are float32 for FAISS
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    with open(PERSISTENCE_PATH, "wb") as f:
        pickle.dump({"index": index, "documents": documents}, f)
    return {"index": index, "documents": documents}


def semantic_search(query: str, store: Dict, top_k: int = 5) -> List[Dict]:
    query_vec = model.encode([query])[0].astype("float32")
    D, I = store["index"].search(np.array([query_vec]), top_k)
    return [store["documents"][i] for i in I[0]]


# ----- Master Aggregator -----
async def aggregate(query: str) -> List[Dict]:
    print(f"Searching for: {query}")
    print("Fetching from multiple sources...")

    # Fetch data with status updates
    ss_data, oa_data, rw_data, who_data = await asyncio.gather(
        fetch_semantic_scholar(query),
        fetch_openalex(query),
        fetch_reliefweb(query),
        fetch_who_iris(query),
    )

    # Report on what we got from each source
    print(
        f"Results: Semantic Scholar({len(ss_data)}), OpenAlex({len(oa_data)}), ReliefWeb({len(rw_data)}), WHO IRIS({len(who_data)})"
    )

    # Debug raw data structures (optional - can be commented out)
    # debug_raw_data(ss_data, "Semantic Scholar")
    # debug_raw_data(oa_data, "OpenAlex")
    # debug_raw_data(rw_data, "ReliefWeb")
    # debug_raw_data(who_data, "WHO IRIS")

    results = []

    # Process and check abstracts for each source
    ss_results = [harmonise_result("semantic_scholar", d) for d in ss_data]
    oa_results = [harmonise_result("openalex", d) for d in oa_data]
    rw_results = [harmonise_result("reliefweb", d) for d in rw_data]
    who_results = [harmonise_result("who_iris", d) for d in who_data]

    # Debug abstract availability
    def check_abstracts(source_results, source_name):
        with_abstracts = sum(
            1 for r in source_results if r.get("abstract") and r.get("abstract").strip()
        )
        without_abstracts = len(source_results) - with_abstracts
        print(
            f"  {source_name}: {with_abstracts} with abstracts, {without_abstracts} without"
        )

        # Show first few abstracts for debugging
        for i, result in enumerate(source_results[:2]):  # Show first 2 results
            abstract = result.get("abstract", "")
            if abstract and abstract.strip():
                preview = abstract[:100] + "..." if len(abstract) > 100 else abstract
                print(f'    Sample {i+1}: "{preview}"')
            else:
                print(f"    Sample {i+1}: [NO ABSTRACT]")

    print("\nAbstract availability by source:")
    check_abstracts(ss_results, "Semantic Scholar")
    check_abstracts(oa_results, "OpenAlex")
    check_abstracts(rw_results, "ReliefWeb")
    check_abstracts(who_results, "WHO IRIS")

    results.extend(ss_results)
    results.extend(oa_results)
    results.extend(rw_results)
    results.extend(who_results)

    total_with_abstracts = sum(
        1 for r in results if r.get("abstract") and r.get("abstract").strip()
    )
    print(f"\nOverall: {total_with_abstracts}/{len(results)} results have abstracts")
    return results


# ----- Main Function -----
async def main():
    print("Multi-Source Research Aggregator")
    print("=" * 40)

    # You can modify the query here or make it interactive
    # Example query 1: wastewater environmental surveillance Vibrio cholerae outbreaks
    # Example query 2: malaria vaccine effectiveness in Africa
    # Example query 3: climate change impact on vector-borne diseases | vector-borne diseases | vector-borne diseases Europe
    # Example query 4: COVID-19 impact on mental health
    # Example query 5: white paint corrugated iron roofs effects
    query_term = "malaria vaccine effectiveness"

    all_results = await aggregate(query_term)

    if not all_results:
        print("No results found.")
        return

    print("\nBuilding vector store for semantic search...")
    store = build_vector_store(all_results)

    print(f"\nTop 5 most relevant results for: '{query_term}'")
    print("-" * 60)

    top_hits = semantic_search(query_term, store, top_k=5)

    for i, result in enumerate(top_hits, 1):
        print(f"\n{i}. {result['source']} | {result['title']}")
        print(f"   URL: {result['url']}")
        if result.get("authors"):
            authors = result["authors"][:3]  # Show first 3 authors
            print(f"   Authors: {', '.join(filter(None, authors))}")
        if result.get("year"):
            print(f"   Year: {result['year']}")
        if result.get("missing_abstract"):
            print("Missing or incomplete abstract")
        print()


if __name__ == "__main__":
    asyncio.run(main())
