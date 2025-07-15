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

# Constants
APP_NAME = "MultiSourceResearchAggregator"
SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
OPENALEX_URL = "https://api.openalex.org/works"
RELIEFWEB_URL = "https://api.reliefweb.int/v1/reports"
WHO_IRIS_OAI_URL = "https://iris.who.int/oai/request"
PERSISTENCE_PATH = "vector_store.pkl"

# Model Initialization TODO: replace by crossencoder
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


# Scholar search engines fetchers
async def fetch_semantic_scholar(query: str, limit: int = 5) -> List[Dict]:
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,authors,year,url",
    }
    async with httpx.AsyncClient(follow_redirects=True) as client:
        r = await client.get(SEMANTIC_SCHOLAR_URL, params=params)
        r.raise_for_status()
        return r.json().get("data", [])


async def fetch_openalex(query: str, limit: int = 5) -> List[Dict]:
    params = {"search": query, "per-page": limit}
    async with httpx.AsyncClient(follow_redirects=True) as client:
        r = await client.get(OPENALEX_URL, params=params)
        r.raise_for_status()
        return r.json().get("results", [])


async def fetch_reliefweb(query: str, limit: int = 5) -> List[Dict]:
    payload = {
        "query": {"value": query},
        "limit": limit,
        "fields": {"include": ["title", "url", "body-html", "source", "date"]},
    }
    async with httpx.AsyncClient(follow_redirects=True) as client:
        r = await client.post(RELIEFWEB_URL, json=payload)
        r.raise_for_status()
        return r.json().get("data", [])


async def fetch_who_iris(query: str, limit: int = 5) -> List[Dict]:
    params = {
        "verb": "ListRecords",
        "metadataPrefix": "oai_dc",
        "set": "default",
        "q": query,
    }
    async with httpx.AsyncClient(follow_redirects=True) as client:
        r = await client.get(WHO_IRIS_OAI_URL, params=params)
        r.raise_for_status()
        xml_data = r.text

    root = ET.fromstring(xml_data)
    ns = {
        "oai": "http://www.openarchives.org/OAI/2.0/",
        "dc": "http://purl.org/dc/elements/1.1/",
    }
    records = []
    for record in root.findall(".//oai:record", ns):
        metadata = record.find(".//oai:metadata", ns)
        if metadata is not None:
            dc = metadata.find("dc:dc", ns)
            if dc is not None:
                title = dc.findtext("dc:title", default="", namespaces=ns)
                creator = dc.findtext("dc:creator", default="", namespaces=ns)
                date = dc.findtext("dc:date", default="", namespaces=ns)
                identifier = dc.findtext("dc:identifier", default="", namespaces=ns)
                description = dc.findtext("dc:description", default="", namespaces=ns)
                records.append(
                    {
                        "title": title,
                        "abstract": description,
                        "authors": [creator] if creator else [],
                        "year": date,
                        "url": identifier,
                        "source": "WHO IRIS",
                    }
                )
    return records[:limit]


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

    ss_data, oa_data, rw_data, who_data = await asyncio.gather(
        fetch_semantic_scholar(query),
        fetch_openalex(query),
        fetch_reliefweb(query),
        fetch_who_iris(query),
    )

    results = []
    results.extend([harmonise_result("semantic_scholar", d) for d in ss_data])
    results.extend([harmonise_result("openalex", d) for d in oa_data])
    results.extend([harmonise_result("reliefweb", d) for d in rw_data])
    results.extend([harmonise_result("who_iris", d) for d in who_data])

    print(f"Found {len(results)} total results")
    return results


# ----- Main Function -----
async def main():
    print("Multi-Source Research Aggregator")
    print("=" * 40)

    # You can modify the query here or make it interactive
    query_term = "malaria vaccine effectiveness in Africa"

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
            print("   ⚠️  Missing or incomplete abstract")
        print()


if __name__ == "__main__":
    asyncio.run(main())
