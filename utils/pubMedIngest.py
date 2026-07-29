import argparse
import requests
import xml.etree.ElementTree as ET
import sys
import os
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from rag.pipeline import RAGPipeline


ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def search_pmids(query: str, max_results: int = 50) -> list[str]:
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
    }
    r = requests.get(ESEARCH_URL, params=params, timeout=10)
    r.raise_for_status()
    return r.json()["esearchresult"]["idlist"]
 
 
def fetch_abstracts(pmids: list[str]) -> list[dict]:
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    }
    r = requests.get(EFETCH_URL, params=params, timeout=30)
    r.raise_for_status()
 
    root = ET.fromstring(r.text)
    records = []
    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        abstract_el = article.find(".//AbstractText")
        title_el = article.find(".//ArticleTitle")
 
        if abstract_el is None or not abstract_el.text:
            continue
 
        records.append({
            "text": abstract_el.text.strip(),
            "source": f"PubMed:{pmid_el.text}" if pmid_el is not None else "PubMed",
            "title": title_el.text if title_el is not None else "",
        })
    return records
 
 
def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Sliding window chunker — keeps context across chunk boundaries."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="PubMed search query")
    parser.add_argument("--max", type=int, default=50, help="Max abstracts to fetch")
    args = parser.parse_args()
 
    print(f"[PubMed] Searching: {args.query!r}")
    pmids = search_pmids(args.query, args.max)
    print(f"[PubMed] Found {len(pmids)} articles")
 
    abstracts = fetch_abstracts(pmids)
    print(f"[PubMed] Fetched {len(abstracts)} abstracts with text")
 
    # Chunk each abstract and flatten
    all_chunks = []
    for rec in abstracts:
        for chunk in chunk_text(rec["text"]):
            all_chunks.append({"text": chunk, "source": rec["source"]})
 
    print(f"[PubMed] Ingesting {len(all_chunks)} chunks into vector store...")
    rag = RAGPipeline()
    n = rag.ingest(all_chunks)
    print(f"[PubMed] Done. {n} chunks indexed.")
 
 
if __name__ == "__main__":
    main()
 