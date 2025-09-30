#!/usr/bin/env python3
"""
Ingest BBC articles into OpenSearch with explicit multi-term recall.
"""
import os, glob, time
from pathlib import Path
from typing import List, Tuple
import requests
import json

from opensearchpy import OpenSearch

# ----------------------------
# Config
# ----------------------------
DATA_DIR   = os.getenv("DATA_DIR", "bbc")
INDEX_NAME = os.getenv("INDEX_NAME", "bbc")

OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9201"))
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER", "")
OPENSEARCH_PASS = os.getenv("OPENSEARCH_PASS", "")
OPENSEARCH_SSL  = os.getenv("OPENSEARCH_SSL", "false").lower() == "true"

# NER server URL
DEFAULT_URL = "http://127.0.0.1:8000/ner"

# ----------------------------
# NER
# ----------------------------
def post_ner(text: str, timeout: float = 5.0) -> dict:
    headers = {"Content-Type": "application/json"}
    payload = {
        "text": text,
    }
    r = requests.post(DEFAULT_URL, headers=headers, data=json.dumps(payload), timeout=timeout)
    # Raise for non-2xx to surface useful diagnostics
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        # try to show server-provided JSON error if present
        try:
            msg = r.json()
        except Exception:
            msg = r.text
        raise SystemExit(f"HTTP {r.status_code} from server: {msg}") from e
    try:
        return r.json()
    except Exception as e:
        raise SystemExit(f"Server returned non-JSON response: {r.text[:2000]}") from e


def extract_normalized_entities(text_file: Path) -> List[str]:
    text = text_file.read_text(encoding="utf-8", errors="ignore")
    ner_result = post_ner(text)

    seen, normalized = set(), []
    for name in ner_result.get("entities", []):
        k = name.lower()
        if k not in seen:
            seen.add(k)
            normalized.append(k)

    return normalized

# ----------------------------
# OpenSearch connection
# ----------------------------
def connect_long() -> OpenSearch:
    http_auth = (OPENSEARCH_USER, OPENSEARCH_PASS) if OPENSEARCH_USER and OPENSEARCH_PASS else None
    client = OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        http_auth=http_auth,
        use_ssl=OPENSEARCH_SSL,
        verify_certs=OPENSEARCH_SSL,
        ssl_assert_hostname=False if OPENSEARCH_SSL else None,
        ssl_show_warn=OPENSEARCH_SSL,
        timeout=60,
        max_retries=3,
        retry_on_timeout=True,
    )
    return client

# ----------------------------
# Index + mapping
# ----------------------------
def ensure_index(client: OpenSearch, index_name: str) -> None:
    if client.indices.exists(index=index_name):
        return
    body = {
        "settings": {
            "analysis": {
                "normalizer": {
                    "lowercase_normalizer": {
                        "type": "custom",
                        "char_filter": [],
                        "filter": ["lowercase"]
                    }
                }
            },
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "content": {"type": "text"},
                "category": {"type": "keyword", "normalizer": "lowercase_normalizer"},
                "filepath": {"type": "keyword"},
                "explicit_terms": {"type": "keyword", "normalizer": "lowercase_normalizer"},
                "explicit_terms_text": {"type": "text"},
                "ingested_at_ms": {"type": "date", "format": "epoch_millis"},
                "doc_version": {"type": "long"}
            }
        },
    }
    client.indices.create(index=index_name, body=body)

# ----------------------------
# Ingest
# ----------------------------
def ingest_bbc(client: OpenSearch, data_dir: str, index_name: str) -> None:
    ensure_index(client, index_name)

    files = sorted(glob.glob(os.path.join(data_dir, "*", "*.txt")))
    if not files:
        print(f"[INFO] No files found under {data_dir}/*/*.txt"); return

    now_ms = int(time.time() * 1000)

    for fp in files:
        p = Path(fp)
        category = p.parent.name
        text = p.read_text(encoding="utf-8", errors="ignore")

        explicit_terms = extract_normalized_entities(p)

        # Stable document id based on relative filepath
        doc_id = p.as_posix()

        doc = {
            "content": text,
            "category": category,
            "filepath": doc_id,
            "explicit_terms": explicit_terms,
            "explicit_terms_text": " ".join(explicit_terms) if explicit_terms else "",
            "ingested_at_ms": now_ms,
            "doc_version": now_ms,  # simple monotonic version; update on re-ingest
        }

        print(f"[INGEST] {doc_id} ({len(text)} chars):\nexplicit terms: {explicit_terms}\n")
        client.index(index=index_name, id=doc_id, body=doc, refresh=False)

    client.indices.refresh(index=index_name)
    print(f"[OK] Ingest complete. Indexed {len(files)} docs into '{index_name}'")

def main():
    client = connect_long()
    ingest_bbc(client, DATA_DIR, INDEX_NAME)

if __name__ == "__main__":
    main()
