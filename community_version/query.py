#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transparent, auditable dual-store BM25 retrieval (LONG = vetted, HOT = unstable) with
parallel queries, deterministic ranking, and explainable matches.

Observability controls (default OFF):
  --observability           Print query JSON and match summaries (no disk writes)
  --save-results PATH       Append compact JSONL per query (safe-by-default off)

Environment (override as needed)
--------------------------------
# LONG (vetted)
OPENSEARCH_LONG_HOST=localhost
OPENSEARCH_LONG_PORT=9200
OPENSEARCH_LONG_USER=admin
OPENSEARCH_LONG_PASS=admin
OPENSEARCH_LONG_SSL=false
LONG_INDEX_NAME=bbc

# HOT (short-term / RL)
OPENSEARCH_HOT_HOST=localhost
OPENSEARCH_HOT_PORT=9202
OPENSEARCH_HOT_USER=admin
OPENSEARCH_HOT_PASS=admin
OPENSEARCH_HOT_SSL=false
HOT_INDEX_NAME=bbc

# Search knobs
SEARCH_SIZE=8
ALPHA=0.5
PREFERENCE_TOKEN=governance-audit-v1
OS_EXPLAIN=false           # if you enable --observability, you can also enable these via env
OS_PROFILE=false

# NER & LLM
NER_URL=http://127.0.0.1:8000/ner
NER_TIMEOUT_SECS=5
MODEL_PATH=~/models/neural-chat-7b-v3-3.Q4_K_M.gguf
"""

from __future__ import annotations

import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests
from opensearchpy import OpenSearch
from opensearchpy.exceptions import TransportError
from llama_cpp import Llama


##############################################################################
# Configuration helpers
##############################################################################

def env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key, str(default)).strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)).strip())
    except Exception:
        return default


def env_str(key: str, default: str) -> str:
    return os.getenv(key, default)


# OpenSearch
OPENSEARCH_LONG_HOST = os.getenv("OPENSEARCH_LONG_HOST", "localhost")
OPENSEARCH_LONG_PORT = int(os.getenv("OPENSEARCH_LONG_PORT", "9201"))
OPENSEARCH_LONG_USER = os.getenv("OPENSEARCH_LONG_USER", "")
OPENSEARCH_LONG_PASS = os.getenv("OPENSEARCH_LONG_PASS", "")
OPENSEARCH_LONG_SSL  = os.getenv("OPENSEARCH_LONG_SSL", "false").lower() == "true"
LONG_INDEX_NAME = env_str("LONG_INDEX_NAME", "bbc")

OPENSEARCH_HOT_HOST = os.getenv("OPENSEARCH_HOT_HOST", "localhost")
OPENSEARCH_HOT_PORT = int(os.getenv("OPENSEARCH_HOT_PORT", "9202"))
OPENSEARCH_HOT_USER = os.getenv("OPENSEARCH_HOT_USER", "")
OPENSEARCH_HOT_PASS = os.getenv("OPENSEARCH_HOT_PASS", "")
OPENSEARCH_HOT_SSL  = os.getenv("OPENSEARCH_HOT_SSL", "false").lower() == "true"
HOT_INDEX_NAME = env_str("HOT_INDEX_NAME", "bbc")


##############################################################################
# NER client
##############################################################################

NER_URL = env_str("NER_URL", "http://127.0.0.1:8000/ner")
NER_TIMEOUT_SECS = float(env_str("NER_TIMEOUT_SECS", "5"))

def post_ner(text: str, timeout: float = NER_TIMEOUT_SECS) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    payload = {"text": text}
    r = requests.post(NER_URL, headers=headers, data=json.dumps(payload), timeout=timeout)
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        try:
            msg = r.json()
        except Exception:
            msg = r.text
        raise SystemExit(f"[NER] HTTP {r.status_code}: {msg}") from e
    try:
        return r.json()
    except Exception as e:
        raise SystemExit(f"[NER] Non-JSON response: {r.text[:800]}") from e


def normalize_entities(ner_result: Dict[str, Any]) -> List[str]:
    """Return normalized (lowercased, de-duped) entity strings."""
    seen, out = set(), []
    for name in ner_result.get("entities", []):
        k = name.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


##############################################################################
# OpenSearch connections (LONG and HOT)
##############################################################################

def connect_long() -> Tuple[OpenSearch, str]:
    http_auth = (OPENSEARCH_LONG_USER, OPENSEARCH_LONG_PASS) if OPENSEARCH_LONG_USER and OPENSEARCH_LONG_PASS else None
    client = OpenSearch(
        hosts=[{"host": OPENSEARCH_LONG_HOST, "port": OPENSEARCH_LONG_PORT}],
        http_auth=http_auth,
        use_ssl=OPENSEARCH_LONG_SSL,
        verify_certs=OPENSEARCH_LONG_SSL,
        ssl_assert_hostname=False if not OPENSEARCH_LONG_SSL else None,
        ssl_show_warn=OPENSEARCH_LONG_SSL,
        timeout=60,
        max_retries=3,
        retry_on_timeout=True,
    )
    return client, LONG_INDEX_NAME

def connect_hot() -> Tuple[OpenSearch, str]:
    http_auth = (OPENSEARCH_HOT_USER, OPENSEARCH_HOT_PASS) if OPENSEARCH_HOT_USER and OPENSEARCH_HOT_PASS else None
    client = OpenSearch(
        hosts=[{"host": OPENSEARCH_HOT_HOST, "port": OPENSEARCH_HOT_PORT}],
        http_auth=http_auth,
        use_ssl=OPENSEARCH_HOT_SSL,
        verify_certs=OPENSEARCH_HOT_SSL,
        ssl_assert_hostname=False if not OPENSEARCH_HOT_SSL else None,
        ssl_show_warn=OPENSEARCH_HOT_SSL,
        timeout=60,
        max_retries=3,
        retry_on_timeout=True,
    )
    return client, HOT_INDEX_NAME


##############################################################################
# Query builder (lexical-first, auditable)
##############################################################################

def build_query(question: str, entities: List[str]) -> Dict[str, Any]:
    """
    - If no entities -> dis_max over a single match on the full question.
    - If entities exist -> dis_max with two bool branches:
        (A) STRICT / AND-style:
            - terms_set on explicit_terms requiring ALL
            - match on explicit_terms_text with operator='and'
            - multi_match on content/category^0.5 with operator='and'
          boost 30.0
        (B) OR-style:
            - terms on explicit_terms (entities)
            - match on explicit_terms_text (joined)
            - multi_match on content/category^0.5
          boost 10.0
    - No fallback full-question clause when entities are present.
    """
    if not entities:
        return {
            "dis_max": {
                "tie_breaker": 0.0,
                "queries": [
                    {"match": {"content": {"query": question}}}
                ]
            }
        }

    joined = " ".join(entities)

    strict_bool = {
        "bool": {
            "should": [
                {
                    "terms_set": {
                        "explicit_terms": {
                            "terms": entities,
                            "minimum_should_match_script": {"source": "params.num_terms"}
                        }
                    }
                },
                {
                    "match": {
                        "explicit_terms_text": {
                            "query": joined,
                            "operator": "and"
                        }
                    }
                },
                {
                    "multi_match": {
                        "query": joined,
                        "fields": ["content^1.0", "category^0.5"],
                        "operator": "and"
                    }
                }
            ],
            "minimum_should_match": 1,
            "boost": 30.0
        }
    }

    or_bool = {
        "bool": {
            "should": [
                {"terms": {"explicit_terms": entities}},
                {"match": {"explicit_terms_text": joined}},
                {"multi_match": {"query": joined, "fields": ["content^1.0", "category^0.5"]}},
            ],
            "minimum_should_match": 1,
            "boost": 10.0
        }
    }

    return {
        "dis_max": {
            "tie_breaker": 0.0,
            "queries": [strict_bool, or_bool]
        }
    }



##############################################################################
# LLM loader and answering
##############################################################################

@lru_cache(maxsize=1)
def load_llm() -> Llama:
    return Llama(
        model_path=str(Path(env_str("MODEL_PATH",
                                    str(Path.home() / "models" / "neural-chat-7b-v3-3.Q4_K_M.gguf"))).expanduser()),
        n_ctx=32768,
        n_threads=max(4, os.cpu_count() or 8),
        temperature=0.2,
        top_p=0.95,
        repeat_penalty=1.2,
        chat_format="chatml",
        verbose=False,
    )


def generate_answer(llm: Llama, question: str, context: str) -> str:
    if not context.strip():
        return "No relevant information found."
    sys_msg = "Answer using ONLY the provided context."
    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"If context lacks the answer, reply exactly with: No relevant information found."
    )

    # print("\n\n=== Prompt to LLM ===")
    # print(user_prompt)
    # print("==========\n\n")

    resp = llm.create_chat_completion(
        messages=[{"role": "system", "content": sys_msg},
                  {"role": "user", "content": user_prompt}],
        temperature=0.2, top_p=0.95, max_tokens=1024,
    )
    return resp["choices"][0]["message"]["content"].strip()


##############################################################################
# Search + ranking utilities
##############################################################################

SEARCH_SIZE = env_int("SEARCH_SIZE", 8)
ALPHA       = float(env_str("ALPHA", "0.5"))
PREFERENCE  = env_str("PREFERENCE_TOKEN", "governance-audit-v1")
DO_EXPLAIN  = env_bool("OS_EXPLAIN", False)
DO_PROFILE  = env_bool("OS_PROFILE", False)

HIGHLIGHT = {
    "fields": {"content": {}},
    "fragment_size": 160,
    "number_of_fragments": 2,
    "pre_tags": ["<em>"],
    "post_tags": ["</em>"]
}

SOURCE_FIELDS = [
    "filepath", "category", "content",
    "explicit_terms", "explicit_terms_text",
    "ingested_at_ms", "doc_version"
]

def search_one(
    label: str,
    client: OpenSearch,
    index_name: str,
    query: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute a single search with observability toggles."""
    body = {
        "query": query,
        "_source": SOURCE_FIELDS,
        "highlight": HIGHLIGHT,
        "size": SEARCH_SIZE,
        "explain": DO_EXPLAIN,
        "profile": DO_PROFILE,
        "track_total_hits": True,
    }
    try:
        res = client.search(index=index_name, body=body, preference=PREFERENCE, request_timeout=10)
    except TransportError as e:
        # Soft-fail: return empty result + diagnostic
        return {
            "_store_label": label,
            "_index_used": index_name,
            "_query": query,
            "_error": f"{e.__class__.__name__}: {getattr(e, 'error', str(e))}",
            "hits": {"total": {"value": 0, "relation": "eq"}, "hits": []}
        }
    res["_store_label"] = label
    res["_index_used"]  = index_name
    res["_query"]       = query
    return res


def rank_hits(res: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Threshold per the spec: keep any hit whose score >= ALPHA * top1.
    We do this per-store to avoid cross-cluster score mixing.
    """
    hits = res.get("hits", {}).get("hits", []) or []
    if not hits:
        return []
    top1 = hits[0]["_score"]
    keep = [h for h in hits if h["_score"] >= ALPHA * top1]
    # Annotate with store for downstream prints
    for h in keep:
        h["_store_label"] = res.get("_store_label", "?")
        h["_index_used"]  = res.get("_index_used", "?")
    return keep


def combine_hits(hits_a: List[Dict[str, Any]], hits_b: List[Dict[str, Any]], max_total: int = 10) -> List[Dict[str, Any]]:
    """
    Combine two per-store lists without pretending cross-store scores are comparable.
    Policy: interleave A and B (stable) while preserving each list's order.
    """
    combined: List[Dict[str, Any]] = []
    ia = ib = 0
    while len(combined) < max_total and (ia < len(hits_a) or ib < len(hits_b)):
        if ia < len(hits_a):
            combined.append(hits_a[ia]); ia += 1
        if len(combined) >= max_total:
            break
        if ib < len(hits_b):
            combined.append(hits_b[ib]); ib += 1
    return combined


def render_observability_summary(res: Dict[str, Any]) -> str:
    err = res.get("_error")
    store = res.get("_store_label", "?")
    idx   = res.get("_index_used", "?")
    total = res.get("hits", {}).get("total", {}).get("value", 0)
    if err:
        return f"[SUMMARY] STORE={store} INDEX={idx} ERROR: {err}"
    return f"[SUMMARY] STORE={store} INDEX={idx} TOTAL={total}"


def render_matches(hits: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("\n================= MATCH EXPLANATIONS =================")
    if not hits:
        lines.append("(no documents kept from either store)")
        lines.append("======================================================\n")
        return "\n".join(lines)
    for i, h in enumerate(hits, 1):
        store = h.get("_store_label", "?")
        idx   = h.get("_index_used", "?")
        fp    = h.get("_source", {}).get("filepath", "<unknown>")
        score = h.get("_score")
        lines.append(f"\n[{i}] STORE={store} INDEX={idx} SCORE={score:.4f}")
        lines.append(f"     DOC={fp}")
        if "highlight" in h and "content" in h["highlight"]:
            frag = h["highlight"]["content"][0]
            lines.append(f"     highlight: {frag}")
        lines.append("")
    lines.append("======================================================\n")
    return "\n".join(lines)


def build_context(hits: List[Dict[str, Any]], max_chars_per_doc: int = 1500) -> str:
    if not hits:
        return ""
    out = []
    for h in hits:
        src = h.get("_source", {})
        fp  = src.get("filepath", "<unknown>")
        store = h.get("_store_label", "?")
        content = (src.get("content") or "").strip()
        snippet = content[:max_chars_per_doc]
        out.append(f"---\nStore: {store}\nDoc: {fp}\n{snippet}\n")
    return "\n".join(out)


def save_results(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


##############################################################################
# Orchestrator
##############################################################################

def ask(llm: Llama, question: str, *, observability: bool = False, save_path: str | None = None) -> Tuple[str, List[Dict[str, Any]]]:
    # 1) NER
    ner = post_ner(question)
    entities = normalize_entities(ner)

    # 2) Build query to match <original>
    query = build_query(question, entities)

    if observability:
        if entities:
            print(f"[NER] entities: {entities}")
            print("\n[QUERY] dis_max (entity path):")
        else:
            print("[NER] No entities detected; using full-question match only.")
            print("\n[QUERY] dis_max (no-entity path):")
        print(json.dumps(query, indent=2))

    # 3) Connect stores
    long_client, long_index = connect_long()
    hot_client,  hot_index  = connect_hot()

    # 4) Execute in parallel
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_long = ex.submit(search_one, "LONG", long_client, long_index, query)
        fut_hot  = ex.submit(search_one, "HOT",  hot_client,  hot_index,  query)
        res_long = fut_long.result()
        res_hot  = fut_hot.result()

    # 5) Rank per store (tolerate 0 hits)
    keep_long = rank_hits(res_long)
    keep_hot  = rank_hits(res_hot)
    combined  = combine_hits(keep_long, keep_hot, max_total=10)

    # 6) Optional observability prints
    if observability:
        print(render_observability_summary(res_long))
        print(render_observability_summary(res_hot))
        print(f"\n[RESULTS] LONG kept={len(keep_long)} of {len(res_long.get('hits',{}).get('hits',[]))}")
        print(f"[RESULTS] HOT  kept={len(keep_hot)} of {len(res_hot.get('hits',{}).get('hits',[]))}")
        print(render_matches(combined))

    # 7) Optional save (compact JSONL)
    if save_path:
        payload = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "question": question,
            "entities": entities,
            "alpha": ALPHA,
            "size": SEARCH_SIZE,
            "preference": PREFERENCE,
            "long": {
                "index": res_long.get("_index_used"),
                "total": res_long.get("hits", {}).get("total", {}).get("value", 0),
                "error": res_long.get("_error"),
                "kept_filepaths": [h.get("_source", {}).get("filepath") for h in keep_long],
            },
            "hot": {
                "index": res_hot.get("_index_used"),
                "total": res_hot.get("hits", {}).get("total", {}).get("value", 0),
                "error": res_hot.get("_error"),
                "kept_filepaths": [h.get("_source", {}).get("filepath") for h in keep_hot],
            },
            "combined_filepaths": [h.get("_source", {}).get("filepath") for h in combined],
            # do NOT dump full content; keep this audit-friendly and light
        }
        save_results(save_path, payload)

    # 8) Build context and answer (short-circuit if none)
    context_block = build_context(combined, max_chars_per_doc=1500)
    if not context_block.strip():
        return "No relevant information found.", []
    
    answer = generate_answer(llm, question, context_block)
    return answer, combined


##############################################################################
# CLI
##############################################################################

def main():
    parser = argparse.ArgumentParser(description="Dual-store BM25 retrieval with parallel LONG/HOT searches (governance-first).")
    parser.add_argument("--question", help="User question to answer.")
    parser.add_argument("--observability", action="store_true", default=True,
                        help="Print query JSON and match summaries (default OFF).")
    parser.add_argument("--save-results", type=str, default=None,
                        help="Append compact JSONL result records to this path (default OFF).")
    args = parser.parse_args()

    llm = load_llm()

    questions: List[str]
    if args.question:
        questions = [args.question]
    else:
        # Demo questions (safe to replace for your corpus)
        questions = [
            "Tell me about the connection between Ernie Wise and Vodafone.",
            "Tell me something about Ernie Wise."
        ]

    for q in questions:
        print("\n" + "=" * 88)
        print(f"QUESTION: {q}")
        print("=" * 88)
        t0 = time.time()
        answer, hits = ask(llm, q, observability=args.observability, save_path=args.save_results)
        dt = time.time() - t0

        print("\n" + "=" * 88)
        print(f"ANSWER: {answer}")
        print("\n" + "=" * 88)
        print(f"\nQuery time: {dt:.2f}s   (stores queried in parallel)")
        print(f"Docs provided to LLM: {len(hits)}\n")


if __name__ == "__main__":
    main()
