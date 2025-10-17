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

# ---------------------------------------------------------------------------
# Reuse your established connectors and index names
# ---------------------------------------------------------------------------
try:
    from common import (
        ask,                    # orchestrator: NER -> dual search (LONG/HOT) -> context -> LLM answer
        load_llm,               # cached Llama loader (llama.cpp)
    )
except Exception as e:
    raise SystemExit(
        "This script expects your existing 'query.py' to be importable from the same folder.\n"
        "Ensure 'query.py' is present next to this file.\n"
        f"Import error: {e}"
    )

import requests
from opensearchpy import OpenSearch
from opensearchpy.exceptions import TransportError
from llama_cpp import Llama


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
            "Windsurf was bought by OpenAI for how much?",
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
