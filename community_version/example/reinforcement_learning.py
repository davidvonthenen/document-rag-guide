#!/usr/bin/env python3
"""
Reinforcement "Hot Memory" Adder + 5-Question Sanity Test (Dual-Store OpenSearch RAG)

What this does:
1) Presents 5 **hardcoded** candidate facts for promotion into the HOT store.
2) For each fact, you can [A]dd or [S]kip. If added:
   - Runs NER to derive `explicit_terms` (normalized, de-duped).
   - Indexes the fact into the HOT OpenSearch index with governance fields.
3) After processing the 5 facts, runs **5 hardcoded test questions** via the
   existing `ask(...)` pipeline from your `query.py` (searches LONG + HOT).

This reuses your existing code paths:
- Imports `ask`, `post_ner`, `normalize_entities`, `connect_hot`,
  `HOT_INDEX_NAME`, and `load_llm` directly from your `query.py`.
- Uses identical field names (`explicit_terms`, `explicit_terms_text`, etc.).
- Uses OpenSearch Python SDK for all indexing and index admin.

Run:
    ./reinforce_hot_hardcoded.py --auto-yes
    ./reinforce_hot_hardcoded.py                      # prompts per fact

Env it honors (same as your existing scripts):
    OPENSEARCH_HOT_HOST, OPENSEARCH_HOT_PORT, OPENSEARCH_HOT_USER, OPENSEARCH_HOT_PASS, OPENSEARCH_HOT_SSL
    HOT_INDEX_NAME (defaults to 'bbc' unless set)
    NER_URL (defaults to http://127.0.0.1:8000/ner)
    MODEL_PATH (GGUF path used by llama.cpp in query.py->load_llm())
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Reuse from your existing query.py (must be importable from same directory)
# ---------------------------------------------------------------------------
try:
    from common import (
        ask,                    # orchestrator: NER -> dual search (LONG/HOT) -> context -> LLM answer
        post_ner,               # NER POST client
        normalize_entities,     # lowercasing + de-dup of NER entities
        connect_hot,            # OpenSearch HOT connection using OPENSEARCH_HOT_* envs
        HOT_INDEX_NAME,         # default HOT index name (env override honored inside query.py)
        load_llm,               # cached Llama loader (llama.cpp)
    )
except Exception as e:
    raise SystemExit(
        "This script expects your existing 'query.py' to be importable from the same folder.\n"
        "Ensure 'query.py' is present next to this file.\n"
        f"Import error: {e}"
    )

from opensearchpy import OpenSearch


# ---------------------------------------------------------------------------
# Hardcoded facts (5) and questions (5)
# Keep these tight and entity-rich so both BM25 and NER paths bite.
# ---------------------------------------------------------------------------

TECH_FACTS = [
    """
    OpenAI has agreed to buy artificial intelligence-assisted coding tool Windsurf for about $3 billion, Bloomberg News reported on Monday, citing people familiar with the matter.
    The deal has not yet closed, the report added.

    OpenAI declined to comment, while Windsurf did not immediately respond to Reuters' requests for comment.

    Windsurf, formerly known as Codeium, had recently been in talks with investors, including General Catalyst and Kleiner Perkins, to raise funding at a $3 billion valuation, according to Bloomberg News.
    
    It was valued at $1.25 billion last August following a $150 million funding round led by venture capital firm General Catalyst. Other investors in the company include Kleiner Perkins and Greenoaks.
    
    The deal, which would be OpenAI's largest acquisition to date, would complement ChatGPT's coding capabilities. The company has been rolling out improvements in coding with the release of each of its newer models, but the competition is heating up.
    
    OpenAI has made several purchases in recent years to boost different segments of its AI products. It bought search and database analytics startup Rockset in a nine-figure stock deal last year, to provide better infrastructure for its enterprise products.
    
    OpenAI's weekly active users surged past 400 million in February, jumping sharply from the 300 million weekly active users in December.
    """,
    """
    Will the Apple Vision Pro be discontinued? It's certainly starting to look that way. In the last couple of months, numerous reports have emerged suggesting that Apple is either slowing down or completely halting production of its flagship headset.

    So, what does that mean for Apple's future in the extended reality market?

    Apple has had a rough time with its Vision Pro headset. Despite incredibly hype leading up to the initial release, and the fact that preorders for the device sold out almost instantly, demand for headset has consistently dropped over the last year.

    In fact, sales have diminished to the point that rumors have been coming thick and fast. For a while now, industry analysts and tech enthusiasts believe Apple might give up on its XR journey entirely and return its focus to other types of tech (like smartphones).

    However, while Apple has failed to achieve its sales targets with the Vision Pro, I don't think they will abandon the XR market entirely. It seems more likely that Apple will view the initial Vision Pro as an experiment, using it to pave the way to new, more popular devices.

    Here's what we know about Apple's XR journey right now.
    """,
    """
    OpenAI sees itself paying a lower share of revenue to its investor and close partner Microsoft by 2030 than it currently does, The Information reported, citing financial documents.

    The news comes after OpenAI this week changed tack on a major restructuring plan to pursue a new plan that would see its for-profit arm becoming a public benefit corporation (PBC) but continue to be controlled by its nonprofit division.

    OpenAI currently has an agreement to share 20% of its top line with Microsoft, but the AI company has told investors it expects to share 10% of revenue with its business partners, including Microsoft, by the end of this decade, The Information reported.

    Microsoft has invested tens of billions in OpenAI, and the two companies currently have a contract until 2030 that includes revenue sharing from both sides. The deal also gives Microsoft rights to OpenAI IP within its AI products, as well as exclusivity on OpenAI's APIs on Azure.

    Microsoft has not yet approved OpenAI's proposed corporate structure, Bloomberg reported on Monday, as the bigger tech company reportedly wants to ensure the new structure protects its multi-billion-dollar investment.

    OpenAI and Microsoft did not immediately return requests for comment.
    """,
    """
    Perplexity, the developer of an AI-powered search engine, is raising a $50 million seed and pre-seed investment fund, CNBC reported. Although the majority of the capital is coming from limited partners, Perplexity is using some of the capital it raised for the company's growth to anchor the fund. Perplexity reportedly raised $500 million at a $9 billion valuation in December.

    Perplexity's fund is managed by general partners Kelly Graziadei and Joanna Lee Shevelenko, who in 2018 co-founded an early-stage venture firm, F7 Ventures, according to PitchBook data. F7 has invested in startups like women's health company Midi. It's not clear if Graziadei and Shevelenko will continue to run F7 or if they will focus all their energies on Perplexity's venture fund.

    OpenAI also manages an investment fund known as the OpenAI Startup Fund. However, unlike Perplexity, OpenAI claims it does not use its own capital for these investments.
    """,
    """
    DeepSeek-R2 is the upcoming AI model from Chinese startup DeepSeek, promising major advancements in multilingual reasoning, code generation, and multimodal capabilities. Scheduled for early 2025, DeepSeek-R2 combines innovative training techniques with efficient resource usage, positioning itself as a serious global competitor to Silicon Valley's top AI technologies.

    In the rapidly evolving landscape of artificial intelligence, a new contender is emerging from China that promises to reshape global AI dynamics. DeepSeek, a relatively young AI startup, is making waves with its forthcoming DeepSeek-R2 model—a bold step in China's ambition to lead the global AI race.

    As Western tech giants like OpenAI, Anthropic, and Google dominate headlines, DeepSeek's R2 model represents a significant milestone in AI development from the East. With its unique approach to training, multilingual capabilities, and resource efficiency, DeepSeek-R2 isn't just another language model—it's potentially a game-changer for how we think about AI development globally.

    What is DeepSeek-R2?
    DeepSeek-R2 is a next-generation large language model that builds upon the foundation laid by DeepSeek-R1. According to reports from Reuters, DeepSeek may be accelerating its launch timeline, potentially bringing this advanced AI system to market earlier than the original May 2025 target.

    What sets DeepSeek-R2 apart is not just its improved performance metrics but its underlying architecture and training methodology. While R1 established DeepSeek as a serious competitor with strong multilingual and coding capabilities, R2 aims to push these boundaries significantly further while introducing new capabilities that could challenge the dominance of models like GPT-4 and Claude.

    DeepSeek-R2 represents China's growing confidence and technical capability in developing frontier AI technologies. The model has been designed from the ground up to be more efficient with computational resources—a critical advantage in the resource-intensive field of large language model development.
    """
]

# Simple QA prompts that exercise each fact once.
TECH_CHECKS = [
    "Windsurf was bought by OpenAI for how much?",
    "What is the status of the Apple Vision Pro?",
    "What is the revenue share agreement between OpenAI and Microsoft?",
    "What is Perplexity's new fund?",
    "What is the significance of DeepSeek-R2?"
]


# ---------------------------------------------------------------------------
# Index admin: ensure HOT index exists with HOT/RL fields
# ---------------------------------------------------------------------------
def ensure_hot_index(client: OpenSearch, index_name: str) -> None:
    """
    Ensures the HOT index exists with mappings compatible with your LONG index,
    plus RL/HOT governance fields.
    """
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
                # Core fields aligned with LONG
                "content": {"type": "text"},
                "category": {"type": "keyword", "normalizer": "lowercase_normalizer"},
                "filepath": {"type": "keyword"},
                "explicit_terms": {"type": "keyword", "normalizer": "lowercase_normalizer"},
                "explicit_terms_text": {"type": "text"},
                "ingested_at_ms": {"type": "date", "format": "epoch_millis"},
                "doc_version": {"type": "long"},

                # HOT/RL governance/TTL hooks
                "hot_promoted_at_ms": {"type": "date", "format": "epoch_millis"},
                "rl_run_id": {"type": "keyword", "normalizer": "lowercase_normalizer"},
                "rl_reward": {"type": "float"},
                "rl_feedback": {"type": "keyword", "normalizer": "lowercase_normalizer"},
                "rl_tags": {"type": "keyword", "normalizer": "lowercase_normalizer"},
                "source": {"type": "keyword", "normalizer": "lowercase_normalizer"},
            }
        },
    }
    client.indices.create(index=index_name, body=body)


# ---------------------------------------------------------------------------
# RL fact indexing
# ---------------------------------------------------------------------------
def _hash_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

def index_rl_fact(
    client: OpenSearch,
    index_name: str,
    *,
    fact_text: str,
    rl_run_id: str,
    category: str = "reinforcement",
    rl_reward: float = 1.0,
    rl_feedback: str = "accept",
    rl_tags: Optional[List[str]] = None,
    source: str = "rl_fact",
) -> Dict[str, Any]:
    """Index a single RL fact into HOT with explicit_terms derived from NER."""
    ner_result = post_ner(fact_text)
    explicit_terms = normalize_entities(ner_result)
    now_ms = int(time.time() * 1000)

    # Stable ID for auditability and dedup across runs
    doc_id = f"rl/{rl_run_id}/{_hash_id(fact_text)}"

    doc = {
        "content": fact_text,
        "category": category,
        "filepath": doc_id,
        "explicit_terms": explicit_terms,
        "explicit_terms_text": " ".join(explicit_terms) if explicit_terms else "",
        "ingested_at_ms": now_ms,
        "doc_version": now_ms,

        "hot_promoted_at_ms": now_ms,
        "rl_run_id": rl_run_id,
        "rl_reward": float(rl_reward),
        "rl_feedback": rl_feedback,
        "rl_tags": rl_tags or ["rl", "hot", "promotion"],
        "source": source,
    }

    client.index(index=index_name, id=doc_id, body=doc, refresh=False)
    return doc


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------
def prompt_yes_no(prompt: str, default_no: bool = True) -> bool:
    default = "N" if default_no else "Y"
    val = input(f"{prompt} [y/N] ").strip().lower() or default.lower()
    return val in ("y", "yes")

def print_doc_indexed(doc: Dict[str, Any]) -> None:
    print(f"\n[INDEXED] filepath={doc.get('filepath')}")
    print(f"  explicit_terms: {doc.get('explicit_terms')}")
    print(f"  hot_promoted_at_ms: {doc.get('hot_promoted_at_ms')}")
    print(f"  rl_run_id={doc.get('rl_run_id')} reward={doc.get('rl_reward')} feedback={doc.get('rl_feedback')}")
    print("")


# ---------------------------------------------------------------------------
# Test runner: run 5 hardcoded questions through ask(...)
# ---------------------------------------------------------------------------
def summarize_hits(hits: List[Dict[str, Any]]) -> Tuple[int, int]:
    hot = sum(1 for h in hits if h.get("_store_label") == "HOT")
    lng = sum(1 for h in hits if h.get("_store_label") == "LONG")
    return hot, lng

def run_test_questions(llm, *, observability: bool, save_path: Optional[str]) -> None:
    print("\n" + "=" * 88)
    print("RUNNING 5 HARD-CODED TEST QUESTIONS")
    print("=" * 88)
    for i, q in enumerate(TECH_CHECKS, 1):
        print("\n" + "-" * 88)
        print(f"[Q{i}] {q}\n")
        answer, hits = ask(llm, q, observability=observability, save_path=save_path)
        hot_cnt, long_cnt = summarize_hits(hits)
        print(f"Answer:\n{answer}\n")
        print(f"[Context docs] HOT={hot_cnt}  LONG={long_cnt}  TOTAL={len(hits)}")
        for j, h in enumerate(hits[:10], 1):
            store = h.get("_store_label", "?")
            idx   = h.get("_index_used", "?")
            fp    = h.get("_source", {}).get("filepath", "<unknown>")
            sc    = h.get("_score")
            print(f"  [{j}] STORE={store} INDEX={idx} SCORE={sc:.4f} DOC={fp}")
        print("-" * 88)
        print("\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Promote 5 hardcoded RL facts into HOT and test with 5 hardcoded questions via ask()."
    )
    parser.add_argument("--hot-index", type=str, default=None,
                   help="Override HOT index name (defaults to HOT_INDEX_NAME from query.py).")
    parser.add_argument("--observability", action="store_true", default=False,
                        help="Print ask() observability (default: ON).")
    parser.add_argument("--save-results", type=str, default=None,
                        help="Append compact JSONL result records from ask() to this path.")
    parser.add_argument("--rl-run-id", type=str, default=None,
                        help="Override RL run id (default: rl-<epoch_ms>).")
    args = parser.parse_args()

    # HOT client + index
    hot_client, resolved_hot_index = connect_hot()
    hot_index = args.hot_index or resolved_hot_index or HOT_INDEX_NAME or "bbc"
    if hot_index != resolved_hot_index:
        print(f"[INFO] HOT index override -> using '{hot_index}' (query.py default is '{resolved_hot_index}')")
    ensure_hot_index(hot_client, hot_index)

    # RL run id
    rl_run_id = args.rl_run_id or f"rl-{int(time.time()*1000)}"
    print(f"[INFO] RL run id: {rl_run_id}")
    print(f"[INFO] HOT index: {hot_index}")

    # Present 5 hardcoded facts
    added = 0
    for i, fact in enumerate(TECH_FACTS, 1):
        print("\n" + "-" * 88)
        print(f"FACT [{i}/5]: {fact}")
        do_add = prompt_yes_no("Add this fact to HOT memory?")
        if not do_add:
            print("  -> skipped")
            continue

        try:
            doc = index_rl_fact(
                hot_client,
                hot_index,
                fact_text=fact,
                rl_run_id=rl_run_id,
                category="reinforcement",
                rl_reward=1.0,
                rl_feedback="accept",
                rl_tags=["rl", "hot", "promotion"],
                source="rl_fact"
            )
            added += 1
            print_doc_indexed(doc)
        except SystemExit as e:
            print(f"  -> indexing failed due to NER/client error: {e}")
        except Exception as e:
            print(f"  -> indexing failed due to unexpected error: {e}")

    # Make additions visible
    hot_client.indices.refresh(index=hot_index)
    print(f"\n[OK] HOT refresh complete. Added {added} of 5 facts.")

    # Load LLM once and run 5 hardcoded test questions
    llm = load_llm()
    run_test_questions(llm, observability=args.observability, save_path=args.save_results)


if __name__ == "__main__":
    main()
