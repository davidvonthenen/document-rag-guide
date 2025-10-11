#!/usr/bin/env python3
"""
HOT → LONG Promoter (Interactive Triage + Manual Copy + Single-Query Deletes)

Flow
----
1) Connect to HOT and LONG OpenSearch clusters using your existing `query.py` helpers.
2) Fetch candidate HOT docs (default: all), show a compact summary for each, and prompt:
      [P]romote to LONG   → mark for promotion
      [D]elete/Expire     → mark for deletion from HOT
      [I]gnore            → leave it in HOT (no action)
      [Q]uit              → stop triage early (process what you've marked so far)
3) After triage:
   - Write an offline filesystem copy of all *promoted* docs (JSONL + txt corpus).
   - Manually copy all *promoted* docs into LONG **in a single bulk request** (no `_reindex`).
   - Remove the *promoted* docs from HOT **in a single delete_by_query** using terms on `_id`.
   - Remove the *expired* docs from HOT **in a single delete_by_query** using terms on `_id`.

Why this matches your stack
---------------------------
- No embeddings; preserves your deterministic, auditable fields.
- Keeps `explicit_terms`/`explicit_terms_text` intact.
- LONG index writes only the governance-approved fields (no RL-only fields).
- Exports a clean JSONL and a txt corpus (by category) for offline audit or re-ingest.

Usage
-----
  ./promote_hot_to_long.py                               # interactive over all HOT docs
  ./promote_hot_to_long.py --rl-run-id rl-1728588123456  # restrict to a run
  ./promote_hot_to_long.py --limit 100 --force           # limit candidates; skip confirmations

Env honored via your existing query.py:
  OPENSEARCH_LONG_* , OPENSEARCH_HOT_* , LONG_INDEX_NAME , HOT_INDEX_NAME
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Reuse your established connectors and index names
# ---------------------------------------------------------------------------
try:
    from common import (
        connect_hot,           # -> (OpenSearch, index_name)
        connect_long,          # -> (OpenSearch, index_name)
        HOT_INDEX_NAME,        # default HOT index
        LONG_INDEX_NAME,       # default LONG index
    )
except Exception as e:
    raise SystemExit(
        "This script expects your existing 'query.py' to be importable in the same folder.\n"
        f"Import error: {e}"
    )

from opensearchpy import OpenSearch
from opensearchpy.exceptions import TransportError

# ---------------------------------------------------------------------------
# Constants / Fields
# ---------------------------------------------------------------------------
SOURCE_FIELDS = [
    # Core fields used by LONG/HOT
    "content", "category", "filepath",
    "explicit_terms", "explicit_terms_text",
    "ingested_at_ms", "doc_version",
    # HOT-only governance
    "hot_promoted_at_ms", "rl_run_id", "rl_reward", "rl_feedback", "rl_tags", "source",
]

# Only these will be written into LONG to preserve stable mapping.
LONG_ALLOWED_FIELDS = {
    "content", "category", "filepath",
    "explicit_terms", "explicit_terms_text",
    "ingested_at_ms", "doc_version",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def env_str(key: str, default: str) -> str:
    return os.getenv(key, default)

def now_ms() -> int:
    return int(time.time() * 1000)

def slugify(s: str) -> str:
    s = re.sub(r"[^\w\-\.]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:80] if len(s) > 80 else s

def ensure_long_index(client: OpenSearch, index_name: str) -> None:
    """
    Create LONG index if missing with the same mapping you used in ingest.
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

def build_hot_selection_query(
    rl_run_id: Optional[str],
    source: Optional[str],
    older_than_ms: Optional[int],
    newer_than_ms: Optional[int],
) -> Dict[str, Any]:
    filters: List[Dict[str, Any]] = []

    if rl_run_id:
        filters.append({"term": {"rl_run_id": rl_run_id}})

    if source:
        filters.append({"term": {"source": source}})

    # Range filter when either bound is provided
    if (older_than_ms is not None) or (newer_than_ms is not None):
        rng: Dict[str, Any] = {}
        if newer_than_ms is not None:
            rng["gte"] = newer_than_ms
        if older_than_ms is not None:
            rng["lte"] = older_than_ms
        filters.append({"range": {"hot_promoted_at_ms": rng}})

    return {"bool": {"filter": filters}} if filters else {"match_all": {}}


def iter_hot_docs(
    client: OpenSearch,
    index_name: str,
    query: Dict[str, Any],
    batch_size: int = 50,
    scroll_ttl: str = "2m",
    limit: Optional[int] = None,
) -> Iterable[Dict[str, Any]]:
    """
    Scroll through HOT candidates. Yields raw hits.
    """
    fetched = 0
    body = {
        "query": query,
        "_source": SOURCE_FIELDS,
        "size": batch_size,
        "sort": ["_doc"],  # recommended for scroll
        "track_total_hits": True,
    }
    res = client.search(index=index_name, body=body, scroll=scroll_ttl, request_timeout=60)
    scroll_id = res.get("_scroll_id")
    try:
        while True:
            hits = res.get("hits", {}).get("hits", []) or []
            if not hits:
                break
            for h in hits:
                yield h
                fetched += 1
                if limit is not None and fetched >= limit:
                    return
            res = client.scroll(scroll_id=scroll_id, scroll=scroll_ttl, request_timeout=60)
            scroll_id = res.get("_scroll_id", scroll_id)
    finally:
        if scroll_id:
            try:
                client.clear_scroll(scroll_id=scroll_id)
            except Exception:
                pass

def print_candidate(hit: Dict[str, Any], idx: int) -> None:
    src = hit.get("_source", {})
    fp = src.get("filepath", "<unknown>")
    cat = src.get("category", "<none>")
    hms = src.get("hot_promoted_at_ms")
    rlid = src.get("rl_run_id")
    terms = src.get("explicit_terms") or []
    snippet = (src.get("content") or "").strip().replace("\n", " ")
    snippet = snippet[:220] + ("…" if len(snippet) > 220 else "")
    print("\n" + "-" * 88)
    print(f"[{idx}] HOT _id={hit.get('_id')}  filepath={fp}  category={cat}  rl_run_id={rlid}  hot_promoted_at_ms={hms}")
    print(f"     explicit_terms: {terms}")
    print(f"     content: {snippet}")

def ask_action() -> str:
    while True:
        ans = input("Action? [P]romote  [D]elete  [I]gnore  [Q]uit > ").strip().lower()
        if ans in ("p", "d", "i", "q"):
            return ans
        print("Please enter one of: p / d / i / q")

def make_long_doc(src: Dict[str, Any]) -> Dict[str, Any]:
    """
    Construct the document body to be stored in LONG.
    Use only LONG_ALLOWED_FIELDS and set fresh ingest/version times.
    """
    out: Dict[str, Any] = {}
    for k in LONG_ALLOWED_FIELDS:
        if k in src and src[k] is not None:
            out[k] = src[k]
    # Ensure explicit_terms_text is present if explicit_terms exists
    if "explicit_terms" in out and out["explicit_terms"] and not out.get("explicit_terms_text"):
        out["explicit_terms_text"] = " ".join(out["explicit_terms"])
    t = now_ms()
    out["ingested_at_ms"] = t
    out["doc_version"] = t
    # Default category if missing
    out.setdefault("category", "reinforcement")
    return out

def bulk_index_long(client: OpenSearch, index_name: str, docs: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Single bulk call to index all docs into LONG.
    Each tuple is (doc_id, body)
    """
    if not docs:
        return {"errors": False, "items": [], "took": 0}
    lines: List[str] = []
    for doc_id, body in docs:
        lines.append(json.dumps({"index": {"_index": index_name, "_id": doc_id}}, ensure_ascii=False))
        lines.append(json.dumps(body, ensure_ascii=False))
    payload = "\n".join(lines) + "\n"
    return client.bulk(body=payload, refresh=True, request_timeout=120)

def delete_by_ids_single_query(client: OpenSearch, index_name: str, ids: List[str]) -> Dict[str, Any]:
    """
    Single delete_by_query using terms on _id.
    """
    if not ids:
        return {"deleted": 0, "total": 0, "took": 0, "version_conflicts": 0}
    q = {"terms": {"_id": ids}}
    return client.delete_by_query(
        index=index_name,
        body={"query": q},
        slices="auto",
        refresh=True,
        wait_for_completion=True,
        conflicts="proceed",
        request_timeout=300,
    )

def write_offline_exports(
    export_dir: Path,
    promoted_hits: List[Dict[str, Any]],
) -> None:
    """
    Create a JSONL (promoted.jsonl) and a text corpus under dataset/<category>/*.txt
    """
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "dataset").mkdir(parents=True, exist_ok=True)

    # JSONL with full HOT sources for audit
    with (export_dir / "promoted.jsonl").open("w", encoding="utf-8") as jf:
        for h in promoted_hits:
            jf.write(json.dumps(h.get("_source", {}), ensure_ascii=False) + "\n")

    # Plain-text corpus (by category) for simple re-ingestion or inspection
    for h in promoted_hits:
        src = h.get("_source", {})
        cat = slugify(src.get("category") or "reinforcement")
        content = (src.get("content") or "").rstrip() + "\n"
        base = src.get("filepath") or h.get("_id") or f"anon-{hash(content)}"
        name = slugify(base) or "doc"
        out_dir = export_dir / "dataset" / cat
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{name}.txt").write_text(content, encoding="utf-8")

    manifest = {
        "ts_ms": now_ms(),
        "count_promoted": len(promoted_hits),
        "note": "promoted.jsonl holds full HOT sources; dataset/ holds plain-text corpus by category.",
    }
    (export_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Promote selected HOT docs into LONG (interactive triage, manual copy, single-query deletes).")
    ap.add_argument("--rl-run-id", type=str, default=None, help="Filter HOT candidates by rl_run_id.")
    ap.add_argument("--source", type=str, default=None, help="Filter by HOT 'source' field (e.g., rl_fact).")
    ap.add_argument("--older-than-ms", type=int, default=None, help="Filter: hot_promoted_at_ms <= this value.")
    ap.add_argument("--newer-than-ms", type=int, default=None, help="Filter: hot_promoted_at_ms >= this value.")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of HOT docs to triage.")
    ap.add_argument("--batch-size", type=int, default=50, help="Scroll batch size (default 50).")
    ap.add_argument("--export-dir", type=str, default=None, help="Where to write offline copies (default: promotion_exports/promote-<epoch_ms>).")
    ap.add_argument("--force", action="store_true", default=False, help="Skip final confirmations.")
    ap.add_argument("--hot-index", type=str, default=None, help="Override HOT index name (default from query.py).")
    ap.add_argument("--long-index", type=str, default=None, help="Override LONG index name (default from query.py).")
    args = ap.parse_args()

    # Connect
    hot_client, hot_index_default = connect_hot()
    long_client, long_index_default = connect_long()

    hot_index = args.hot_index or hot_index_default or HOT_INDEX_NAME or "bbc"
    long_index = args.long_index or long_index_default or LONG_INDEX_NAME or "bbc"

    print(f"[INFO] HOT index : {hot_index}")
    print(f"[INFO] LONG index: {long_index}")

    # Ensure LONG exists with expected mapping (no-ops if present)
    ensure_long_index(long_client, long_index)

    # Build selection query
    sel_query = build_hot_selection_query(
        rl_run_id=args.rl_run_id,
        source=args.source,
        older_than_ms=args.older_than_ms,
        newer_than_ms=args.newer_than_ms,
    )

    # Triage loop
    promoted: List[Dict[str, Any]] = []
    expired_ids: List[str] = []
    ignored_ids: List[str] = []
    processed = 0

    print("\n[INFO] Starting triage. Fetching HOT candidates...")
    for hit in iter_hot_docs(
        hot_client, hot_index, sel_query,
        batch_size=args.batch_size, scroll_ttl="2m", limit=args.limit
    ):
        processed += 1
        print_candidate(hit, processed)
        action = ask_action()
        if action == "q":
            print("[INFO] Quitting triage by user request.")
            break
        if action == "i":
            ignored_ids.append(hit.get("_id"))
            continue
        if action == "d":
            expired_ids.append(hit.get("_id"))
            continue
        if action == "p":
            promoted.append(hit)
            continue

    print("\n" + "=" * 88)
    print(f"[SUMMARY] Processed={processed}  Promote={len(promoted)}  Delete={len(expired_ids)}  Ignore={len(ignored_ids)}")
    print("=" * 88)

    # Exports directory
    export_dir = Path(args.export_dir or (Path("promotion_exports") / f"promote-{now_ms()}")).expanduser()
    print(f"[INFO] Export dir: {export_dir}")

    # Confirm before mutating clusters
    if not args.force:
        ans = input("Proceed with export + LONG bulk index + HOT deletions? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("Aborted.")
            return

    # 1) Write offline copies (promoted only)
    if promoted:
        write_offline_exports(export_dir, promoted)
        print(f"[OK] Offline copy written to: {export_dir}")
    else:
        print("[OK] No promoted docs → skipping offline export.")

    # 2) Manual copy to LONG (single bulk request)
    if promoted:
        to_index: List[Tuple[str, Dict[str, Any]]] = []
        for h in promoted:
            src = h.get("_source", {})
            long_body = make_long_doc(src)
            # Use filepath as stable id (same pattern as ingest)
            doc_id = long_body.get("filepath") or h.get("_id")
            to_index.append((doc_id, long_body))

        bulk_resp = bulk_index_long(long_client, long_index, to_index)
        errors = bool(bulk_resp.get("errors"))
        items = bulk_resp.get("items", [])
        took  = bulk_resp.get("took", 0)
        ok_cnt = sum(1 for it in items if "index" in it and 200 <= it["index"].get("status", 500) < 300)
        print(f"[LONG BULK] took={took}ms  ok={ok_cnt}/{len(items)}  errors={errors}")
        if errors:
            # Show a couple of errors
            shown = 0
            for it in items:
                st = it.get("index", {}).get("status")
                if st and st >= 300:
                    print("  error item:", it)
                    shown += 1
                    if shown >= 5:
                        break
    else:
        print("[LONG BULK] No promoted docs → skipping LONG indexing.")

    # 3) Single-query deletes in HOT
    # 3a) Delete the *promoted copies* in HOT in one delete_by_query
    if promoted:
        promoted_ids = [h.get("_id") for h in promoted if h.get("_id")]
        resp_promoted_del = delete_by_ids_single_query(hot_client, hot_index, promoted_ids)
        print(f"[HOT DELETE promoted] total={resp_promoted_del.get('total')} deleted={resp_promoted_del.get('deleted')} took={resp_promoted_del.get('took')}ms")
    else:
        print("[HOT DELETE promoted] No promoted docs → skipping.")

    # 3b) Delete *expired* in HOT in one delete_by_query
    if expired_ids:
        resp_expired_del = delete_by_ids_single_query(hot_client, hot_index, expired_ids)
        print(f"[HOT DELETE expired]  total={resp_expired_del.get('total')} deleted={resp_expired_del.get('deleted')} took={resp_expired_del.get('took')}ms")
    else:
        print("[HOT DELETE expired] No expired docs → skipping.")

    # Final refresh for visibility
    try:
        long_client.indices.refresh(index=long_index)
    except TransportError as e:
        print(f"[WARN] LONG refresh failed: {e}")
    try:
        hot_client.indices.refresh(index=hot_index)
    except TransportError as e:
        print(f"[WARN] HOT refresh failed: {e}")

    # Write simple action logs
    (export_dir / "ignored_ids.jsonl").write_text("\n".join(ignored_ids) + ("\n" if ignored_ids else ""), encoding="utf-8")
    (export_dir / "expired_ids.jsonl").write_text("\n".join(expired_ids) + ("\n" if expired_ids else ""), encoding="utf-8")
    if promoted:
        promoted_ids = [h.get("_id") for h in promoted if h.get("_id")]
        (export_dir / "promoted_ids.jsonl").write_text("\n".join(promoted_ids) + "\n", encoding="utf-8")

    print("\n[OK] Promotion run complete.")


if __name__ == "__main__":
    main()
