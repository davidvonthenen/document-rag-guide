#!/usr/bin/env python3
"""
HOT TTL Reaper — expire HOT-store docs older than a configurable age

What it does
------------
Deletes documents from the HOT OpenSearch index whose `hot_promoted_at_ms`
is older than a cutoff (now - TTL). Default TTL is 2 hours.

Why this fits your stack
------------------------
- Reuses your existing OpenSearch connection + index naming from `query.py`
  (connects with OPENSEARCH_HOT_* envs and HOT_INDEX_NAME).
- Preserves your governance fields; only targets docs by the auditable
  `hot_promoted_at_ms` timestamp you already stamp on promotion.
- Uses OpenSearch `delete_by_query` for efficient server-side cleanup.

Usage
-----
  # Dry run (see what WOULD be deleted), default TTL=2h
  ./hot_ttl_reaper.py --dry-run

  # Delete with explicit TTL; skip confirmation
  ./hot_ttl_reaper.py --ttl 90m --force

  # Override HOT index name if needed
  ./hot_ttl_reaper.py --ttl 1d --hot-index my_hot

Accepted TTL formats
--------------------
- Single or composite: "7200s", "120m", "2h", "1d", "1h30m", "2h15m20s"
- Bare integer => seconds

Requires
--------
- `query.py` available alongside this file providing: connect_hot, HOT_INDEX_NAME
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional

# Reuse your existing HOT connection + index naming
try:
    from common import connect_hot, HOT_INDEX_NAME
except Exception as e:
    raise SystemExit(
        "This tool expects your existing 'query.py' to be importable from the same folder.\n"
        f"Import error: {e}"
    )

from opensearchpy import OpenSearch
from opensearchpy.exceptions import TransportError


# -----------------------------
# Utility helpers
# -----------------------------
_UNIT_TO_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}

def parse_ttl_seconds(ttl_str: str) -> int:
    """
    Parse TTL like "2h", "90m", "1d", "1h30m", "2h15m20s" or bare "7200" (seconds).
    Returns seconds (int). Raises ValueError on nonsense.
    """
    ttl_str = ttl_str.strip().lower()
    if ttl_str.isdigit():
        sec = int(ttl_str)
        if sec <= 0:
            raise ValueError("TTL must be > 0 seconds")
        return sec

    parts = re.findall(r"(\d+)\s*([smhd])", ttl_str)
    if not parts:
        raise ValueError(f"Unrecognized TTL format: {ttl_str}")

    total = 0
    for qty, unit in parts:
        total += int(qty) * _UNIT_TO_SECONDS[unit]
    if total <= 0:
        raise ValueError("TTL must be > 0 seconds")
    return total

def ms_to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()


# -----------------------------
# Expiration core
# -----------------------------
def build_expire_query(cutoff_ms: int) -> Dict[str, Any]:
    """
    Match docs strictly older than cutoff by 'hot_promoted_at_ms'.
    (Docs missing this field will NOT match, which is what we want.)
    """
    return {"range": {"hot_promoted_at_ms": {"lt": cutoff_ms}}}

def preview_expiring(
    client: OpenSearch,
    index_name: str,
    query: Dict[str, Any],
    size: int = 10
) -> List[Dict[str, Any]]:
    body = {
        "query": query,
        "sort": [{"hot_promoted_at_ms": {"order": "asc"}}],
        "_source": ["filepath", "hot_promoted_at_ms", "rl_run_id", "rl_tags", "source"],
        "size": size,
        "track_total_hits": True,
    }
    res = client.search(index=index_name, body=body, request_timeout=30)
    return res.get("hits", {}).get("hits", []) or []

def count_expiring(client: OpenSearch, index_name: str, query: Dict[str, Any]) -> int:
    res = client.count(index=index_name, body={"query": query})
    return int(res.get("count", 0))

def delete_expiring(
    client: OpenSearch,
    index_name: str,
    query: Dict[str, Any],
    *,
    slices: str = "auto",
    refresh: bool = True,
    proceed_on_conflict: bool = True,
) -> Dict[str, Any]:
    kwargs = {
        "index": index_name,
        "body": {"query": query},
        "slices": slices,
        "refresh": refresh,
        "wait_for_completion": True,
    }
    if proceed_on_conflict:
        kwargs["conflicts"] = "proceed"
    return client.delete_by_query(**kwargs)


# -----------------------------
# CLI
# -----------------------------
def main():
    p = argparse.ArgumentParser(description="Expire HOT-store docs older than a TTL (default 2h).")
    p.add_argument("--ttl", type=str, default="2h",
                   help='Time-to-live window. Examples: "2h" (default), "90m", "1d", "1h30m", or "7200" (seconds).')
    p.add_argument("--hot-index", type=str, default=None,
                   help="Override HOT index name (defaults to HOT_INDEX_NAME from query.py).")
    p.add_argument("--dry-run", action="store_true", default=False,
                   help="Do not delete; show count + a preview sample.")
    p.add_argument("--force", action="store_true", default=False,
                   help="Skip confirmation prompt before deletion.")
    p.add_argument("--preview-size", type=int, default=10,
                   help="Number of sample docs to show in dry-run/preview (default 10).")
    args = p.parse_args()

    # Connect to HOT
    hot_client, default_hot_idx = connect_hot()
    hot_index = args.hot_index or default_hot_idx or HOT_INDEX_NAME or "bbc"

    # Compute cutoff
    try:
        ttl_seconds = parse_ttl_seconds(args.ttl)
    except ValueError as e:
        raise SystemExit(f"Invalid --ttl value: {e}")
    now_ms = int(time.time() * 1000)
    cutoff_ms = now_ms - (ttl_seconds * 1000)

    print(f"[INFO] HOT index     : {hot_index}")
    print(f"[INFO] TTL           : {args.ttl}  (~{ttl_seconds} seconds)")
    print(f"[INFO] Cutoff (UTC)  : {ms_to_iso(cutoff_ms)}  (hot_promoted_at_ms < cutoff)")

    # Build query & count
    query = build_expire_query(cutoff_ms)
    try:
        n = count_expiring(hot_client, hot_index, query)
    except TransportError as e:
        raise SystemExit(f"Count failed: {e}")

    if n == 0:
        print("[OK] Nothing to expire. HOT is clean.")
        return

    print(f"[INFO] Matching docs : {n}")

    # Always show a small preview set
    try:
        hits = preview_expiring(hot_client, hot_index, query, size=args.preview_size)
    except TransportError as e:
        raise SystemExit(f"Preview search failed: {e}")

    if hits:
        print("\n[PREVIEW] Oldest matching documents:")
        for i, h in enumerate(hits, 1):
            src = h.get("_source", {})
            fp = src.get("filepath", "<unknown>")
            ts = src.get("hot_promoted_at_ms")
            run = src.get("rl_run_id")
            tags = src.get("rl_tags")
            src_tag = src.get("source")
            iso = ms_to_iso(ts) if isinstance(ts, int) else str(ts)
            print(f"  [{i}] hot_promoted_at: {iso}  filepath: {fp}  rl_run_id: {run}  tags: {tags}  source: {src_tag}")
    else:
        print("\n[PREVIEW] No previewable docs (unexpected).")

    if args.dry_run:
        print("\n[DRY-RUN] Exiting without deletion.")
        return

    # Confirm unless forced
    if not args.force:
        ans = input(f"\nDelete {n} document(s) from '{hot_index}'? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("Aborted.")
            return

    # Delete
    try:
        resp = delete_expiring(hot_client, hot_index, query)
    except TransportError as e:
        raise SystemExit(f"Delete-by-query failed: {e}")

    deleted = resp.get("deleted", 0)
    total   = resp.get("total", 0)
    took    = resp.get("took", 0)
    vconf   = resp.get("version_conflicts", 0)
    fails   = resp.get("failures", [])

    print("\n[RESULT]")
    print(f"  total processed     : {total}")
    print(f"  deleted             : {deleted}")
    print(f"  version conflicts   : {vconf}")
    print(f"  took (ms)           : {took}")

    if fails:
        print("\n  failures (truncated):")
        for f in fails[:5]:
            print(f"    - {f}")

    # Make deletions visible immediately
    try:
        hot_client.indices.refresh(index=hot_index)
    except TransportError as e:
        print(f"[WARN] Refresh after delete failed (continuing): {e}")

    print("\n[OK] Expiration run complete.")


if __name__ == "__main__":
    main()
