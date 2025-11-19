# Document RAG: Enterprise Version

Retrieval-Augmented Generation (RAG) has become a critical pattern for grounding Large Language Model (LLM) responses in real-world data, improving both accuracy and reliability. Yet, conventional RAG implementations often default to vector databases, which come with drawbacks: hallucinations, opaque ranking logic, and challenges with regulatory compliance.

![Enterprise Document RAG](./images/enterprise_version.png)

In contrast, Document-centric RAG built on **lexical search (BM25)** offers a transparent and deterministic alternative. This mirrors capabilities produced by [Graph-based RAG and Knowledge Graphs](https://neo4j.com/blog/genai/what-is-graphrag/) but using Document search as the focal point. By prioritizing explicit term matching and leveraging an **external NER service** to extract normalized entities that are stored alongside documents, this architecture produces retrievals that are observable, reproducible, and audit-ready. Every query can be explained—down to which entity or phrase matched and why a document was returned.

## Key Benefits of Document-based RAG

* **Transparency and Explainability**: Each match is traceable through explicit queries, field names, and highlights—no hidden embedding math.
* **Determinism and Auditability**: Deterministic analyzers and explicit keyword fields (for normalized entities) ensure reproducible relevance decisions under BM25.
* **Governance and Compliance**: Observable retrieval paths simplify regulatory adherence and policy enforcement.
* **Bias and Risk Mitigation**: External NER models can be tailored to the domain, making keyword selection explicit and reviewable.

## Dual-Memory Architecture

The enterprise Document RAG agent uses a **HOT (unstable) + long-term store** design:

| Memory Type          | Infrastructure                   | Data Stored                                           | Purpose                                                         |
| -------------------- | -------------------------------- | ----------------------------------------------------- | --------------------------------------------------------------- |
| **Long-term Memory** | Persistent OpenSearch index      | Curated, validated documents                          | Authoritative knowledge base; compliance-ready                  |
| **HOT (unstable)**   | High-performance OpenSearch node | Unstable/unverified facts and user data | Governance boundary, retention variations, operational locality |

## Business Impact

Deploying Document-first RAG in the enterprise yields clear strategic advantages:

* **Operational clarity** via parallel LT+HOT queries and deterministic BM25 scoring with entity-level filters; storage tiers exist primarily for governance and isolation, not latency wins per se.
* **Improved compliance and risk control** thanks to explainable retrieval logic and complete data lineage.
* **Scalability and resilience** via leveraging enterprise storage practices.

By grounding AI systems in observable document retrieval instead of opaque embeddings, enterprises can increase trustworthiness, compliance, and operational clarity—while meeting real-world performance requirements without relying on per-request data movement.

This guide provides an enterprise-oriented reference for implementing Document-based RAG architectures, enabling organizations to build faster, clearer, and fully governable AI solutions.

# 2. Ingesting Your Data Into Long-Term Memory

> **Same core pipeline, enterprise-grade surroundings.**
> This section mirrors the community version but emphasizes enterprise priorities such as audit trails, schema governance, and storage economics.

## Why We Start with Clean Knowledge

Long-term memory is the system's **source of truth**. Anything that lands here must be:

1. **Authoritative**: derived from validated, trusted documents.
2. **Traceable**: every document is stored with explicit provenance and metadata.
3. **Governance-ready**: aligned with organizational taxonomies, compliance policies, and audit requirements.

## Four-Step Ingestion Pipeline

| Stage                | What Happens                                                                           | Enterprise Add-Ons                                                                             |
| -------------------- | -------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **1. Parse**         | Raw content (text files, PDFs, tickets) is loaded into the pipeline.                   | Compute content hashes for tamper detection and store file paths for lineage.                  |
| **2. Slice**         | Documents are ingested **as full docs** and additionally **chunked by paragraph by default**. | Preserve offsets and category metadata if you later enable alternative slicing strategies.                            |
| **3. Extract Terms** | An **external NER HTTP service** returns normalized entities (lowercased, deduped).    | Maintain the NER service version separately; terms are stored, not the model name.             |
| **4. Persist**       | Documents are indexed into OpenSearch with explicit term fields.                       | Each record carries `filepath`/`URI`, `ingested_at_ms`, and numeric `doc_version` for observability. |

## Reference Code Snapshot *(aligned with enterprise edition)*

```python
# ingest_bbc.py - trimmed to show stored fields and stable ID
doc_id = p.as_posix()  # stable per file for clean re-index/promotion
now_ms = int(time.time() * 1000)

doc = {
    "content": text,
    "category": category,
    "filepath": doc_id,
    "explicit_terms": explicit_terms,                         # exact keyword field (lowercased)
    "explicit_terms_text": " ".join(explicit_terms) if explicit_terms else "",
    "ingested_at_ms": now_ms,                                 # epoch_millis
    "doc_version": now_ms                                     # monotonic version for traceability
}
client.index(index=index_name, id=doc_id, body=doc, refresh=False)
```

The ingestion script ensures idempotency at the document level via a **stable `_id` = file path**. In enterprise settings, you can add a **batch ID** around each run to trace or roll back entire ingests.

## Operational Knobs You Control

| Variable          | Typical Value               | Why You Might Change It                             |
| ----------------- | --------------------------- | --------------------------------------------------- |
| `DATA_DIR`        | `bbc`                       | Point to NFS, S3-mount, or SharePoint sync.         |
| `INDEX_NAME`      | `bbc`                       | Route to a different long-term index/alias.         |
| `DEFAULT_URL`     | `http://127.0.0.1:8000/ner` | Direct to your NER service or a domain-tuned model. |
| `OPENSEARCH_PORT` | `9201`                      | Match your long-term node or cluster ingress.       |

### Implementation Considerations

* The reference pipeline **calls an external NER service** and performs **NER-only enrichment** during ingest. NER HTTP failures halt the run with explicit diagnostics from the client. If you need fail-soft behavior, catch the exception and store an empty `explicit_terms` list.
* Index **settings and mappings** explicitly set a lowercase normalizer for keyword fields and `number_of_replicas: 0`. BM25 uses OpenSearch defaults; change analyzers/mappings only with a controlled reindex plan.
* Provenance metadata is explicit and minimal: `filepath`/`URI`, `ingested_at_ms`, and `doc_version`. The NER model name is not persisted in documents; track it at the service level (e.g., `/health`) for audits.
* **HOT → LT promotion policy:** the only time data moves from HOT back into LT is when there's (1) enough positive reinforcement to warrant promotion **or** (2) a trusted human-in-the-loop has verified the data.

## Additional Notes

* *Version everything.* Use the numeric `doc_version` and your VCS/infra configs to record mapping and analyzer changes you might reindex later.

With clean, well-labeled documents in long-term memory, every downstream RAG query inherits stable, auditable provenance. Next, we'll cover how to materialize vetted subsets into **HOT (unstable)** for governance boundaries, TTL-based lifecycle, and operational isolation.

# 4. Promotion of Long-Term Memory into **HOT (unstable)**

> **Goal:** Maintain a governed, low-retention variations working set near the application while preserving provenance and compliance. This tiering exists primarily for **governance, isolation, and policy asymmetry**—not because query latency on LT is inherently slow.

## Why Promote?

* **retention variations control.** HOT absorbs high write rates, experiments, and unstable facts without impacting LT's mappings, heap, or latency characteristics.
* **Policy asymmetry.** LT stays conservative (strict analyzers, WORM-like behavior); HOT can use permissive schemas, faster refresh, TTL/rollover, and relaxed matching.
* **Operational locality.** Keep the active topic set colocated with serving paths and GPUs while **leaving LT untouched**; throughput improves by isolating churn, not by per-request copying.

## Enterprise Twist vs. Community Guide

| Stage               | Community Edition                          | Enterprise Edition                                                                                            |
| ------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| **Detect entities** | Extract keywords/NER in the app process    | **External NER HTTP service** produces normalized entities (lowercased, de-duplicated).                       |
| **Fetch docs**      | Direct BM25 search on long-term index      | **Filter by normalized entities** (`terms`/`terms_set` on `explicit_terms`), optional `ingested_at_ms` window |
| **Transfer data**   | Python script copies docs into cache index | HOT **calls `/_reindex`** with a *remote source* (LT) + filter; HOT "pulls" only matching docs                |
| **TTL management**  | Simple cron delete                         | **Eviction script** runs `delete_by_query` on `hot_promoted_at` threshold (rate-limited, capped)              |

## Promotion Flow (Enterprise)

1. **Entity set is selected out-of-band** — from recent activity, topics, or operator input using the external NER service.
2. **Promotion filter** targets `explicit_terms` (and optionally a time window on `ingested_at_ms`).
3. **HOT `/_reindex` from remote LT** executes with that filter and **stamps `hot_promoted_at` (epoch_millis)** on each copied doc.
4. **Serving**: the application **queries LT and HOT in parallel** (see query code) and, by default, uses the external BM25 re-ranker to order the merged hit list; there is **no inline reindexing** during a user question.
5. **Eviction job** removes docs whose `hot_promoted_at` exceeds the configured TTL.

```json
// Example promotion doc snippet
{
  "filepath": "/mnt/ingest/bbc/politics/file123.txt",
  "category": "politics",
  "content": "…",
  "explicit_terms": ["ernie wise", "vodafone"],
  "doc_version": 1727370000000,
  "hot_promoted_at": 1727373600000
}
```

> **Reverse path (HOT → LT):** Promotion **only** occurs when (1) sufficient positive reinforcement warrants it **or** (2) a trusted human-in-the-loop verifies the data.

## Operational Knobs

| Variable                      | Typical Value    | Purpose                                                |
| ----------------------------- | ---------------- | ------------------------------------------------------ |
| `TTL_MINUTES`                 | `30`             | Eviction threshold for `hot_promoted_at`.              |
| `DEST_INDEX`                  | `bbc`            | Target HOT index/alias that receives promoted docs.    |
| `REINDEX_REQUESTS_PER_SECOND` | `-1` (unlimited) | Rate limit for the promotion `/_reindex` task.         |
| `PROMOTE_WINDOW_SECONDS`      | `0` (disabled)   | Only promote docs with recent `ingested_at_ms` if > 0. |

### Replication: the Enterprise Upgrade Path

Default is **on-demand `/_reindex`** with explicit filters. If you need continuous movement at scale:

1. **CCR (Cross-Cluster Replication)** can mirror indices; use filters or separate pipelines to limit scope.
2. **ISM policies** enforce lifecycle/retention alongside (or in place of) the eviction script.
3. **Event-driven promotions** (e.g., Kafka CDC) can trigger filtered reindexing when upstream systems change.

```json
PUT /bbc_cache/_ccr/follow?leader_cluster=longterm&leader_index=bbc
```

### Where to Run **HOT**

Your choice of backing store governs speed and operational flexibility:

| Option                     | Speed                         | Caveats                                                                | Best for                                   |
| -------------------------- | ----------------------------- | ---------------------------------------------------------------------- | ------------------------------------------ |
| **In-memory index**        | 🚀 Fastest, all docs in RAM   | Limited by host memory; volatile on restart.                           | Demos, PoCs                                |
| **Local NVMe SSD**         | ⚡ Near-memory once warmed     | Data tied to node; rescheduling or failover harder.                    | Bare-metal, fixed clusters                 |
| **ONTAP FlexCache volume** | ⚡ Micro-second reads at scale | Requires NetApp ONTAP; gains portability and rescheduling flexibility. | Production Kubernetes or multi-site setups |

**Why FlexCache Helps Enterprises**

* **Elastic capacity** beyond physical RAM without pipeline redesigns.
* **Portability** — cache volumes can follow pods across nodes/AZs.
* **Governance** — SnapMirror and thin provisioning aid audit and cost control.

In short: entity-filtered **`/_reindex` + TTL eviction** gives speed-through-isolation and determinism today; adding **CCR + ISM + FlexCache** layers in **resilience and governance** when scale and ops require it.

## 4. Implementation Guide

For a reference, please check out the following: [enterprise_version/README.md](./enterprise_version/README.md)

# 5. Conclusion

Document-based RAG turns retrieval-augmented generation from a black-box trick into a transparent, governed architecture. By grounding retrieval in explicit lexical search (BM25) and observable NER-derived terms—and materializing the **working set** into **HOT (unstable)** when appropriate—you get answers that are:

* **Governed performance.** Parallel queries to LT and HOT keep UX consistent **without per-request copying**; tiers exist primarily for governance, isolation, and policy asymmetry—any speedup is a secondary effect of locality and reduced churn.
* **Clearer.** Every match is traceable through explicit fields and analyzer settings (e.g., `explicit_terms`, `explicit_terms_text`) with human-readable **highlights**; auditors can replay how a result matched.
* **Safer.** Opaque embeddings are replaced with deterministic, explainable retrieval logic; hallucinations and hidden bias are easier to detect and fix.
* **Compliant.** Built-in provenance metadata (`filepath`/`URI`, `ingested_at_ms`, `doc_version`) makes regulatory alignment and retention policies straightforward.

The enterprise path centers on **on-demand `/_reindex` executed on HOT with a remote source pointing to LT**, filtered by normalized entities, plus a **TTL eviction job** keyed on `hot_promoted_at`. Query serving does **not** trigger reindex; the orchestrator searches LT and HOT in parallel and (by default) feeds the merged hits through the external BM25 re-ranker before presenting context to the LLM. **HOT → LT promotion occurs only** when (1) sufficient positive reinforcement warrants it **or** (2) a trusted human-in-the-loop has verified the data. When needed, **Cross-Cluster Replication (CCR)**, **Index State Management (ISM)**, and **NetApp FlexCache** add 24/7 resilience, lifecycle control, and high-speed caching at scale.

## Next Steps

1. **Clone the repo.** The reference code and docs live at `github.com/your-org/document-rag-guide`. Try it locally with Docker Compose.
2. **Swap in your search backend.** All queries use standard OpenSearch DSL; adapt for Elasticsearch or your preferred lexical engine as needed.
3. **Feed it live data.** Point the ingest pipeline at a corpus (news feeds, Jira exports, PDFs) and run the NER service; **materialize HOT via filtered `/_reindex` out-of-band** (not in the request path).
4. **Tune the thresholds.** Adjust `α` (relative scoring cutoff), `TTL_MINUTES` (eviction window), `REINDEX_REQUESTS_PER_SECOND`, and optional `PROMOTE_WINDOW_SECONDS` until behavior matches your domain.
5. **Share lessons.** File issues, submit pull requests, or post a case study. This guide improves with community input and enterprise feedback.

Document-based RAG isn't a prototype—it's running code with governance baked in. Bring it into your stack and start building AI you can trust.
