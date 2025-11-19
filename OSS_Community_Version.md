# Document RAG: Open Source Community Version

Most "RAG" stacks start with vectors and end with questions from auditors: *why did that snippet surface, which fields mattered, and can we reproduce the response tomorrow?* Document RAG flips the playbook. We begin with **OpenSearch BM25** and an **external NER service** so retrieval is explicit, auditable, and grounded in text and metadata you control—then add dense signals only if they earn their keep. This mirrors capabilities produced by [Graph-based RAG and Knowledge Graphs](https://neo4j.com/blog/genai/what-is-graphrag/) but using Document search as the focal point.

![Community Document RAG](./images/community_version.png)

**Why lexical-first beats black-box vectors**

* **Explainability on day one.** We treat entities and keywords as first-class citizens and query them directly. You can point to the exact fields and terms that fired—no hand-waving about "semantic neighbors."
* **Deterministic, governable search.** We use explicit **keyword normalizers** on entity fields, default **BM25** scoring, and **highlights** for legibility.
* **Reduce hallucinations without hiding the ball.** Answers are grounded in retrieved documents (not vector vibes), and every claim traces back to source evidence.

**How this stack works**

At ingest, an external HTTP NER service (spaCy model `en_core_web_sm` by default) extracts entities from document text and writes them into two fields: `explicit_terms` (**keyword** with lowercase normalizer) and `explicit_terms_text` (**text**). In the search path, if a user question yields entities, we build a `dis_max` query with **two branches**:

1. **Strict AND (entity co-occurrence):** `terms_set` on `explicit_terms` requiring **all** detected entities to be present (AND semantics). This makes "why it matched" painfully clear.
2. **Soft OR (blend):** an OR-style branch that considers `terms`/`match`/`multi_match` across `explicit_terms_text` and `content` to surface near-misses.

We set `tie_breaker` to `0.0`, so the best branch wins without cross-branch score leakage. If **no** entities are detected, we fall back to a straightforward BM25 `match` on the full question. We also request **highlights** on `content` so reviewers see the exact snippets that triggered the hit.

> Note: the current code **does not enforce phrase-scoped matching** (`match_phrase`) for entities; the strictness comes from `terms_set` on the keyword entity field, not from phrase queries.

After both LT and HOT respond, `community_version/common.py` can (and does, by default) run an **external BM25 re-ranker** that re-scores the merged hit list. The re-ranker is lightweight (`bm25s` Python library), keeps score math outside OpenSearch for transparency, and ensures the LLM sees a single auditable ordering instead of two separately ranked pools. Disable it only if you need to inspect raw OpenSearch scores.

**Dual-store design (HOT vs. Long) with OpenSearch**

| Layer                    | Where it lives                                  | What it stores                               | Why it matters                                                                 |
| ------------------------ | ----------------------------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------ |
| **Long-term (LT)**       | Dedicated OpenSearch index on durable storage   | Curated, vetted content + stable NER fields  | Authoritative source with provenance; ideal for compliance and audits          |
| **HOT (unstable) cache** | Separate OpenSearch index on isolated resources | Fresh, session-scoped context and live feeds | Isolates blast radius and policy; natural place to enforce TTLs and test facts |

Promotion is **explicit and outside the query path**: the retrieval code queries **LT and HOT in parallel** (read-only). Moving data **from HOT to LT** occurs **only** when there's sufficient positive reinforcement of the facts *or* a trusted human-in-the-loop has verified them.

**Business impact**

* **Transparency & accountability:** Each result includes **highlighted** excerpts and matches on explicit fields, so you can always explain *why* a document surfaced.
* **Compliance by design:** Clear lifecycles (**HOT TTL** vs. long-term retention) simplify GDPR/HIPAA-style obligations.
* **Lower risk, higher signal:** Grounded retrieval reduces fabrication while keeping dense signals as an **optional, declared add-on**—not a hidden default.

The sections that follow show how to **ingest** with NER enrichments (stable mappings; NER service **required by default**), **retrieve** with strict entity co-occurrence via `terms_set` under `dis_max` (no named branches) and **highlights**, and **operate LT ↔ HOT promotion workflows** (outside the query path) with optional TTL/ISM policies—all with observability hooks that make auditors smile.

## 2. Ingesting Your Data Into Long-Term Memory

### Why We Start With Deterministic Indexing

Long-term memory is the system's **source of truth**. Every document that lands here must be clean, auditable, and ready for governance. That means explicit mappings, predictable analyzers, and transparent enrichments. The ingestion pipeline does four things in a single pass:

1. **Parses raw text** from articles, manuals, or tickets.
2. **Attaches metadata + stable IDs** (`category` from the parent folder; stable `_id` from the file's path under `DATA_DIR`).
3. **Extracts named entities** via an **external HTTP NER service**.
4. **Persists text + entity fields** into OpenSearch with a lowercase **normalizer** and stable mappings.

Do this well once, and every downstream RAG query inherits the same deterministic, audit-friendly provenance.

### Pipeline at a Glance

The reference `ingest.py` makes it concrete:

* Documents are read from disk, `category` is derived from the parent folder, and the full file body goes into `content`.
* Entities returned by the NER service are **lower-cased and de-duplicated** and indexed into both:

  * `explicit_terms` (**keyword**, with lowercase normalizer)
  * `explicit_terms_text` (**text**, BM25-scored)

* Paragraph-level slices are emitted alongside each document. By default, `community_version/ingest.py` writes full documents into `bbc` (configurable via `INDEX_NAME`) and deterministic paragraph chunks into `bbc-chunks` (`CHUNK_INDEX_NAME`). Each chunk carries the parent filepath, chunk index, and chunk count so you can trace a span back to the source file without manual bookkeeping.

We also store **stable operational metadata**:

* `ingested_at_ms` (epoch millis)
* `doc_version` (monotonic integer; set to the current timestamp for re-ingest/upsert workflows)

This representation supports strict **AND** matching on entities (via `terms_set` against `explicit_terms`) and lexical BM25 recall, both with clear "why did this match" stories.

```
Document {
   filepath               // keyword
   category               // keyword (lowercase normalizer)
   content                // text
   explicit_terms         // keyword (lowercase normalizer)
   explicit_terms_text    // text
   ingested_at_ms         // date (epoch_millis)
   doc_version            // long
}
```

Documents are **inserted** by stable `_id` (the path under `DATA_DIR`). They are not immutable; repeat runs replace prior versions in place with an updated `doc_version`.

### Paragraph-level chunking in practice

Paragraph slices become the default retrieval unit in the community stack: the query helpers point at the chunk index so each BM25 hit already maps to a small span. Because `chunk_index`, `chunk_count`, and `parent_filepath` are stored on each record, governance reviews can jump from a retrieved paragraph to the original file instantly. Adjust `split_into_paragraphs` in `community_version/ingest.py` if your corpus needs a different slicing heuristic (e.g., headers or semantic sentences) while keeping the metadata contract intact.

### Step-by-Step Walkthrough

| Stage                         | What happens                                                                           | Code cue                                   |
| ----------------------------- | -------------------------------------------------------------------------------------- | ------------------------------------------ |
| **1. Bootstrap index**        | Create index with explicit mapping + lowercase normalizer if missing.                  | `ensure_index()`                           |
| **2. Walk the dataset**       | For every `.txt` file, derive `category` from the folder.                              | `glob(os.path.join(DATA_DIR,"*","*.txt"))` |
| **3. Read text**              | Raw file body stored under `content`.                                                  | `p.read_text()`                            |
| **4. Call NER service**       | POST full text to `DEFAULT_URL` (`/ner`); labels are enforced server-side.             | `post_ner()`                               |
| **5. Normalize entities**     | Lowercase + dedupe; produce `explicit_terms` and `explicit_terms_text`.                | `extract_normalized_entities()`            |
| **6. Persist doc**            | Index with stable `_id` = path under `DATA_DIR`; include `ingested_at_ms/doc_version`. | `client.index()`                           |
| **7. Batch refresh**          | One refresh after the loop (not per doc).                                              | `indices.refresh()`                        |
| **8. Logs for observability** | Console prints filepath and extracted entities during ingest.                          | `print(...)`                               |

### Operational Knobs You Control

| Environment variable                 | Purpose                                                     |
| ------------------------------------ | ----------------------------------------------------------- |
| `DATA_DIR`                           | Root directory of raw `.txt` files (organized by category). |
| `INDEX_NAME`                         | Name of the OpenSearch index to create/target.              |
| `OPENSEARCH_HOST/PORT/USER/PASS/SSL` | Connection details for your OpenSearch cluster.             |

> Note: **FUTURE TODO** The ingest script calls a **hardcoded** NER endpoint (`DEFAULT_URL = "http://127.0.0.1:8000/ner"`). Change that constant in the script if your NER service runs elsewhere. The spaCy model is configured on the **NER service** via `SPACY_MODEL`, not in the ingest script.

### Implementation Considerations

The NER step is performed by an **external HTTP service** (spaCy by default). For governance, deploy a **domain-trained** model on that service (product codes, regulation IDs, case numbers, etc.). The current ingest script **does not write** `ner_status` or `ner_model` into each document; auditability is provided by:

* deterministic index mappings (including a lowercase normalizer),
* stable `_id` and `doc_version`,
* `ingested_at_ms` for temporal provenance,
* and NER service configuration/logs (where the active model is declared via `SPACY_MODEL`).

If you want fail-soft behavior when the NER service is unavailable, add exception handling around `post_ner()` and index with empty `explicit_terms`; the reference code **currently requires** the NER service and exits on NER HTTP errors.

## 3. Reinforcement & Data Promotion (HOT → Long)

### Why a feedback loop matters

Reinforcement learning (RL) data contained in HOT would let high-value items **graduate** to long-term (LT). In the **community reference code**, we **do not** mutate LT based on HOT activity. LT remains the authoritative store; HOT is a governed cache that warms via entity-based promotion and **evicts by TTL**. This keeps provenance stable and audits clean. If you later enable HOT → LT, do it as an explicit workflow with defined criteria.

### Signals and scoring (document-RAG style)

There is **no in-index confidence scoring** in the reference implementation.

What we actually track today:

* `hot_promoted_at` (epoch millis) on HOT documents — written during promotion.
* Optional app-level metrics (outside OpenSearch) if you choose to log cache hits or human reviews.
* Eviction is driven by time (TTL), not by a score.

If you need scores later, store them **out-of-band** (analytics DB) and keep long-term documents unchanged.







### Promotion workflow

| Stage                   | Trigger                                  | Action                                                                                                                                      |
| ----------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Cache write**      | App detects entities via `/ner`          | A **HOT-side promotion job** issues `/_reindex` **pulling from LT** with a `terms` filter on `explicit_terms` (OR); sets `hot_promoted_at`. |
| **2. Serve results**    | Subsequent queries                       | Retrieval queries **LT and HOT in parallel**; strictness (AND over entities) is enforced at **query time** via `terms_set`.                 |
| **3. Evict**            | `hot_promoted_at` older than TTL         | A scheduled `delete_by_query` removes expired HOT docs.                                                                                     |
| **4. Promote HOT → LT** | Sufficient reinforcement **or** human OK | Only then copy into LT (explicit, audited job). The reference code does **not** implement this step automatically.                          |
| **5. Audit**            | After promotion/eviction                 | Evidence comes from reindex payloads/responses, `hot_promoted_at`, and eviction job outputs.                                                |




### Code cues (Python)

**Evict expired HOT docs by TTL (matches reference script)**

```python
from opensearchpy import OpenSearch
import time, os, json

OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9202"))
OPENSEARCH_SSL  = os.getenv("OPENSEARCH_SSL", "false").lower() == "true"

client = OpenSearch(
    hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
    use_ssl=OPENSEARCH_SSL, verify_certs=OPENSEARCH_SSL, timeout=60, max_retries=3, retry_on_timeout=True
)

INDEX_NAME  = "bbc"               # HOT index/alias
TTL_MINUTES = 30                  # align with your ops policy

now_ms      = int(time.time() * 1000)
threshold   = now_ms - TTL_MINUTES * 60 * 1000

body = {
  "query": {
    "bool": {
      "filter": [
        {"exists": {"field": "hot_promoted_at"}},
        {"range":  {"hot_promoted_at": {"lt": threshold}}}
      ]
    }
  }
}

resp = client.delete_by_query(
    index=INDEX_NAME,
    body=body,
    params={"conflicts": "proceed", "refresh": "true", "wait_for_completion": "true"}
)
print(json.dumps(resp, indent=2))
```

> Note: Promotion is executed by a **HOT-side job** using `/_reindex` (remote source = LT) with an entity `terms` filter. The NER service only returns entities; it **does not** promote.

### Tunables you should expose

| Variable / setting            | Default / Source           | What it controls                                                |
| ----------------------------- | -------------------------- | --------------------------------------------------------------- |
| `TTL_MINUTES`                 | Eviction script env/CLI    | Lifetime of HOT records before time-based deletion.             |
| `REINDEX_MAX_DOCS`            | Promotion job env          | Caps number of promoted docs per reindex call.                  |
| `PROMOTE_WINDOW_SECONDS`      | Promotion job env          | Optional recency filter on `ingested_at_ms` during promotion.   |
| `SOURCE_INDEX` / `DEST_INDEX` | Promotion job env          | LT (source) and HOT (dest) index names.                         |
| `REINDEX_REFRESH` / `WAIT`    | Promotion job env          | Reindex endpoint behavior (searchable on completion; blocking). |
| `REQUESTS_PER_SECOND`         | Eviction or reindex params | Rate limiting for delete or reindex tasks.                      |
| `reindex.remote.allowlist`    | HOT config                 | Required allowlist for the LT host:port during promotion.       |

### Governance & safety wins

* **Authoritative remains authoritative.** HOT never overwrites LT automatically. No hidden "vetted" flags, no drifting source of truth.
* **Deterministic lifecycle.** Promotion is explicit (reindex payload + `hot_promoted_at`); eviction is explicit (TTL query).
* **Explainable path.** You can show the exact entity filter that warmed HOT and the job outputs that cleaned it up.

### Field notes

Start with entity-triggered warming and **time-based eviction**. Promotion uses **OR** over `explicit_terms` to avoid false negatives; apply your **AND** constraints at retrieval using `terms_set`. If you later add reinforcement or human approval, keep those signals **outside** the long-term index and promote to LT **only** when there's enough positive reinforcement **or** a trusted human-in-the-loop signs off.

## 4. Implementation Guide

For a reference, please check out the following: [community_version/README.md](./community_version/README.md)

## 5. Conclusion

Document-first RAG gives you more than quick answers—it gives you answers you can **defend**.

* **Transparency & explainability** are built-in: explicit fields (`explicit_terms`, `explicit_terms_text`), entity filters (`terms_set`), and **highlights** show *exactly* why a document matched.
* **Accountability & risk control** improve because retrieval is deterministic (fixed mappings, lowercase normalizer, BM25), and promotion/eviction behavior is configured and reproducible.
* **Compliance** turns into an operational posture: a durable long-term store, a TTL-governed **HOT** cache keyed by `hot_promoted_at`, and observable promotion/eviction events.

Pair that with the dual-store layout—**HOT (unstable) cache** for today's topics plus **durable long-term memory** for provenance—and you get a system that's **responsive now** and **auditable later**.

### Your next steps

1. **Clone the repo and run the stack.** Stand up OpenSearch, create indices, ingest, and run STRICT entity queries (`terms_set`) to see **highlights** in action (with the parallel LT+HOT plan).
2. **Swap the NER.** Configure the NER service (`SPACY_MODEL` or your model) to fit your domain; if you want per-doc audit fields, extend the mapping to include them.
3. **Enable HOT cache (promotion + TTL).** Trigger a **HOT-side** `/_reindex` from **LT** using an entity `terms` filter, stamp `hot_promoted_at`, and schedule TTL eviction. Log hits/validations **out of band**. Promotion **from HOT to LT** happens **only** after sufficient positive reinforcement **or** trusted human approval.
4. **(Optional) Add vectors—explicitly.** If you introduce dense fields, document the scorers and how you mix them with BM25 (e.g., RRF or a linear blend). Keep lexical as the default, explainable baseline.
5. **Share what you learn.** PRs with analyzer tweaks, relevance sweeps, and governance playbooks make the ecosystem better.

Document-based RAG isn't a thought experiment; it's running code with clear governance wins. Spin it up, measure it, and raise the bar for responsible retrieval.
