# InSARBench: A Benchmark for Source-Aware Retrieval in Insurance Documents

InSARBench is a Chinese insurance evidence-retrieval benchmark for studying evidence localization and source consistency across insurers. The resource includes a passage corpus, an original question set, two paired challenge sets, document summaries, and the core source-aware retrieval modules.

Resource publication date: 2026-11-02.

## Repository contents

| File | Contents |
| --- | --- |
| `corpus.jsonl.gz.part0001` ... `corpus.jsonl.gz.part0010` | Ten ordered parts of the compressed corpus: 130,473 passages from 1,700 documents, covering 40,590 document-page pairs and 17 insurers |
| `questions_original.json` | 1,908 questions with reference answers and evidence annotations |
| `questions_colloquial.json` | 500 paired questions: 400 colloquial rewrites and 100 unchanged controls |
| `questions_implicit_source.json` | The same 500 IDs, with insurer names removed from 200 questions and accompanying fluency adjustments |
| `document_profiles.json` | Document metadata and summaries: 1,773 records covering 1,700 distinct filenames |
| `data_manifest.json` | Record counts, data transformations, part order, and SHA-256 checksums |
| `DATA_SHA256SUMS.txt` | Checksums for the distributed data files |
| `code/source_aware_retriever.py` | Query-conditioned company filtering, three-channel fusion, reranking, and relevance verification/retry |
| `code/build_metadata_index.py` | Metadata augmentation and index construction |

The code files were renamed from `ours_agent.py` and `init_embedding_index.py`, respectively. Data filenames and data contents are unchanged by this organization.

## 1. Download and reconstruct the corpus

Download the repository ZIP from the anonymous resource page and extract it. Run the following Python 3 code from the extracted repository root.

The ten corpus parts are consecutive byte ranges of one gzip archive. Join them in the manifest order before decompressing; they are not independently extractable archives. Downloading the repository ZIP retrieves all parts together.

```python
from pathlib import Path
import gzip
import hashlib
import json
import shutil

root = Path(".")
manifest = json.loads((root / "data_manifest.json").read_text(encoding="utf-8"))
spec = manifest["archives"]["corpus.jsonl.gz"]

digest = hashlib.sha256()
with (root / "corpus.jsonl.gz").open("wb") as out:
    for name in spec["parts"]:
        data = (root / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == spec["part_checksums"][name], name
        out.write(data)
        digest.update(data)
assert digest.hexdigest() == spec["sha256"]

with gzip.open(root / "corpus.jsonl.gz", "rb") as src:
    with (root / "corpus.jsonl").open("wb") as dst:
        shutil.copyfileobj(src, dst)

assert hashlib.sha256((root / "corpus.jsonl").read_bytes()).hexdigest() == spec["uncompressed_sha256"]
for name, info in manifest["files"].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == info["sha256"], name
print("Data verified; corpus.jsonl is ready.")
```

## 2. Understand the data

### Passage corpus

Each JSONL record has `id`, `text`, and `source`. Source metadata includes `company`, `kind`, `type`, `product_name`, `category_paths`, `filename`, and `page`.

### Question sets

Each question contains `id`, `question`, `answer`, `category`, `granularity`, and `evidence`. Evidence entries include `filename`, `file_page`, and document attributes. Match evidence to passages by the exact **(filename, page)** pair, not by filename alone.

Shared IDs connect each challenge question to its original question and to the other challenge variant. Answers and evidence labels are unchanged across variants. The implicit-source set has 248 question texts that differ from the colloquial set, including the 200 insurer-name removals and fluency adjustments.

Only local `evidence.file_path` values were removed when preparing the question files. Other question/evidence fields are preserved, and every annotated document-page pair resolves to the corpus.

### Document profiles

Profiles contain `filename`, `company`, `kind`, `type`, `product_name`, and `summary`. All 1,773 rows are retained, including multiple records for some filenames. For the supplied summary-index workflow, index the records as supplied; document-level fusion maps the ranked summary results back to filenames. Do not silently substitute a different summary-selection policy.

## 3. Prepare the retrieval runtime

The two code files are integration modules for an existing retrieval runtime. Before using the index and retrieval examples, make the following modules importable in your Python environment, for example by adding the runtime's module directory to `PYTHONPATH`.

| Runtime module | Interface used by the supplied source |
| --- | --- |
| `config` | Milvus connection, embedding/reranking configuration, retrieval limits, model paths, and LLM configuration |
| `utils` | `load_json`, `get_last_name`, and passage/profile conversion helpers |
| `retriever_milvus` | `Retriever` and `Reranker` |
| `query_metadata` | `predict_query_metadata(question)`, returning a `predictions` object with a nullable `company` |
| `chatbot` | `ChatModel.async_chat(messages, is_response_json=True)` |

The retrieval runtime uses Milvus with dense-vector and BM25 retrieval, an embedding encoder, a reranker, and an LLM endpoint for metadata inference and relevance verification. Use your own service credentials in the runtime configuration.

Set the metadata candidate source to the reconstructed corpus, and use the **same encoder and metadata representation for indexing and querying**. The local workflow uses BGE-large-zh-v1.5-compatible embeddings and a BGE-reranker-large-compatible reranker. Record the exact encoder checkpoint, reranker, LLM identifier, and runtime parameters for each run; replacing a checkpoint defines a different experimental configuration.

## 4. Build the passage and summary indexes

`code/build_metadata_index.py` contains `add_text_metadata` and `run`. It uses the configured `Retriever` backend.

Prepare two indexes using the same embedding configuration:

1. **Passage index:** load `corpus.jsonl`, augment the passage text with its category path and filename, then index the passages.
2. **Summary index:** load `document_profiles.json`, augment each summary with company, kind, type, product name, and filename, then index the profiles.

Before running the script, edit its `run()` settings for the chosen input. For the passage pass, use:

```python
from pathlib import Path

corpus_path = str(Path("corpus.jsonl").resolve())
corpus_data_name = "corpus"
is_text_add_metadata = True
```

For the summary pass, use:

```python
corpus_path = str(Path("document_profiles.json").resolve())
corpus_data_name = "profiles"
is_text_add_metadata = True
```

These values replace the corresponding assignments inside `run()`; they are not command-line flags. With metadata augmentation enabled, the script appends `_metadata`, giving collection bases `corpus_metadata` and `profiles_metadata`. Also set `embedding_model` to the model identifier used by your runtime. The backend may append model-specific suffixes to the collection names, so reuse the same model and collection arguments at query time.

Run once for each configured input:

```text
python code/build_metadata_index.py
```

The source script defaults to `force_rebuild_index = True`, which recreates the selected collection. Use dedicated collection names for this benchmark. For an existing populated index, set this flag to `False` before running. Confirm that the indexes contain 130,473 passage records and 1,773 profile records.

The metadata augmentation in this source serializes list-valued passage metadata with underscores, prepends the metadata to the indexed text, and retains the source attributes used for filtering and evidence matching.

## 5. Run source-aware retrieval

The main entry point is `new_try` in `code/source_aware_retriever.py`.

The workflow is:

1. Predict query metadata from the question text. Apply a hard company filter only when a company is predicted; otherwise search without that filter.
2. Retrieve 500 dense passage candidates, 500 BM25 passage candidates, and up to 500 hybrid summary results.
3. Fuse passage rankings with document-summary rankings using reciprocal-rank contributions. The main call uses `k = 60` and `doc_weight = 0.5`: each passage channel has weight 0.25 and the document channel has weight 0.5.
4. Rerank the fused candidates. The number passed to the reranker is controlled by the backend's `rerank_num` setting.
5. Verify the top ten retrieved passages with the LLM. If no passage is judged relevant, rewrite the sparse and dense queries, retrieve again, rerank, and remove previously rejected passage IDs. The retry limit is three.
6. Return ranked passage results, summary results, and query metadata.

Verification is triggered for every question from retrieved content. No gold filename, answer, or evidence label is read by `new_try`. If the verification service fails, the source retains the current retrieval results.

Example, after configuring the runtime and building the two indexes:

```python
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("code").resolve()))
from source_aware_retriever import new_try
from retriever_milvus import Retriever, Reranker

# Match the model identifier and collection bases used in the index-building step.
embedding_model = "BAAI/bge-large-zh-v1.5"
passages = Retriever(embedding_model, collection_name="corpus_metadata")
profiles = Retriever(embedding_model, collection_name="profiles_metadata")
reranker = Reranker("bge-reranker-large")

qa_path = Path("questions_original.json")
questions = json.loads(qa_path.read_text(encoding="utf-8"))
results = []
for item in questions:
    # Pass question text only; keep annotations outside the retrieval call.
    prediction = new_try(
        {"question": item["question"]},
        passages,
        profiles,
        reranker,
    )
    results.append({
        "id": item["id"],
        "question": item["question"],
        "search_results": prediction["search_results"][:20],
        "query_metadata": prediction["query_metadata"],
    })

Path("retrieval_results.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
```

Run the same procedure on `questions_colloquial.json` and `questions_implicit_source.json`, writing separate result files. `new_try_no_verify` provides a diagnostic configuration that omits relevance verification and retries.

## 6. Evaluate and compare runs

Join predictions to the selected question file by `id`. Use gold evidence only in this evaluation stage. Keep passage ranking positions intact: do not collapse repeated document pages before selecting the top-k passages.

| Measure | Definition |
| --- | --- |
| Recall@1/5/10/20 | Fraction of annotated evidence pages covered by the first k passages, averaged over questions |
| MRR@20 | Reciprocal rank of the first evidence-page match within the first 20 passages; zero when absent |
| WrongSrc@10 | Fraction of the returned top-ten passages whose company labels have no overlap with the evidence companies, averaged over questions |
| AnyWrongSrc@10 | Fraction of questions with at least one wrong-company passage in the first ten results |
| DistinctPages@10 | Number of distinct (filename, page) pairs among the first ten passages, averaged over questions |

For WrongSrc, use the actual number of returned passages (at most ten) as the per-question denominator. Empty results have Recall/MRR zero and WrongSrc zero; report empty-result counts separately so that absence of results is not interpreted as successful source selection. Missing company metadata should be checked rather than counted as a correct source.

For robustness comparisons, evaluate the original questions restricted to the shared 500 IDs as well as both variants. For a company-removal analysis, identify the 200 intended insurer-removal cases separately from additional fluency changes.

Candidate-exclusion diagnostics require recording the candidate documents before passage ranking. Compute FalseExcl from whether the candidate filter removes all annotated evidence documents; it cannot be recovered reliably from final top-k results alone.

## Run records

For each experiment, record the question filename/hash, corpus hash, encoder and model versions, metadata augmentation setting, index identifiers, retrieval/fusion parameters, reranking limit, verification/retry settings, and per-query ranked results. The data checksums in `data_manifest.json` provide the reference data version.
