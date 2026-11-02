# InSARBench: Insurance Evidence Retrieval Benchmark

This resource provides the retrieval corpus, three question sets, and document metadata/summaries for InSARBench.

Published: 2026-11-02. This is the actual resource publication date. Resource checksums and transformations are recorded in `data_manifest.json` and `DATA_SHA256SUMS.txt`.

## Files

| File | Description |
| --- | --- |
| `corpus.jsonl.gz.part0001` through `corpus.jsonl.gz.part0010` | Ten consecutive parts of a losslessly compressed corpus: 130,473 passages, 1,700 documents, 40,590 document-page pairs, 17 insurers |
| `questions_original.json` | 1,908 original questions, answers, categories and evidence annotations |
| `questions_colloquial.json` | 500 paired questions, comprising 400 rewrites and 100 unchanged controls |
| `questions_implicit_source.json` | The same 500 IDs; insurer names removed from 200 questions, plus fluency adjustments; 248 question texts differ from the colloquial set |
| `document_profiles.json` | 1,773 metadata/summary records covering 1,700 distinct document filenames; duplicates retained |
| `data_manifest.json` | Counts, transformations, source/release SHA-256 hashes, corpus part order and hashes |
| `DATA_SHA256SUMS.txt` | SHA-256 checksums for all data release files |

## Download and reconstruct the corpus

Download the repository ZIP from the anonymous resource page, extract it, and open a terminal in the extracted directory. The ten parts exist because the anonymous service limits individual file sizes. They are byte ranges of one gzip file, not separate archives. You do not need to download them individually if you use the ZIP download.

Run this with Python 3:

```python
from pathlib import Path
import gzip, hashlib, json, shutil
root = Path('.')
manifest = json.loads((root / 'data_manifest.json').read_text(encoding='utf-8'))
spec = manifest['archives']['corpus.jsonl.gz']
digest = hashlib.sha256()
with (root / 'corpus.jsonl.gz').open('wb') as out:
    for name in spec['parts']:
        data = (root / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == spec['part_checksums'][name], name
        out.write(data)
        digest.update(data)
assert digest.hexdigest() == spec['sha256']
with gzip.open(root / 'corpus.jsonl.gz', 'rb') as src, (root / 'corpus.jsonl').open('wb') as dst:
    shutil.copyfileobj(src, dst)
assert hashlib.sha256((root / 'corpus.jsonl').read_bytes()).hexdigest() == spec['uncompressed_sha256']
print('Verified corpus.jsonl is ready.')
```

## Schema and evidence matching

Corpus JSONL records contain `id`, `text`, and `source`. Source fields include `company`, `kind`, `type`, `product_name`, `category_paths`, `filename`, and `page`.

Question JSON records contain `id`, `question`, `answer`, `category`, `granularity`, and `evidence`. Evidence entries contain `filename`, `file_page`, `company`, `kind`, `type`, and `product_name`. Match passages to evidence using both the exact filename and page number. Ranking positions are passage positions; do not deduplicate pages before Recall/MRR scoring. Shared question IDs link both challenge variants to the original set. Reference answers and evidence labels are unchanged across variants. All annotated filenames and pages resolve to the corpus.

Only local `evidence.file_path` values were removed from question records. IDs, question text, reference answers, categories and all other evidence fields are preserved. Corpus and document-profile content is preserved without changes. Raw PDFs are not included.

Document profiles contain `filename`, `company`, `kind`, `type`, `product_name`, and `summary`. The 73 records beyond the unique document count are retained to preserve the original summary resource; the method release must document the actual indexing and duplicate-handling policy.

## Reuse terms

No new data/model license is assigned by this upload. Distribution permission has been confirmed by the resource contributors; reuse terms and upstream model licensing should be documented separately. The OpenReview submission license should not be assumed to license these resources.
