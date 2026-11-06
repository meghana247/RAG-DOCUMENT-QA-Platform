# Retrieval evaluation

Scores the retrieval pipeline against a fixed corpus and a labelled question set,
so a retrieval change is judged by numbers rather than by reading a few answers.

```bash
make up                                      # or any migrated Postgres + pgvector
python -m eval.run_eval                      # index eval/corpus if needed, then score
python -m eval.run_eval --reingest           # after changing chunking or the embedding model
```

It needs only `DATABASE_URL`. No LLM is called and no real secrets are required;
placeholders are filled in for settings that retrieval never uses.

## Corpus

Three NIST publications, chosen because they are public, stable, easy to explain,
and hard in the ways real documents are hard: dense tables of identifiers
(`GOVERN 1.6`, `GV.SC`), heavy acronym use, and two documents (the AI RMF and its
Generative AI Profile) that cover overlapping ground and compete for the same
queries.

| File | Document | Pages | Chunks |
|---|---|---|---|
| `nist-ai-rmf-1.0.pdf` | [NIST AI 100-1](https://doi.org/10.6028/NIST.AI.100-1), AI Risk Management Framework 1.0 (Jan 2023) | 48 | 122 |
| `nist-ai-600-1-genai-profile.pdf` | [NIST AI 600-1](https://doi.org/10.6028/NIST.AI.600-1), AI RMF Generative AI Profile (Jul 2024) | 64 | 184 |
| `nist-csf-2.0.pdf` | [NIST CSWP 29](https://doi.org/10.6028/NIST.CSWP.29), Cybersecurity Framework 2.0 (Feb 2024) | 32 | 80 |

Works of the US federal government are not subject to copyright in the United
States (17 U.S.C. § 105), so the files are committed as-is.

## Golden set

`golden_set.json` holds 36 questions: 31 answerable and 5 that the corpus cannot
answer.

| Category | Count | What it tests |
|---|---|---|
| exact-identifier | 6 | `GOVERN 1.6`, `GV.SC`: tokens embeddings handle poorly |
| acronym | 3 | TEVV, CBRN, ERM |
| definition | 8 | direct "what is X" questions |
| paraphrase | 10 | wording that shares few words with the source passage |
| multi-hop | 4 | answers that need two separate passages |
| unanswerable | 5 | off-topic questions where the right result is *nothing* |

Each answerable question lists **evidence items**: facts a correct answer needs.
A chunk satisfies an item when it comes from the named document and contains every
listed substring (ignoring case, whitespace and ligatures). An item can list
`any_of` alternatives when the corpus states a fact in more than one place, such as
a definition in prose and again in a summary table.

Relevance is judged per evidence item, not per chunk. Chunks overlap, so one fact
often appears in two neighbouring chunks; counting both as separately relevant
would penalise MMR for removing the duplicate. As a result every metric is bounded
by 1.0.

Before scoring, every label is checked against the indexed chunks. A label that
matches nothing would read as a retrieval miss when it is really a labelling or
chunking error, so the run stops instead.

## Metrics

| Metric | Meaning |
|---|---|
| hit@k | question had at least one relevant passage in the top k |
| recall@k | share of the question's evidence items found in the top k |
| precision@k | relevant passages / k |
| MRR | 1 / rank of the first relevant passage |
| nDCG | rank-weighted quality: relevant passages near the top score higher |
| refusal | unanswerable questions where retrieval returned nothing |

Ranking metrics average over the answerable questions only; refusal is reported on
its own.

## Latest results

`top_k = 6`, `all-MiniLM-L6-v2` embeddings, `ms-marco-MiniLM-L-6-v2` reranker, CPU.
The full per-question output is in `results/retrieval.json`.

| Configuration | hit@6 | recall@6 | MRR | nDCG | refusal | p50 ms |
|---|---|---|---|---|---|---|
| vector only | 0.581 | 0.532 | 0.399 | 0.426 | 0.80 | 11 |
| keyword only | 0.226 | 0.226 | 0.210 | 0.214 | 1.00 | 4 |
| hybrid (RRF) | 0.645 | 0.597 | 0.455 | 0.484 | 0.80 | 10 |
| hybrid + MMR | 0.645 | 0.597 | 0.396 | 0.442 | 0.80 | 13 |
| **hybrid + MMR + rerank** (production default) | **0.645** | **0.613** | **0.539** | **0.540** | **0.80** | 569 |

What the numbers say:

- **Hybrid beats either retriever alone.** It finds the right passage for 64.5% of
  questions, against 58.1% for vector-only and 22.6% for keyword-only.
- **The reranker improves ordering, not coverage.** Hit rate is unchanged, but the
  first relevant passage moves up (MRR 0.396 → 0.539), at about 0.5 s of CPU per
  query.
- **MMR on its own lowers MRR** (0.455 → 0.396) on this set. It only pays for
  itself once the reranker runs after it.
- **Keyword-only search is weak** because `websearch_to_tsquery` requires *every*
  question word to appear in a chunk. This is the clearest thing to improve next.
- **One off-topic question gets through:** "What is the maximum GDPR fine for a
  data breach?" clears the similarity floor on generic privacy passages.

With 31 answerable questions, one question is worth about 0.03 of hit rate, so
treat differences smaller than that as noise.

## Adding questions

1. Find the answer passage in the corpus and copy a short, distinctive phrase from
   it as a `contains` substring.
2. Run the eval. If the label matches no chunk, the run says so and names it.
3. When a miss looks wrong, check the `retrieved` list in `results/retrieval.json`.
   If retrieval returned a passage that genuinely answers the question, add it as
   an `any_of` alternative; if not, it is a real miss and the label stays.
