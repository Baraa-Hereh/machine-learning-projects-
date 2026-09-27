# Data Audit Report — Complaint Routing

## 1. Source

| Item | Value |
|---|---|
| Origin | CFPB Consumer Complaint Database (U.S. federal work, public domain) |
| Copy used | Hugging Face `davidheineman/consumer-finance-complaints-large`, 11 parquet shards |
| Period | 2020-03-01 to 2021-03-31 |
| Why this period | Same window as public reference work, so results can be sanity-checked |

**Why not other sources**

| Source | Reason rejected |
|---|---|
| Kaggle preprocessed copy | Text already lowercased, stripped of punctuation, digits and stopwords by an undocumented process; cannot be reproduced at inference time |
| Official CFPB download | Narratives were removed from public distribution in September 2026 |
| Kaggle raw copy | 500-row sample only |
| Data Rescue Project archive | Requires login; downloads kept failing |

**Note:** shards are not random samples. Row counts per shard for the target
period ranged from 714 to 47,738. All 11 shards are used.

## 2. Cleaning pipeline

Run with `python src/pipeline.py`. Steps, in order:

| # | Step | Rows after | Removed |
|---|---|---|---|
| 1 | Extract period + non-empty narrative | 201,746 | — |
| 2 | Parse dates, drop invalid | 201,746 | 0 |
| 3 | Unique `complaint_id` | 201,746 | 0 |
| 4 | Mask PII with `[PII]` | 201,746 | 0 |
| 5 | Drop texts that appear under more than one product | 198,832 | 2,914 |
| 6 | Drop duplicate texts, keep earliest | 167,825 | 31,007 |

**Order rationale**
- PII masking runs before duplicate checks: two texts differing only in an
  email become identical after masking.
- Conflicts are removed before duplicates: deduplicating first would keep one
  copy with an arbitrary label and hide the conflict.
- The earliest copy is kept: a text's date is when it first appeared, which
  keeps the time split honest.
- Everything that removes rows runs before the split, to prevent the same
  text landing in both train and test.

**PII found and masked**

| Type | Matches |
|---|---|
| CFPB `XXXX` masks | 2,543,867 |
| Email | 52 |
| Phone | 6 |
| Digit runs of 8+ | 35 |

CFPB redaction is broad but not complete. The same masking must run in the
inference script, since new complaints will not be pre-masked.

## 3. Language

No non-Latin script found. 242 texts flagged for few common English words
were inspected: all English, heavily masked. Nothing removed.

## 4. Text length (words, first shard)

| Percentile | 50% | 75% | 90% | 95% | 99% | Max |
|---|---|---|---|---|---|---|
| Words | 139 | 248 | 395 | 496 | 973 | 5,613 |

**Decision:** `max_len = 400`. Covers 90% of complaints in full; longer ones
are truncated, not dropped. To be validated against the 100 ms latency limit.

## 5. Split

Time-based, not random: the model is trained on the past and used on the
future, and complaint topics shift over time.

| Split | Period | Rows |
|---|---|---|
| Train | 2020-03 to 2020-12 | 125,707 |
| Val | 2021-01 to 2021-02 | 26,509 |
| Test | 2021-03 | 15,609 |

## 6. Class distribution

| Product | Train | Val | Test |
|---|---|---|---|
| Credit reporting | 60,531 | 12,109 | 6,633 |
| Debt collection | 20,366 | 4,817 | 3,094 |
| Credit card or prepaid card | 14,383 | 2,713 | 1,520 |
| Mortgage | 10,734 | 2,507 | 1,590 |
| Checking or savings account | 8,281 | 1,904 | 1,142 |
| Money transfer / virtual currency | 4,406 | 1,067 | 878 |
| Vehicle loan or lease | 3,223 | 668 | 360 |
| Payday / title / personal loan | 1,937 | 369 | 201 |
| Student loan | 1,846 | 355 | 191 |

- Imbalance: about 33 to 1 between the largest and smallest class.
- A model predicting "Credit reporting" for everything scores about 48%
  accuracy on train, which is why Macro F1 is the primary metric.
- Distribution drifts over time: Credit reporting falls from 48% to 42%;
  Debt collection and Money transfer rise.
- Smallest classes have about 200 test examples; per-class F1 will be
  reported with confidence intervals.

## 7. Known limitations

- Labels are chosen by the consumer, not by an expert; some are noisy
  (e.g. 1,089 complaints with sub-product "I do not know" in the first shard).
- The period is exceptional (COVID-19). Retraining on a broader, more recent
  window is recommended before production use; the training script supports it.
- Categories follow the pre-August-2023 complaint form.

## 8. Open questions for client

- Is per-decision explanation a regulatory requirement?
- Would the client relax size/latency limits for pretrained models?
- Can small departments (student loans, personal loans) be merged?
