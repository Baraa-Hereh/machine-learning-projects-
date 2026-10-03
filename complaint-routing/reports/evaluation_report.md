# Complaint Routing — Evaluation Report

**Date:** 2 October 2026
**Delivered model:** TF-IDF (50,000 words and word pairs) + logistic regression, run `tfidf_50k_ng12_cw50_C1`

## 1. Summary

| | Result |
|---|---|
| Test Macro F1 | **0.757** (evaluated once) |
| Validation Macro F1 | 0.756 |
| Original baseline (validation) | 0.736 |
| Target (validation) | 0.766, **not met** (0.9 points short) |
| Best neural network, same features and weights | 0.7466, 0.9 points below the delivered model |
| Model size | 5.4 MB (limit 40 MB) ✔ |
| Latency, CPU | median 1.24 ms, p95 2.05 ms (limit 100 ms) ✔ |

The delivered model improves on the original baseline by 2.1 points and meets both technical constraints with a wide margin. It did not reach the 0.766 target. Error analysis shows the largest remaining source of error is a genuine overlap between two departments (section 7). Section 9 lists options to go further.

## 2. Objective and constraints

- **Task:** route English complaint text to one of nine departments, with a confidence score.
- **Main metric:** Macro F1, so every department counts equally regardless of size. Secondary: Weighted F1 and automation rate.
- **Target:** Macro F1 at least 3 points above the TF-IDF + logistic regression baseline (0.736 → 0.766).
- **Technical constraints:** CPU only, under 100 ms per complaint, model at most 40 MB.
- **Business constraints:** personal data stays in-house; low-confidence complaints go to staff; on model failure all complaints go to staff; explanations are desirable.
- **Out of scope:** pretrained transformer models (they break the size and latency limits), other languages, UI, system integration, automatic retraining.

## 3. Data

Public CFPB consumer complaints, filed 2020-03-01 to 2021-03-31, after removing duplicates, contradictions and empty texts, with personal information masked.

| Split | Period | Complaints |
|---|---|---|
| Train | 2020-03 to 2020-12 | 125,707 |
| Validation | 2021-01 to 2021-02 | 26,509 |
| Test | 2021-03 | 15,609 |

The split is by time, so evaluation reflects how the model performs on complaints filed after its training period. Department sizes are unbalanced (largest to smallest about 33:1).

## 4. Method

1. Baseline first: TF-IDF + logistic regression, plus a majority-class reference (Macro F1 0.070).
2. Neural networks (GRU, mean of word embeddings, attention), each run with 5 seeds, early stopping on validation Macro F1, and the training-set score tracked alongside to detect memorisation.
3. Each experiment changed one thing, with the meaning of every possible outcome written down before running.
4. All model choices were made on validation. The test set was used once, on the frozen model, by a script that refuses to run twice.

## 5. Experiments

### 5.1 Neural networks

| Network | Change | Mean Macro F1 (5 seeds) | Seed spread |
|---|---|---|---|
| GRU | starting point | 0.708 | 1.1 |
| Mean of embeddings | remove word order | 0.705 | 0.2 |
| Attention | learned word weights instead of equal | 0.707 | 0.7 |
| Attention + dropout 0.3 | reduce memorisation | 0.711 | 0.5 |
| Attention + dropout 0.5 | stronger dose | 0.712 | 0.3 |
| Mean + dropout 0.5 + word pairs, 50k vocabulary | same features as baseline | 0.731 | 0.3 |
| … + softer class weights (power 0.5) | same weights as delivered model | 0.747 | 0.5 |

### 5.2 Why the networks fell short

Hypotheses were tested one at a time:

| Hypothesis | Test | Finding |
|---|---|---|
| Word order adds little | remove order (mean of embeddings) | Same score as GRU: order adds nothing for this task |
| Equal word weights dilute key words | attention instead of mean | Same score: not the cause |
| Memorisation caps the score | dropout 0.3 and 0.5 (on the attention network) | Gap to training shrank, validation did not rise: memorisation exists but is not the cap |
| The gap comes from features, not the network | baseline restricted to the network's features (20k single words) | Baseline dropped to 0.714, level with the networks |

A 2×2 check on the baseline isolated the feature that mattered:

| | Single words | Words + pairs |
|---|---|---|
| 20k features | 0.714 | 0.726 |
| 50k features | 0.716 | 0.736 |

**Word pairs** (for example `student loan`, `credit report`) drive the gain, and need enough vocabulary room. Given the same features and class weights, the network matched the linear model to within one point but did not beat it, while being larger, slower and not exactly reproducible on GPU.

### 5.3 Improving the baseline

The baseline over-predicted small departments (high recall, low precision), a sign that fully balanced class weights were too strong.

| Change | Validation Macro F1 | Train-sample Macro F1 | Gap |
|---|---|---|---|
| Original (balanced weights, C = 1) | 0.736 | 0.846 | 0.110 |
| Class weights raised to power 0.5 | **0.756** | 0.874 | 0.117 |
| … with C = 0.25 | 0.750 | 0.804 | 0.054 |
| … with C = 4 | 0.753 | 0.944 | 0.191 |

Softer class weights gave +2.0 points. Regularisation strength `C` controlled memorisation clearly but barely moved validation, so C = 1 was kept. Part of the remaining gap reflects the time shift between training (2020) and validation (2021).

## 6. Final model

### 6.1 Test results (per department)

| Department | Precision | Recall | F1 | Complaints |
|---|---|---|---|---|
| Checking or savings account | 0.73 | 0.80 | 0.76 | 1,142 |
| Credit card or prepaid card | 0.73 | 0.81 | 0.77 | 1,520 |
| Credit reporting | 0.88 | 0.84 | 0.86 | 6,633 |
| Debt collection | 0.78 | 0.77 | 0.78 | 3,094 |
| Money transfer | 0.84 | 0.77 | 0.80 | 878 |
| Mortgage | 0.91 | 0.95 | 0.93 | 1,590 |
| Payday loan, title loan, personal loan | 0.50 | 0.54 | 0.52 | 201 |
| Student loan | 0.71 | 0.76 | 0.73 | 191 |
| Vehicle loan or lease | 0.59 | 0.74 | 0.66 | 360 |
| **Macro average** | 0.74 | 0.77 | **0.757** | 15,609 |

Test (0.757) matches validation (0.756): the validation estimate was reliable and the model held up on a later month. Per-department scores moved by at most about 3 points; the smallest departments have only ~200 test complaints, so their scores are less certain.

### 6.2 Confidence and thresholds

Confidence was checked against actual accuracy on validation. Between confidence 0.4 and 0.8 the model is slightly under-confident; above 0.8 it is well matched. The threshold table (actual accuracy, validation):

| Threshold | Routed automatically | Accuracy on those | Sent to staff |
|---|---|---|---|
| 0.5 | 87.3% | 86.8% | 12.7% |
| 0.6 | 78.6% | 89.1% | 21.4% |
| 0.7 | 68.8% | 91.4% | 31.2% |
| 0.8 | 57.3% | 93.4% | 42.7% |
| 0.9 | 40.3% | 95.7% | 59.7% |

The threshold is a business decision: it trades staff workload against misrouted complaints.

### 6.3 Constraints

- **Size:** 5.4 MB.
- **Latency:** measured on 500 validation complaints after warm-up: median 1.24 ms, p95 2.05 ms, max 5.79 ms. The model must be loaded once and kept in memory.

## 7. Error analysis (validation)

| True department | Predicted | Errors |
|---|---|---|
| Credit reporting | Debt collection | 1,019 |
| Debt collection | Credit reporting | 615 |
| Credit reporting | Credit card | 338 |
| Credit reporting | Mortgage | 190 |
| Credit card | Credit reporting | 189 |
| Credit reporting | Vehicle loan | 184 |

- The credit reporting ↔ debt collection pair accounts for about a third of all errors; the six largest pairs, all involving credit reporting, for more than half.
- **Cause:** an unpaid debt sent to collection usually also appears on the credit report, so one complaint often describes both, and the consumer chooses the department when filing.
- A sample of 10 complaints from the largest pair was classified independently by two readers, who agreed on 7 of 8 rated cases. About one error in nine was a clear model error; the rest were genuine overlap or uninformative text.
- Explanations exposed the same overlap elsewhere: the word `student` pushes towards credit reporting, because many credit-reporting complaints mention student loans.

**Limits of this analysis:** a small sample, and readers without financial-domain expertise. A larger review by a domain expert would give a firmer estimate of the achievable ceiling.

## 8. Conclusion

The delivered model reaches Macro F1 0.757 on test, 2.1 points above the original baseline, within all technical constraints, with explanations and a confidence score for every decision. The 0.766 target was not met. The evidence points to category overlap in the labels, rather than model capacity, as the main limit: neural networks given the same information did not do better.

## 9. Options to go further

| Option | Expected effect | Cost |
|---|---|---|
| Route low-confidence complaints to staff (threshold) | Higher accuracy on automated complaints (e.g. 89% at threshold 0.6) | Staff time on the remainder |
| Expert review of labels for the overlapping departments | Firmer ceiling estimate; cleaner training data | Domain-expert time |
| Merge or redefine overlapping departments | Removes the largest error source | Process change |
| Pretrained language model | Likely higher accuracy | Breaks size and latency limits; needs GPU or relaxed limits |

## 10. Open questions for the client

1. What threshold fits your staff capacity?
2. Are explanations required for every decision by regulation?
3. Would you relax the size and latency limits for higher accuracy?
4. Could small departments (student and personal loans) be merged, or credit reporting and debt collection be redefined?

## 11. Limitations

- Trained on 2020 to early 2021 complaints; topics shift over time, so performance should be monitored after deployment.
- English only.
- Related word forms are treated as separate words.
- Neural-network results on GPU are not bit-for-bit reproducible with the same seed; the delivered model is.
