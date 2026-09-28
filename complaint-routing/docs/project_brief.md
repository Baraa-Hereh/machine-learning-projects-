# Project Brief — Complaint Routing

## Client and problem

| Item | Value |
|---|---|
| Client | Financial services company |
| Problem | Complaints are read and routed to departments by hand |
| Goal | Route each complaint automatically to the right department |
| Input | Complaint text written by the customer (English) |
| Output | Department label + confidence |

## Success metric

| Role | Metric |
|---|---|
| Primary | Macro F1 |
| Secondary | Weighted F1, automation rate |

Macro F1 is primary because classes are highly imbalanced (about 33 to 1).
A model that sends everything to the largest department scores about 48%
accuracy but only 0.07 Macro F1.

## Target

Baseline (TF-IDF + logistic regression, validation): **Macro F1 = 0.736**.

The network must meet all three:

1. Mean Macro F1 over 5 seeds >= **0.766** (baseline + 3 points)
2. The gain must exceed the network's seed spread (max - min),
   measured on the first 5-seed run
3. No seed below the baseline

## Constraints

**Technical**
- CPU only, no GPU
- Latency < 100 ms per complaint
- Model size <= 40 MB

**Business**
- Personal data never leaves the company
- Model routes automatically; no human confirmation
- Low confidence -> human agent (threshold set on validation data)
- Model failure -> all complaints go to a human agent
- Each decision should be explainable (desired, not required)

## Scope

**In scope**
- English complaints
- Trained model + inference script
- Reproducible training script (retrain on new data or departments)
- Evaluation report

**Out of scope**
- Other languages
- User interface
- Integration with company systems
- Automated retraining and monitoring
- Pretrained transformer models (violate size and latency limits)

## Data

- Source: CFPB Consumer Complaint Database (public domain), via the
  Hugging Face copy `davidheineman/consumer-finance-complaints-large`
- Period: 2020-03-01 to 2021-03-31
- Details and all cleaning decisions: `reports/data_audit.md`

## Open questions for client

- Is per-decision explanation a regulatory requirement?
- Would the client relax size and latency limits for higher accuracy
  (pretrained models)?
- Can small departments (student loans, personal loans) be merged?

## Deliverables

Model, inference script, training script, evaluation report, handoff document.
