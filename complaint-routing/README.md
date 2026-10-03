# Complaint Routing

Routes English consumer-finance complaints to one of nine departments, with a confidence score, a runner-up department, and the words that drove the decision.

**Delivered model:** TF-IDF (words and word pairs) + logistic regression.
**Test Macro F1:** 0.757 (validation 0.756). **Size:** 5.4 MB. **Latency:** median 1.2 ms, p95 2.1 ms per complaint on CPU.

See `reports/evaluation_report.md` for the full evaluation.

---

## Departments

Checking or savings account · Credit card or prepaid card · Credit reporting, credit repair services, or other personal consumer reports · Debt collection · Money transfer, virtual currency, or money service · Mortgage · Payday loan, title loan, or personal loan · Student loan · Vehicle loan or lease

---

## Setup

Python 3.12, Linux or WSL2.

```bash
python -m venv ~/venvs/complaint-routing
source ~/venvs/complaint-routing/bin/activate
pip install -r requirements.txt
```

Inference needs only `scikit-learn`, `joblib`, `numpy`, `pandas` and `pyyaml`. TensorFlow is listed because the repository also contains the neural-network experiments.

### GPU note (neural experiments only)

On WSL2 with `tensorflow[and-cuda]`, TensorFlow may fail to load `libcusolver.so.11` even though the file is installed. Add its folder to the library path in the environment's activate script:

```bash
echo 'export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cusolver/lib:$LD_LIBRARY_PATH"' >> ~/venvs/complaint-routing/bin/activate
```

For Jupyter, register a kernel that carries the same variable:

```bash
python -m ipykernel install --user --name complaint-routing \
  --display-name "Python (complaint-routing GPU)" \
  --env LD_LIBRARY_PATH "$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cusolver/lib"
```

---

## Usage

### In your service (recommended)

Load the model **once** at startup and keep it in memory. Loading takes seconds; routing one complaint takes about 1–2 ms.

```python
from pathlib import Path
from predict import load_model, route

model, names = load_model(Path("configs/config.yaml"))
result = route(model, names, "I paid off my student loan but it still shows as late on my credit report")
```

### From the command line (testing only)

```bash
python src/predict.py "complaint text here"
```

This reloads the model on every call, so it is slow by design. Do not use it per complaint in production.

### Output

```json
{
  "department": "Credit reporting, credit repair services, or other personal consumer reports",
  "probability": 0.656,
  "runner_up": "Student loan",
  "runner_up_probability": 0.33,
  "top_words": ["student", "report", "reports", "late", "credit"],
  "latency_ms": 1.2
}
```

| Field | Meaning |
|---|---|
| `department` | Predicted department |
| `probability` | Model confidence for that department (0–1) |
| `runner_up`, `runner_up_probability` | Second most likely department, useful for ambiguous complaints |
| `top_words` | Up to five words or word pairs that pushed the decision towards `department` |
| `latency_ms` | Routing time for this complaint (command line only) |

---

## Choosing a confidence threshold

The model routes a complaint automatically when its confidence is at or above a threshold you choose; the rest go to staff. Measured on the validation set (26,509 complaints):

| Threshold | Routed automatically | Accuracy on those | Sent to staff |
|---|---|---|---|
| 0.5 | 87.3% | 86.8% | 12.7% |
| 0.6 | 78.6% | 89.1% | 21.4% |
| 0.7 | 68.8% | 91.4% | 31.2% |
| 0.8 | 57.3% | 93.4% | 42.7% |
| 0.9 | 40.3% | 95.7% | 59.7% |

Pick the highest threshold your staff capacity allows. Example: if staff can review at most 25% of complaints, use 0.6.

Between confidence 0.4 and 0.8 the model is slightly under-confident (it is right more often than its score says); above 0.8 the scores match reality. The table above uses actual accuracy, so it does not depend on this.

If the model cannot be loaded or fails, route all complaints to staff.

---

## Reproducing the results

First get the raw data: download the 11 parquet files from the [dataset page](https://huggingface.co/datasets/davidheineman/consumer-finance-complaints-large/tree/main) and place them **directly** in `data/raw/cfpb_hf/` (no subfolders). This is how the delivered results were produced.

Alternatively, with `huggingface_hub`:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('davidheineman/consumer-finance-complaints-large', repo_type='dataset', local_dir='data/raw/cfpb_hf', allow_patterns='*.parquet')"
```

This keeps the dataset's own folder layout. If the files end up in a subfolder, move them directly into `data/raw/cfpb_hf/` so they match `data.raw_glob` in `configs/config.yaml`.

Then:

```bash
python src/pipeline.py          # extract and clean data, build time-based splits
python src/baseline.py          # train and save the delivered model, report validation scores
python src/evaluate_test.py     # evaluate once on the test set (refuses to run twice)
```

The delivered model is fixed by these settings in `configs/config.yaml` under `baseline`:

```yaml
max_features: 50000
ngram_range: [1, 2]
min_df: 2
class_weight_power: 0.5
C: 1.0
```

The model file and reports are named from these settings (`tfidf_50k_ng12_cw50_C1`). Changing them trains a different model under a different name.

Logistic regression has no randomness, so retraining gives the same scores exactly.

### Neural-network experiments

```bash
python src/train.py             # settings under model and train in configs/config.yaml
```

Run names are built from the settings (for example `mean_sdrop50_ng12_v50k_cw50`). Results on GPU vary slightly between runs even with the same seed.

---

## Data

Public CFPB consumer complaints (US government work, public domain), from the Hugging Face dataset [`davidheineman/consumer-finance-complaints-large`](https://huggingface.co/datasets/davidheineman/consumer-finance-complaints-large), filed 2020-03-01 to 2021-03-31. Personal information is masked during cleaning. The raw parquet files go in `data/raw/cfpb_hf/`; `src/pipeline.py` builds the cleaned data and the splits from them.

Project scope and requirements: [`docs/project_brief.md`](docs/project_brief.md). Data quality checks: [`reports/data_audit.md`](reports/data_audit.md).

| Split | Period | Complaints |
|---|---|---|
| Train | 2020-03 to 2020-12 | 125,707 |
| Validation | 2021-01 to 2021-02 | 26,509 |
| Test | 2021-03 | 15,609 |

`data/` and `models/` are generated and not stored in git.

---

## Known limitations

- **Credit reporting and debt collection overlap.** A collection debt usually also appears on the credit report, so one complaint often describes both. This pair causes about a third of all errors. See the evaluation report.
- Small departments (Payday loan, Vehicle loan) have the lowest scores (F1 0.52 and 0.66 on test).
- Trained on 2020–early 2021 complaints. Topics shift over time; performance should be re-checked on recent data before and after deployment.
- English only.
- The model treats related word forms (for example `report` and `reports`) as different words.

---

## Project structure

```
complaint-routing/
├── configs/config.yaml        settings for data, baseline and neural models
├── docs/                      project brief, delivery notes
├── notebooks/                 data audit, baseline analysis, attention and error analysis
├── reports/                   evaluation report and run reports (json)
├── src/
│   ├── extract.py, clean.py, pipeline.py   data preparation
│   ├── baseline.py            delivered model: train and validate
│   ├── evaluate_test.py       one-time test evaluation
│   ├── predict.py             routing for new complaints
│   ├── model.py, train.py     neural-network experiments
├── tests/
└── requirements.txt
```
