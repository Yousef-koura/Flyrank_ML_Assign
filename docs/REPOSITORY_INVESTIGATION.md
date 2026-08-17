# FlyRank ML Internship — Repository Technical Investigation

- **Date of investigation:** 2026-08-17
- **Branch inspected:** `main` (commit `fdb7009`, "Complete Week 3 Search Intelligence Data Contract")
- **Method:** Static inspection only. No code was executed against the warehouse; the reference pipeline was **not** re-run (no `data/processed/`, `outputs/model_results.json`, or `outputs/summary.json` exist in this clone, so the committed outputs could not be regenerated here). Claims about the raw CSV were verified with read-only pandas. Notebook execution state was verified from each notebook's saved outputs and `execution_count` metadata.
- **How to read this report:** Facts verified from repo evidence are stated directly. Items I could not verify are marked `Not verified from repository evidence.` "Inference" marks analytical conclusions, not observed facts.

---

## 1. What this project is

This is the **starter repository for the FlyRank ML Internship** (Applied Search Intelligence track — Google Search ranking & discoverability). It is a template repo: each intern clones it, does weekly assignments in `work/`, and submits their own repo URL. The internship asks each intern to pick an analysis "lane", then follow a fixed workflow:

```
research question -> data contract -> signal audit -> baseline score -> lane model
-> validation -> ranked action recommendations -> deployed research paper
```

The educational payload is:

- **A runnable reference pipeline** (`scripts/01–05` + `run_all.py`) that demonstrates the whole workflow on a small anonymized starter dataset, and whose committed outputs serve as "the shape of what your capstone should produce".
- **Three guided starter notebooks** (`notebooks/01–03`) that walk through running the pipeline, reading a decision tree, a leakage lesson, and connecting to the full ~79M-row warehouse release on Hugging Face via DuckDB.
- **Weekly skeleton notebooks** (`work/notebooks/w01…w07`, `capstone.ipynb`) that the intern fills in, one per assignment.
- **A skill library** (`skills/`) that instructs AI coding assistants how to help on each assignment.
- **A data-safety layer**: `.gitignore` blocks datasets by default and CI fails any commit containing one.

The underlying business context (from `skills/flyrank/flyrank-context/SKILL.md`): FlyRank builds content-as-infrastructure, then watches search data and optimizes. The valuable decision the internship targets is *"out of thousands of pages, which one should a human fix FIRST?"* — a ranking problem where a learned model can beat FlyRank's existing hand-written product flags.

---

## 2. Repository organization

```
Flyrank_ML_Assign/
├── .github/workflows/
│   ├── smoke-test.yml          CI: runs the reference pipeline + data-leak guard + paper_url check
│   └── data-path-smoke.yml     CI: executes notebook 03 against the live HF release (needs HF_TOKEN secret)
├── data/raw/content_refresh_anonymized.csv    The one dataset that ships (30,000 × 44, anonymized)
├── docs/
│   ├── data-dictionary.md                    All 44 columns + warehouse release notes (authoritative)
│   ├── ml-intern-dataset-and-lane-guide.md   The lanes, capstone workflow, validation rules (authoritative)
│   ├── ml-core-foundation-framework.md       Educational: ML-as-a-system reference
│   ├── intern-free-tooling-guide.md          Zero-budget tool stack
│   └── flyrank-seo-research-march-2026.pdf   FlyRank's public data report (used by w06 assignment)
├── notebooks/
│   ├── 01_first_look_and_discovery.ipynb        Week 1 guided notebook (executed by intern)
│   ├── 02_your_first_readable_model.ipynb       Week 2 guided notebook (executed by intern)
│   └── 03_working_with_the_full_release.ipynb   Week 3+ DuckDB/HF workflow (never executed in this clone)
├── outputs/                      Committed examples showing target shape
│   ├── model_report.md                        Committed pipeline report (RF Precision@50 = 0.740)
│   ├── refresh_queue_sample.csv               First 200 rows of the ranked queue
│   └── charts/*.svg                           Five generated charts
├── scripts/                       Reference pipeline (to be copied, not edited)
│   ├── ml_utils.py               Shared paths, helpers, feature lists (the one editable file)
│   ├── 01_prepare_features.py    clean + feature vector + label
│   ├── 02_baseline_score.py      transparent rule-based refresh score + reason codes
│   ├── 03_train_model.py         logistic regression / decision tree / random forest, client-holdout split
│   ├── 04_evaluate_and_export.py blend model+baseline -> ranked queue, charts, Markdown report
│   ├── 05_build_pdf_report.py    reportlab PDF
│   └── run_all.py                runs 01→05
├── skills/                        AI-assistant skill library (router: skills/README.md)
│   └── flyrank/                   flyrank-context + flyrank-data (project-specific)
├── submission/                    paper_url.txt placeholder + README
├── work/                          The intern's workspace
│   ├── README.md
│   ├── capstone_report_template.md
│   └── notebooks/  w01…w07 + capstone.ipynb    Weekly skeletons (3 completed, 7 untouched)
├── AGENTS.md / CLAUDE.md          Agent instructions -> skills/README.md router
├── DATA_USE.md                    Data rules / public-safety contract
├── GUIDE.md                       Operating manual
├── SETUP.md                       GitHub + Colab + HF setup
├── README.md                      Front door
├── requirements.txt               pandas, numpy, scikit-learn, matplotlib, reportlab, duckdb, huggingface_hub
└── .env                           Present locally but gitignored (used by w03 to hold HF_TOKEN). Not in git.
```

Additional observation: `data/processed/` does not exist in this clone, and `outputs/` contains only the committed examples (`model_results.json`, `summary.json`, `refresh_queue.csv`, and the PDF are gitignored). **The reference pipeline has not been run in this clone**, so the committed `outputs/model_report.md` reflects a run at commit time, not the current local state.

---

## 3. What has already been completed (work chronology)

Git history (`git log --oneline`):

| Commit | Date | Content |
|---|---|---|
| `aa05346` | 2026-07-09 | Initial commit — full starter repo template |
| `edc8caa` | 2026-07-09 | Notebook 01 assignment (Week 1 "My Discovery" section added) |
| `aeb1c24` | 2026-07-09 | Notebook 02 experiment (Week 2 "My Experiment" section added) |
| `b77c3c8` | 2026-07-14 | Week 1 research question notebook (`w01_research_question.ipynb`) |
| `12c61b7` | 2026-07-27 | Week 2 ML task framing (`w02_ml_task_framing.ipynb`) |
| `e00cbaf` | 2026-07-27 | Week 3 data contract (`w03_data_contract.ipynb`) — first pass |
| `fdb7009` | 2026-07-27 | Week 3 data contract — completion |

So the intern has completed, in order: **Week 1 (research question), Week 2 (task framing), Week 3 (data contract)**. Everything from **ML-05 onward is an untouched skeleton**.

---

## 4. What each important notebook/script/document does

### 4.1 Weekly skeletons in `work/notebooks/`

**`w01_research_question.ipynb` (ML-02) — COMPLETE.** Lane chosen: **Refresh / Content Opportunity Scoring**. States the decision (which pages to review first), who acts (SEO specialists), the action (review/refresh/monitor), and cost of wrong calls (wasted review time vs missed opportunity). Verified data facts: 30,000 pages, 54.2% declining, 52.7% beyond page 1. Claims discipline spelled out (observed/directional, no causal claims). Executed in a local `Machine_Learning_env` kernel.

**`w02_ml_task_framing.ipynb` (ML-03) — COMPLETE.** Frames the lane as a **scoring** task with **Precision@K** as the defended metric. Unit of analysis: one content page for one client. Identifies the proxy as a "content opportunity score based on observed SEO signals"; explicitly states the starter data has no true refresh-success outcome, so the proxy supports prioritization only. Executed.

**`w03_data_contract.ipynb` (ML-04) — COMPLETE (the intern's most substantial work).** Builds a page-level feature frame from the **warehouse release** (not the starter CSV):

- Downloads `fact_content_daily_performance/month=2026-03/data_0.parquet` from HF (`FlyRank/internship-warehouse`), 9,841,378 rows.
- Verifies grain (report_date × client × content unique) and date span (2026-03-01 → 2026-03-31, 31 days).
- Verifies GSC availability: only 36.7% of March rows have `gsc_data_available = TRUE`.
- Aggregates to page level: **331,437 pages**, 5 features (`march_gsc_impressions`, `march_gsc_clicks`, `march_gsc_ctr`, `march_ga4_sessions`, `march_scroll_events`).
- Defines proxy: `opportunity_proxy = (march_gsc_impressions > 0) & (march_gsc_clicks == 0)` → 32.6% positive.
- Runs a deliberate leakage experiment: score = proxy itself → **Precision@1000 = 1.000**; honest score = `impressions/(clicks+1)` → **Precision@1000 = 0.572**.
- Auths via `.env` (local `HF_TOKEN`), so it requires `python-dotenv` (not in `requirements.txt`).

**`w03_feature_leakage_check.ipynb` (ML-05) — EMPTY skeleton** (only prompt markdown, zero executed code).

**`w04_signal_audit.ipynb` (ML-06) — EMPTY skeleton.**

**`w04_baseline_score.ipynb` (ML-07) — EMPTY skeleton.** (Its prompt asks for `work/outputs/baseline_action_score.csv`.)

**`w05_model.ipynb` (ML-08) — EMPTY skeleton.**

**`w06_validation_audit.ipynb` (ML-09) — EMPTY skeleton.** (Its prompt asks for an audit of two findings from `docs/flyrank-seo-research-march-2026.pdf`.)

**`w07_action_playbook.ipynb` (ML-10) — EMPTY skeleton.**

**`capstone.ipynb` (ML-11) — EMPTY skeleton** (question / data / methodology / results / limitations / recommendations / artifacts sections all prompt-only).

### 4.2 Guided starter notebooks in `notebooks/`

**`01_first_look_and_discovery.ipynb` — EXECUTED by the intern.** Runs `scripts/run_all.py` (live output: baseline Precision@50 = 0.240, RF = 0.740, "3.1x"), then three "discoveries" (search_volume↔impressions corr ≈ 0.001; CTR by position tier; word count by trend). The intern's Week-1 "My Discovery" repeats the correlation on `impressions_90d > 0` (0.0012).

**`02_your_first_readable_model.ipynb` — EXECUTED by the intern.** Hand rule (stale × visible × impressions) vs depth-2 decision tree; Precision@20 hand rule 0.900 vs tree 0.550; Precision@50 0.680 vs 0.600. Leakage demo: adding `trend_pct` (label-derived) → Precision@50 = 1.000. Intern's "My Experiment": depth-3 tree (P@20 0.700, P@50 0.720), concluding deeper trees don't help.

**`03_working_with_the_full_release.ipynb` — NEVER EXECUTED in this clone** (all code cells `execution_count=None`). It teaches the DuckDB-over-HF workflow (dim_clients, dim_content, fact_content_daily_performance, fact_content_daily_performance_sample, fact_content_query_90d), a momentum feature query, query-mix signals, and a first honest model with a train/test split.

### 4.3 Reference pipeline scripts

See §6 (baseline), §7 (models) below. Summary of roles:

- `01_prepare_features.py` — loads raw CSV, coerces types, fills, filters (`impressions_90d > 0`, `content_age_days >= 90`), dedupes by `content_id`, defines `is_declining_label = trend_direction == "down"`, adds log/flag features, writes `data/processed/refresh_feature_vector.csv` + `feature_metadata.json`.
- `02_baseline_score.py` — computes a weighted visibility/freshness/position/depth score, reason codes, suggested actions, writes `data/processed/baseline_refresh_queue.csv` + `baseline_metadata.json`.
- `03_train_model.py` — feature matrix (numeric + one-hot categorical), client-holdout or stratified-row split, trains 3 sklearn classifiers, evaluates vs baseline on the test fold, refits on all data for the final queue probabilities, writes `data/processed/model_predictions.csv` + `outputs/model_results.json`.
- `04_evaluate_and_export.py` — merges baseline + model into `final_refresh_score = 100·(0.70·best_model_probability + 0.30·normalized_baseline_score)`, reason codes, actions, confidence labels, ranked queue, 5 SVG charts, `outputs/model_report.md`, `outputs/summary.json`.
- `05_build_pdf_report.py` — renders `outputs/flyrank_refresh_model_results.pdf` (reportlab).
- `run_all.py` — orchestrates 01→05, prints summary.

### 4.4 Documents

- `README.md` / `SETUP.md` / `GUIDE.md` — setup + operating manual (see §11 for mismatches).
- `DATA_USE.md` — data rules; the anchor for `.gitignore` and CI guards.
- `docs/data-dictionary.md` — authoritative column reference + warehouse release notes. All its key numbers were verified against the raw CSV: 30,000 rows × 44 columns, 32 clients, 54.2% declining, `avg_position == 0` in 1,205 rows, missing `search_volume` 2,468, missing `word_count` 7,699, `feedly article` ~100% missing keyword data, `keyword article` ~28.3% missing word_count, 119 rows with `scroll_rate > 100`, 23 with `ai_traffic_pct > 100`, `impressions_90d >= 1`, `content_age_days >= 90`.
- `docs/ml-intern-dataset-and-lane-guide.md` — the authoritative process doc (lanes, capstone workflow, validation rules, thresholds).
- `docs/flyrank-seo-research-march-2026.pdf` — FlyRank's 36-page public data report ("State of AI-Driven SEO"). Text extracted: 10 confirmed findings, 2 reversed myths, 2 nuanced, 1 debunked, ML appendix (correlations, k=5 k-means archetypes, RF feature importance for health score, logistic growth model, distributions). Content was confirmed by text extraction; original PDF is not machine-readable by this agent beyond that.
- `work/capstone_report_template.md` — 8-section template mapping to the grading rubric. No `work/capstone_report.md` has been created yet.

---

## 5. Data pipeline

### 5.1 Two data sources

1. **Starter CSV** `data/raw/content_refresh_anonymized.csv` (30,000 × 44, 32 clients, trailing-90-day snapshot). Used by the reference pipeline and by `w01`/`w02`.
2. **Warehouse release** `hf://datasets/FlyRank/internship-warehouse` (gated, build `v20260703`; ~79M daily fact rows, 519,606 content items, 104 clients, dates 2025-01-27 → 2026-06-30). Used by `w03_data_contract` (March 2026 partition only) and by notebook 03. The warehouse itself was **not** queried during this investigation.

### 5.2 Reference pipeline data flow (verified from `scripts/`)

```
data/raw/content_refresh_anonymized.csv           30,000 × 44
  └─ 01_prepare_features.py
       type coercion; numerics → 0, categoricals → "unknown"
       filter: impressions_90d > 0 AND content_age_days >= 90
       dedupe by content_id
       label: is_declining_label = (trend_direction == "down")   [16,262 rows, 54.2%]
       features: log_impressions/clicks/sessions/ai_sessions, has_clicks, has_ai_sessions,
                 measurable_opportunity  →  52 columns
  ▼  data/processed/refresh_feature_vector.csv
  └─ 02_baseline_score.py
       visibility = pct_rank(log1p(impressions_90d))
       freshness_risk = pct_rank(days_since_last_update)
       position_opportunity = (1−norm(avg_position clip 1–50)) × visibility × (position>0)
       depth_gap = (1−pct_rank(word_count)) × visibility
       baseline_refresh_score = 0.40·vis + 0.30·fresh + 0.25·pos + 0.05·depth  (clip 0–1)
       reason codes (6 rules) + suggested action + rank
  ▼  data/processed/baseline_refresh_queue.csv
  └─ 03_train_model.py
       feature matrix: MODEL_NUMERIC_FEATURES (18) + one-hot MODEL_CATEGORICAL_FEATURES (8)
       split: client_holdout (~20% of clients) fallback stratified_row_holdout; seed 42
       fit LR / DT / RF on train; evaluate on test; refit all on full data for queue
  ▼  data/processed/model_predictions.csv  +  outputs/model_results.json
  └─ 04_evaluate_and_export.py
       final_refresh_score = 100·(0.70·best_model_probability + 0.30·normalized_baseline)
       final reason codes, suggested action, confidence (high/medium/low via p80/p50)
       sort by (final score, impressions, sessions) desc → final_rank
  ▼  outputs/refresh_queue.csv, outputs/model_report.md, outputs/charts/*.svg
  └─ 05_build_pdf_report.py
  ▼  outputs/flyrank_refresh_model_results.pdf
```

Feature definitions live in `scripts/ml_utils.py`: `MODEL_NUMERIC_FEATURES` and `MODEL_CATEGORICAL_FEATURES`. **Confirmed leak-safety:** neither `trend_pct` nor `trend_direction` is in the model feature lists (both are label-derived). IDs (`content_id`, `client_id`) are used only for grouping/splits.

### 5.3 Intern's own (capstone-direction) data flow — from `w03_data_contract.ipynb`

```
HF warehouse → March-2026 partition parquet (9,841,378 daily rows)
  └─ grain check (report_date × client × content unique)
  └─ GSC availability filter context (36.7% of rows gsc_data_available)
  └─ group by (client_hash_id, content_hash_id) → 331,437 pages
       features: march_gsc_impressions, march_gsc_clicks, march_gsc_ctr,
                 march_ga4_sessions, march_scroll_events
  └─ proxy: opportunity_proxy = (impressions > 0) & (clicks == 0)   (32.6% positive)
  └─ leakage demo: leaked_score = proxy → P@1000 = 1.000
  └─ honest score: impressions/(clicks+1) → P@1000 = 0.572
```

Notable: the intern's frame has **no content metadata** (word count, intent, keyword context) and **no query-mix features** yet; it is a pure monthly-aggregate of 5 signals. No per-client window alignment against `dim_clients.gsc_data_start` / `ga4_data_start` was performed (the notebook notes GA4 unavailability qualitatively, but does not filter on `ga4_data_available`).

---

## 6. Baseline

There are two distinct baselines in the repo:

### 6.1 Reference-pipeline baseline (`scripts/02_baseline_score.py`) — implemented, evaluated

- **Inputs:** the 52-column feature vector.
- **Rules/thresholds (reason codes):**
  - `stale_visible_page`: `days_since_last_update >= 180` and `impressions_90d >= 500`
  - `declining_with_demand`: `trend_direction == "down"` and `impressions_90d >= 100`
  - `thin_visible_page`: `0 < word_count < 1200` and `impressions_90d >= 250`
  - `page_one_decay_risk`: `0 < avg_position <= 10` and `content_age_days >= 180`
  - `low_ctr_visible_page`: `impressions_90d >= 500`, `0 < avg_position <= 20`, `ctr < 0.5`
  - `low_engagement_visible_page`: `sessions_90d >= 30` and (0 < engagement_rate < 30 or 0 < scroll_rate < 30)
  - fallback `general_refresh_review`
- **Scoring formula (weights):** `0.40·visibility + 0.30·freshness_risk + 0.25·position_opportunity + 0.05·depth_gap`, clipped to [0,1]. (Note: the reason codes do not alter the score; only the four sub-scores do. `trend_direction` therefore only feeds *labels/reasons*, not the ranking — so the ranking itself is leak-free with respect to the label.)
- **Ranking logic:** `baseline_refresh_score` descending; ties broken by `rank(method="first")`.
- **Evaluated Precision@50 (holdout test clients):** **0.240** (committed `outputs/model_report.md` + executed notebook 01 output). ROC AUC 0.627, avg precision 0.468.
- **Limitations:** The rule only captures univariate thresholds — no interactions; the score correlates mostly with raw visibility, so it can rank high-traffic pages first regardless of actual decline risk. It is nonetheless a defensible *transparent* baseline for the decision (limited-capacity review prioritization) precisely because it is a simple, inspectable rule with reason codes.

### 6.2 Intern's Week-3 "honest score" baseline (`w03_data_contract.ipynb`)

`march_gsc_impressions / (march_gsc_clicks + 1)`, Precision@1000 = **0.572**. This is a **same-window heuristic**, not the final capstone baseline; the ML-07 baseline notebook (which would define the lane's real baseline) is empty.

---

## 7. ML models

### 7.1 Reference pipeline models (`scripts/03_train_model.py`) — trained and evaluated

| Model | Features | Target | Preprocessing | Hyperparameters (fixed, seed 42) | Split | ROC AUC | Avg precision | Precision@50 |
|---|---|---|---|---|---|---|---|---|
| Logistic regression | 18 numeric + one-hot 8 categorical (~40+ dummy cols) | `is_declining_label` | StandardScaler, class_weight balanced, max_iter 1000 | — | client_holdout | 0.700 | 0.522 | 0.400 |
| Decision tree | same | same | class_weight balanced | max_depth 5, min_samples_leaf 50 | client_holdout | 0.742 | 0.575 | 0.540 |
| Random forest | same | same | class_weight balanced_subsample | n_estimators 200, max_depth 10, min_samples_leaf 25 | client_holdout | 0.750 | 0.618 | 0.740 |
| **Baseline rules** | — | same | — | — | same test fold | 0.627 | 0.468 | **0.240** |

All numbers above come from the committed `outputs/model_report.md` and the executed output of notebook 01 (identical), i.e., they were produced by an actual pipeline run at commit time. Results are **not reproducible in this clone without running the pipeline** (artifacts are gitignored), but the pipeline is deterministic (seeds 42; no cross-validation) so a re-run should reproduce the same test clients and close numbers. README/GUIDE warn that RF Precision@50 can land 0.68–0.74 across library versions; the stable claim is the ~3× lift.

- **Interpretability:** best model = RF, selected by test-set Precision@50. Top features (committed report): `days_with_impressions` 0.158, `log_impressions_90d` 0.128, `avg_position` 0.109, `content_age_days` 0.096, then char/word count, log clicks, ctr, scroll_rate, days_with_sessions.
- **Caveats:** (a) model selection uses the **test set** (`best_model` = max test Precision@50), a mild selection-on-test bias; (b) final queue probabilities come from models refit on **all** data; (c) a single holdout, no CV, so no fold-level variance or stability analysis exists; (d) the label is a same-window proxy (see §10 risks).

### 7.2 Models discussed but not implemented in the intern's work

None yet. `w05_model.ipynb` is empty. The intern's work has no trained model of their own yet — only the reference pipeline's (which they ran in notebook 01).

---

## 8. Validation and generalization

The **only implemented validation** is in the reference pipeline (`scripts/03_train_model.py`):

- **Split type:** grouped by client — `client_holdout`. ~20% of the 32 clients (6–7 clients) are held out; no page from a held-out client appears in training. Fallback to `stratified_row_holdout` only if a clean client split can't satisfy both classes in both folds (unused in practice).
- **Grouping variable:** `client_id`.
- **Can clients appear in both train and test?** No, when `client_holdout` is used (verified from committed report: `split_strategy: client_holdout`).
- **Cross-validation:** none. Single deterministic 80/20 client split, seed 42.
- **Evaluation rows:** the held-out test pages (~20% of rows).
- **K values:** 20, 50, 100 (Precision@K); plus accuracy/precision/recall/F1 at 0.5 and ROC AUC / avg precision.
- **Tie policy:** `precision_at_k` sorts by score descending; ties broken by stable sort order (`sort_values` default quicksort on score only). The baseline rank uses `method="first"`.
- **Fold-level performance / variance:** none exist (single split).
- **Sensitivity / feature-exclusion experiments:** none in the pipeline. (Notebook 02's leakage experiment — adding `trend_pct` → P@50 = 1.000 — is the only feature-level experiment, and it is in-sample teaching.)

**What the current evaluation actually tests:** generalization to **unseen clients** (whole-client holdout) on the starter slice. This is stronger than a random row split but still only one draw of the client split, with no repeated/out-of-time validation, and it is on the teaching label, not a future-window label.

**`w06_validation_audit.ipynb` (ML-09) is empty** — the intern has not yet audited the reference paper's findings nor re-run their (nonexistent) model under an honest split. The reference pipeline's client-holdout design is the pattern the capstone is expected to follow.

---

## 9. Reproducibility

**What would reproduce:**
- Reference pipeline: `pip install -r requirements.txt` then `python scripts/run_all.py` (~1 min). Deterministic seeds (42). Outputs gitignored but regenerated. The raw CSV is committed, so a fresh clone can run it. The committed `outputs/model_report.md` and `outputs/refresh_queue_sample.csv` are consistent with a run.
- Starter notebooks 01/02: self-contained, need the raw CSV only.

**What would NOT reproduce without gaps:**
1. **`w03_data_contract.ipynb` requires a Hugging Face token** (`HF_TOKEN` via `.env` → `python-dotenv`, or env var). `.env` exists locally but is gitignored; a fresh clone has no token. `python-dotenv` is **not** in `requirements.txt`.
2. **Hard-coded data path:** the notebook downloads `fact_content_daily_performance/month=2026-03/data_0.parquet` via `hf_hub_download`. If the partition spans multiple files (`data_1.parquet`, …), only part of March is loaded. `Not verified from repository evidence` — HF was not queried. Also, the file layout is hard-coded and will break if the release reorganizes.
3. **Warehouse access is gated** (instant approval) and the full-table iteration rule (iterate on `_sample` / single month) is followed only partially — the intern pulled the full March partition (~9.8M rows) instead of `fact_content_daily_performance_sample`.
4. **Environment assumptions:** notebooks were authored/executed in a local `Machine_Learning_env` conda kernel; skeleton notebooks specify plain `Python 3`. Notebook 03 needs `duckdb`/`huggingface_hub` installed in the kernel (its install cell is `%pip install -q duckdb huggingface_hub`).
5. **No committed pipeline artifacts:** `outputs/model_results.json`, `summary.json`, `refresh_queue.csv`, PDF are gitignored — the committed `model_report.md` cannot be cross-checked against its JSON source in this clone.
6. **No lockfile / no pinned versions** — `requirements.txt` uses loose `>=` pins; README itself acknowledges library-version sensitivity (RF 0.68–0.74).
7. **Missing file reference:** `01_prepare_features.py` error message points at `00_export_bigquery_anonymized.mjs`, which does not exist in this repo (mentor-side tooling).
8. **CI runs the pipeline but not the intern's notebooks** (see §10), so nothing verifies `work/notebooks/*` re-run cleanly.

---

## 10. Git / CI / project engineering

- **`.github/workflows/smoke-test.yml`:** on push to `main` + PRs. (a) Runs `python scripts/run_all.py` on the sample, asserts `outputs/model_results.json`, `outputs/refresh_queue.csv`, `outputs/model_report.md` exist. (b) `data-leak-check`: fails on any committed `.parquet/.zip/.tar/.feather` and on any CSV other than `data/raw/content_refresh_anonymized.csv` and `outputs/refresh_queue_sample.csv`. (c) Soft-check on `submission/paper_url.txt` (placeholder allowed with a note).
- **`.github/workflows/data-path-smoke.yml`:** executes `notebooks/03_working_with_the_full_release.ipynb` against the live release via nbclient, gated on the `HF_TOKEN` repo secret (skips on forks). Runs on push (path-filtered), weekly schedule, manual dispatch. **Notably, notebook 03 has never been executed in this clone locally, and this workflow only triggers when notebook 03 / requirements / the workflow change.**
- **What CI tests:** the reference pipeline end-to-end + dataset leak guard + paper-URL format. It does **not** execute `work/notebooks/*`, does not lint, and has no unit tests. There is no validation that the intern's own assignments run.
- **Git hygiene:** 8 commits, clean working tree, `.env` correctly untracked, no datasets committed. Branch is `main`; nothing to indicate force-push or rebase issues.

---

## 11. Documentation vs Implementation

### Documentation vs Implementation

| Claim (where) | Reality (evidence) | Classification |
|---|---|---|
| "Your skeleton notebooks are already in `notebooks/`" (`work/README.md` table) | Skeletons are in **`work/notebooks/`**; `notebooks/` holds only the 3 guided starter notebooks | Documentation outdated |
| "Skeleton notebooks live in `work/notebooks/`" (`docs/ml-intern-dataset-and-lane-guide.md` §1) | Matches actual layout | Confirmed / matches |
| Data dictionary counts: 30,000 × 44, 32 clients, 54.2% down, 1,205 `avg_position==0`, 2,468 missing SV, 7,699 missing WC, missingness-by-type, rates >100 | All verified against the raw CSV | Confirmed / matches |
| `is_declining_label = trend_direction == "down"`; `trend_direction`/`trend_pct` never features | Confirmed in `01_prepare_features.py` + `ml_utils.py` feature lists | Confirmed / matches |
| Baseline formula + reason codes (guide §5, `data-dictionary.md`) | Matches `02_baseline_score.py` exactly | Confirmed / matches |
| Final score `100·(0.70·model + 0.30·baseline)` + final reason codes | Matches `04_evaluate_and_export.py` exactly | Confirmed / matches |
| Starter results table: baseline 0.240 / LR 0.400 / DT 0.540 / RF 0.740 (guide §5) | Matches committed `outputs/model_report.md` and executed notebook 01 output | Confirmed / matches |
| "verified from `outputs/model_results.json`" (guide §5) | `model_results.json` is gitignored and absent from this clone — cannot be verified against it here | Ambiguous |
| `data/processed/` exists / regenerates | Does not exist in this clone; regenerates on run | Confirmed (on first run) |
| `outputs/refresh_queue.csv` produced by pipeline (GUIDE diagram) | Produced but gitignored; only `refresh_queue_sample.csv` committed | Confirmed / matches (design) |
| w03 data contract uses the "iteration" sample table (flyrank-data skill guidance) | Used the full March partition instead of `_sample` — acceptable but heavier | Potential issue (minor) |
| w03 leakage demo "leaked 1.000 vs honest 0.572" | Matches executed notebook outputs | Confirmed / matches |
| `work/` = intern space; capstone deliverable in `work/` | `work/capstone_report.md` does not exist; capstone notebook empty | Implementation incomplete |
| Notebooks w04–w07 & capstone "filled" (self-check boxes unchecked) | All are empty skeletons (0 executed code cells) | Implementation incomplete |
| README claims the intern's repo is a public clone for submission | This is a local repo on branch `main`; remote presence/URL not verified | Not verified from repository evidence |
| Notebook 03 CI smoke test exercises the weeks-3+ path | Workflow exists; has never been observed to run here; notebook never executed locally | Ambiguous |

---

## 12. Capstone investigation (current state)

**Business problem:** Content that ranks in Google search quietly decays; teams notice too late and have more pages than review capacity.
**Business objective:** Prioritize which pages a human SEO reviewer should look at **first**.
**Business decision / action:** Review → refresh/expand/protect/monitor a page; cost of wrong call = wasted review effort (FP) or missed visibility (FN).
**Intended user:** SEO specialist / content manager.
**Unit of analysis:** one content page (per client), per the intern's `w01`/`w02`/`w03` framing.
**Available data:** starter CSV (used in w01/w02) and warehouse March-2026 partition (used in w03). Not yet: dim_content metadata, query-mix features, per-client window alignment.
**Features:** the intern has 5 March features (`march_gsc_impressions/clicks/ctr/ga4_sessions/scroll_events`). The reference pipeline demonstrates a fuller 26-feature set.
**Target/proxy:** `opportunity_proxy = (impressions > 0) & (clicks == 0)` — a same-window, current-state bucket. The docs themselves flag such proxies as beginner stand-ins; a future-outcome label (features from prior 90d → decline/recovery over next 30d) is explicitly the "stronger" shape.
**Ranking approach:** score pages and rank; Precision@K is the accepted metric.
**Evaluation metric:** Precision@K (K = 1000 in w03 demo; 20/50/100 in the reference pipeline).
**Baseline:** not yet defined by the intern for the capstone lane (ML-07 empty). Reference-pipeline baseline exists as a template.
**ML models:** none of the intern's own yet (ML-08 empty). Reference-pipeline RF is the demonstrated template.
**Split strategy:** reference pipeline uses client-holdout; the intern has not yet designed their own split.
**Validation strategy:** none yet (ML-09 empty).
**Final output:** a ranked "Content Opportunity Score" queue with actions + reason codes + confidence, then a deployed public research paper (URL into `submission/paper_url.txt`, still placeholder).

**Does the repo implement the documented concept?** Partially, up to Week 3:

```
SEO page
  → observed search/engagement signals          (w01/w02 framing; w03: 5 March signals)
  → decision-time features                      (w03: March page-level features; April excluded)  ✔
  → Content Opportunity Score                   (w03: honest_score demo; final scoring NOT built)  ✘
  → prioritized pages                           (w03: top-K demo; no queue artifact)              ✘
  → human SEO review                            (documented intent only)                          ✘
```

The pipeline through feature construction and a leakage demonstration exists; the actual scoring model, validation, ranked queue, and playbook do not.

---

## 13. Leakage investigation (detailed)

From `w03_data_contract.ipynb` (verified from executed outputs):

- **Proxy:** `opportunity_proxy = 1` when `march_gsc_impressions > 0 AND march_gsc_clicks == 0`, else 0. 107,901 / 331,437 pages positive (32.6%).
- **Construction:** derived from the same two March signals that are also features.
- **Features allowed at decision time:** the 5 March page-level aggregates. April is explicitly excluded from features.
- **Features that could leak the proxy:** `march_gsc_clicks` and `march_gsc_impressions` *are* the label's building blocks — i.e., the proxy is a deterministic function of the features. Feeding the proxy (or `clicks == 0` directly) back in is trivially "perfect".
- **How the experiment was implemented:** `leaked_score = opportunity_proxy`, rank by it, P@1000 = **1.000**; then honest score `impressions/(clicks+1)`, P@1000 = **0.572**. The difference (1.000 vs 0.572) is presented as the cost of leakage.
- **Does leakage remain?** In the *reference pipeline*, no — `trend_pct`/`trend_direction` are excluded from model features and the baseline's *score* doesn't use `trend_direction` (only reason codes). In the *intern's* design, there is no model yet; the bigger structural issue is that the proxy is **defined on the same window and the same columns as the features**, so even the "honest" 0.572 is a mechanical artifact of ranking by a monotone function of the label's components rather than evidence of genuine signal — and a future-outcome proxy has not yet been defined. The leakage experiment itself is sound as a teaching demonstration, but it demonstrates label-as-feature, not the subtler window-overlap/query-table leakage the docs warn about.

---

## 14. Current state assessment

### Completed
- Week 1 (ML-02), Week 2 (ML-03), Week 3 data contract (ML-04) notebooks — filled, executed, committed.
- Starter notebooks 01/02 executed with the intern's own discovery/experiment sections.
- Reference pipeline (scripts 01–05) shipped, CI-covered, with committed example outputs (baseline 0.240 → RF 0.740).
- Data-safety layer (gitignore + CI leak guard) in place and clean.

### In Progress
- Nothing in the intern's lane is mid-implementation beyond Week 3. The reference pipeline is the only end-to-end implementation.

### Planned (documented, not implemented)
- ML-05 leakage/feature-vector check, ML-06 signal audit, ML-07 baseline, ML-08 model, ML-09 validation audit, ML-10 action playbook, ML-11 capstone + deployed paper.
- `work/capstone_report.md`, `work/outputs/`, figures, `submission/paper_url.txt` real URL.

### Unknown
- Whether the intern has HF warehouse access beyond the local `.env` (token exists locally; not part of repo).
- Whether the March partition `data_0.parquet` is the complete partition (single-file vs multi-file).
- Remote repo status / CI status of the intern's fork (no evidence in this clone).
- Full contents of `docs/ml-core-foundation-framework.md` beyond its structure (educational; skimmed only).
- Whether the committed `outputs/model_report.md` matches a fresh run in today's library versions (README predicts 0.68–0.74 possible).

### Risks / Issues
1. **Proxy validity (highest methodological risk):** the intern's proxy is a same-window, feature-derived bucket. It cannot support a "which pages will need review" claim and makes "improvement" numbers partly circular. The docs already point at the stronger `prior-90d features → next-30d outcome` shape; adopting it should happen before modeling.
2. **No client-grouped or time-aware validation yet** in the intern's lane; the reference pipeline's single client-holdout has no CV or variance estimate.
3. **Leakage surface is larger than exercised:** query-table window overlap and per-client `gsc/ga4_data_start` misalignment are documented risks the intern hasn't addressed.
4. **Reproducibility gaps:** unlisted `python-dotenv` dependency, hard-coded HF file path, gated data access, no lockfile, gitignored pipeline artifacts.
5. **Skeleton notebooks use a different kernelspec** (`Python 3` vs `Machine_Learning_env`), and nothing in CI executes `work/notebooks/*`.
6. **Model selection on the test set** in the reference pipeline (minor teaching trade-off).
7. **Documentation drift:** `work/README.md` points at `notebooks/` for skeletons (actual: `work/notebooks/`); guide references a gitignored `model_results.json` for verification.

### Recommended Next Step
**Complete ML-05 (`work/notebooks/w03_feature_leakage_check.ipynb`)** — the next unfilled notebook in the sequence, directly downstream of the finished data contract. Concretely: formalize the March-2026 page-level feature vector (decide missing-value flags for GA4-vs-GSC availability using `gsc_data_available`/`ga4_data_available`, add content metadata from `dim_content`), and run a genuine leakage audit that (a) checks the current proxy's determinism-from-features problem, (b) tests window alignment against `dim_clients.gsc_data_start`/`ga4_data_start`, and (c) produces an honest feature/label split before any model is built. Do not start this implementation in this session — it is a recommendation for the next engineer.

---

## Appendix A — Files inspected

- All 10 notebooks under `work/notebooks/` (full cell-by-cell dump including outputs).
- All 3 starter notebooks under `notebooks/`.
- All 7 scripts under `scripts/`.
- All top-level docs (README, GUIDE, SETUP, DATA_USE, AGENTS, CLAUDE, requirements.txt, .gitignore, LICENSE).
- `docs/data-dictionary.md`, `docs/ml-intern-dataset-and-lane-guide.md`, `docs/intern-free-tooling-guide.md`, `docs/ml-core-foundation-framework.md` (structure), research PDF (text extraction).
- `outputs/model_report.md`, `outputs/refresh_queue_sample.csv`, `outputs/charts/*` (via report), `.github/workflows/*`, `submission/*`, `work/capstone_report_template.md`, `work/README.md`, `skills/*` (router + context + data skills).
- Raw CSV verified with read-only pandas.
