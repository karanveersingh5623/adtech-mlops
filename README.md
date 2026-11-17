# AdTech ML — 7-Day End-to-End Capstone

Build, test, explain, and demonstrate a small **contextual advertising and recommendation system** in seven days. This repository is designed for **two developers learning together**, with approximately **2–4 hours per person per day** (target: 3 hours/day).

The outcome is **one connected, runnable application**, not seven unrelated notebooks: impression and behavioral data → validation and features → click-through-rate (CTR) prediction → candidate filtering and ranking → HTTP API → tests, latency measurements, and a monitoring plan.

> **Project status:** Learning scaffold. The deterministic synthetic-data generator, example CSV files, and generator tests are included. Training, ranking, audience filtering, serving, and monitoring are intentionally **not implemented yet**. The daily deliverables below define what the contributors will build.
>
> **Scope:** Educational prototype, not a production advertising platform. All included user IDs and events are synthetic. No production performance, revenue uplift, or causal business improvement is claimed.

## Table of contents

1. [Project goal and success criteria](#project-goal-and-success-criteria)
2. [System overview](#system-overview)
3. [Tech stack and scope](#tech-stack-and-scope)
4. [Repository structure](#repository-structure)
5. [Quick start](#quick-start)
6. [Data dictionary and modeling rules](#data-dictionary-and-modeling-rules)
7. [Seven-day roadmap](#seven-day-roadmap)
8. [Working together](#working-together)
9. [Quality gates and definition of done](#quality-gates-and-definition-of-done)
10. [Evaluation and performance](#evaluation-and-performance)
11. [Risks, limitations, and stretch goals](#risks-limitations-and-stretch-goals)
12. [Final demonstration](#final-demonstration)

## Project goal and success criteria

**Scenario:** A content site has several eligible advertisements. Given impression-time context, estimate the likelihood that a user will click each candidate and return a ranked list. Build a small behavioral-event pipeline to illustrate how recent interactions can inform audience eligibility.

By the end of Day 7, we aim to demonstrate:

- [ ] A validated, reproducible dataset and documented feature/label definitions.
- [ ] A constant-probability baseline and a trained CTR classifier.
- [ ] Held-out evaluation with metrics appropriate for imbalanced clicks.
- [ ] Deterministic top-*K* ranking of **eligible** ad candidates.
- [ ] A minimal behavioral-event aggregation and audience-eligibility component using **past events only**.
- [ ] A locally runnable `POST /rank` API with input validation and model version.
- [ ] Unit tests, integration tests, and reproducible setup commands.
- [ ] Measured local inference/API latency, plus documented monitoring and retraining strategy.
- [ ] A joint demo in which **both contributors can explain and modify the code**.

**Completion standard:** A passing test suite and a live demo are required. Merely training a model in a notebook is not sufficient.

## System overview

```text
Historical ad impressions                User behavior events
         |                                      |
         v                                      v
  Validate + explore                     Validate + aggregate
         |                                      |
         v                                      v
  Feature preparation                 Past-only behavior features
         |                                      |
         v                                      v
  Train CTR classifier                 Audience-eligibility rules
         |                                      |
         v                                      |
  Evaluate + save model                          |
         |                                      |
         +----------------+---------------------+
                          |
              Candidate ad requests
                          |
                 Filter eligible ads
                          |
                 Model CTR scoring
                          |
                   Top-K ranking
                          |
                    FastAPI /rank
                          |
             Tests + latency + metrics
```

**Important distinctions:**

- **CTR prediction** estimates the chance of a click on an impression.
- **Audience eligibility** identifies which audiences or categories a user may qualify for based on historical behavior. The first implementation is **rule-based**, not a separate trained conversion model.
- **Ranking** sorts eligible candidates by predicted click probability. A real advertising platform might additionally account for bids, budgets, pacing, eligibility, fairness, and expected revenue; those are outside the core one-week implementation.
- **Conversion prediction** is different from predicting clicks. The `converted` column exists in our synthetic data, but it is **not** a feature for the impression-time CTR model.

## Tech stack and scope

| Layer | Core choice | Why |
|---|---|---|
| Language | Python 3.10+ | Practical data/ML/service development |
| Data handling | Pandas, NumPy | Tabular transformations and validation |
| Models | scikit-learn logistic regression | Explainable, fast baseline and preprocessing pipeline |
| Optional model | PyTorch | Small neural classifier for comparison, **only after the baseline works** |
| API | FastAPI, Pydantic, Uvicorn | Typed HTTP requests and model serving |
| Tests | `unittest` (already present); optional `pytest` | Reproducibility, validation, API and ranking checks |
| Packaging | Docker (optional if time permits) | Reproducible local runtime |
| Collaboration | Git, GitHub, pull requests | Reviewable changes and shared ownership |
| Performance | Python timers / local load script | Measured p50/p95 latency and throughput under stated conditions |

**Not required for the seven-day MVP:** Kafka cluster, Kubernetes cluster, cloud deployment, hosted feature store, distributed training, real-time bidding infrastructure, real A/B experiment, or production GPU serving. We will discuss where they would fit without making them prerequisites.

## Repository structure

### Currently included

```text
adtech_7_day_capstone/
├── README.md                     # This plan
├── docs/
│   └── DAY1.md                   # Day 1 guided exercise
├── data/
│   ├── ad_impressions.csv        # Generated synthetic impression rows
│   └── behavior_events.csv       # Generated synthetic activity events
├── src/
│   ├── __init__.py
│   └── generate_synthetic.py     # Deterministic dataset generator
├── tests/
│   └── test_generate_synthetic.py
├── explore_data.py              # Day 1 exercise starter
├── requirements-day1.txt        # Initial data dependency
└── .gitignore
```

### Suggested structure to create during the week

```text
src/
├── data_validation.py          # Day 1
├── features.py                 # Days 1–2
├── train.py                    # Day 2
├── evaluate.py                 # Day 2
├── ranking.py                  # Day 3
├── audiences.py                # Day 4
├── api.py                      # Day 5
└── benchmark.py                # Day 6
models/                          # Local artifacts; avoid committing large binaries
reports/
├── data_profile.md
├── model_card.md
└── latency.md
tests/
├── test_data_validation.py
├── test_features.py
├── test_ranking.py
├── test_audiences.py
└── test_api.py
Dockerfile                       # Optional Day 5/6 deliverable
requirements.txt                 # Consolidate as packages are introduced
```

Names under **suggested structure** are a plan, not existing files. Do not expect them to run until they have been implemented.

## Quick start

### 1. Clone or extract

After pushing this project to GitHub:

```bash
git clone <YOUR_REPOSITORY_URL>
cd adtech_7_day_capstone
```

If you have the downloaded starter ZIP instead, extract it and open the `adtech_7_day_capstone` folder in your editor.

### 2. Create an environment

```bash
python3 -m venv .venv
source .venv/bin/activate       # macOS/Linux
python -m pip install --upgrade pip
python -m pip install -r requirements-day1.txt
```

Windows PowerShell activation: `.venv\Scripts\Activate.ps1`.

### 3. Generate the sample data

```bash
python -m src.generate_synthetic \
  --out data --users 300 --impressions 15000 --seed 42
```

This creates `data/ad_impressions.csv` and `data/behavior_events.csv`. Sample copies are already included in the starter archive. The script is deterministic for a fixed seed and arguments.

### 4. Run the existing tests

```bash
python -m unittest discover -s tests -v
```

### 5. Begin Day 1

```bash
python explore_data.py
```

**VS Code debugger tip:** Open the project directory as the workspace, or set `"cwd": "${workspaceFolder}"` in `.vscode/launch.json`. Relative CSV paths are resolved from the process working directory, not automatically from the script location. For more robust file loading, use `Path(__file__).resolve().parent / "data" / "ad_impressions.csv"` in a root-level script.

## Data dictionary and modeling rules

### `data/ad_impressions.csv`

| Column | Meaning | CTR modeling guidance |
|---|---|---|
| `request_id` | Unique impression identifier | Use for integrity/deduplication, **not** a predictive feature |
| `event_time` | UTC impression timestamp | May derive time features available at request time |
| `user_id` | Synthetic user identifier | Avoid raw high-cardinality ID in the first baseline |
| `ad_id` | Candidate ad identifier | Optional later feature; avoid fitting on accidental ID patterns |
| `ad_category` | Candidate ad category | Initial feature |
| `device` | Mobile or desktop | Initial feature |
| `hour` | UTC hour, 0–23 | Initial feature |
| `clicked` | 0 or 1 observed click outcome | **Target**, never an input feature |
| `converted` | 0 or 1 later conversion outcome | **Never** input to impression-time CTR prediction |

### `data/behavior_events.csv`

Fields include `event_id`, `event_time`, `request_id`, `user_id`, `ad_id`, `category`, and `event_type` (`impression`, `click`, or `conversion`).

**Time rule:** A prediction at time *t* may only use information known **before or at t**. A user's later click or conversion may be used as a training label after it becomes available, but cannot be included as an input feature for that same historical prediction.

**Synthetic generator assumptions:**

- Four ad categories: finance, sports, shopping, travel.
- Two devices: desktop and mobile.
- User preferences, matching categories, and hour influence generated click probabilities.
- Conversion can occur only after a click **in this generator**; this is not a universal rule for real advertising attribution.
- The data is intentionally simple and learnable. Performance on it does **not** demonstrate real-world production performance.
- If `clicked` is missing, don't silently count it as zero. Investigate the missing label, exclude invalid rows from evaluated labeled samples where appropriate, and report the exclusion.

## Seven-day roadmap

**Time budget:** Each person targets 45 minutes learning, 90 minutes coding, and 45 minutes validation/review per day. If only two hours are available, prioritize the **required deliverable** and skip optional enhancements.

### Day 1 — Understand the data and define the prediction problem

**Learn:** impressions, clicks, conversions, CTR; supervised learning; features versus target; Python/Pandas DataFrames; data types; missing values; duplicate IDs; leakage and class imbalance.

**Implement:**

1. Generate/read both CSV files.
2. Inspect `shape`, `head`, `dtypes`, missing counts, unique users, duplicate `request_id` values, label values and hour range.
3. Calculate overall CTR and CTR by device and ad category, including impression counts.
4. Decide which columns are safe to use for a prediction at impression time.
5. Write down data limitations and a baseline hypothesis.

**Required deliverable:** An executable data-profiling script or notebook, a short `reports/data_profile.md`, and basic data-validation tests.

**Think and discuss:** Does higher historical CTR for a category mean the category causes more clicks? What happens when a click label is missing? Why is `converted` leakage for CTR prediction?

**Day 1 done when:** Both collaborators can reproduce the data profile and explain the chosen features and target without reading code aloud.

### Day 2 — Train and evaluate a simple CTR model

**Learn:** binary classification, logistic regression, probabilities, preprocessing, categorical encoding, train/validation/test splits, overfitting, log loss, ROC-AUC and PR-AUC.

**Implement:**

1. Define a trivial baseline (predict the training click rate).
2. Split data reproducibly; ensure preprocessing is fitted **only on training data**.
3. Train a scikit-learn `Pipeline` with imputation/encoding as needed and logistic regression.
4. Evaluate against baseline on held-out data; save evaluation results and the fitted pipeline.
5. Verify that unseen categorical values can be processed at prediction time.

**Required deliverable:** `src/train.py`, `src/evaluate.py`, saved local model artifact, `reports/model_card.md` with actual measured metrics, and focused tests.

**Think and discuss:** Why can accuracy be misleading when most impressions are not clicked? Why choose log loss for predicted probabilities? If rows share users or chronology, what could make a random split optimistic? Keep split limitations explicit.

**Day 2 done when:** One command trains the pipeline, another reproduces its held-out evaluation, and the report compares baseline versus trained model.

### Day 3 — Turn probabilities into an ad ranking

**Learn:** classification versus ranking, top-*K*, eligibility constraints, tie-breaking, probability calibration, ranking metrics and candidate-set assumptions.

**Implement:**

1. Define a small input schema describing context and candidate ads.
2. Implement candidate filtering (e.g., enabled category and campaign eligibility).
3. Score candidates through the saved CTR pipeline.
4. Sort by score descending, with stable tie-breaking; return top-*K*.
5. Add tests for empty candidates, invalid *K*, deterministic order and exclusion of ineligible ads.

**Required deliverable:** `src/ranking.py`, a command-line demonstration, and ranking tests.

**Optional:** Train a tiny PyTorch classifier and compare log loss, ranking output, complexity and latency. **Do not replace a working baseline merely to introduce neural networks.**

**Think and discuss:** A classifier scores one impression; a ranker compares multiple *eligible* candidates. How do we build valid candidate features without using future interactions? Why might the best CTR score not maximize revenue?

**Day 3 done when:** Given a request with several candidates, the system returns a deterministic ranked list and test cases prove eligibility is enforced.

### Day 4 — Aggregate behavior and build a simple audience filter

**Learn:** event schemas, deduplication, windowed aggregation, feature freshness, online/offline consistency, time leakage, user history and basic audience rules.

**Implement:**

1. Load `behavior_events.csv`, parse timestamps, and validate types and unique `event_id` values.
2. Derive behavior features from events **strictly before a chosen scoring timestamp** (e.g., recent category clicks).
3. Create simple, documented audience-eligibility rules based on past behavior.
4. Integrate eligible categories with the ranking pipeline; test that later events don't influence earlier results.
5. Explain the trade-off between event-by-event processing and short batching windows.

**Required deliverable:** `src/audiences.py`, batch event-to-eligibility workflow, and tests proving no future-event leakage.

**Scope note:** This is a rule-based audience filter with synthetic data, **not** a trained audience conversion-propensity model. A learned conversion gate is a future extension requiring properly defined targets and evaluation.

**Day 4 done when:** A simulated user's past events can influence which ads are eligible, while future events cannot.

### Day 5 — Expose the ranking pipeline as an API

**Learn:** HTTP/JSON, request/response schemas, Pydantic validation, model loading, synchronous inference, errors, health checks, and batch versus real-time predictions.

**Implement:**

1. Create a FastAPI app with `GET /health` and `POST /rank`.
2. Load the model once at startup, not on every request.
3. Validate user context and candidate inputs; return ranked candidates and model version.
4. Return clear errors for malformed requests and missing model artifacts.
5. Add API integration tests and document `uvicorn` startup and sample requests.

**Required deliverable:** `src/api.py`, automated endpoint tests, and a working localhost demo.

**Optional:** Add a `Dockerfile` after local tests pass.

**Day 5 done when:** A new developer can start the service from the README and get the same response structure.

### Day 6 — Reliability, performance and ML operations

**Learn:** unit versus integration tests, p50/p95/p99 latency, throughput, class prevalence, prediction drift, input drift, data quality, model versioning, retraining triggers and rollback.

**Implement:**

1. Execute all tests and fix failures.
2. Benchmark warmed-up local requests under a stated load; report p50/p95 latency and requests/sec **as measured**, not guessed.
3. Log structured request outcomes without recording raw user identifiers or sensitive payloads.
4. Document monitoring: API errors, latency, input distributions, score distributions, label quality, delayed outcomes and model freshness.
5. Describe a reproducible retrain → evaluate → promote/rollback workflow.

**Required deliverable:** `reports/latency.md`, expanded tests, model/version information, and monitoring/retraining notes.

**Optional:** Container build/smoke test and lightweight CI (tests on pull requests).

**Day 6 done when:** You can demonstrate correctness, report actual local performance, and explain what could fail after deployment.

### Day 7 — Integrate, demonstrate and defend design choices

**Learn:** explaining ambiguous requirements, debugging under time pressure, defending algorithm choices, reviewing another engineer's code, and explaining trade-offs across data/model/API layers.

**Implement:**

1. Start from a fresh checkout/environment and run the documented setup.
2. Run generation → profiling → training → evaluation → API → ranking.
3. Each contributor independently fixes or extends one small function with tests.
4. Swap code-review roles and explain the full system verbally.
5. Record limitations, known issues and sensible next steps in the README or project notes.

**Required deliverable:** Passing tests, reproducible live demo, clean Git history, and a short design/retrospective note.

**Day 7 done when:** Both developers can run, debug, and explain the system without depending on code generation to answer basic implementation questions.

## Working together

### GitHub workflow

1. Create the repository (suggested name: `adtech-ml-capstone`).
2. Protect `main` if desired and avoid direct commits to it.
3. Create small issues for each day's deliverables; use branches such as `day1/data-profile` or `day3/ranking`.
4. Open focused PRs, link the issue, and attach test output and a short explanation of the change.
5. Have the other developer review before merging. Avoid merging a PR if neither person understands how it works.
6. At the end of each day, run the main branch's smoke tests together.

### Suggested division of work

| Day | Contributor A leads | Contributor B leads | Together |
|---|---|---|---|
| 1 | Data exploration and CTR | Data validation and tests | Review data assumptions and labels |
| 2 | Preprocessing and model training | Baseline and evaluation | Compare metrics and leakage risks |
| 3 | Candidate scoring/ranking | Eligibility logic and ranking tests | Demo and peer review |
| 4 | Event aggregation | Audience rules and temporal tests | Verify past-only features |
| 5 | HTTP endpoint and schemas | API tests and client examples | End-to-end request walkthrough |
| 6 | Benchmark and model monitoring | Reliability tests and documentation | Performance review |
| 7 | Integration demo | Independent debugging challenge | Swap roles and present full system |

**Rotate roles when possible.** Leading a task is not exclusive ownership: both should read the PR, run the tests, and explain the implementation.

### Daily collaboration rhythm (per person, target 3 hours)

- **45 minutes — Learn:** review the day's concepts and agree on success criteria.
- **90 minutes — Build:** implement a small, testable vertical slice.
- **45 minutes — Validate:** test, peer-review, document one trade-off, and merge.

If AI coding tools are used, treat their output as an untrusted draft: inspect code, run tests, and verify every important assumption. Each contributor should also complete some independent implementation and debugging.

## Quality gates and definition of done

A day's work is complete only if:

- The relevant code runs from a documented command.
- The change is covered by focused tests where practical.
- New dependencies are documented and pinned or constrained appropriately.
- Generated data is reproducible; no private customer information or API secrets are committed.
- The PR explains assumptions, trade-offs, and known limitations.
- The other contributor can understand and reproduce the result.

### Cross-cutting checks

- **Leakage:** `clicked`, `converted`, and future events never enter the impression-time feature vector.
- **Integrity:** IDs, event timestamps, label domains and required fields are validated.
- **Evaluation:** Compare against a trivial baseline and report split method and limitations.
- **Serving:** Preserve training/inference preprocessing parity by saving the fitted pipeline.
- **Reliability:** Handle invalid requests, missing artifacts and empty candidate lists explicitly.
- **Privacy:** Use synthetic identifiers and minimize logging. Real user identifiers require a separate privacy review.

## Evaluation and performance

### Offline ML metrics

| Metric | Why it matters |
|---|---|
| Log loss | Evaluates quality of probabilistic click predictions |
| ROC-AUC | Measures discrimination across thresholds |
| PR-AUC | Especially informative with rare positive clicks; compare with base rate |
| Calibration / Brier score (optional) | Checks whether predicted probabilities correspond to observed rates |
| Accuracy (secondary) | May be misleading on imbalanced labels |

**Data split:** Start with a reproducible holdout and record the method. For more realistic evaluation, consider chronological splits and grouping by user to limit leakage or optimistic reuse. Do not report metrics without explaining what the split represents. Fit encoders and other learned preprocessing only on the training portion.

### Ranking

The core demo ranks eligible candidates by model-estimated CTR. This is a straightforward baseline. Formal ranking metrics such as NDCG require a suitable relevance-labeled candidate set; do not claim offline ranking improvement solely from a CSV of independently observed impressions.

### Service metrics

Record request count, error rate, p50/p95 latency, requests/sec, model version and environment. Note whether results include serialization and API overhead and whether the server was warmed up. All performance numbers must come from actual runs.

### Business metrics

CTR, conversion rate, audience coverage and revenue/yield are useful real-world goals. Our synthetic offline experiment **cannot establish** real-world changes to any of them. A real A/B test would require randomized traffic assignment and appropriate measurement.

## Risks, limitations, and stretch goals

| Risk or limitation | Practical approach |
|---|---|
| One-week time budget | Prioritize a working scikit-learn baseline, ranking, API and tests |
| Synthetic click labels | Clearly report that patterns were generated; don't claim external validity |
| No real auction, bids or budgets | Rank by predicted CTR only; document the missing economic objective |
| Simplified audience rules | Implement clear temporal rules; do not present as trained conversion prediction |
| Repeated users and time effects | Document split assumptions and, if possible, compare to temporal evaluation |
| Dataset lacks full candidate exposure information | Avoid claiming unbiased real ranking evaluation |
| Optional neural model takes too long | Defer PyTorch comparison until required deliverables are complete |
| Time pressure | Drop Docker/CI polish before dropping correctness tests or API functionality |

**After Day 7, optional extensions:** public click dataset experiment, calibrated deep CTR model with embeddings, trained audience conversion model, richer contextual features, a streaming broker, online feature store, counterfactual/off-policy evaluation, load testing at scale, and cloud/Kubernetes deployment.

## Final demonstration

A successful demo should show, in this order:

```text
1. Generate sample impression + event data.
2. Display data checks and descriptive CTR breakdowns.
3. Train baseline and CTR model; show held-out metrics.
4. Build audience eligibility from past behavior events.
5. Submit request with candidate ads to POST /rank.
6. Show eligibility filtering, ranked results, and model version.
7. Run unit/integration tests.
8. Show measured API latency and explain monitoring/retraining.
9. Have each contributor explain one major design choice and fix a small bug.
```

**Final guiding principle:** Understand the problem → make the data trustworthy → choose the simplest useful model → build a working system → measure and test it → explain what it can and cannot do.

---

**License:** Not yet selected. Add a license before inviting public reuse or contributions outside the project team.
