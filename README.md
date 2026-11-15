# driftwatch

Model monitoring and safe deployment for a credit-scoring model. driftwatch
watches a production model batch by batch (schema and data-quality checks,
PSI, KS, chi-square and MMD for data drift, a label-free estimate of today's
AUC, ADWIN on calibration once outcomes arrive), turns detector signals into
alerts with persistence and cooldowns, keeps every model version with its
lineage in a registry, and releases new versions through shadow and canary
stages whose statistical guardrails roll a bad release back on their own.
When drift is confirmed it retrains, and a champion-challenger gate decides
whether the challenger is worth rolling out.

Everything runs on a laptop CPU with no API key and no network at runtime:
Python 3.12 or newer, `uv`, scikit-learn, scipy, FastAPI, SQLite.

## Why

Credit risk is the hard case for monitoring. The label (default within
twelve months) arrives long after the decision, so performance cannot simply
be measured; the applicant mix moves with marketing and the economy, and a
moved mix is not the same as a broken model; the pipeline feeding the model
breaks in ordinary ways (an outage nulls the bureau score, income starts
arriving annual instead of monthly); and a bad release costs money in
proportion to the applications it decides. A monitor that alerts on every
shift trains its operators to ignore it, and one that waits for labels is
months late.

driftwatch is built around those constraints and measured against a seeded
simulator in which every row's true probability of default and every
incident's start batch are known, so detection delay, false alarms,
estimation error and rollback decisions are scored against ground truth
rather than eyeballed.

## Architecture

```
 origination pipeline           driftwatch service: one ControlLoop behind FastAPI
 --------------------           ------------------------------------------------------------
 POST /v1/batches   --------->  coerce + schema check --> champion PD (and candidate PD
 (one day of applications)        |                       during a rollout) --> decisions
                                  v                                    |
                               data quality: nulls, ranges,            v
                               unit changes, unseen categories    Monitor (decision time):
                                  |                               KS / chi-square / PSI per
                                  |                               feature, MMD, score KS, CBPE
                                  +----------------+--------------+
                                                   v
                                     AlertPolicy: 2-of-3 persistence, cooldowns,
                                     drift explained by a quality incident suppressed
                                                   |
 POST /v1/batches/{id}/labels --> Monitor (label time): ADWIN on y - p, label rate,
 (outcomes, 30 batches later)     DDM (advisory); rollout label guardrails
                                                   |
                                                   v
                     RetrainTrigger (blocked by pipeline incidents, backs off after a
                     rejection) --> refit and recalibrated challengers --> gate (paired
                     log loss and DeLong AUC, alpha/2 each) --> Rollout: shadow -> 5% ->
                     25% -> 50% -> production. Guardrails: runtime, rank agreement,
                     approval rate, calibration, discrimination; above 5% needs labels.
                                                   |
                     ModelRegistry (SQLite, hashed artifacts, lineage, audited stages)
                     Store (SQLite: batches with features, labels, alerts, rollouts)
```

`docs/ARCHITECTURE.md` describes the modules and both data paths;
`docs/EVALUATION.md` describes the simulator, every metric and how to
reproduce the numbers.

## Quickstart

```
make setup        # uv sync (Python 3.12 or newer, runtime and dev dependencies)
make bench        # all benchmarks plus the threshold gate, a few minutes on 2 worker processes
make serve        # API and dashboard on http://127.0.0.1:8774
make replay       # second shell: streams a concept-drift scenario into the service
```

Open http://127.0.0.1:8774 for the dashboard. On first start the service
trains the reference model on a seeded clean history and registers it as
version 1, which takes a few seconds. `make lint` and `make test` run ruff, the house
text rules and the test suite.

## Measured results

Produced by `make bench` (seed 2026, two worker processes, each with one
BLAS and one OpenMP thread). The block below is generated from
`benchmarks/results.json`; `benchmarks/RESULTS.md` has every table, including
the per-detector matrix and the estimation errors per stream.

<!-- RESULTS:BEGIN -->
| Metric | Measured |
|---|---:|
| False alarms per 10,000 drift-free batches (alert policy) | 0.00 |
| `concept_abrupt`: recall, median delay (batches) | 1.00, 32.0 |
| `concept_gradual`: recall, median delay (batches) | 1.00, 53.5 |
| `covariate_abrupt`: recall, median delay (batches) | 1.00, 1.0 |
| `covariate_gradual`: recall, median delay (batches) | 1.00, 11.0 |
| `covariate_subtle`: recall, median delay (batches) | 1.00, 1.5 |
| `dq_new_category`: recall, median delay (batches) | 1.00, 0.0 |
| `dq_nulls`: recall, median delay (batches) | 1.00, 0.0 |
| `dq_schema_break`: recall, median delay (batches) | 1.00, 0.0 |
| `dq_unit_change`: recall, median delay (batches) | 1.00, 0.0 |
| `prior_shift`: recall, median delay (batches) | 1.00, 32.0 |
| CBPE error of the AUC change, covariate_abrupt (MAE; naive estimator in brackets) | 0.0055 (0.0357) |
| CBPE error of the AUC change, concept_abrupt (MAE; naive estimator in brackets) | 0.0255 (0.0277) |
| Planted bad releases rolled back | 1.000 |
| Good releases rolled back (false rollback rate) | 0.000 |
| Champion-challenger gate decisions correct | 1.000 |
<!-- RESULTS:END -->

What the numbers say:

* **False alarms.** The alert policy raised no alert in 10,000 drift-free
  batches (25 control streams of 400 daily batches). Raw detectors fire more
  often (per 10,000 batches: CBPE 40, MMD 31, label rate 12, score KS 9,
  chi-square 7, KS 4, DDM 1); 2-of-3 persistence removes those single-batch
  flukes, at the cost of one batch of delay on label-free drift.
* **Detection.** All 100 drifted streams (10 per type) were detected within
  the 80-batch window, every first alert named the right kind of problem, and
  no stream alerted before its onset. Data-quality incidents alert on the
  batch they start. Concept drift changes nothing a model can see before
  outcomes arrive, and prior shift moves the score distribution only slightly
  at 400 rows a batch (the raw score KS fires on 4 of 10 streams, a median 36
  batches after onset), so both are caught by the label detectors a median 32
  batches after onset, 30 of which are the label delay.
* **Label-free estimation.** CBPE tracks the change in AUC under covariate
  shift with a mean absolute error of 0.0055, where assuming no change errs by
  0.0357. It cannot see concept drift (0.0255 against 0.0277) because its one
  assumption, that P(default | x) is unchanged, is exactly what broke, and it
  is worse than no estimate when a feature disappears (schema break: 0.0220
  against 0.0103). On stationary data it reads 0.0157 AUC high because the
  model is overconfident in its highest-risk decile (mean PD 0.335 against a
  true 0.309). The alarm compares CBPE with its own reference band, so that
  level bias cancels there.
* **Deployment safety.** 72 of 72 planted bad releases were rolled back and 24
  of 24 routine retrains were promoted, with the registry in the right state
  every time. Inverted scores, an income-units bug, a dropped calibration
  step and a 5 percent crash rate were rolled back in the first shadow batch,
  before deciding a single application. A model trained on 1,200 rows, worse
  only on outcomes (noise-free AUC 0.742 against the champion's 0.784), was
  rolled back by the label-based discrimination guardrail after a median 40
  batches and 702.5 applications decided at 5 percent traffic.
* **Retraining.** All 42 gate decisions were correct against ground truth.
  After concept drift a refit was promoted in 6 of 6 runs and recovered 0.056
  noise-free AUC over keeping the champion; after prior shift the
  recalibrated champion was promoted in 6 of 6 runs (log loss better by
  0.0163, AUC unchanged, as label shift predicts); after covariate shift every
  challenger was rejected because the champion stays calibrated; a unit-change
  incident blocked retraining in 6 of 6 runs; no retrain started on a control
  stream.

## How it works

**Data.** `driftwatch.data` draws applications from a latent creditworthiness
factor that drives the bureau score, utilisation and delinquencies, with age,
income, loan size, tenure and an affordability ratio (FOIR) on top, and
defaults from an explicit scorecard-like log-odds with one interaction and one
non-linearity. Ten scenarios each change one mechanism at a known batch:
covariate shift (abrupt, gradual, subtle), prior shift (by rejection sampling,
so P(x | y) is exact), concept drift (abrupt, gradual), and four pipeline
incidents. The reference model (histogram gradient boosting with Platt
calibration) is trained on a clean 60-batch history.

**Detectors.** One reference profile per model version; nulls and unseen
categories are kept out of the drift statistics because they are
data-quality findings with their own checks.

| Detector | Needs labels | What it is for |
|---|---|---|
| schema | no | missing, renamed or retyped columns |
| null rate, range, unit change, unseen category | no | pipeline incidents, with a statistical and a practical floor |
| KS (continuous), chi-square (discrete, categorical) | no | per-feature drift, Bonferroni across features |
| PSI | no | effect size, alarm at 0.25 |
| MMD, random Fourier features, Gamma null | no | joint drift of the numeric features |
| score KS | no | prediction drift |
| CBPE | no | expected AUC from calibrated probabilities, 3 sigma band |
| ADWIN on `y - p` | yes | calibration drift, blind to a mere change of mix |
| label rate | yes | default-rate change, two-sided binomial |
| DDM | yes | advisory only: see `docs/EVALUATION.md` |

**Deployment safety.** During a rollout both models score every row and the
candidate decides only its canary share, so the guardrails draw power from
all traffic while exposure stays small. Label-free guardrails (runtime
errors, rank agreement with the champion, a confidence sequence on the paired
approval-rate difference) act from the first shadow batch; label guardrails
(calibration, DeLong non-inferiority with alpha spent over every look) act
when outcomes arrive. Traffic above 5 percent opens only after the candidate
is shown non-inferior, and a candidate that cannot show it within 150 batches
is rolled back.

**Retraining.** Performance and label alerts trigger a retrain; label-free
alerts only when they recur; schema and data-quality alerts block it, since
training on a broken pipeline teaches the model the bug. The training window
starts just before the evidence. A full refit and the champion recalibrated
on recent labels are gated at alpha/2 each against the champion on the newest
labelled slice; a passing challenger goes to shadow with its lineage (parent,
window, trigger, gate metrics) in the registry.

## API

| Method and path | Purpose |
|---|---|
| `POST /v1/batches` | score, route and monitor a batch; returns decisions, alerts, rollout action |
| `POST /v1/batches/{batch_id}/labels` | outcomes for every application of an earlier batch |
| `GET /v1/timeline` | per-batch drift statistics, estimates and realised metrics |
| `GET /v1/alerts`, `POST /v1/alerts/{id}/ack` | alert list and acknowledgement |
| `GET /v1/registry/models`, `GET /v1/registry/models/{version}` | versions, lineage chain, audit events |
| `GET /v1/rollout`, `POST /v1/rollout/{version}/start`, `POST /v1/rollout/abort` | rollout state and control |
| `POST /v1/retrain` | train and gate challengers now |
| `GET /health`, `GET /metrics` | health and Prometheus text metrics |

Writes need `X-API-Key` when `APP_API_KEY` is set. Every response carries
`X-Request-ID` and every log line is JSON with the same id.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `APP_API_KEY` | unset | require this key on write endpoints |
| `DRIFTWATCH_DATA_DIR` | `data` | registry, artifacts and SQLite store |
| `DRIFTWATCH_AUTO_RETRAIN` | `true` | let alerts start retraining |
| `DRIFTWATCH_MAX_ROWS` | `5000` | largest accepted batch |
| `DRIFTWATCH_HISTORY_SEED` | `7` | seed of the clean history the first model is trained on |
| `DRIFTWATCH_LOG_LEVEL` | `INFO` | log level |

## Docker

```
docker build -t driftwatch:local .
docker compose up                    # API on port 8774, data in a named volume
docker compose --profile demo up     # plus a replay client streaming concept drift
```

The image is multi-stage, runs as a non-root user and has a health check.

## Limitations

* The data is simulated. The generator is statistically motivated and
  documented, but real applicant data has dependencies, seasonality and
  reporting quirks it does not have; the detection numbers are a property of
  these scenarios at 400 rows per batch, not a guarantee.
* Labels are assumed to arrive for every application, including rejected
  ones (as bureau-reported outcomes on loans taken elsewhere can provide).
  Real lenders mostly observe outcomes of approved loans only, which biases
  realised AUC and the rollout's label guardrails; selective labels are not
  modelled.
* The label delay is a fixed 30 batches; real outcome windows are longer and
  vary by product.
* CBPE carries a level bias of about 0.016 AUC on this model and fails by
  design under concept drift; the EM prior estimate is noisy on single
  400-row batches and wrong under covariate shift.
* Detector state (ADWIN window, DDM, alert history) is in memory and restarts
  with the service; a restart also rolls back any rollout in progress.
* One deployment slot: one production model and one candidate at a time.

## Roadmap

* Selective labels: outcomes only for approved applications, with a small
  random-approval holdout for unbiased monitoring.
* Pooled EM over a rolling window of batches for the default-rate estimate.
* Persist detector state so a restart resumes monitoring instead of
  re-learning it.
* Several deployment slots (one per product or region) behind one registry.

## Layout

```
src/driftwatch/
  schema.py              feature contract, coercion of raw batches
  data/                  generator and scenarios (seeded, ground truth per row)
  model.py, training.py  gradient boosting with Platt calibration; time-ordered splits
  stats.py               KS, chi-square, PSI, AUC, expected AUC, DeLong, confidence sequences
  detectors/             drift, quality, label-free estimation, performance
  alerts.py, monitor.py  signals to alerts; all detectors per batch
  registry.py, store.py  versions, lineage, audit; operational state
  deploy/                rollout guardrails; retraining trigger and gate
  loop.py                the control loop the API and benchmarks drive
  api/                   FastAPI app, schemas, metrics, logging, dashboard
  bench/                 benchmark tasks, aggregation, report, threshold gate
tests/                   unit, property-based, integration and API tests
benchmarks/              results.json, RESULTS.md, thresholds.json, history/
```
