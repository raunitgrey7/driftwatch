# Report

Project: driftwatch, model monitoring and safe deployment for a credit-scoring
model (spec 14). Built and measured on macOS-26.5.1-arm64-arm-64bit-Mach-O
(15 CPUs, Apple Silicon) with Python 3.13.17, numpy 2.5.4, scipy 1.18.1 and
scikit-learn 1.9.1. The benchmark numbers below come from `make bench` run in
a fresh clone of commit 01efb52 from a clean environment (`env -i` with only
`HOME` and a minimal `PATH`), generated at 2026-10-25T05:46:54+00:00; the
same files are committed as `benchmarks/results.json` and
`benchmarks/RESULTS.md`.

## What was built

* `driftwatch.data`: a seeded simulator of daily batches of 400 credit
  applications. A latent creditworthiness factor drives the bureau
  variables; income, loan, tenure and FOIR follow documented distributions;
  default follows an explicit scorecard-like log-odds, so every row has a
  known true PD. Ten scenarios change one mechanism each at a known batch:
  covariate shift (abrupt, gradual, subtle), prior shift (rejection sampling
  on the label, P(x | y) exact), concept drift (abrupt, gradual) and four
  pipeline incidents (nulls, a x12 unit change, a renamed column in percent,
  a new category).
* `driftwatch.model` and `training`: histogram gradient boosting with Platt
  calibration and a per-version approval cut-off, trained on a time-ordered
  split of a clean 60-batch history.
* `driftwatch.stats`: exact O(n log m) two-sample KS from the sample side,
  Stephens-corrected p-values, chi-square, PSI, midrank AUC, the
  ratio-of-expectations AUC under per-row probabilities, DeLong paired AUC
  variance, a normal-mixture confidence sequence, a Gamma null fit.
* `driftwatch.detectors`: schema, null-rate, range, unit-change and
  unseen-category checks; KS or chi-square plus PSI per feature with a
  Bonferroni correction; MMD with random Fourier features and a bootstrapped
  Gamma null; score KS; CBPE with a bootstrapped band and the EM prior
  estimate; ADWIN on calibration residuals; DDM on decision errors
  (advisory); a binomial label-rate test.
* `driftwatch.alerts` and `monitor`: signals for every evaluation, an alert
  policy with k-of-m persistence, cooldowns and suppression of drift that a
  data-quality incident explains; the per-batch and per-label-delivery
  monitor.
* `driftwatch.registry` and `store`: SQLite registry with versions, parents,
  lineage, gate metrics, an audited stage machine, atomic promotion and
  rollback, and SHA-256-verified artifacts; SQLite store of batches (features
  as `npz`, no pickles), labels, alerts and rollouts.
* `driftwatch.deploy`: shadow and canary rollout with five guardrails
  (runtime, rank agreement, approval rate by confidence sequence, calibration,
  DeLong non-inferiority with alpha spent over every look), traffic above 5
  percent gated on label evidence, automatic rollback; a retraining trigger
  blocked by pipeline incidents with a backoff after rejection, a refit and a
  recalibrated-champion challenger, and the paired champion-challenger gate.
* `driftwatch.loop`: the control loop the API and the benchmarks both drive;
  it rolls back rollouts interrupted by a restart.
* `driftwatch.api`: FastAPI with typed Pydantic schemas, `/health`,
  `/metrics` in Prometheus text format, JSON logs with request ids, API-key
  auth on writes when `APP_API_KEY` is set, and a static dashboard (drift
  heatmap, estimated against realised AUC, approval rate, rollout guardrails,
  registry, alerts) without a build step.
* `driftwatch.replay` and `cli`: `driftwatch serve`, `replay`, `scenarios`
  and `bench`.
* `driftwatch.bench`: the benchmark (detection, estimation, deployment
  safety, retraining) on a two-process pool, `results.json`, `RESULTS.md`,
  the README block and a 54-rule threshold gate.
* Packaging: Makefile (`setup`, `lint`, `test`, `bench`, `serve`, plus
  `replay`), multi-stage non-root Dockerfile with a health check,
  docker-compose with an optional replay profile, GitHub Actions CI running
  lint, tests and the benchmark gate.

## Measured results

Command for every number in this section: `make bench`
(`uv run python -m driftwatch.bench`), seed 2026, two worker processes. The
key in `benchmarks/results.json` is given where it helps.

### Reference model (`champion`)

| Metric | Value |
|---|---|
| Training rows, calibration rows | 16,000, 4,000 |
| Realised AUC on the 4,000-row holdout | 0.7905 |
| Noise-free AUC on a fresh 8,000-row sample | 0.7843 |
| Mean PD against true default rate, fresh sample | 0.0819 against 0.0802 |
| Approval cut-off and approval rate, fresh sample | PD 0.1165, 0.801 |
| Top-decile reliability (mean PD against mean true PD) | 0.335 against 0.309 |

### Detection (`detection`)

| Drift type | Recall | Precision | Median delay | p90 delay | Diagnosis |
|---|---:|---:|---:|---:|---:|
| covariate_abrupt | 1.00 | 1.000 | 1.0 | 1.0 | 1.00 |
| covariate_gradual | 1.00 | 1.000 | 11.0 | 13.0 | 1.00 |
| covariate_subtle | 1.00 | 1.000 | 1.5 | 2.0 | 1.00 |
| prior_shift | 1.00 | 1.000 | 32.0 | 33.1 | 1.00 |
| concept_abrupt | 1.00 | 1.000 | 32.0 | 33.1 | 1.00 |
| concept_gradual | 1.00 | 1.000 | 53.5 | 56.1 | 1.00 |
| dq_nulls | 1.00 | 1.000 | 0.0 | 0.0 | 1.00 |
| dq_unit_change | 1.00 | 1.000 | 0.0 | 0.0 | 1.00 |
| dq_schema_break | 1.00 | 1.000 | 0.0 | 0.0 | 1.00 |
| dq_new_category | 1.00 | 1.000 | 0.0 | 0.0 | 1.00 |

Ten streams per type, delays in batches after the onset (label delay 30).
False alarms of the alert policy on 10,000 drift-free batches: 0 (0.00 per
10,000). Raw detector fires per 10,000 control batches: CBPE 40, MMD 31,
label rate 12, score KS 9, chi-square 7, KS 4, DDM 1.

### Label-free estimation (`estimation_post_onset`, `control.estimation`)

| Stream | AUC change error, CBPE | AUC change error, naive | CBPE level bias | Default rate MAE, mean PD | Default rate MAE, EM |
|---|---:|---:|---:|---:|---:|
| control | 0.0046 | 0.0085 | 0.0157 | 0.0021 | 0.0269 |
| covariate_abrupt | 0.0055 | 0.0357 | 0.0196 | 0.0036 | 0.2885 |
| covariate_gradual | 0.0056 | 0.0307 | 0.0192 | 0.0031 | 0.2318 |
| prior_shift | 0.0068 | 0.0078 | 0.0220 | 0.0510 | 0.0276 |
| concept_abrupt | 0.0255 | 0.0277 | 0.0410 | 0.0453 | 0.0429 |
| dq_schema_break | 0.0220 | 0.0103 | -0.0063 | 0.0075 | 0.0590 |

All eleven streams are in `benchmarks/RESULTS.md`.

### Deployment safety (`deployment`)

| Release | Runs | Rolled back | Median batches to decision | Median exposure | True AUC | Decided by |
|---|---:|---:|---:|---:|---:|---|
| inverted_score | 12 | 1.00 | 1.0 | 0 | 0.2150 | rank agreement |
| unit_bug | 12 | 1.00 | 1.5 | 0 | 0.7737 | approval rate |
| miscalibrated | 12 | 1.00 | 1.0 | 0 | 0.7849 | approval rate |
| crash_5pct | 12 | 1.00 | 1.0 | 0 | 0.7842 | runtime |
| no_bureau | 12 | 1.00 | 2.0 | 0 | 0.6905 | rank agreement |
| small_sample | 12 | 1.00 | 40.0 | 702.5 | 0.7419 | discrimination |
| retrain_same | 12 | 0.00 | 76.0 | 3026 | 0.7850 | promoted |
| retrain_more | 12 | 0.00 | 56.0 | 2345.5 | 0.7908 | promoted |

Pooled: 72 of 72 bad releases rolled back, 24 of 24 good releases promoted,
false rollback rate 0.000, registry in the correct final state in 96 of 96
runs. Excess defaults approved by the small-sample release before rollback:
1.25 per run on average.

### Retraining (`retraining`)

| Scenario | Retrains per run | Gate promote rate | Gate correct | New production | Median batches onset to production | True AUC lift | True log-loss gain | Winner |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| concept_abrupt | 1.00 | 1.00 | 1.00 | 6 of 6 | 97.5 | 0.0560 | 0.0426 | refit 6 |
| concept_gradual | 1.00 | 1.00 | 1.00 | 6 of 6 | 121.0 | 0.0500 | 0.0394 | refit 6 |
| prior_shift | 1.00 | 1.00 | 1.00 | 6 of 6 | 97.5 | -0.0001 | 0.0163 | recalibrate 6 |
| covariate_abrupt | 3.00 | 0.00 | 1.00 | 0 of 6 | n/a | n/a | n/a | none |
| dq_unit_change | 0.00 | n/a | n/a | 0 of 6 | n/a | n/a | n/a | none |
| control | 0.00 | n/a | n/a | 1 of 6 (forced) | n/a | -0.0000 | 0.0003 | recalibrate 1 |

All 42 gate decisions were correct against ground truth. Control streams got
one forced retrain each; 5 of 6 were rejected and the one promotion was
within the 0.0005 log-loss tolerance of the champion, so it is judged
acceptable.

### Cost

34,000 monitored batches; 161.5 s wall time for the whole benchmark on two
processes in that run (136.0 s and 107.2 s in earlier full runs on the same
code paths, depending on machine load), 5.253 ms of single-thread CPU per
monitored batch including generating, scoring and every detector
(`timing.monitor_ms_per_batch`).

## Quality gate history

The 54 rules in `benchmarks/thresholds.json` were written before the first
full run. Runs on the way:

1. Commit a349a6d, first full run: 53 of 54 rules passed. Failed:
   `FAIL retraining.control.triggered_retrains_per_run = 0.333333 (want <= 0.2)`.
   Output kept in `benchmarks/history/run1-a349a6d-results.json`,
   `run1-a349a6d-RESULTS.md` and `run1-a349a6d-gate.txt` (the last produced
   by `uv run python -m driftwatch.bench --check-only` on that results file).
   Cause: every early performance false alarm, nine before an onset on the
   drift streams and the two that started retrains on control streams, came
   from DDM within 50 batches of the first labels. The threshold was not
   changed; DDM was given a 4,000-row burn-in and made advisory (commit
   dea8cf0). DDM also detected nothing in the matrix that other detectors
   had not caught first, so nothing was lost.
2. Commit dea8cf0 and every run since: 54 of 54. The detection, deployment
   and retraining sections of the last three runs are identical, as the
   harness's determinism predicts.

## Findings recorded during the build

* **Prior-shift ground truth was wrong in the first version of the
  generator.** It recorded the pre-shift PD for rows accepted by the label
  rejection step; the true PD of an accepted row is the posterior
  `a1 q / (a1 q + a0 (1 - q))`. Noticed because every prior-shift retrain was
  rejected and the ground truth agreed. Fixed before the first full run
  (commit a349a6d), with a test that the mean true PD matches the realised
  default rate under prior shift.
* **Retraining could not fix prior shift** until a second challenger was
  added: the champion recalibrated on recent labels. A full refit on a few
  thousand recent rows loses more discrimination than it gains in level; the
  recalibrated champion keeps the ranking and wins 6 of 6 runs.
* **DDM pages falsely during warm-up** (above).
* **CBPE has a level bias of 0.0157 AUC** from an overconfident top decile;
  the alarm is self-referenced and the change error is the honest accuracy
  measure, both documented in `docs/EVALUATION.md`.
* **Raw MMD and CBPE fire above their nominal rates** (31 and 40 per 10,000
  against 10 and 13.5); KS, chi-square, score KS and the label-rate test are
  at or near nominal. The 2-of-3 persistence rule is what keeps the policy at
  zero false alarms.
* **Hypothesis found two bugs**: records with no fields collapsed to a
  zero-row batch (the id column now always survives, and empty batches are
  refused); and the unit-change message named x10 for a x12 change whose
  batch median came out at 11.2 (the closest factor is now chosen).
* **scikit-learn's binning crashes on an all-NaN column**, which is how the
  first version dropped features; dropped features are now sliced out of the
  matrix.
* **The `.gitignore` rule `data/` also matched `src/driftwatch/data`**, so the
  generator was not committed until a lint run in a fresh clone exposed it
  (commit 01efb52). Commits before 01efb52 cannot be built from a clone.
* **Speed**: computing the KS statistic from the sample side only and using
  `scipy.special` ufuncs instead of `scipy.stats` distribution objects, plus
  vocabulary codes instead of the booster's native categorical encoder, cut
  the monitor's cost per batch during development (profiled, not
  benchmarked separately); the final per-batch cost is the `timing` figure
  above.

## Live run over HTTP

`DRIFTWATCH_DATA_DIR=<scratch dir> uv run driftwatch serve --port 8774`, then
`uv run driftwatch replay --api http://127.0.0.1:8774 --scenario concept_abrupt --batches 200 --onset 40`
on the final code (the quiet progress lines every 20 batches are omitted):

```
batch   40 2026-02-10 drift weight 1.00 | quiet
batch   72 2026-03-14 drift weight 1.00 | alert label_drift:labels; alert performance:model
batch   92 2026-04-03 drift weight 1.00 | retrain v2 (refit) passed: challenger has lower log loss and non-inferior AUC
batch   97 2026-04-08 drift weight 1.00 | rollout advance
batch  105 2026-04-16 drift weight 1.00 | alert label_drift:labels
batch  128 2026-05-09 drift weight 1.00 | rollout advance
batch  133 2026-05-14 drift weight 1.00 | rollout advance
batch  138 2026-05-19 drift weight 1.00 | rollout promote
batches=200 labelled=170 alerts=3 retrains=1 rollout_events=4
```

Afterwards `/metrics` reported `driftwatch_champion_version 2`,
`driftwatch_rows_scored_total{version="1"} 53228` and `{version="2"} 26772`,
and `/v1/registry/models/2` showed lineage `[2, 1]`, training window
`[37, 62]` and the audited stages registered, shadow, canary, production.
The dashboard HTML and script were served with status 200 and the script
passes `node --check`.

## Docker

`docker build -t driftwatch:local .` built in 18.7 s with the base images
cached; `docker run -p 18774:8774 driftwatch:local` answered `/health` within
three polls one second apart, `docker exec ... id` printed
`uid=999(driftwatch)`, and the image is 639 MB. The image was removed
afterwards. The compose demo profile was not run.

## Tests

140 tests (`make test`, 51.3 s in the clean-clone run): statistics against
scipy, scikit-learn or brute-force definitions (property-based with
hypothesis), the generator's one-mechanism contract for every scenario,
detectors quiet on clean data and firing on their mechanism, the alert policy,
the model's calibration and determinism, the registry stage machine and
tampering, each guardrail on planted defects, the retraining trigger and
gate, the control loop end to end (concept drift to promotion, a bad release
to rollback, restart recovery, incidents blocking retraining), the HTTP layer
(validation, auth, status codes, metrics, request ids), benchmark
determinism across runs and worker counts, README and results consistency,
and the house text rules. Adversarial cases: arbitrary JSON scalars in any
column, broken schemas over HTTP, a tampered artifact, crashing and inverted
releases. Determinism: same seed, same batch and the same benchmark output.

## Tail of `make lint` (fresh clone, clean environment)

```
uv run ruff check .
All checks passed!
uv run ruff format --check .
64 files already formatted
uv run python scripts/check_text.py
text checks: 0 problems
```

## Tail of `make test` (fresh clone, clean environment)

```
uv run pytest -q
........................................................................ [ 51%]
....................................................................     [100%]
140 passed in 51.27s
```

## Tail of `make bench` (fresh clone, clean environment)

```
wrote benchmarks/results.json in 161.5 s
PASS control.batches = 10000 (want >= 10000)
PASS control.false_alarms_per_10k = 0 (want <= 5)
...
PASS deployment.bad_rollback_rate = 1 (want >= 0.99)
PASS deployment.false_rollback_rate = 0 (want <= 0.2)
PASS deployment.registry_correct_rate = 1 (want >= 0.9)
PASS deployment.inverted_score.exposure_median = 0 (want <= 0)
PASS deployment.unit_bug.exposure_median = 0 (want <= 0)
PASS deployment.miscalibrated.exposure_median = 0 (want <= 0)
PASS deployment.crash_5pct.exposure_median = 0 (want <= 0)
PASS retraining.gate_correct_rate = 1 (want >= 0.8)
PASS retraining.control.triggered_retrains_per_run = 0 (want <= 0.2)
PASS retraining.dq_unit_change.triggered_retrains_per_run = 0 (want <= 0)
PASS retraining.concept_abrupt.true_auc_lift_after_promotion_mean = 0.056036 (want >= 0.03)
PASS retraining.prior_shift.true_logloss_gain_after_promotion_mean = 0.016338 (want >= 0)
54/54 threshold rules passed
```

The same clone then ran `make serve`, which answered `/health` with
`{"status":"ok","version":"0.1.0",...,"champion_version":1,...}`.

## Known gaps and what was not achieved

* All evaluation is on simulated data from one generator and one model
  family; no public credit dataset was used.
* Selective labels are not modelled: outcomes arrive for every application,
  including rejected ones, after a fixed 30 batches.
* DDM is implemented and evaluated, but it does not page: as a pager it gave
  false alarms during warm-up and added no detection here.
* CBPE's level bias (0.0157 AUC) is reported, not corrected. The EM prior
  estimate is unpooled, noisy on 400-row batches and wrong under covariate
  shift; a rolling pooled version is on the roadmap.
* The MMD Gamma null and the CBPE band are anti-conservative per batch (see
  above); the persistence rule compensates, the nominal levels do not hold.
* Proving non-inferiority is slow: routine retrains needed a median 56 to 76
  batches to reach production with a 0.01 AUC margin and Bonferroni-spent
  looks.
* Detector state is in memory; a restart re-learns it and rolls back any
  rollout in progress. One deployment slot; SQLite with a single writer.
* The dashboard was verified as served, as syntactically valid and by
  rendering the loaded page in headless Chrome with no console errors; its
  buttons were not driven by an automated browser test.
* No LLM component was built.
* The docker-compose demo profile was not run, only the image build and a
  container health check.
