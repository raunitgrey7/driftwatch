# Evaluation

Every number in the README and in `REPORT.md` comes from `make bench`, which
writes `benchmarks/results.json` and `benchmarks/RESULTS.md`. This document
explains the data, the protocol and the metric definitions behind them.

## What is measured

1. **Detection**: does each kind of change raise the right alert, how soon,
   and how often do drift-free streams raise false ones?
2. **Label-free estimation**: how close is the estimate of today's AUC and
   default rate, made before any outcome is known, to the truth?
3. **Deployment safety**: are planted bad releases rolled back, how many
   applications do they decide first, and are routine retrains let through?
4. **Retraining**: does the trigger start retrains for the right reasons, and
   does the champion-challenger gate promote exactly the challengers that are
   truly better?

## Data: a seeded credit-application simulator

Real monitoring data has no ground truth for "when did the drift start" or
"what was this applicant's true risk", so the benchmark uses a simulator in
which both are known. `src/driftwatch/data/generator.py` documents each
equation; in short:

* A latent creditworthiness `z ~ N(0, 1)` drives the bureau score
  (`700 + 60 z` plus noise, clipped to 300..900), revolving utilisation
  (logistic in `z`) and delinquencies (Poisson with log-rate linear in `z`),
  so those observed features are correlated the way bureau data is.
* Age is `21 + Gamma(3.2, 5.0)`; income is log-normal (median 42,000 INR a
  month) with a concave age profile and employment offsets; the loan is a
  log-normal multiple of income; tenure is 12 to 60 months; the instalment
  follows the annuity formula at 15 percent a year and FOIR (existing
  obligations plus the new instalment, over income) measures affordability.
* Default within twelve months is `Bernoulli(sigmoid(eta(x)))` with `eta` a
  scorecard-like function of the observed features (bureau score,
  utilisation, FOIR, delinquencies, inquiries, income, a quadratic in age,
  employment type, thin file, purpose and channel effects, and a
  self-employed by FOIR interaction). The coefficients give a default rate of
  about 8 percent and a true-PD AUC near 0.80 (`tests/test_generator.py`
  checks both bounds).
* A few columns are missing completely at random at baseline rates of 0.4 to
  1.2 percent, so null-rate checks have a non-zero reference to beat.

Each batch is one day: 400 applications. Outcomes arrive 30 batches later.

### Scenarios

| Scenario | Mechanism | What changes |
|---|---|---|
| `control` | none | nothing |
| `covariate_abrupt` | P(x) | partner campaign: latent mean -0.45, sd 0.8, younger, income median x0.78, partner channel 48 percent; overnight |
| `covariate_gradual` | P(x) | the same, ramped linearly over 40 batches |
| `covariate_subtle` | P(x) | 30 percent of the same shift, overnight |
| `prior_shift` | P(y) | default rate x1.75 by rejection sampling on the label, so P(x given y) is exact |
| `concept_abrupt` | P(y given x) | FOIR weight x2.5, bureau weight x0.4, self-employed +0.7 log-odds; overnight |
| `concept_gradual` | P(y given x) | the same, ramped over 40 batches |
| `dq_nulls` | pipeline | 35 percent of `bureau_score` null |
| `dq_unit_change` | pipeline | `monthly_income` delivered as annual income (x12) |
| `dq_schema_break` | pipeline | `credit_utilization` renamed `credit_utilization_pct` and sent in percent |
| `dq_new_category` | pipeline | a new `employment_type` value `gig` for most contract and some self-employed rows |

Pipeline incidents corrupt the delivered columns only; the true PD is
computed on the clean data, because the world did not change. Under prior
shift the recorded true PD is the posterior after the rejection step, not
the pre-shift PD (an early version recorded the latter; see `REPORT.md`).

Every batch is generated from `SeedSequence(seed, spawn_key=(index,))`, so a
drifted stream equals the control stream with the same seed before its onset
(`tests/test_generator.py` checks this for every scenario).

### Reference model

`build_champion` trains histogram gradient boosting on batches 0 to 39 of a
clean history (seed 7, 16,000 rows), fits Platt scaling and the 80 percent
approval cut-off on batches 40 to 49, and profiles batches 40 to 59 as the
drift reference. Its measured quality is the first table of
`benchmarks/RESULTS.md` (holdout AUC 0.7905; noise-free AUC 0.7843 and mean
PD 0.0819 against a true default rate of 0.0802 on a fresh 8,000-row sample).

## Protocol

| Section | Streams | Length | Notes |
|---|---|---|---|
| Detection, control | 25 | 400 batches | 10,000 drift-free batches for false alarms |
| Detection, drift | 10 per drift type, 100 in all | 240 batches | onset uniform in [100, 140], seeded; window 80 batches |
| Deployment safety | 8 release kinds x 4 trained instances x 3 traffic streams = 96 | up to 220 batches | stationary traffic; stops when the rollout decides |
| Retraining | 6 scenarios x 6 streams = 36 | 300 batches | onset at batch 80; control streams get one forced retrain at batch 150 |

All streams use distinct seeds derived from the benchmark seed (2026). Labels
always arrive 30 batches after their batch. The detection section runs the
`Monitor` directly; the deployment and retraining sections run the full
`ControlLoop` with a real registry in a temporary directory, so stage
transitions are part of what is measured.

## Metric definitions

### Detection (alert policy)

* An event is **detected** when any alert opens in `[onset, onset + 80)`.
  **Delay** is the alert's batch minus the onset batch, so label-based
  detection includes the 30-batch label delay.
* **Recall** is detected events over streams of that type.
* Every alert before the onset is a **false alarm**. **Precision** per drift
  type is detected events over detected events plus the false alarms on that
  type's streams.
* **Diagnosis rate**: the share of detected events whose first alert category
  names the mechanism: data drift, prediction drift or estimated performance
  for covariate shift; label drift, performance or prediction drift for prior
  shift; performance or label drift for concept drift; schema or data quality
  for pipeline incidents.
* **False alarms per 10,000 batches**: all alerts on the control streams over
  the control batches, times 10,000.

### Detection (raw detectors)

The same rules applied to each detector's own fired signals with no
persistence, cooldown or suppression. On control streams a detector counts
once per batch in which any of its signals fired. Nominal per-batch rates
follow from the configured thresholds and can be compared with the measured
ones:

| Detector | Configured threshold | Nominal per 10,000 batches | Measured |
|---|---|---:|---:|
| KS (7 continuous features) | p below 0.001 / 16 per feature | at most 4.4 | 4 |
| chi-square (9 discrete or categorical) | p below 0.001 / 16 per feature | at most 5.6 | 7 |
| score KS | p below 0.001 | 10 | 9 |
| label rate | two-sided p below 0.001 | at most 10 | 12 |
| MMD | Gamma-approximated p below 0.001 | 10 | 31 |
| CBPE | estimate below reference mean minus 3 sd | 13.5 | 40 |

KS, chi-square, score KS and the label-rate test hold their nominal rates.
MMD and CBPE do not: the Gamma fit underestimates the far tail of the MMD
null, and the CBPE band is centred and scaled on a finite 8,000-row reference.
Neither reaches an alert because a second firing within three batches is
needed, but the nominal levels of those two should not be read as calibrated.

### Label-free estimation

Per batch, with the champion's scores `s` and calibrated PDs `p` (the same
numbers here):

* **Expected AUC** (the truth used for scoring): the AUC of `s` expected under
  the true PDs `q` of the batch's rows, `stats.expected_auc(s, q)`. It removes
  the Bernoulli noise of 400 outcomes.
* **CBPE**: the same formula with the model's own `p` in place of `q`.
* **Naive**: the AUC realised on the champion's reference holdout, assumed
  constant.
* **Level error**: mean absolute difference from the expected AUC; also
  reported against the realised AUC of the batch's 400 labels, which adds
  the sampling noise of 400 outcomes (on control streams the naive estimate
  is 0.0105 from the expected AUC but 0.0345 from the realised one).
* **Change error**: `|(CBPE_t - CBPE_ref) - (expected_t - expected_ref)|`, the
  error in the estimated change since the reference period; the naive change
  error is `|expected_t - expected_ref|`. A constant level bias cancels here,
  as it does in the alarm.
* **Default rate**: the mean PD and the EM prior estimate, each against the
  mean true PD of the batch.

Drift streams are scored on batches at or after the onset, control streams on
all batches.

### Deployment safety

* **Rolled back**, **promoted**: the final rollout status of each run.
* **Registry correct**: a bad release ends `rolled_back` with the champion
  still in production, or a good release ends in production with the
  champion archived.
* **Batches to decision**: batches the rollout ran until it promoted or
  rolled back.
* **Exposure**: applications decided by the candidate before that.
* **Excess defaults approved**: over rows the candidate decided, defaulters it
  approved that the champion would have rejected, minus the reverse.
* **True AUC**: the release's noise-free AUC on a fixed fresh 8,000-row sample.

### Retraining

* **Triggered retrains per run** (forced ones counted separately).
* **Gate correct**: each gate decision is judged on a fresh 20-batch sample
  from the distribution at decision time, with the true PD. The challenger
  *should* be promoted when its expected log loss is lower than the
  champion's by more than 0.001 and its expected AUC is not lower by more
  than 0.005; a promotion is *acceptable* when its expected log loss is within
  0.0005 of the champion's or better, with the same AUC condition. A decision
  is correct when it promoted an acceptable challenger or rejected one that
  should not have been promoted.
* **Batches from onset to production**, and on the batches after the first
  promotion the **true AUC lift** and **true log-loss gain** of the scores
  actually used against the original champion's on the same rows.
* **Promoted challengers**: which kind passed the gate (refit or recalibrate).

## Reproduce

```
make bench                                    # full run, writes benchmarks/, updates the README block, runs the gate
uv run python -m driftwatch.bench --quick --out /tmp/dw --no-check   # small run, same code paths
uv run python -m driftwatch.bench --check-only                       # re-apply the gate to results.json
uv run python -m driftwatch.bench --workers 1                        # single process
```

The output does not depend on scheduling: tasks are independent and seeded
and results are aggregated in task order. `tests/test_bench.py` runs a small
configuration twice in one process and once on two processes and requires
identical detection, deployment, retraining and headline sections. Wall time
depends on machine load; `results.json` records it under `timing`.

## Quality gate

`benchmarks/thresholds.json` holds 54 rules over the `headline` view of the
results, each with a reason. A missing or null metric fails its rule. The
rules were written before the first full run as product requirements. That
run failed one of them, `retraining.control.triggered_retrains_per_run`
(0.333 against at most 0.2); its results and gate output are kept in
`benchmarks/history/`, and the cause and the fix are in the findings below.
No threshold has been changed since.

## Findings

**DDM is not usable as a pager here.** In the first full run all nine false
alarms before an onset, and both spontaneous retrains on control streams,
came from DDM within 50 batches of the first labels. Its running minimum is
set on a few thousand noisy rows and the 3-sigma rule is then checked on every
row. The control streams showed the same rate per stream; only their length
diluted it per 10,000 batches. In the detector matrix DDM also caught nothing
that ADWIN or the label-free detectors had not caught first (recall 0 on
abrupt concept drift, where ADWIN has 1.0 at 32 batches). DDM now tracks its
minimum only after 4,000 rows and is advisory: its signal is kept on the
timeline, it never opens an alert. The policy then raised no false alarm in
10,000 control batches, and the raw DDM rate is 1 per 10,000.

**Watching calibration residuals separates broken models from changed
populations.** DDM on decision errors fired on every abrupt covariate shift
(the error rate rises when riskier applicants arrive, even though the model
is right), while ADWIN on `y - p` never fired on any covariate scenario and
caught concept drift and prior shift on every stream.

**CBPE tracks changes, not levels.** It reads 0.0157 AUC high on stationary
data because the model is overconfident in its highest-risk decile: on a
fresh sample that decile predicts a mean PD of 0.335 against a true 0.309,
while the other nine deciles are within 0.003 (`champion.calibration_deciles`
in `results.json`). CBPE assumes every probability is exact, so an
overconfident tail inflates its AUC. As a change estimate under covariate shift it is
accurate (0.0055 against 0.0357 for assuming no change); under concept drift
it barely beats the naive estimate (0.0255 against 0.0277), and after a
schema break it is worse (0.0220 against 0.0103), because in both its
assumption that the model's probabilities are right is what failed.

**The default rate cannot be estimated without knowing the shift type.**
The mean PD is right under covariate shift (MAE 0.0036) and wrong under prior
shift (0.0510); the EM prior estimate is the reverse (0.0276 under prior
shift, 0.2885 under covariate shift). Unlabelled data cannot tell the two
apart, which is one more reason the label-based detectors exist.

**Proving a retrain is safe takes time.** With a 0.01 AUC non-inferiority
margin, alpha spent over every possible look and a 30-batch label delay, a
same-recipe retrain needed a median 76 batches to be promoted and one trained
on twice the data 56. Bad releases were decided faster: the four gross
defects in one batch, the small-sample model in a median 40.

**Covariate shift spends compute, not trust.** It triggered 3 retrains per
run (recurring drift alerts and the label-rate test), every one rejected by
the gate and judged correctly rejected by the ground truth, so the champion,
which stays calibrated under covariate shift, kept serving.

## Limitations of this evaluation

* Simulated data from one generator and one model family. The relative
  behaviour of detectors is more transferable than the absolute delays.
* Labels for every application, a fixed 30-batch delay and a fixed batch size
  of 400. With fewer rows per batch every label-free detector slows down.
* The tests inside a batch assume independent rows; real batches have
  correlated rows (one dealer, one campaign).
* Each configuration ran once with 10 streams per drift type; recall of 1.00
  on 10 streams is consistent with a true recall somewhat below 1.
