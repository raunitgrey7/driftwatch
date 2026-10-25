# Architecture

driftwatch is one Python package, `driftwatch`, with a FastAPI service on top
and a benchmark harness beside it. The service and the benchmarks drive the
same object, `loop.ControlLoop`, so what the benchmark measures (rollbacks,
retrains, alerts) is what the service does.

```
            schema ---- data ---- model/training
               \          \           |
                \          v          v
              detectors (drift, quality, estimation, performance)
                         |
                       alerts ---- monitor
                                     |
        registry ---- store ---- loop (ControlLoop) ---- deploy (rollout, retrain)
                                     |
                       api (FastAPI)      bench (tasks, report, gate)
```

## Modules

### `driftwatch.schema`

The input contract: sixteen features, each with a kind (continuous,
discrete, categorical) that decides its drift test, and vocabularies for the
categoricals. `coerce` turns a raw column dictionary (anything JSON can
carry) into a typed `Batch` and returns every violation as a `SchemaIssue`
instead of raising: a missing column becomes an all-null column, an
unexpected column is dropped, an unparseable number becomes null. A
monitoring system has to describe a broken batch, not refuse it. `Batch`
caches vocabulary codes per categorical column, which the encoder, the drift
profile and the quality checks share.

### `driftwatch.data`

`generator.py` holds the applicant distribution (`Population`) and the
default mechanism (`Concept`) as frozen dataclasses, so a drift is a second
instance of one of them and a gradual drift is a linear interpolation.
`scenarios.py` builds ten scenarios on top and a `CreditStream` that
generates batch `i` of a stream from its own `SeedSequence(seed,
spawn_key=(i,))`: any batch can be regenerated alone, and a drifted stream
equals the control stream with the same seed before its onset. Every batch
carries the delivered columns (possibly corrupted), the labels, the true PD
of every row and the drift weight; only the benchmark reads the last three.

### `driftwatch.model` and `driftwatch.training`

`CreditModel` is histogram gradient boosting (categoricals as vocabulary
codes, nulls handled natively) with Platt scaling fitted on a later slice and
an approval cut-off set at training time (the PD that approves 80 percent of
the calibration population), so every version carries its own credit policy.
`Scorer` is the protocol every model and every planted defect implements.
`training.build_champion` splits a clean history by time: 40 batches to fit,
10 to calibrate, the last 20 as the drift reference, with the training rows
as the MMD null pool (disjoint from the reference anchor).

### `driftwatch.stats`

Per-batch statistics with the reference side precomputed: the KS statistic
evaluated on the sample's distinct values only (exact, O(n log m)), the
Stephens-corrected Kolmogorov p-value, chi-square of homogeneity, PSI,
midrank AUC, the ratio-of-expectations AUC under per-row probabilities (the
core of CBPE), DeLong's paired AUC variance, the normal-mixture confidence
sequence and a Gamma fit for the MMD null. p-values use `scipy.special`
ufuncs, not `scipy.stats` distribution objects, whose per-call overhead was
a third of the batch cost.

### `driftwatch.detectors`

* `drift.py`: `UnivariateDrift` (KS or chi-square plus PSI per feature from
  one fitted profile), `ScoreDrift` (KS on the model's output) and
  `MMDDetector` (RBF kernel via 128 random Fourier features, bandwidth by the
  median heuristic, null distribution per batch size bootstrapped from the
  pool and summarised by a Gamma fit, cached per size).
* `quality.py`: null-rate, range, unit-change and unseen-category checks and
  the conversion of schema issues into signals.
* `estimation.py`: CBPE (expected AUC, default rate, accuracy and bad rate of
  approvals from calibrated PDs) with an alarm band bootstrapped from the
  reference, and the Saerens-Latinne-Decaestecker EM prior estimate.
* `performance.py`: ADWIN with cut points at batch boundaries over the
  calibration residual, DDM over decision errors in vectorised form, and the
  label-rate binomial test.

### `driftwatch.alerts` and `driftwatch.monitor`

Every detector evaluation is a `Signal` (fired or not), so the timeline keeps
statistics below the alarm line. `AlertPolicy` keys signals by (category,
scope) and applies a `Rule` per category: persistence k of the last m
evaluations, then a cooldown. Drift on a feature with an active data-quality
alert is suppressed, and DDM is advisory. `Monitor.observe` runs the
decision-time detectors on one batch and `Monitor.observe_labels` the
label-time ones; a `Reference` holds every fitted profile of one model
version.

### `driftwatch.registry` and `driftwatch.store`

`ModelRegistry` keeps versions in SQLite (stage, parent, lineage JSON, gate
metrics, reason) and artifacts as files named by their SHA-256, verified on
every load. Stage changes follow an explicit state machine and are written to
an `events` table with reason and actor; `promote` and `rollback` change two
rows in one transaction. The registry stores a `ModelBundle`: the scorer plus
its monitoring `Reference`, so a promoted model brings its own baselines.
`Store` keeps the operational state: batches with their coerced features as
`npz` payloads (numbers and codes, never pickles), labels, label reports,
alerts and rollouts. `":memory:"` gives the same store to the benchmark.

### `driftwatch.deploy`

* `rollout.py`: `Rollout`, the state machine of one candidate against the
  champion, with the ramp (shadow, 5, 25 and 50 percent, steps above 5
  percent gated on label evidence), stable hash routing per application id,
  and the five guardrails.
* `retrain.py`: `RetrainTrigger` (which alerts start, block or back off a
  retrain, and from which batch the training window starts), the
  recalibration challenger, and `champion_challenger_gate`.

### `driftwatch.loop`

`ControlLoop` owns one production model, at most one candidate, a monitor
built from the production version's reference, a trigger and a store. It is
the only place where alerts, guardrail decisions and registry transitions
meet. On construction it rolls back any version left in shadow or canary,
because a rollout's guardrail state does not survive a restart.

### `driftwatch.api`

`create_app` wires settings, JSON logging with request ids, Prometheus text
metrics, exception mapping (`NotFoundError` 404, `ConflictError` 409,
`InvalidInputError` 422) and the routes. `Services` holds the loop behind one
lock because the loop is stateful and FastAPI runs sync handlers on a thread
pool. The dashboard is static HTML, CSS and JavaScript served from
`api/static`, with no build step and no external resources.

### `driftwatch.bench`

Tasks for the three benchmark sections (`detection`, `deployment`,
`retraining`), each a pure function of its seeded task; `run.py` runs them on
a spawn-context process pool, aggregates them in task order and computes the
flat `headline` view; `report.py` renders `RESULTS.md` and the README block;
`check.py` applies `benchmarks/thresholds.json`.

## Data flow of one batch

1. `POST /v1/batches` with JSON records; Pydantic checks sizes and that every
   value is a JSON scalar.
2. `records_to_columns` and `coerce` produce a typed batch and schema issues.
3. The champion scores every row. During a rollout the candidate scores every
   row too; rows in the canary share (hash of the application id) take the
   candidate's decision, failed candidate rows fall back to the champion.
4. The monitor evaluates schema, quality, drift, prediction drift and CBPE;
   the policy turns fired signals into alerts; the rollout evaluates its
   label-free guardrails and may roll the candidate back.
5. The store records the batch summary, the coerced features, both models'
   scores and a quality flag; alerts are stored and fed to the trigger.
6. If the trigger is due, challengers are trained on the labelled window,
   gated and, on a pass, registered and put in shadow.

## Data flow of one label delivery

1. `POST /v1/batches/{id}/labels` with one outcome per application id; the
   service checks the ids cover the batch exactly.
2. The loop joins the outcomes to the scores used at decision time (from
   memory, or from the stored payload after a restart).
3. If the batch was scored by the current champion (or by the candidate that
   has since been promoted), ADWIN, DDM and the label-rate test run on it.
4. If the batch was scored during the running rollout, the calibration and
   discrimination guardrails run; the rollout may promote or roll back.
5. The trigger learns the latest labelled batch, which can make a retrain due.

## Deployment

`make serve` runs uvicorn on 127.0.0.1:8774 with the data under `data/`. The
Docker image runs the same command on 0.0.0.0 as a non-root user, with the
data directory as a volume. One process per deployment slot: the loop is not
shared across processes, and the SQLite files are not meant for concurrent
writers.
