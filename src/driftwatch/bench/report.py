"""Markdown tables from ``benchmarks/results.json``.

The README's results block is rendered by ``readme_table`` from the same
document, and a test re-renders it to prove the README matches the JSON.
"""

from __future__ import annotations

from typing import Any

BEGIN, END = "<!-- RESULTS:BEGIN -->", "<!-- RESULTS:END -->"


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|",
    ]
    return lines + ["| " + " | ".join(r) + " |" for r in rows]


def champion_table(results: dict[str, Any]) -> list[str]:
    c = results["champion"]
    rows = [
        ["Training rows, calibration rows", f"{c['train_rows']}, {c['calibration_rows']}"],
        ["Boosting iterations (early stopping)", _fmt(c["boosting_iterations"])],
        [f"Realised AUC on the {c['holdout_rows']}-row holdout", _fmt(c["holdout_auc"], 4)],
        ["Noise-free AUC on a fresh 8,000-row sample", _fmt(c["true_auc_fresh_sample"], 4)],
        [
            "Mean PD against true default rate, fresh sample",
            f"{_fmt(c['mean_pd_fresh_sample'], 4)} against {_fmt(c['true_default_rate_fresh_sample'], 4)}",
        ],
        [
            "Approval cut-off (PD) and approval rate, fresh sample",
            f"{_fmt(c['approval_cutoff_pd'], 4)}, {_fmt(c['approval_rate_fresh_sample'], 3)}",
        ],
    ]
    deciles = c["calibration_deciles"]
    reliability = ", ".join(f"{d['mean_pd']:.3f}/{d['mean_true_pd']:.3f}" for d in deciles)
    return [
        *_table(["Reference model", "Measured"], rows),
        "",
        f"Reliability by decile of predicted PD (mean predicted / mean true, fresh sample): {reliability}.",
    ]


def detection_table(results: dict[str, Any]) -> list[str]:
    rows = []
    for name, m in results["detection"]["drift"].items():
        rows.append(
            [
                name,
                _fmt(m["recall"], 2),
                _fmt(m["precision"], 3),
                _fmt(m["delay_median"], 1),
                _fmt(m["delay_p90"], 1),
                _fmt(m["diagnosis_rate"], 2),
                _fmt(m["false_alarms_before_onset"]),
            ]
        )
    header = [
        "Drift type",
        "Recall",
        "Precision",
        "Median delay (batches)",
        "p90 delay",
        "Diagnosis",
        "False alarms before onset",
    ]
    return _table(header, rows)


def control_table(results: dict[str, Any]) -> list[str]:
    c = results["detection"]["control"]
    rows = [["alert policy (all categories)", _fmt(c["false_alarms_per_10k"], 2)]]
    rows += [[f"raw `{name}`", _fmt(value, 2)] for name, value in c["raw_fires_per_10k"].items()]
    return [
        f"{c['batches']} drift-free batches on {c['streams']} control streams.",
        "",
        *_table(["Signal", "False alarms per 10,000 batches"], rows),
    ]


def detector_matrix(results: dict[str, Any]) -> list[str]:
    drift = results["detection"]["drift"]
    names = sorted({d for m in drift.values() for d in m["raw"]})
    rows = []
    for scenario, m in drift.items():
        cells = []
        for d in names:
            r = m["raw"].get(d)
            cells.append("." if not r or r["recall"] == 0 else f"{r['recall']:.1f} @{_fmt(r['delay_median'], 0)}")
        rows.append([scenario, *cells])
    return [
        "Raw detector recall within the window, with the median delay in batches after `@`.",
        "",
        *_table(["Drift type", *names], rows),
    ]


def estimation_table(results: dict[str, Any]) -> list[str]:
    rows = []
    entries = [("control", results["detection"]["control"]["estimation"])]
    entries += [(n, m["estimation_post_onset"]) for n, m in results["detection"]["drift"].items()]
    for name, e in entries:
        rows.append(
            [
                name,
                _fmt(e.get("auc_change_cbpe"), 4),
                _fmt(e.get("auc_change_naive"), 4),
                _fmt(e.get("auc_cbpe_vs_expected"), 4),
                _fmt(e.get("auc_naive_vs_expected"), 4),
                _fmt(e.get("auc_cbpe_bias"), 4),
                _fmt(e.get("auc_cbpe_vs_realized"), 4),
                _fmt(e.get("auc_naive_vs_realized"), 4),
                _fmt(e.get("rate_meanpd_vs_true"), 4),
                _fmt(e.get("rate_em_vs_true"), 4),
            ]
        )
    header = [
        "Stream",
        "AUC change error, CBPE",
        "AUC change error, naive",
        "AUC level MAE, CBPE",
        "AUC level MAE, naive",
        "CBPE level bias",
        "CBPE vs realised",
        "Naive vs realised",
        "Default rate MAE, mean PD",
        "Default rate MAE, EM prior",
    ]
    note = (
        "Errors are mean absolute errors per batch against the AUC expected under the true PD (noise-free), "
        "except the two 'realised' columns, which compare with the AUC measured on the batch's 400 labels. "
        "The change error compares the estimated change since the reference period with the true change."
    )
    return [*_table(header, rows), "", note]


def deployment_table(results: dict[str, Any]) -> list[str]:
    rows = []
    for kind, m in results["deployment"]["per_kind"].items():
        guard = ", ".join(f"{k} {v}" for k, v in sorted(m["guardrails"].items()))
        rows.append(
            [
                kind,
                m["class"],
                _fmt(m["runs"]),
                _fmt(m["rolled_back_rate"], 2),
                _fmt(m["registry_correct_rate"], 2),
                _fmt(m["batches_to_decision_median"], 1),
                _fmt(m["exposure_median"], 0),
                _fmt(m["excess_defaults_approved_mean"], 1),
                _fmt(m["release_true_auc_mean"], 4),
                guard,
            ]
        )
    header = [
        "Release",
        "Class",
        "Runs",
        "Rolled back",
        "Registry correct",
        "Median batches to decision",
        "Median exposure (rows)",
        "Excess defaults approved",
        "True AUC",
        "Decided by",
    ]
    p = results["deployment"]["pooled"]
    summary = f"Bad releases rolled back: {p['bad_rollback_rate']:.3f} of {p['bad_runs']}; good releases promoted: {p['good_promotion_rate']:.3f} of {p['good_runs']} (false rollback rate {p['false_rollback_rate']:.3f})."
    return [*_table(header, rows), "", summary]


def retraining_table(results: dict[str, Any]) -> list[str]:
    rows = []
    for name, m in results["retraining"]["per_scenario"].items():
        rows.append(
            [
                name,
                _fmt(m["runs"]),
                _fmt(m["triggered_retrains_per_run"], 2),
                _fmt(m["gate_promote_rate"], 2),
                _fmt(m["gate_correct_rate"], 2),
                _fmt(m["forced_retrains"]),
                _fmt(m["forced_correct_rate"], 2),
                _fmt(m["runs_with_new_production"]),
                _fmt(m["batches_onset_to_production_median"], 1),
                _fmt(m["true_auc_lift_after_promotion_mean"], 4),
                _fmt(m["true_logloss_gain_after_promotion_mean"], 4),
                ", ".join(f"{k} {v}" for k, v in m["promoted_challengers"].items()) or "none",
            ]
        )
    header = [
        "Scenario",
        "Runs",
        "Retrains per run",
        "Gate promote rate",
        "Gate correct",
        "Forced retrains",
        "Forced correct",
        "Runs with new production",
        "Median batches onset to production",
        "True AUC lift after promotion",
        "True log-loss gain after promotion",
        "Promoted challengers",
    ]
    pooled = results["retraining"]["pooled"]
    return [
        *_table(header, rows),
        "",
        f"Gate decisions judged against ground truth: {pooled['gate_decisions']}, correct rate {pooled['gate_correct_rate']:.3f}.",
    ]


def results_markdown(results: dict[str, Any]) -> str:
    env, timing = results["environment"], results["timing"]
    parts = [
        "# Benchmark results",
        "",
        f"Generated by `make bench` at {env['generated_at']} (commit {env['commit']}) on {env['platform']}, Python {env['python']}, "
        f"{results['config']['workers']} worker processes; {timing['batches_monitored']} batches monitored in {timing['total_seconds']} s wall time.",
        "",
        "## Reference model",
        "",
        *champion_table(results),
        "",
        "## Drift detection (alert policy)",
        "",
        *detection_table(results),
        "",
        "## False alarms on drift-free streams",
        "",
        *control_table(results),
        "",
        "## Detector matrix",
        "",
        *detector_matrix(results),
        "",
        "## Label-free performance estimation",
        "",
        *estimation_table(results),
        "",
        "## Deployment safety: planted releases",
        "",
        *deployment_table(results),
        "",
        "## Retraining and the champion-challenger gate",
        "",
        *retraining_table(results),
        "",
    ]
    return "\n".join(parts)


def readme_table(results: dict[str, Any]) -> str:
    """The compact table embedded in README.md between the RESULTS markers."""
    h = results["headline"]
    rows = [
        ["False alarms per 10,000 drift-free batches (alert policy)", _fmt(h["control.false_alarms_per_10k"], 2)],
    ]
    for name in results["detection"]["drift"]:
        rows.append(
            [
                f"`{name}`: recall, median delay (batches)",
                f"{_fmt(h[f'detection.{name}.recall'], 2)}, {_fmt(h[f'detection.{name}.delay_median'], 1)}",
            ]
        )
    rows += [
        [
            "CBPE error of the AUC change, covariate_abrupt (MAE; naive estimator in brackets)",
            f"{_fmt(h.get('estimation.covariate_abrupt.auc_change_cbpe'), 4)} ({_fmt(h.get('estimation.covariate_abrupt.auc_change_naive'), 4)})",
        ],
        [
            "CBPE error of the AUC change, concept_abrupt (MAE; naive estimator in brackets)",
            f"{_fmt(h.get('estimation.concept_abrupt.auc_change_cbpe'), 4)} ({_fmt(h.get('estimation.concept_abrupt.auc_change_naive'), 4)})",
        ],
        ["Planted bad releases rolled back", _fmt(h["deployment.bad_rollback_rate"], 3)],
        ["Good releases rolled back (false rollback rate)", _fmt(h["deployment.false_rollback_rate"], 3)],
        ["Champion-challenger gate decisions correct", _fmt(h["retraining.gate_correct_rate"], 3)],
    ]
    return "\n".join(_table(["Metric", "Measured"], rows))
