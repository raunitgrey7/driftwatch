"""driftwatch: monitoring and safe deployment for a credit-scoring model.

The package is organised around the life of one model in production:

* ``driftwatch.data`` generates a seeded applicant stream with drift and data
  incidents injected at known batches, so every detector can be scored
  against ground truth.
* ``driftwatch.model`` trains the reference (champion) model on the clean
  period.
* ``driftwatch.detectors`` holds the statistics: schema and data-quality
  checks, PSI, KS, chi-square and MMD for data drift, ADWIN and DDM for
  performance drift, and confidence-based performance estimation.
* ``driftwatch.alerts`` turns raw detector signals into alerts with
  persistence and cooldowns, ``driftwatch.monitor`` runs all of it per batch.
* ``driftwatch.registry`` and ``driftwatch.deploy`` cover the release side:
  versions with lineage, shadow and canary rollout with statistical
  guardrails, automatic rollback, and the retraining champion-challenger gate.
"""

__version__ = "0.1.0"
