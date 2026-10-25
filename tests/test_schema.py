"""Coercion must describe broken batches instead of raising, and keep row alignment."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from driftwatch.schema import (
    CATEGORICAL_FEATURES,
    FEATURES,
    ID_COLUMN,
    NULL_CODE,
    NUMERIC_FEATURES,
    UNSEEN_CODE,
    coerce,
    concat_batches,
    records_to_columns,
)
from tests.conftest import stream_batch


def _clean_columns(n: int = 3) -> dict[str, list]:
    columns: dict[str, list] = {ID_COLUMN: [f"a{i}" for i in range(n)]}
    for spec in FEATURES:
        columns[spec.name] = [spec.vocabulary[0]] * n if spec.vocabulary else [1.0] * n
    return columns


def test_clean_batch_has_no_issues_and_typed_arrays():
    batch, issues = coerce(_clean_columns())
    assert issues == []
    assert all(batch.numeric[n].dtype == np.float64 for n in NUMERIC_FEATURES)
    assert all(batch.categorical[n].dtype == object for n in CATEGORICAL_FEATURES)
    assert batch.n_rows == 3


def test_missing_and_unexpected_columns_are_reported():
    columns = _clean_columns()
    columns["utilisation_pct"] = columns.pop("credit_utilization")
    batch, issues = coerce(columns)
    kinds = {(i.kind, i.column) for i in issues}
    assert ("missing_column", "credit_utilization") in kinds
    assert ("unexpected_column", "utilisation_pct") in kinds
    assert np.isnan(batch.numeric["credit_utilization"]).all()


def test_unparseable_numbers_become_null_with_a_type_issue():
    columns = _clean_columns(4)
    columns["monthly_income"] = ["45000", "45k", True, float("inf")]
    batch, issues = coerce(columns)
    issue = next(i for i in issues if i.column == "monthly_income")
    assert issue.kind == "type_mismatch"
    assert issue.fraction == pytest.approx(0.75)
    assert batch.numeric["monthly_income"][0] == 45000.0
    assert np.isnan(batch.numeric["monthly_income"][1:]).all()


def test_columns_of_different_length_are_rejected():
    columns = _clean_columns()
    columns["age"] = [30.0]
    with pytest.raises(ValueError, match="different lengths"):
        coerce(columns)


def test_records_to_columns_fills_missing_keys_with_none():
    columns = records_to_columns([{"a": 1, "b": 2}, {"a": 3}])
    assert columns == {ID_COLUMN: [None, None], "a": [1, 3], "b": [2, None]}


def test_records_without_fields_keep_their_row_count():
    """Regression found by hypothesis: [{}] used to collapse to a zero-row batch."""
    batch, issues = coerce(records_to_columns([{}, {}]))
    assert batch.n_rows == 2
    assert batch.ids.tolist() == ["row-0", "row-1"]
    assert sum(i.kind == "missing_column" for i in issues) == len(FEATURES)


def test_codes_mark_unseen_and_null_and_survive_take():
    columns = _clean_columns(3)
    columns["employment_type"] = ["salaried", "gig", None]
    batch, _ = coerce(columns)
    codes = batch.codes("employment_type")
    assert codes.tolist() == [0, UNSEEN_CODE, NULL_CODE]
    assert batch.take(np.array([2, 0])).codes("employment_type").tolist() == [NULL_CODE, 0]


def test_concat_batches_stacks_rows_in_order():
    a, b = stream_batch("control", 0), stream_batch("control", 1)
    both = concat_batches([a, b])
    assert both.n_rows == a.n_rows + b.n_rows
    np.testing.assert_array_equal(both.numeric["age"][: a.n_rows], a.numeric["age"])
    with pytest.raises(ValueError):
        concat_batches([])


json_scalar = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(-(10**12), 10**12),
    st.floats(allow_nan=True, allow_infinity=True),
    st.text(max_size=8),
)


@settings(max_examples=60, deadline=None)
@given(
    st.lists(
        st.dictionaries(st.sampled_from([*NUMERIC_FEATURES, *CATEGORICAL_FEATURES, "junk"]), json_scalar, max_size=6),
        min_size=1,
        max_size=12,
    )
)
def test_coerce_never_raises_on_arbitrary_json_records(records):
    """Adversarial input: any JSON scalars in any columns give a batch of the right length."""
    batch, issues = coerce(records_to_columns(records))
    assert batch.n_rows == len(records)
    for name in NUMERIC_FEATURES:
        values = batch.numeric[name]
        assert len(values) == len(records)
        assert not np.isinf(values).any()
    assert all(0.0 <= i.fraction <= 1.0 for i in issues)
    assert all(math.isfinite(i.fraction) for i in issues)
