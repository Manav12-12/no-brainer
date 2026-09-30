import pytest
from hypothesis import given
from hypothesis import strategies as st

from sentinel.jev.interface import DecisionUnavailable, QuestionResult
from sentinel.jev.typesafe_backend import normalize_probabilities


@pytest.mark.property
@given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
def test_noul_distribution_always_sums_to_one(value: float) -> None:
    result = QuestionResult(
        type="noul",
        selected=value,
        probabilities={"false": 1 - value, "true": value},
    )
    assert sum(result.probabilities.values()) == pytest.approx(1.0)


@pytest.mark.property
def test_invalid_distribution_is_rejected() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        QuestionResult(type="choice", selected="a", probabilities={"a": 0.2, "b": 0.2})


@pytest.mark.property
@given(
    st.floats(min_value=0.05, max_value=0.95, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.98, max_value=1.02, allow_nan=False, allow_infinity=False),
)
def test_near_unit_distributions_normalize(weight: float, total: float) -> None:
    normalized, _ = normalize_probabilities(
        {"a": weight * total, "b": (1.0 - weight) * total}, "choice"
    )
    assert sum(normalized.values()) == pytest.approx(1.0)


@pytest.mark.property
@given(
    st.one_of(
        st.floats(
            min_value=0.1,
            max_value=0.979,
            allow_nan=False,
            allow_infinity=False,
        ),
        st.floats(
            min_value=1.021,
            max_value=2.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
)
def test_out_of_band_distributions_are_rejected(total: float) -> None:
    with pytest.raises(DecisionUnavailable, match="outside normalization tolerance"):
        normalize_probabilities({"a": total / 2, "b": total / 2}, "choice")
