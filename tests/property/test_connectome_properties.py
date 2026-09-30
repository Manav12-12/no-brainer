import pytest
from hypothesis import given
from hypothesis import strategies as st

from sentinel.connectome.loader import synthetic_connectome
from sentinel.connectome.shuffle import degree_preserving_shuffle


@pytest.mark.property
@given(st.integers(min_value=0, max_value=10_000))
def test_shuffle_preserves_directed_degree_sequences(seed: int) -> None:
    graph = synthetic_connectome()
    shuffled = degree_preserving_shuffle(graph, seed, swaps=10)
    original_in = sorted(dict(graph.in_degree()).values())
    original_out = sorted(dict(graph.out_degree()).values())
    assert sorted(dict(shuffled.in_degree()).values()) == original_in
    assert sorted(dict(shuffled.out_degree()).values()) == original_out
