from self_evolving_v1.case_depth import _one_sided_greater


def test_one_sided_exact_probability() -> None:
    assert _one_sided_greater(0, 0) == 1.0
    assert _one_sided_greater(2, 0) == 0.25
    assert _one_sided_greater(3, 0) == 0.125
    assert _one_sided_greater(1, 1) == 0.75
