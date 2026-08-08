from beartype import BeartypeStrategy

from plum._bear import is_bearable


def test_is_bearable_basic_types():
    assert is_bearable(1, int)
    assert not is_bearable("a", int)
    assert is_bearable("a", str)


def test_is_bearable_union():
    assert is_bearable(1, int | str)
    assert is_bearable("a", int | str)
    assert not is_bearable(1.0, int | str)


def test_is_bearable_class_hierarchy():
    class Num:
        pass

    class Real(Num):
        pass

    assert is_bearable(Real(), Num)
    assert not is_bearable(Num(), Real)


def test_is_bearable_uses_on_strategy():
    # `_bear.py` explicitly opts in to the `O(n)` strategy for correctness over
    # the default `O(1)` strategy's speed.
    assert is_bearable.keywords["conf"].strategy is BeartypeStrategy.On
