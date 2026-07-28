import itertools
import random
from collections.abc import Iterable
from functools import partial
from typing import Literal

import pytest

import plum
from .util import benchmark


def assert_cache_performance(f, f_native):
    # Time the performance of a native call.
    dur_native = benchmark(f_native, (1,), n=250, burn=10)

    def resolve_registrations():
        for f in plum.Function._instances:
            f._resolve_pending_registrations()

    def setup_no_cache():
        plum.clear_all_cache()
        resolve_registrations()

    # Time the performance of a cache miss.
    dur_first = benchmark(f, (1,), n=250, burn=10, setup=setup_no_cache)

    # Time the performance of a cache hit.
    plum.clear_all_cache()
    resolve_registrations()
    dur = benchmark(f, (1,), n=250, burn=10)

    # A cached call should not be more than 50 times slower than a native call.
    assert dur <= 50 * dur_native

    # A first call should not be more than 2000 times slower than a cached call.
    assert dur_first <= 2000 * dur

    # The cached call should be at least 4 times faster than a first call.
    assert dur <= dur_first / 4


def test_cache_function(dispatch: plum.Dispatcher):
    def f_native(x):
        pass

    @dispatch
    def f(x):
        pass

    @dispatch
    def f(x: int | float):
        pass

    @dispatch
    def f(x: int | float | str):
        pass

    # Test performance.
    assert_cache_performance(f, f_native)

    # Test cache correctness.
    assert f(1) is None

    @dispatch
    def f(x: int):
        return 1

    assert f(1) == 1


# This class needs to be in the global scope, otherwise it cannot its methods cannot
# obtains a reference to it.


class A:
    _dispatch = plum.Dispatcher()

    @_dispatch
    def __call__(self, x: int):
        pass

    @_dispatch
    def __call__(self, x: str):
        pass

    @_dispatch
    def go(self, x: int):
        pass

    @_dispatch
    def go(self, x: str):
        pass

    @_dispatch
    def go_again(self, x: int):
        pass

    @_dispatch
    def go_again(self, x: str):
        pass


def test_cache_class():
    class ANative:
        def __call__(self, x):
            pass

        def go(self, x):
            pass

        def go_again(self, x):
            pass

    a_native = ANative()
    a = A()

    # Test performance of calls.
    assert_cache_performance(a, a_native)

    # Test performance of method calls.
    assert_cache_performance(lambda x: a.go(x), lambda x: a_native.go(x))

    # Test performance of static calls.
    assert_cache_performance(
        lambda x: A.go_again(a, x),
        lambda x: ANative.go_again(a_native, x),
    )


def test_cache_clearing(dispatch: plum.Dispatcher):
    @dispatch
    def f(x: int):
        return 1

    @dispatch
    def f(x: float):
        return 2

    assert len(f._cache) == 0
    assert len(f._resolver) == 0

    assert f(1) == 1
    # Check that cache is used.
    assert len(f._cache) == 1
    assert len(f._resolver) == 2

    # Clear via the dispatcher.
    dispatch.clear_cache()
    assert len(f._cache) == 0
    assert len(f._resolver) == 0

    # Run the function again.
    assert f(1) == 1
    assert len(f._cache) == 1
    assert len(f._resolver) == 2

    # Clear via `clear_all_cache`.
    plum.clear_all_cache()
    assert len(f._cache) == 0
    assert len(f._resolver) == 0

    # Run the function one last time.
    assert f(1) == 1
    assert len(f._cache) == 1
    assert len(f._resolver) == 2


def test_cache_unfaithful(dispatch: plum.Dispatcher):
    @dispatch
    def f(x: int):
        return 1

    @dispatch
    def f(x: list[int]):
        return 2

    # Since `f` is not faithful, no method should be cached.
    assert f(1) == 1
    assert f([1]) == 2
    assert len(f._cache) == 0
    # The methods to consider are cached instead, one bucket per argument type.
    assert set(f._verify_cache) == {(int,), (list,)}


def test_type_dispatch_is_cached(dispatch):
    @dispatch
    def g(x: type[int]):
        return "type[int]"

    @dispatch
    def g(x: type[str]):
        return "type[str]"

    g._resolve_pending_registrations()
    assert g._resolver.is_cacheable and not g._resolver.is_faithful
    assert len(g._cache) == 0
    assert g(int) == "type[int]"
    assert g(str) == "type[str]"
    assert len(g._cache) == 2  # keyed by identity, one entry per class


def test_faithful_class_args_share_one_entry(dispatch):
    @dispatch
    def h(x: object):
        return "object"

    h._resolve_pending_registrations()
    assert h._resolver.is_faithful
    for cls in (int, str, float, list, dict):
        assert h(cls) == "object"
    assert len(h._cache) == 1  # faithful ⇒ keyed on type(x)=type, one bucket


def test_literal_dispatch_is_cached(dispatch):
    @dispatch
    def f(x: Literal[1]):
        return "one"

    @dispatch
    def f(x: Literal[2]):
        return "two"

    @dispatch
    def f(x: int):
        return "int"

    f._resolve_pending_registrations()
    assert f._resolver.is_cacheable and not f._resolver.is_faithful
    assert len(f._cache) == 0

    assert f(1) == "one"
    assert f(2) == "two"
    assert f(3) == "int"
    # Distinct literal values must not collide: one entry each.
    assert len(f._cache) == 3
    # And the cached answers must still be the right ones.
    assert (f(1), f(2), f(3)) == ("one", "two", "int")
    assert len(f._cache) == 3


def test_literal_dispatch_keeps_bool_and_int_apart(dispatch):
    # Beartype matches `x` against `Literal[v]` iff `isinstance(x, type(v))` and
    # `x == v`, so `True` matches `Literal[1]` but `1` does not match `Literal[True]`.
    assert plum._bear.is_bearable(True, Literal[1])
    assert not plum._bear.is_bearable(1, Literal[True])

    @dispatch
    def f(x: Literal[True]):
        return "true"

    @dispatch
    def f(x: int):
        return "int"

    assert f(True) == "true"
    assert f(1) == "int"
    assert f(0) == "int"
    # `True == 1`, so only the type slot of the key keeps these entries apart.
    assert len(f._cache) == 3


def test_literal_dispatch_covers_subclasses(dispatch):
    class MyInt(int):
        pass

    @dispatch
    def f(x: Literal[1]):
        return "one"

    @dispatch
    def f(x: int):
        return "int"

    # A subclass instance does match a `Literal`, so its value must be keyed.
    assert f(MyInt(1)) == "one"
    assert f(MyInt(2)) == "int"
    assert f(MyInt(1)) == "one"


def test_literal_dispatch_subclass_with_untrustworthy_equality(dispatch):
    """A subclass may define a non-transitive `__eq__`, so its value cannot be keyed.

    `W(1) == W(2)` is `True` with equal hashes, so value-keying puts them in the
    same cache bucket — yet `W(1) == 1` and `W(2) != 1`, so they must dispatch
    differently. Only identity is fine enough to key such an argument.
    """

    class W(int):
        def __hash__(self):
            return 0

        def __eq__(self, other):
            if type(other) is W:
                return True
            return int.__eq__(self, other)

    @dispatch
    def f(x: Literal[1]):
        return "one"

    @dispatch
    def f(x: int):
        return "int"

    assert f(W(1)) == "one"
    assert f(W(2)) == "int"
    # And in the other warm-up order.
    f.clear_cache()
    assert f(W(2)) == "int"
    assert f(W(1)) == "one"


def test_literal_dispatch_uses_identity_for_subclasses(dispatch):
    """End-to-end exercise of the `_Identity` fallback through actual dispatch."""
    from plum._type import _Identity

    class MyInt(int):
        pass

    @dispatch
    def f(x: Literal[1]):
        return "one"

    @dispatch
    def f(x: int):
        return "int"

    a, b = MyInt(1), MyInt(1)
    assert f(a) == "one"
    assert f(b) == "one"
    # Two equal-but-distinct subclass instances get separate, identity-keyed entries.
    assert len(f._cache) == 2
    assert all(isinstance(k[0][-1], _Identity) for k in f._cache)


def test_aspect_key_survives_clear_cache_with_reregister(dispatch):
    """`clear_cache(reregister=True)` installs a fresh, faithful resolver whose
    methods are only pending. The key callable used for the next call must be the
    one that resolver ends up with, not a stale faithful `type`."""

    @dispatch
    def f(x: Literal[1]):
        return "one"

    @dispatch
    def f(x: int):
        return "int"

    assert f(1) == "one"
    f.clear_cache(reregister=True)
    assert f(2) == "int"
    assert f(1) == "one"


def test_literal_dispatch_cache_is_bounded(dispatch):
    """A `Literal` method plus caller-controlled values must not grow the cache
    without bound. Dispatch stays correct past the limit; only the memoisation
    stops."""
    from plum._function import _VALUE_CACHE_LIMIT

    @dispatch
    def f(x: Literal["ready"]):
        return "ready"

    @dispatch
    def f(x: str):
        return "other"

    for i in range(_VALUE_CACHE_LIMIT + 100):
        assert f(f"v{i}") == "other"

    assert len(f._cache) == _VALUE_CACHE_LIMIT
    # Beyond the limit, resolution still gives the right answer.
    assert f("ready") == "ready"
    assert f("v0") == "other"
    assert len(f._cache) == _VALUE_CACHE_LIMIT


def test_identity_only_dispatch_cache_is_not_bounded(dispatch):
    """The bound applies to `VALUE` resolvers only: classes are bounded already."""
    from plum._function import _VALUE_CACHE_LIMIT

    @dispatch
    def f(x: type[int]):
        return "type[int]"

    @dispatch
    def f(x: object):
        return "object"

    n = _VALUE_CACHE_LIMIT + 10
    for i in range(n):
        assert f(type(f"C{i}", (), {})) == "object"

    assert len(f._cache) > _VALUE_CACHE_LIMIT


def test_literal_dispatch_unhashable_argument(dispatch):
    @dispatch
    def f(x: Literal[1]):
        return "one"

    @dispatch
    def f(x: object):
        return "object"

    # An unhashable argument must not make the cache key raise.
    assert f([1, 2]) == "object"
    assert f({"a": 1}) == "object"
    assert f(1) == "one"


def test_literal_and_type_dispatch_mixed(dispatch):
    from plum._type import Aspect

    @dispatch
    def f(x: Literal[1]):
        return "one"

    @dispatch
    def f(x: type[int]):
        return "type[int]"

    @dispatch
    def f(x: object):
        return "object"

    f._resolve_pending_registrations()
    assert f._resolver.aspects == {Aspect.IDENTITY, Aspect.VALUE}

    assert f(1) == "one"
    assert f(int) == "type[int]"
    assert f(2) == "object"
    assert f(str) == "object"
    assert len(f._cache) == 4


def test_late_type_registration_invalidates_plain_type_keys(dispatch):
    # The one path that could mix key shapes in a single dict: entries accumulated
    # while the function was faithful are keyed on `type(x)`, but registering a
    # `type[X]` method rebinds the key callable to the identity-aware one.
    @dispatch
    def f(x: object):
        return "object"

    f._resolve_pending_registrations()
    assert f(int) == "object"
    assert f(str) == "object"
    # Faithful: both class arguments land in the same `(type,)` bucket.
    assert set(f._cache) == {(type,)}

    @dispatch
    def f(x: type[int]):
        return "type[int]"

    # Registration is lazy, so resolving it is what invalidates the cache.
    f._resolve_pending_registrations()
    assert f._cache == {}

    assert f(int) == "type[int]"
    assert f(str) == "object"
    # No stale `(type,)` key survives, and the two classes — which dispatch
    # differently — no longer share a key.
    assert (type,) not in f._cache
    assert len(set(f._cache)) == 2


def test_late_literal_registration_invalidates_plain_type_keys(dispatch):
    @dispatch
    def f(x: object):
        return "object"

    f._resolve_pending_registrations()
    assert f(1) == "object"
    assert f(2) == "object"
    assert set(f._cache) == {(int,)}

    @dispatch
    def f(x: Literal[1]):
        return "one"

    f._resolve_pending_registrations()
    assert f._cache == {}

    assert f(1) == "one"
    assert f(2) == "object"
    # No stale `(int,)` key survives, and the two values — which dispatch
    # differently — no longer share a key.
    assert (int,) not in f._cache
    assert len(set(f._cache)) == 2


def test_type_dispatch_does_not_capture_a_value_slot(dispatch):
    from plum._type import Aspect

    @dispatch
    def g(x: type[int]):
        return "type[int]"

    g._resolve_pending_registrations()
    assert g._resolver.aspects == {Aspect.IDENTITY}
    assert g(int) == "type[int]"
    # No value slot for a `type[X]`-only resolver: distinct values of the same
    # non-class type are indistinguishable to its key.
    assert g._resolver._arg_key(1) == g._resolver._arg_key(2)
    # Classes still are, since that is the aspect it does ask for.
    assert g._resolver._arg_key(int) != g._resolver._arg_key(str)


def test_empty_tuple_hint_dispatch_tier_one(dispatch):
    """`tuple[()]` must not be treated as faithful: it matches on length."""

    @dispatch
    def f(x: tuple[()]):
        return "empty"

    @dispatch
    def f(x: tuple):
        return "tuple"

    assert f(()) == "empty"
    f.clear_cache()
    # Warming with a non-empty tuple must not poison the `tuple` bucket.
    assert f((1,)) == "tuple"
    assert f(()) == "empty"


def test_empty_tuple_hint_dispatch_tier_two(dispatch):
    """The same, with an uncacheable method forcing the verify cache."""

    @dispatch
    def f(x: tuple[()]):
        return "empty"

    @dispatch
    def f(x: tuple):
        return "tuple"

    @dispatch
    def f(x: list[int]):  # makes the resolver uncacheable -> tier two
        return "list"

    assert f(()) == "empty"
    f.clear_cache()
    assert f((1,)) == "tuple"
    assert f(()) == "empty"
    assert f([1]) == "list"


# The verify cache: an uncacheable function cannot memoise a method, but it can
# memoise which methods are worth considering for given bare argument types.


def test_verify_cache_narrows_the_methods_considered(dispatch):
    @dispatch
    def f(x: list[int]):
        return "list[int]"

    @dispatch
    def f(x: list[str]):
        return "list[str]"

    @dispatch
    def f(x: int):
        return "int"

    assert f([1]) == "list[int]"
    assert f(["a"]) == "list[str]"
    assert f(1) == "int"

    # Uncacheable, so no method is memoised.
    assert len(f._cache) == 0
    # But the methods that could possibly match are, bucketed by bare types.
    assert set(f._verify_cache) == {(list,), (int,)}
    # A `list` argument can never match `int`, and vice versa.
    assert len(f._verify_cache[(list,)][0]) == 2
    assert len(f._verify_cache[(int,)][0]) == 1


def test_verify_cache_is_invalidated_by_registration(dispatch):
    @dispatch
    def f(x: list[int]):
        return "list[int]"

    # Warm the bucket for `list` arguments.
    assert f([1]) == "list[int]"
    assert set(f._verify_cache) == {(list,)}

    @dispatch
    def f(x: list[str]):
        return "list[str]"

    # The new method must make it into the bucket for `list`.
    assert f(["a"]) == "list[str]"
    assert f([1]) == "list[int]"


def test_verify_cache_clearing(dispatch: plum.Dispatcher):
    @dispatch
    def f(x: list[int]):
        return "list[int]"

    assert f([1]) == "list[int]"
    assert len(f._verify_cache) == 1

    dispatch.clear_cache()
    assert len(f._verify_cache) == 0

    assert f([1]) == "list[int]"
    assert len(f._verify_cache) == 1

    plum.clear_all_cache()
    assert len(f._verify_cache) == 0


def test_verify_cache_preserves_ambiguity(dispatch):
    @dispatch
    def f(x: list[int], y: object):
        return 1

    @dispatch
    def f(x: object, y: list[int]):
        return 2

    # Cold: nothing is cached yet.
    with pytest.raises(plum.AmbiguousLookupError):
        f([1], [1])
    # The bucket is now warm, and the same call must still be ambiguous.
    assert set(f._verify_cache) == {(list, list)}
    with pytest.raises(plum.AmbiguousLookupError):
        f([1], [1])

    # Warming the very same bucket with an unambiguous call changes nothing.
    assert f([1], ["a"]) == 1
    with pytest.raises(plum.AmbiguousLookupError):
        f([1], [1])


def test_verify_cache_preserves_not_found(dispatch):
    @dispatch
    def f(x: list[int]):
        return 1

    @dispatch
    def f(x: tuple[int, ...]):
        return 2

    for _ in range(2):  # Cold, then warm.
        with pytest.raises(plum.NotFoundLookupError) as e:
            f(["a"])
        # The error reports all methods, not just the narrowed ones.
        assert len(e.value.methods) == 2


def test_verify_cache_preserves_precedence(dispatch):
    @dispatch(precedence=1)
    def f(x: list[int], y: object):
        return 1

    @dispatch
    def f(x: object, y: list[int]):
        return 2

    # Ambiguous but for the precedence, cold and warm.
    assert f([1], [1]) == 1
    assert f([1], [1]) == 1


def test_verify_cache_handles_varargs_and_arities(dispatch):
    @dispatch
    def f(x: int, *xs: list[int]):
        return "varargs"

    @dispatch
    def f(x: int):
        return "one"

    @dispatch
    def f(x: int, y: int):
        return "two"

    assert f(1) == "one"
    assert f(1, 2) == "two"
    assert f(1, [1]) == "varargs"
    assert f(1, [1], [2]) == "varargs"
    # A fixed-arity method is only ever in the bucket of its own arity. The varargs
    # method can match any arity, but only for arguments its varargs type admits:
    # it is in the `(int,)` bucket, where the varargs go unused, but not in the
    # `(int, int)` one, where `2` can never be a `list[int]`.
    assert len(f._verify_cache[(int,)][0]) == 2
    assert len(f._verify_cache[(int, int)][0]) == 1
    assert len(f._verify_cache[(int, list)][0]) == 1
    assert len(f._verify_cache[(int, list, list)][0]) == 1


# When a bucket can be put in resolution order, the first method that matches is the
# one full resolution would select, so the call can return on it.

_mro_dispatch = plum.Dispatcher()


class _MroBase:
    def f(self, x):
        return "base"


class _MroSub(_MroBase):
    @_mro_dispatch
    def f(self, x: list[int]):
        return "sub"


def test_verify_cache_orders_a_comparable_bucket(dispatch):
    @dispatch
    def f(x: list):
        return "list"

    @dispatch
    def f(x: list[int]):
        return "list[int]"

    assert f([1]) == "list[int]"
    assert f([1]) == "list[int]"
    assert f(["a"]) == "list"

    methods, _, first_wins = f._verify_cache[(list,)]
    assert first_wins
    # Most specific first.
    assert [m.signature.types for m in methods] == [(list,), (list[int],)]


def test_verify_cache_does_not_order_an_incomparable_bucket(dispatch):
    @dispatch
    def f(x: list[int]):
        return "list[int]"

    @dispatch
    def f(x: list[str]):
        return "list[str]"

    assert f([1]) == "list[int]"
    # Neither signature is below the other, and the empty list matches both, so the
    # bucket cannot return on a first match.
    assert not f._verify_cache[(list,)][2]
    with pytest.raises(plum.AmbiguousLookupError):
        f([])


def test_verify_cache_settles_an_unordered_bucket_on_a_unique_match(dispatch):
    @dispatch
    def f(x: list[int]):
        return "list[int]"

    @dispatch
    def f(x: list[str]):
        return "list[str]"

    # An unordered bucket can still settle a call that matches exactly one of its
    # methods: that method is the only candidate full resolution can collect.
    assert f([1]) == "list[int]"
    assert f(["a"]) == "list[str]"
    assert not f._verify_cache[(list,)][2]

    # Several matches fall through to full resolution, which finds the ambiguity.
    with pytest.raises(plum.AmbiguousLookupError):
        f([])
    # So does no match at all, which reports every method, not just the bucket.
    with pytest.raises(plum.NotFoundLookupError) as e:
        f([1.0])
    assert len(e.value.methods) == 2


def test_verify_cache_ordering_ignores_precedence(dispatch):
    @dispatch(precedence=5)
    def f(x: list):
        return "list"

    @dispatch
    def f(x: list[int]):
        return "list[int]"

    # A fully comparable bucket is never ambiguous, so precedence never gets a say:
    # specificity decides, exactly as in full resolution.
    assert f._verify_cache is not None
    assert f([1]) == "list[int]"
    assert f([1]) == "list[int]"
    assert f._verify_cache[(list,)][2]


def test_verify_cache_orders_equally_specific_methods_by_registration(dispatch):
    @dispatch(precedence=1)
    def f(x: list[int]):
        return "first"

    @dispatch
    def f(x: list[int]):
        return "second"

    # The two signatures are below each other but not equal, since they differ in
    # precedence. Full resolution keeps the last registered one, and so must the
    # ordering, cold and warm.
    assert f([1]) == "second"
    assert f([1]) == "second"
    assert f._verify_cache[(list,)][2]
    assert f._resolver.resolve(([1],)).implementation([1]) == "second"


def test_verify_cache_leaves_a_large_bucket_unordered(dispatch):
    hints = [
        list[int],
        list[str],
        list[bool],
        list[float],
        list[bytes],
        list[complex],
        list[frozenset],
        list[tuple],
        list[dict],
        list[set],
        list[object],
    ]
    f = None
    for i, hint in enumerate(hints):

        def impl(x, _i=i):
            return _i

        impl.__name__ = "big"
        f = dispatch.multi(plum.Signature(hint))(impl)

    # Ordering is quadratic in the size of the bucket, so a large one is left to the
    # unique-match path, which resolves it exactly as full resolution does.
    assert len(f.methods) > 10
    assert f([1]) == 0
    assert not f._verify_cache[(list,)][2]
    assert f(["a"]) == 1


def test_verify_cache_first_match_preserves_the_mro_fallback():
    # `["a"]` lands in the same bucket as `[1]` but matches nothing in it, so the
    # first-match path has to fall through to the walk up the MRO.
    sub = _MroSub()
    assert sub.f([1]) == "sub"
    assert sub.f(["a"]) == "base"
    assert sub.f(["a"]) == "base"


# A differential fuzz test for the verify cache. Random method sets over uncacheable
# hints are dispatched over a corpus of values, and every outcome is compared against
# full resolution over all methods, cold and warm. Whatever a bucket does, it has to
# agree with the resolver it stands in for.

_FUZZ_HINTS = (
    list[int],
    list[str],
    list[object],
    list,
    tuple[int, ...],
    dict[str, int],
    Iterable[int],
    list[int] | int,
    Literal[1],
    int,
    object,
)

# Beartype checks a container by sampling an element, so a heterogeneous container
# would make `match` itself random and the comparison meaningless. Every container
# here is homogeneous.
_FUZZ_VALUES = (
    [],
    [1],
    [1, 2],
    ["a"],
    [[1]],
    (1, 2),
    (),
    {"a": 1},
    {"a": "b"},
    1,
    2,
    "a",
    object(),
)


def _fuzz_function(rng):
    """Build a function from a random set of methods over `_FUZZ_HINTS`.

    Returns the function, a map from implementation to the index of the method that
    implements it, and the arities the methods were built for.
    """
    impls = {}
    f = None
    for i in range(rng.randint(1, 4)):

        def impl(*args, _i=i):
            return _i

        impl.__name__ = "fuzzed"
        types = [rng.choice(_FUZZ_HINTS) for _ in range(rng.randint(1, 2))]
        varargs = rng.choice(_FUZZ_HINTS) if rng.random() < 0.2 else plum.Missing
        signature = plum.Signature(
            *types, varargs=varargs, precedence=rng.choice((0, 0, 1))
        )
        if f is None:
            f = plum.Function(impl)
        f.register(impl, signature=signature, precedence=None)
        impls[impl] = i
    f._resolve_pending_registrations()
    return f, impls


def _resolved_index(f, args, impls):
    """The index of the method full resolution selects, using no cache at all."""
    return impls[f._resolver.resolve(args).implementation]


def _outcome(call, impls):
    """Run `call` and reduce whatever it does to a comparable value."""
    try:
        return ("value", call())
    except plum.AmbiguousLookupError as e:
        return ("ambiguous", tuple(sorted(impls[m.implementation] for m in e.methods)))
    except plum.NotFoundLookupError as e:
        return ("not found", tuple(sorted(impls[m.implementation] for m in e.methods)))


def test_verify_cache_differential_fuzz():
    rng = random.Random(20240611)
    uncacheable = 0

    for _ in range(300):
        f, impls = _fuzz_function(rng)
        uncacheable += not f._resolver.is_cacheable

        for n in (1, 2, 3):
            for args in itertools.product(_FUZZ_VALUES, repeat=n):
                if n > 1 and rng.random() > 0.15:
                    continue

                # The reference: full resolution over every method, no cache at all.
                reference = _outcome(partial(_resolved_index, f, args, impls), impls)

                # Cold: every call rebuilds the bucket from scratch.
                f.clear_cache(reregister=False)
                cold = _outcome(partial(f, *args), impls)
                # Warm: the bucket built by the call above is reused.
                warm = _outcome(partial(f, *args), impls)

                assert cold == reference, (args, f.methods)
                assert warm == reference, (args, f.methods)

    # The fast paths under test only run for an uncacheable resolver, so the corpus
    # has to actually produce those.
    assert uncacheable > 200
