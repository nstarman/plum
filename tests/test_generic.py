"""Dispatch on user-defined :class:`typing.Generic` subclasses."""

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from typing import Annotated, Any, Generic, Literal, Optional, TypeVar, Union

import pytest

from beartype.door import TypeHint
from beartype.vale import Is

from plum import Dispatcher, NotFoundLookupError
from plum._signature import Signature
from plum._type import (
    _aspects,
    _is_generic_hint,
    is_cacheable,
    is_faithful,
    resolve_type_hint,
)

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
U = TypeVar("U")


class Box(Generic[T]):
    def __init__(self, v: Any) -> None:
        self.v = v


class CoBox(Generic[T_co]):
    def __init__(self, v: Any) -> None:
        self.v = v


class Pair(Generic[T, U]):
    def __init__(self, a: Any, b: Any) -> None:
        self.a = a
        self.b = b


class IntBox(Box[int]):
    def __init__(self) -> None:
        super().__init__(1)


class StrBox(Box[str]):
    def __init__(self) -> None:
        super().__init__("a")


def test_dispatch_on_user_generic() -> None:
    """`Box[int]` and `Box[str]` are distinguished via `__orig_class__`."""
    dispatch = Dispatcher()

    @dispatch
    def f(x: Box[int]) -> str:
        return "int"

    @dispatch
    def f(x: Box[str]) -> str:
        return "str"

    assert f(Box[int](1)) == "int"
    assert f(Box[str]("a")) == "str"


def test_bare_fallback_coexists() -> None:
    """A bare `Box` method catches instances with no `__orig_class__`."""
    dispatch = Dispatcher()

    @dispatch
    def f(x: Box) -> str:
        return "bare"

    @dispatch
    def f(x: Box[int]) -> str:
        return "int"

    assert f(Box[int](1)) == "int"
    assert f(Box(1)) == "bare"
    # A parametrised signature is strictly more specific than the bare one.
    assert Signature(Box[int]) <= Signature(Box)
    assert not Signature(Box) <= Signature(Box[int])


def test_no_matching_parameter_is_not_found() -> None:
    """Without a bare fallback, an unmatched parameter is a lookup error."""
    dispatch = Dispatcher()

    @dispatch
    def f(x: Box[int]) -> str:
        return "int"

    with pytest.raises(NotFoundLookupError):
        f(Box[str]("a"))
    with pytest.raises(NotFoundLookupError):
        f(Box(1.0))


def test_subclass_of_parametrised_generic() -> None:
    """`class IntBox(Box[int])` satisfies `Box[int]` via `__orig_bases__`."""
    dispatch = Dispatcher()

    @dispatch
    def f(x: Box[int]) -> str:
        return "int"

    @dispatch
    def f(x: Box[str]) -> str:
        return "str"

    assert f(IntBox()) == "int"
    assert f(StrBox()) == "str"


def test_multi_parameter_generic() -> None:
    dispatch = Dispatcher()

    @dispatch
    def f(x: Pair[int, str]) -> str:
        return "int,str"

    @dispatch
    def f(x: Pair[str, int]) -> str:
        return "str,int"

    assert f(Pair[int, str](1, "a")) == "int,str"
    assert f(Pair[str, int]("a", 1)) == "str,int"


def test_nested_generic() -> None:
    dispatch = Dispatcher()

    @dispatch
    def f(x: Box[list[int]]) -> str:
        return "list[int]"

    @dispatch
    def f(x: Box[list[str]]) -> str:
        return "list[str]"

    assert f(Box[list[int]]([1])) == "list[int]"
    assert f(Box[list[str]](["a"])) == "list[str]"


def test_variance_is_beartypes_call() -> None:
    """Plum holds no opinion on variance: assert what beartype actually does."""
    dispatch = Dispatcher()

    @dispatch
    def f(x: CoBox[int]) -> str:
        return "int"

    # Beartype's subtype ordering is the single source of truth.
    covariant_ok = bool(TypeHint(CoBox[bool]) <= TypeHint(CoBox[int]))
    assert covariant_ok, "beartype changed its variance rule; update this test"
    assert f(CoBox[bool](True)) == "int"

    # Beartype applies the same rule to an *invariant* `TypeVar`.
    assert bool(TypeHint(Box[bool]) <= TypeHint(Box[int]))


def test_generic_is_uncacheable() -> None:
    """A parametrised user generic must route to the tier-two verify cache."""
    assert _aspects(resolve_type_hint(Box[int])) is None
    assert not is_cacheable(Box[int])
    assert not is_faithful(Box[int])
    # The bare class stays faithful.
    assert is_faithful(Box)


def test_generic_dispatch_uses_verify_cache() -> None:
    """Tier two serves generic dispatch; no generics-specific cache is added."""
    dispatch = Dispatcher()

    @dispatch
    def f(x: Box[int]) -> str:
        return "int"

    @dispatch
    def f(x: Box[str]) -> str:
        return "str"

    assert f(Box[int](1)) == "int"
    # `@dispatch` returns the `Function` itself.
    assert f._resolver.aspects is None, "the resolver must be uncacheable"
    assert f._verify_cache, "the verify cache did not populate"
    # The bucket keys on the bare runtime type and must not exclude either method.
    # A bucket is `(methods, entries, first_wins)`; only the methods matter here.
    assert f._verify_cache[(Box,)][0] == f._resolver.methods
    # Nothing is memoised in tier one for an uncacheable resolver.
    assert not f._cache
    # Warm dispatch still selects correctly.
    assert f(Box[str]("a")) == "str"
    assert f(Box[int](2)) == "int"


def test_might_match_includes_generic_methods() -> None:
    """`might_match` must never exclude a `Box[int]` method for a runtime `Box`."""
    for value in (Box[int](1), Box[str]("a"), Box(1), IntBox(), StrBox()):
        assert Signature(Box[int]).might_match((value,))
        assert Signature(Box).might_match((value,))
    assert not Signature(Box[int]).might_match((1,))


def test_resolve_type_hint_recurses_into_generics() -> None:
    """`resolve_type_hint` rebuilds user generics without warning."""
    assert resolve_type_hint(Box[int]) == Box[int]
    assert resolve_type_hint(Box[list[int]]) == Box[list[int]]
    assert resolve_type_hint(Pair[int, str]) == Pair[int, str]
    assert resolve_type_hint(Box) is Box


def test_resolve_type_hint_leaves_special_forms_alone() -> None:
    """Special forms must be untouched by the new generic branch."""
    is_positive = Is[lambda x: x > 0]
    for hint in (
        Annotated[int, "meta"],
        # The legacy spellings are deliberate: they are the special forms the new
        # generic branch must not touch.
        Union[int, str],  # noqa: UP007
        Optional[int],  # noqa: UP007, UP045
        Callable[[int], str],
        Literal[1, 2],
        type[int],
        list[int],
        dict[str, int],
        int,
        Any,
        Annotated[int, is_positive],
    ):
        # Compare with beartype's semantics rather than by identity: plum already
        # normalises `typing.Callable` to `collections.abc.Callable`, which is the
        # same hint. What matters is that the new generic branch changes nothing.
        assert TypeHint(resolve_type_hint(hint)) == TypeHint(hint)

    assert resolve_type_hint(is_positive) is is_positive


def test_no_warning_for_generic_hints(recwarn: pytest.WarningsRecorder) -> None:
    """Registering a generic method must not warn about unresolvable hints."""
    dispatch = Dispatcher()

    @dispatch
    def f(x: Box[int]) -> str:
        return "int"

    assert f(Box[int](1)) == "int"
    assert [w for w in recwarn.list if "Could not" in str(w.message)] == []


def test_generic_defined_in_an_exec_namespace() -> None:
    """A generic whose module is `builtins`, as in a doctest, still dispatches.

    `exec` without a `__name__` leaves `Box.__module__ == "builtins"`, and
    `Box[int]` is then a `typing._GenericAlias` — so a module-name test mistakes it
    for a `typing` special form. The gate keys on `Generic` inheritance instead.
    """
    ns: dict[str, Any] = {}
    exec(
        "from typing import Generic, TypeVar\n"
        'T = TypeVar("T")\n'
        "class Box(Generic[T]):\n"
        "    def __init__(self, v): self.v = v\n",
        ns,
    )
    box = ns["Box"]
    assert box.__module__ == "builtins"
    assert _is_generic_hint(box[int])
    assert not _is_generic_hint(box)

    dispatch = Dispatcher()

    @dispatch
    def f(x: box[int]) -> str:
        return "int"

    @dispatch
    def f(x: box[str]) -> str:
        return "str"

    assert f(box[int](1)) == "int"
    assert f(box[str]("a")) == "str"


def test_gate_excludes_builtins_abcs_and_special_forms() -> None:
    """Only true `Generic` subclasses take the `__orig_class__` path."""
    for hint in (
        list[int],
        dict[str, int],
        tuple[int, ...],
        set[int],
        Sequence[int],
        Iterable[int],
        Mapping[str, int],
        re.Pattern[str],
        AbstractContextManager[int],
        type[int],
        Literal[1],
        Annotated[int, "meta"],
        Union[int, str],  # noqa: UP007
        int,
        Box,
    ):
        assert not _is_generic_hint(hint), hint

    for hint in (Box[int], Pair[int, str], CoBox[int], Box[list[int]]):
        assert _is_generic_hint(hint), hint


def test_parametrised_builtins_still_dispatch() -> None:
    """The pre-existing structural path for builtins is unchanged."""
    dispatch = Dispatcher()

    @dispatch
    def f(x: list[int]) -> str:
        return "list[int]"

    @dispatch
    def f(x: list[str]) -> str:
        return "list[str]"

    assert f([1]) == "list[int]"
    assert f(["a"]) == "list[str]"


def test_generic_with_annotated_and_union() -> None:
    """A user generic composes with the special forms it sits beside."""
    dispatch = Dispatcher()

    @dispatch
    def f(x: Box[int] | Box[str]) -> str:
        return "either"

    @dispatch
    def f(x: Box[float]) -> str:
        return "float"

    assert f(Box[int](1)) == "either"
    assert f(Box[str]("a")) == "either"
    assert f(Box[float](1.0)) == "float"


def test_generic_varargs() -> None:
    dispatch = Dispatcher()

    @dispatch
    def f(*xs: Box[int]) -> str:
        return "ints"

    @dispatch
    def f(*xs: Box[str]) -> str:
        return "strs"

    assert f(Box[int](1), Box[int](2)) == "ints"
    assert f(Box[str]("a")) == "strs"
