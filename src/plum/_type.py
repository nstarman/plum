__all__ = (
    "PromisedType",
    "ModuleType",
    "type_mapping",
    "resolve_type_hint",
    "is_faithful",
    "is_cacheable",
    "cache_key",
)

import abc
import enum
import sys
import typing
import warnings
from collections.abc import Callable, Hashable, Iterable
from functools import reduce
from operator import or_
from types import UnionType
from typing import Literal, TypeGuard, TypeVar, cast, final, get_args, get_origin

from beartype.vale._core._valecore import BeartypeValidator

from ._mypyc import mypyc_attr

T = TypeVar("T", bound="ResolvableType")


@mypyc_attr(native_class=False)
class ResolvableType(type):
    """A resolvable type that will resolve to `type` after `type` has been delivered via
    :meth:`.ResolvableType.deliver`. Before then, it will resolve to itself.

    Args:
        name (str): Name of the type to be delivered.
    """

    def __init__(self, name: str, /) -> None:
        type.__init__(self, name, (), {})
        self._type: type | None = None

    def __new__(cls: type[T], name: str) -> T:
        return type.__new__(cls, name, (), {})

    def deliver(self: T, delivered_type: type, /) -> T:
        """Deliver the type.

        Args:
            delivered_type (type): Type to deliver.

        Returns:
            :class:`ResolvableType`: `self`.
        """
        self._type = delivered_type
        return self

    def resolve(self: T) -> type | T:
        """Resolve the type.

        Returns:
            type: If no type has been delivered, this will return itself. If a type
                `type` has been delivered via :meth:`.ResolvableType.deliver`, this will
                return that type.
        """
        return self if self._type is None else self._type


@final
@mypyc_attr(native_class=False)
class PromisedType(ResolvableType):
    """A type that is promised to be available when you will you need it.

    Args:
        name (str, optional): Name of the type that is promised. Defaults to
            `"SomeType"`.
    """

    def __init__(self, name: str = "SomeType") -> None:
        ResolvableType.__init__(self, f"PromisedType[{name}]")
        self._name = name

    def __new__(cls, name: str = "SomeType") -> "PromisedType":
        # `ResolvableType.__new__` rather than `super().__new__` so `mypyc` can compile
        # this (it cannot generate `object.__new__` for a non-extension class).
        return ResolvableType.__new__(cls, f"PromisedType[{name}]")

    def __repr__(self) -> str:
        return f"<class 'plum.PromisedType[{self._name}]'>"


TModuleType = TypeVar("TModuleType", bound="ModuleType")


@final
@mypyc_attr(native_class=False)
class ModuleType(ResolvableType):
    """A type from another module.

    Args:
        module (str): Module that the type lives in.
        name (str): Name of the type that is promised.
        allow_fail (bool, optional): If the type is does not exist in `module`,
            do not raise an `AttributeError`.
        condition (Callable[[], bool], optional): A callable that can check a condition,
            like a package version. This callable will be run whenever `module` has been
            imported. Only if the callable returns `True`, `name` will be imported
            from `module`.
        faithful (bool, optional): If set, set the dunder `__faithful__` of the type to
            this value upon retrieval.
    """

    def __init__(
        self,
        module: str,
        name: str,
        *,
        allow_fail: bool = False,
        condition: Callable[[], bool] | None = None,
        faithful: bool | None = None,
    ) -> None:
        if module in {"__builtin__", "__builtins__"}:
            module = "builtins"
        super().__init__(f"ModuleType[{module}.{name}]")
        self._name = name
        self._module = module
        self._allow_fail = allow_fail
        self._condition = condition
        self._faithful = faithful

    def __new__(
        cls: type[TModuleType], module: str, name: str, **kwargs: object
    ) -> TModuleType:
        return ResolvableType.__new__(cls, f"ModuleType[{module}.{name}]")

    def deliver(self: TModuleType, delivered_type: type, /) -> TModuleType:
        return_value = super().deliver(delivered_type)
        if self._faithful is not None:
            # Only set `delivered_type.__faithful__` if it is not already set to a
            # different value.
            if (
                # Use `hasattr` instead of `_has_dunder_faithful` so `mypy` remains
                # aware that `delivered_type` is a `type` and won't complain about
                # `delivered_type.__name__`.
                hasattr(delivered_type, "__faithful__")
                and delivered_type.__faithful__ != self._faithful
            ):
                raise TypeError(
                    f"`{delivered_type.__name__}.__faithful__` is already set and "
                    f"would be changed by `{self.__name__}` to a different value."
                )
            delivered_type.__faithful__ = self._faithful  # type: ignore[attr-defined]
        return return_value

    def retrieve(self) -> bool:
        """Attempt to retrieve the type from the reference module.

        Returns:
            bool: Whether the retrieval succeeded.
        """
        if self._type is None and self._module in sys.modules:
            # If a condition is given, check the condition before attempting to import.
            if self._condition is not None and not self._condition():
                return False

            retrieved: object = sys.modules[self._module]
            for name in self._name.split("."):
                # If `retrieved` does not contain `name` and `self._allow_fail` is
                # set, then silently fail.
                if not hasattr(retrieved, name) and self._allow_fail:
                    return False
                retrieved = getattr(retrieved, name)
            # We expect this to be a type, so we cast it.
            self.deliver(cast(type, retrieved))
        return self._type is not None


def _is_hint(x: object) -> bool:
    """Check if an object is a type hint.

    Args:
        x (object): Object.

    Returns:
        bool: `True` if `x` is a type hint and `False` otherwise.
    """
    try:
        if x.__module__ == "builtins":
            # Check if `x` is a subscripted built-in. We do this by checking the module
            # of the type of `x`.
            x = type(x)
        return x.__module__ in {
            "types",  # E.g., `tuple[int]`
            "typing",
            "collections.abc",  # E.g., `Callable`
            "typing_extensions",
        }
    except AttributeError:
        return False


def _is_generic_hint(x: object, /) -> bool:
    """Check if an object is a parametrised :class:`typing.Generic` subclass.

    The test is that the origin is a class inheriting :class:`typing.Generic`, which
    is exactly the population whose parameter cannot be recovered from the value: such
    an instance records it in `__orig_class__` instead. Everything else is excluded by
    that same test rather than by name — the parametrised builtins (`list[int]`), the
    abstract base classes (`Sequence[int]`), `re.Pattern[str]` and
    `AbstractContextManager[int]` are not `Generic` subclasses and beartype checks
    them structurally, while `Annotated`, `Union`, `Optional`, `Literal` and
    `type[X]` have origins that are not classes at all. Plum's own parametric types
    have no origin.

    Deliberately *not* keyed on `__module__` (as :func:`_is_hint` is): a generic
    declared in an `exec`'d namespace, a doctest among them, reports its module as
    `builtins`, and its subscript is a `typing._GenericAlias`, so a module test
    mistakes it for a `typing` special form.

    Args:
        x (object): Object.

    Returns:
        bool: `True` if `x` is a parametrised user generic and `False` otherwise.
    """
    origin = get_origin(x)
    return (
        isinstance(origin, type)
        and issubclass(origin, typing.Generic)
        and get_args(x) != ()
    )


def _has_generic_hint(x: object, /) -> bool:
    """Check whether a parametrised user generic occurs anywhere in the hint `x`.

    This walks the arguments, so it also finds generics nested inside a union, an
    `Annotated`, or a parametrised builtin. It is computed once per signature, at
    registration, to keep the check off the matching path.

    Args:
        x (object): Type hint.

    Returns:
        bool: `True` if `x` contains a parametrised user generic.
    """
    return _is_generic_hint(x) or any(_has_generic_hint(a) for a in get_args(x))


def _hashable(x: object | type) -> TypeGuard[Hashable]:
    """Check if an object is hashable.

    Args:
        x (object): Object to check.

    Returns:
        bool: `True` if `x` is hashable and `False` otherwise.
    """
    try:
        hash(x)
        return True
    except TypeError:
        return False


type_mapping: dict[type, type] = {}
"""dict: When running :func:`resolve_type_hint`, map keys in this dictionary to the
values."""


def resolve_type_hint(x: object, /) -> object:
    """Resolve all :class:`ResolvableType` in a type or type hint.

    Args:
        x (type or type hint): Type hint.

    Returns:
        type or type hint: `x`, but with all :class:`ResolvableType`\\s resolved.
    """
    if _hashable(x) and isinstance(x, type) and x in type_mapping:
        return resolve_type_hint(type_mapping[x])
    elif _is_hint(x):
        origin = get_origin(x)
        args = get_args(x)
        if args == ():
            # `origin` might not make sense here. For example, `get_origin(Any)`
            # is `None`. Since the hint wasn't subscripted, the right thing is
            # to return the hint itself.
            return x
        if origin is UnionType:  # The new union syntax was used.
            return reduce(or_, (resolve_type_hint(arg) for arg in args))
        else:
            # Do not resolve the arguments for `Literal`s.
            if origin is not Literal:
                resolved_args = resolve_type_hint(args)
                assert isinstance(resolved_args, tuple)
                args = resolved_args

            # Ensure origin is not `None` before indexing.
            assert origin is not None
            return origin[args]

    elif x is None or x is Ellipsis:
        return x

    elif isinstance(x, tuple):
        return tuple(resolve_type_hint(arg) for arg in x)
    elif isinstance(x, list):
        return [resolve_type_hint(arg) for arg in x]
    elif isinstance(x, type):
        if not isinstance(x, ResolvableType):
            return x
        elif isinstance(x, ModuleType) and not x.retrieve():
            # If the type could not be retrieved, then just return the
            # wrapper. Namely, `x.resolve()` will then return `x`, which
            # means that the below call will result in an infinite
            # recursion.
            return x

        return resolve_type_hint(x.resolve())

    # This sits below the plain-`type` case on purpose. `resolve_type_hint` runs twice
    # per call for any method with a return annotation (via `convert`), and a plain
    # type such as `int` is by far the commonest argument. Testing it here means such
    # a type returns above without ever paying for the origin lookup; a parametrised
    # user generic is not a `type`, so it still reaches this branch.
    elif _is_generic_hint(x):
        # A parametrised user generic, e.g. `Box[int]`. Rebuild it from its origin so
        # that a `ResolvableType` nested in its arguments is resolved too.
        origin = get_origin(x)
        assert origin is not None
        resolved_args = tuple(resolve_type_hint(arg) for arg in get_args(x))
        return origin[resolved_args]

    # For example, `Is[lambda x: x > 0]` is an example of a `BeartypeValidator`.
    # We shouldn't resolve those.
    elif isinstance(x, BeartypeValidator):
        return x

    else:
        warnings.warn(
            f"Could not resolve the type hint of `{x}`. "
            f"I have ended the resolution here to not make your code break, but some "
            f"types might not be working correctly. "
            f"Please open an issue at https://github.com/beartype/plum.",
            stacklevel=2,
        )
    return x


UNION_TYPES = (typing.Union, UnionType, typing.Optional)


class Aspect(enum.Enum):
    """A property of an argument its dispatch cache key must capture.

    Faithful types need none of these (`type(x)` suffices). Each member names an
    extra thing `cache_key` must encode so that a category of non-faithful types
    becomes cacheable. `IDENTITY` supports `type[X]` and `VALUE` supports
    `Literal[...]`. Further members are additive.
    """

    IDENTITY = "identity"
    VALUE = "value"


_NO_ASPECTS: "frozenset[Aspect]" = frozenset()
_IDENTITY: "frozenset[Aspect]" = frozenset({Aspect.IDENTITY})
_VALUE: "frozenset[Aspect]" = frozenset({Aspect.VALUE})
_ALL_ASPECTS: "frozenset[Aspect]" = frozenset(Aspect)


class _SupportsDunderFaithful(typing.Protocol):
    __faithful__: bool


def _has_dunder_faithful(x: type, /) -> TypeGuard[_SupportsDunderFaithful]:
    """Check whether `x` has the `__faithful__` attribute."""
    return hasattr(x, "__faithful__")


class _Identity:
    """Identity cache-key wrapper for an object whose hash or equality cannot be
    trusted.

    A class cannot be used as a cache key directly: its hash and equality come from
    its metaclass, so a metaclass with a custom `__eq__` would make distinct classes
    collide (silent wrong hit) and one whose classes are unhashable would make the
    key raise `TypeError`. The same applies to an unhashable value. This wrapper keys
    on `id`, sidestepping both, and holds a reference to `obj` so its `id` is not
    reused while the entry lives.
    """

    __slots__ = ("obj",)

    def __init__(self, obj: object, /) -> None:
        self.obj = obj

    def __hash__(self) -> int:
        return id(self.obj)

    def __eq__(self, other: object, /) -> bool:
        return type(other) is _Identity and self.obj is other.obj


def _identity(x: object, /) -> object | None:
    """The identity component of `cache_key` for `x`.

    `None` for non-classes. For a class, the class itself when its metaclass is plain
    `type` (whose hash is id-based and equality is identity — already safe and fast),
    otherwise the metaclass-safe `_Identity` wrapper.
    """
    if not isinstance(x, type):
        return None
    return x if type(x) is type else _Identity(x)


_LITERAL_TYPES: frozenset[type] = frozenset({bool, int, str, bytes, type(None)})
"""The types a `Literal` argument may legally have, exactly."""

_LITERAL_BASES: tuple[type, ...] = (int, str, bytes, enum.Enum)
"""Bases of those types. `bool` and `NoneType` cannot be subclassed."""


def _value(x: object, /) -> object | None:
    """The value component of `cache_key` for `x`.

    Beartype matches `x` against `Literal[v]` exactly when `isinstance(x, type(v))`
    and `x == v`. The first half is settled by `type(x)`, which the key already
    carries; this slot settles the second half.

    An `x` that is not an instance of any legal `Literal` type can never match any
    `Literal`, so `type(x)` alone determines the answer and the slot is `None` — this
    is also what keeps unhashable arguments (a `list`, say) out of the key.

    Only an `x` of one of those types *exactly* is keyed on its value. Such an `x`
    has the built-in `__eq__` and `__hash__`, under which equal keys really do imply
    equal `x == literal` for every literal. A *subclass* instance can also match
    (`is_bearable(MyInt(1), Literal[1])` is `True`), but its `__eq__` and `__hash__`
    are user code and may be non-transitive, so two arguments could share a key while
    matching different literals. It is therefore keyed on its identity instead, which
    is strictly finer than its value and so never collides. This is the same
    precaution :class:`_Identity` already takes for classes.

    Args:
        x (object): Value.

    Returns:
        object or None: The value component of the cache key for `x`.
    """
    if type(x) in _LITERAL_TYPES:
        return x
    if isinstance(x, _LITERAL_BASES):
        # Note: this caches per object rather than per value. `Enum` members are
        # singletons, so for them the two coincide.
        return _Identity(x)
    return None


def cache_key(
    x: object, /, aspects: "frozenset[Aspect]" = _ALL_ASPECTS
) -> tuple[object, ...]:
    """Cache key for a value `x`, capturing the requested `aspects`.

    For any hint `t` with `is_cacheable(t)`, whether `x` matches `t` depends only on
    `cache_key(x)`, so a dispatch result for `x` can be memoised under this key. A
    resolver passes only the aspects its own types need, so it never captures more
    than necessary.

    The exact width of the returned tuple and the order of its slots are an
    implementation detail: every aspect added to :class:`Aspect` adds a slot to the
    default key. Only the contract is stable: equal keys imply the same match result.

    Note that the identity and value slots keep a strong reference to `x` — necessarily,
    since that is what makes `id`-based hashing safe. A function dispatching on
    `type[X]` or `Literal` therefore accumulates one cache entry per distinct argument
    class or value, and pins that class or value, for the function's lifetime;
    dynamically created classes are not collected. Call `f.clear_cache()` (or
    :func:`plum.clear_all_cache`) to release them. Because a `Literal` argument's value
    is typically caller-supplied, a function dispatching on one stops caching once it
    holds `plum._function._VALUE_CACHE_LIMIT` entries; further arguments resolve
    normally.

    Args:
        x (object): Value to compute a cache key for.
        aspects (frozenset[:class:`Aspect`], optional): Aspects to capture. Defaults
            to every aspect.

    Returns:
        tuple: Cache key for `x`.
    """
    key: tuple[object, ...] = (type(x),)
    if Aspect.IDENTITY in aspects:
        key += (_identity(x),)
    if Aspect.VALUE in aspects:
        key += (_value(x),)
    return key


_ARG_KEYS: "dict[frozenset[Aspect], Callable[[object], object]]" = {
    _NO_ASPECTS: type,
    _IDENTITY: lambda x: (type(x), _identity(x)),
    _VALUE: lambda x: (type(x), _value(x)),
    _IDENTITY | _VALUE: lambda x: (type(x), _identity(x), _value(x)),
}
"""`cache_key` specialised to each combination of aspects, for :class:`.Resolver` to
bind on the hot path. Testing `Aspect` membership per call costs ~80 ns per aspect
(hashing an `Enum` member is not cheap), which is the bulk of a cached dispatch; these
do the same work with the aspects already decided. One entry per subset of `Aspect`;
`test_arg_keys_agree_with_cache_key` holds them to `cache_key`."""


def is_faithful(x: object, /) -> bool:
    """Check whether a type hint is faithful.

    A type or type hint `t` is _faithful_ if, for all `x`::

        isinstance(x, t) == issubclass(type(x), t)

    i.e. matching depends only on `type(x)`. Faithful types are cacheable with a plain
    `type(x)` key. You can control faithfulness by setting `__faithful__`::

        class UnfaithfulType:
            __faithful__ = False

    `type[X]` is *not* faithful (its match depends on class identity); see
    :func:`is_cacheable`.

    Args:
        x (type or type hint): Type hint.

    Returns:
        bool: Whether `x` is faithful or not.
    """
    return _aspects(resolve_type_hint(x)) == _NO_ASPECTS


def is_cacheable(x: object, /) -> bool:
    """Check whether a type hint is cacheable.

    `t` is _cacheable_ if, for all `x`, whether `x` matches `t` is a function of
    :func:`cache_key(x) <cache_key>` alone. Every faithful type is cacheable; in
    addition `type[X]` is cacheable but not faithful (its match `issubclass(x, X)`
    depends on the class identity of `x`, which `cache_key` captures), and so is
    `Literal[...]` (its match depends on the value of `x`, likewise captured).

    Args:
        x (type or type hint): Type hint.

    Returns:
        bool: Whether `x` is cacheable or not.
    """
    return _aspects(resolve_type_hint(x)) is not None


def _combine(items: "Iterable[object]", /) -> "frozenset[Aspect] | None":
    """Union the aspects of `items` (each resolved); `None` if any is uncacheable."""
    acc = _NO_ASPECTS
    for item in items:
        sub = _aspects(resolve_type_hint(item))
        if sub is None:
            return None
        acc |= sub
    return acc


def _aspects(x: object, /) -> "frozenset[Aspect] | None":
    """Classify a **resolved** hint into the cache aspects it needs, or `None`.

    `frozenset()` = faithful (type-key suffices); `{IDENTITY}` = `type[X]`;
    `{VALUE}` = `Literal[...]`; a union is
    the union of its members (`None` if any member is uncacheable); everything else
    that is not a plainly faithful type is `None` (uncacheable). This is the single
    classifier `is_faithful` and `is_cacheable` derive from.
    """
    if _is_hint(x):
        origin = get_origin(x)
        args = get_args(x)
        if args == ():
            if origin is tuple:
                # `tuple[()]` is the one hint that is subscripted yet has no
                # arguments: `get_args(tuple[()])` is `()` on Python >= 3.11. It
                # matches on the *length* of the value, not its type, so it is
                # neither faithful nor cacheable. Bare `typing.Tuple` is
                # indistinguishable from it here and is conservatively lumped in.
                return None
            # Unsubscripted hints tend to be faithful: `Any`, `List`, `Callable`, ...
            return _NO_ASPECTS
        if origin is type:
            # `type[X]`: cacheable via the identity component of the cache key.
            return _IDENTITY
        if origin is Literal:
            # `Literal[...]`: cacheable via the value component of the cache key.
            return _VALUE
        if origin in UNION_TYPES:
            return _combine(args)
        return None

    elif x is None or x == Ellipsis:
        return _NO_ASPECTS

    elif isinstance(x, (tuple, list)):
        return _combine(x)

    elif isinstance(x, type):
        if _has_dunder_faithful(x):
            return _NO_ASPECTS if x.__faithful__ else None
        # Fallback: default `__instancecheck__` ⇒ faithful.
        faithful = type(x).__instancecheck__ in {
            type.__instancecheck__,
            abc.ABCMeta.__instancecheck__,
        }
        return _NO_ASPECTS if faithful else None

    elif _is_generic_hint(x):
        # A parametrised user generic. Whether a value matches depends on its
        # `__orig_class__`, which no bounded cache key captures, so this is
        # uncacheable and routes to the tier-two verify cache. Ordered below the
        # plain-`type` case for the same reason as in `resolve_type_hint`.
        return None

    else:
        warnings.warn(
            f"Could not determine whether `{x}` is faithful or cacheable. "
            f"I have concluded that it is neither, so your code might run "
            f"with subpar performance. "
            f"Please open an issue at https://github.com/beartype/plum.",
            stacklevel=2,
        )
    return None
