"""Infer `__orig_class__` for bare instantiations of a user generic."""

__all__ = ["generic"]

import functools
import warnings
from collections.abc import Callable
from typing import Any, TypeVar, overload

T = TypeVar("T")


@overload
def generic(cls: type[T], /) -> type[T]: ...


@overload
def generic(cls: None = ..., /) -> "Callable[[type[T]], type[T]]": ...


def generic(
    cls: "type[T] | None" = None, /
) -> "type[T] | Callable[[type[T]], type[T]]":
    """Make bare instantiation of a :class:`typing.Generic` subclass record its
    parametrisation.

    Plum dispatches on a parametrised user generic through `__orig_class__`, the
    attribute Python sets when you write `Box[int](1)`. A *bare* `Box(1)` sets
    nothing, so it can only ever match the unparametrised `Box`. This decorator
    closes that gap: after decoration, `Box(1)` behaves like `Box[T](1)`, where `T`
    is whatever the class's `__infer_type_parameter__` returns for the
    freshly-constructed instance.

    The class **must** define `__infer_type_parameter__` as a :func:`classmethod`
    taking the instance and returning the type parameter, or a tuple of them for a
    multi-parameter generic. It is the class's own business how it infers: reading
    the value's type, as below, is only the obvious choice.

    Explicit subscription still wins. Python assigns `__orig_class__` *after*
    `__init__` returns, so `Box[str](1)` overwrites the inferred `Box[int]`.

    The parametrisation is built from `type(self)`, so a generic subclass of a
    decorated class infers its own class rather than the decorated one.

    Two situations are reported rather than raised, because neither is worth
    breaking a constructor over:

    - A class whose `__slots__` omit `'__orig_class__'` cannot carry the attribute
      at all. That is a :class:`RuntimeWarning` at decoration time, and another
      when a first instance is built.
    - `__infer_type_parameter__` returning something the class cannot be
      subscripted by is a :class:`RuntimeWarning` at construction time. The
      instance is returned intact, unparametrised.

    Note that this moves work into the constructor: every instantiation now runs
    the inference and builds a parametrisation. Dispatch on the result is
    :class:`~plum.Aspect`-cacheable and so as fast as any other generic dispatch,
    but construction itself is not free.

    Args:
        cls (type, optional): Class to decorate. Omit it to use the decorator with
            parentheses, as `@generic()`.

    Returns:
        type: `cls`, with its `__init__` wrapped.

    Example:
        >>> from typing import Generic, TypeVar
        >>> from plum import generic
        >>> T = TypeVar("T")

        >>> @generic
        ... class Box(Generic[T]):
        ...     def __init__(self, value):
        ...         self.value = value
        ...
        ...     @classmethod
        ...     def __infer_type_parameter__(cls, instance):
        ...         return type(instance.value)

        >>> Box(1).__orig_class__ == Box[int]
        True
    """
    if cls is None:
        # `@generic()`, with parentheses.
        return generic

    if not hasattr(cls, "__infer_type_parameter__"):
        raise TypeError(
            f"`@generic` requires `{cls.__name__}` to define "
            f"`__infer_type_parameter__(cls, instance)` as a `classmethod`."
        )

    slots = cls.__dict__.get("__slots__")
    if slots is not None and "__orig_class__" not in slots:
        warnings.warn(
            f"`{cls.__name__}` defines `__slots__` without `'__orig_class__'`, so "
            f"its instances cannot carry that attribute and dispatch will not "
            f"route them to parametrised methods. Add `'__orig_class__'` to "
            f"`__slots__`.",
            RuntimeWarning,
            stacklevel=2,
        )

    original_init = cls.__init__

    @functools.wraps(original_init)
    def __init__(self: Any, *args: Any, **kw_args: Any) -> None:
        original_init(self, *args, **kw_args)
        # Use `type(self)`, not `cls`: a subclass must infer its own class.
        actual = type(self)
        try:
            parameter = actual.__infer_type_parameter__(self)
            # `object.__setattr__` so that a frozen dataclass, or any class with a
            # custom `__setattr__`, still gets the attribute.
            object.__setattr__(self, "__orig_class__", actual[parameter])
        except (TypeError, AttributeError) as e:
            warnings.warn(
                f"`@generic` could not set `__orig_class__` on a "
                f"`{actual.__name__}` instance: {type(e).__name__}: {e}. Dispatch "
                f"will not route it to parametrised methods.",
                RuntimeWarning,
                stacklevel=2,
            )

    cls.__init__ = __init__  # type: ignore[method-assign]
    return cls
