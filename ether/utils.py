"""Utility functions for the Ether package.

This module contains shared utility functions that can be used across
different modules in the Ether package.
"""

import inspect
from abc import ABCMeta
from collections.abc import Callable
from datetime import UTC, datetime
from types import FunctionType
from typing import Any, get_type_hints

from ether._errors.status import ErrorStatus


def has_valid_arg_types(
    fn: Callable,
    expected_types: list[str],
    allow_extra: bool = False,
) -> ErrorStatus:
    """Check if a function has valid argument types."""
    is_passing = ErrorStatus(success=True, message="<PASSING>")

    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    type_hints = get_type_hints(fn, globals(), locals())
    is_passing.metadata["params"] = [
        {
            "name": p.name,
            "annotation": p.annotation,
            "type": type_hints.get(p.name, None),
            "default": p.default,
        }
        for p in params
    ]
    if not allow_extra and len(params) != len(expected_types):
        is_passing.success = False
        is_passing.message = f"Expected {len(expected_types)} arguments, got {len(params)}"
        return is_passing

    for idx, param in enumerate(is_passing.metadata["params"]):
        arg_name = param["name"]
        arg_type = param["type"]
        arg_type_str = arg_type.__name__ if arg_type is not None else None

        if arg_type is None:
            is_passing.success = False
            is_passing.message = f"Missing type hint for {arg_name}"
            return is_passing
        elif arg_type_str is not None and "." not in expected_types[idx]:  # Match only the class name
            arg_type_str = arg_type_str.split(".")[-1]
        if arg_type_str != expected_types[idx]:
            is_passing.success = False
            is_passing.message = f"Expected {expected_types[idx]} for {arg_name}, got {arg_type_str}"
            return is_passing
    return is_passing


def rfc3339_now() -> str:
    """Generate RFC 3339 timestamp with microsecond precision.

    Uses microsecond precision and 'Z' suffix for better cross-platform
    compatibility and to avoid clock skew issues in distributed systems.

    Returns:
        RFC 3339 formatted timestamp string
    """
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def final(obj: Any) -> Any:
    """Mark a method or property as final (non-overridable).

    - If decorating a function (or class/staticmethod), the mark is set on the function.
    - If decorating a property object, the mark is set on any available accessor(s).
    """
    return _Guard.final(obj)


class _Guard(ABCMeta):
    """Metaclass that prevents overriding members marked as final, including properties.

    Protection mechanisms:

    - [x] @final: Mark a method or property as final (non-overridable).
    - [ ] @sealed: Mark a class as sealed (cannot be inherited from).
    - [ ] @readonly: Mark a property as readonly (can be read but not modified after initialization).
    - [ ] @internal: Mark a method or property as internal (only accessible within the same module).

    Notes:
    - `@final` on a property doesn't guard against `x = property(...)` pattern
    """

    __SENTINEL = object()

    def __new__(cls, name, bases, class_dict):  # type: ignore[no-untyped-def]
        final_names = set()
        seen = set()

        # Walk full MRO of all bases to find any final members (methods or properties)
        for base in bases:
            for c in getattr(base, "__mro__", ()):
                if c in (object,) or c in seen:
                    continue
                seen.add(c)

                for key, value in vars(c).items():
                    # properties: final if any accessor is final
                    if isinstance(value, property):
                        if cls.__property_is_final(value):
                            final_names.add(key)
                        continue

                    # methods/classmethod/staticmethod
                    func = cls.__unwrap_callable(value)
                    if func is not None and cls.is_final(func):
                        final_names.add(key)

        # Block any shadowing of final members
        if any(key in final_names for key in class_dict):
            raise RuntimeError("Cannot override final member")

        return super().__new__(cls, name, bases, class_dict)

    # ---- helpers ----

    @staticmethod
    def __unwrap_callable(value: Any) -> FunctionType | None:
        """Return underlying function for functions/classmethods/staticmethods, else None."""
        if isinstance(value, FunctionType):
            return value
        func = getattr(value, "__func__", None)  # classmethod/staticmethod
        return func if isinstance(func, FunctionType) else None

    @classmethod
    def __property_is_final(cls, prop: property) -> bool:
        """A property is 'final' if any accessor is marked final."""
        accessors = (prop.fget, prop.fset, prop.fdel)
        return any(a is not None and cls.is_final(a) for a in accessors)

    @classmethod
    def is_final(cls, obj: Any) -> bool:
        if isinstance(obj, property):
            return cls.__property_is_final(obj)
        func = getattr(obj, "__func__", obj)
        return getattr(func, "__final", None) is cls.__SENTINEL

    @classmethod
    def final(cls, obj: Any) -> Any:
        """
        Mark a method or property as final (non-overridable).

        - If decorating a function (or class/staticmethod), the mark is set on the function.
        - If decorating a property object, the mark is set on any available accessor(s).
        """
        # property: mark all available accessors
        if isinstance(obj, property):
            if obj.fget is not None:
                setattr(obj.fget, "__final", cls.__SENTINEL)
            if obj.fset is not None:
                setattr(obj.fset, "__final", cls.__SENTINEL)
            if obj.fdel is not None:
                setattr(obj.fdel, "__final", cls.__SENTINEL)
            return obj

        # methods / classmethod / staticmethod
        func = getattr(obj, "__func__", obj)
        setattr(func, "__final", cls.__SENTINEL)
        return obj
