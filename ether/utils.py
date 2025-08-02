"""Utility functions for the Ether package.

This module contains shared utility functions that can be used across
different modules in the Ether package.
"""

from abc import ABCMeta
from datetime import UTC, datetime
from types import FunctionType
from typing import Any


def rfc3339_now() -> str:
    """Generate RFC 3339 timestamp with microsecond precision.

    Uses microsecond precision and 'Z' suffix for better cross-platform
    compatibility and to avoid clock skew issues in distributed systems.

    Returns:
        RFC 3339 formatted timestamp string
    """
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


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
