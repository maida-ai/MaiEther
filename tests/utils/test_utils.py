"""Tests for utility functions."""

from datetime import datetime, timedelta

import pytest

from ether.utils import _Guard, rfc3339_now


class TestRfc3339Now:
    """Test the rfc3339_now utility function."""

    def test_rfc3339_now_format(self) -> None:
        """Test that rfc3339_now() generates correct RFC 3339 format."""
        timestamp = rfc3339_now()

        # Should end with Z
        assert timestamp.endswith("Z")

        # Should be parseable as RFC 3339
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        assert dt.tzinfo is not None

        # Should have microsecond precision
        assert "." in timestamp
        microseconds = timestamp.split(".")[1].replace("Z", "")
        assert len(microseconds) == 6  # 6 digits for microseconds

    def test_rfc3339_now_timezone(self) -> None:
        """Test that rfc3339_now() uses UTC timezone."""
        timestamp = rfc3339_now()

        # Convert Z to +00:00 for fromisoformat
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

        # Should be in UTC (offset should be 0)
        assert dt.tzinfo.utcoffset(dt) == timedelta(0)  # type: ignore[union-attr]

    def test_rfc3339_now_uniqueness(self) -> None:
        """Test that rfc3339_now() generates unique timestamps."""
        timestamps = [rfc3339_now() for _ in range(10)]

        # All timestamps should be unique (microsecond precision should ensure this)
        assert len(set(timestamps)) == len(timestamps)


class TestGuard:
    """Test the _Guard metaclass for access control."""

    def test_final_method_decorator(self) -> None:
        """Test that @_Guard.final decorator works correctly."""

        class Parent(metaclass=_Guard):
            @_Guard.final
            def final_method(self) -> str:
                return "parent"

            def normal_method(self) -> str:
                return "parent"

        # Check that the method is marked as final
        assert _Guard.is_final(Parent.final_method)
        assert not _Guard.is_final(Parent.normal_method)

    def test_final_method_cannot_be_overridden(self) -> None:
        """Test that final methods cannot be overridden in subclasses."""

        class Parent(metaclass=_Guard):
            @_Guard.final
            def final_method(self) -> str:
                return "parent"

        # Attempting to override should raise RuntimeError
        with pytest.raises(RuntimeError, match="Cannot override final"):

            class Child(Parent):
                def final_method(self) -> str:
                    return "child"

    def test_non_final_methods_can_be_overridden(self) -> None:
        """Test that non-final methods can be overridden normally."""

        class Parent(metaclass=_Guard):
            def normal_method(self) -> str:
                return "parent"

        class Child(Parent):
            def normal_method(self) -> str:
                return "child"

        # Should not raise any error
        child = Child()
        assert child.normal_method() == "child"

    def test_final_methods_inherited_from_multiple_bases(self) -> None:
        """Test that final methods from multiple base classes are protected."""

        class Base1(metaclass=_Guard):
            @_Guard.final
            def method1(self) -> str:
                return "base1"

        class Base2(metaclass=_Guard):
            @_Guard.final
            def method2(self) -> str:
                return "base2"

        class Parent(Base1, Base2):
            pass

        # Attempting to override either final method should raise RuntimeError
        with pytest.raises(RuntimeError, match="Cannot override final"):

            class Child(Parent):
                def method1(self) -> str:
                    return "child1"

        with pytest.raises(RuntimeError, match="Cannot override final"):

            class Child2(Parent):
                def method2(self) -> str:
                    return "child2"

    def test_final_methods_with_different_names(self) -> None:
        """Test that final methods with different names don't conflict."""

        class Parent(metaclass=_Guard):
            @_Guard.final
            def method1(self) -> str:
                return "method1"

            @_Guard.final
            def method2(self) -> str:
                return "method2"

        class Child(Parent):
            def new_method(self) -> str:
                return "new"

        # Should not raise any error
        child = Child()
        assert child.new_method() == "new"

    def test_final_methods_in_deep_inheritance(self) -> None:
        """Test that final methods are protected in deep inheritance chains."""

        class Base(metaclass=_Guard):
            @_Guard.final
            def final_method(self) -> str:
                return "base"

        class Parent(Base):
            pass

        class Child(Parent):
            pass

        # Attempting to override at any level should raise RuntimeError
        with pytest.raises(RuntimeError, match="Cannot override final"):

            class GrandChild(Child):
                def final_method(self) -> str:
                    return "grandchild"

    def test_final_methods_with_parameters(self) -> None:
        """Test that final methods with parameters work correctly."""

        class Parent(metaclass=_Guard):
            @_Guard.final
            def final_method(self, param: str) -> str:
                return f"parent_{param}"

        # Should not raise any error for method definition
        parent = Parent()
        assert parent.final_method("test") == "parent_test"

        # Attempting to override should raise RuntimeError
        with pytest.raises(RuntimeError, match="Cannot override final"):

            class Child(Parent):
                def final_method(self, param: str) -> str:
                    return f"child_{param}"

    def test_final_methods_with_properties(self) -> None:
        """Test that final methods work alongside properties."""

        class Parent(metaclass=_Guard):
            @property
            def normal_property(self) -> str:
                return "property"

            @_Guard.final
            def final_method(self) -> str:
                return "final"

        class Child(Parent):
            @property
            def normal_property(self) -> str:
                return "child_property"

        # Should not raise any error
        child = Child()
        assert child.normal_property == "child_property"
        assert child.final_method() == "final"

    def test_final_methods_with_static_methods(self) -> None:
        """Test that final methods work alongside static methods."""

        class Parent(metaclass=_Guard):
            @staticmethod
            def static_method() -> str:
                return "static"

            @_Guard.final
            def final_method(self) -> str:
                return "final"

        class Child(Parent):
            @staticmethod
            def static_method() -> str:
                return "child_static"

        # Should not raise any error
        assert Child.static_method() == "child_static"
        child = Child()
        assert child.final_method() == "final"

    def test_final_methods_with_class_methods(self) -> None:
        """Test that final methods work alongside class methods."""

        class Parent(metaclass=_Guard):
            @classmethod
            def class_method(cls) -> str:
                return "class"

            @_Guard.final
            def final_method(self) -> str:
                return "final"

        class Child(Parent):
            @classmethod
            def class_method(cls) -> str:
                return "child_class"

        # Should not raise any error
        assert Child.class_method() == "child_class"
        child = Child()
        assert child.final_method() == "final"

    def test_final_methods_with_abstract_methods(self) -> None:
        """Test that final methods work alongside abstract methods."""
        from abc import ABC, abstractmethod

        class Parent(ABC, metaclass=_Guard):
            @abstractmethod
            def abstract_method(self) -> str:
                pass

            @_Guard.final
            def final_method(self) -> str:
                return "final"

        class Child(Parent):
            def abstract_method(self) -> str:
                return "implemented"

        # Should not raise any error
        child = Child()
        assert child.abstract_method() == "implemented"
        assert child.final_method() == "final"

    def test_final_methods_with_multiple_decorators(self) -> None:
        """Test that final decorator works with other decorators."""

        class Parent(metaclass=_Guard):
            @_Guard.final
            @staticmethod
            def static_final() -> str:
                return "static_final"

            @_Guard.final
            @classmethod
            def class_final(cls) -> str:
                return "class_final"

        # Should not raise any error
        assert Parent.static_final() == "static_final"
        assert Parent.class_final() == "class_final"

        # Attempting to override should raise RuntimeError
        with pytest.raises(RuntimeError, match="Cannot override final"):

            class Child(Parent):
                @staticmethod
                def static_final() -> str:
                    return "child_static"

    def test_final_methods_with_inheritance_and_new_methods(self) -> None:
        """Test that new methods can be added alongside final methods."""

        class Parent(metaclass=_Guard):
            @_Guard.final
            def final_method(self) -> str:
                return "final"

        class Child(Parent):
            def new_method(self) -> str:
                return "new"

        class GrandChild(Child):
            def another_new_method(self) -> str:
                return "another_new"

        # Should not raise any error
        grandchild = GrandChild()
        assert grandchild.final_method() == "final"
        assert grandchild.new_method() == "new"
        assert grandchild.another_new_method() == "another_new"

    def test_final_methods_with_slots(self) -> None:
        """Test that final methods work with classes using __slots__."""

        class Parent(metaclass=_Guard):
            __slots__ = ("attr",)

            def __init__(self):
                self.attr = "value"

            @_Guard.final
            def final_method(self) -> str:
                return "final"

        class Child(Parent):
            def new_method(self) -> str:
                return "new"

        # Should not raise any error
        child = Child()
        assert child.attr == "value"
        assert child.final_method() == "final"
        assert child.new_method() == "new"

    def test_final_methods_with_metaclass_conflicts(self) -> None:
        """Test that _Guard works with other metaclasses."""

        class OtherMeta(type):
            pass

        # This should work - _Guard should be the primary metaclass
        class Parent(metaclass=_Guard):
            @_Guard.final
            def final_method(self) -> str:
                return "final"

        # Should not raise any error
        parent = Parent()
        assert parent.final_method() == "final"

    def test_final_methods_with_inheritance_from_non_guard_class(self) -> None:
        """Test that final methods work when inheriting from non-Guard classes."""

        class RegularParent:
            def normal_method(self) -> str:
                return "normal"

        class GuardParent(metaclass=_Guard):
            @_Guard.final
            def final_method(self) -> str:
                return "final"

        class Child(RegularParent, GuardParent):
            def normal_method(self) -> str:
                return "child_normal"

        # Should not raise any error
        child = Child()
        assert child.normal_method() == "child_normal"
        assert child.final_method() == "final"

        # Attempting to override final method should raise RuntimeError
        with pytest.raises(RuntimeError, match="Cannot override final"):

            class GrandChild(Child):
                def final_method(self) -> str:
                    return "grandchild_final"

    def test_is_final(self) -> None:
        """Test that is_final() works correctly."""

        class Parent(metaclass=_Guard):
            @_Guard.final
            def final_method(self) -> str:
                return "final"

            @_Guard.final  # type: ignore[prop-decorator]
            @property
            def final_property(self) -> str:
                return "final"

            def normal_method(self) -> str:
                return "normal"

            @property
            def normal_property(self) -> str:
                return "normal"

        assert _Guard.is_final(Parent.final_method)
        assert _Guard.is_final(Parent.final_property)
        assert not _Guard.is_final(Parent.normal_method)
        assert not _Guard.is_final(Parent.normal_property)

    def test_final_property_pre(self) -> None:
        """Test that final properties work correctly."""

        class Parent(metaclass=_Guard):
            @property
            @_Guard.final
            def final_pre_property(self) -> str:
                return "final"

        with pytest.raises(RuntimeError, match="Cannot override final"):

            class Child(Parent):
                @property
                def final_pre_property(self) -> str:
                    return "child_final"

    def test_final_property_post(self) -> None:
        """Test that final properties work correctly."""

        class Parent(metaclass=_Guard):
            @property
            def final_property(self) -> str:
                return "final"

            @final_property.setter
            def final_property(self, value: str) -> None:
                pass

            @_Guard.final
            @final_property.deleter  # type: ignore[name-defined]
            def final_property(self) -> None:
                pass

        with pytest.raises(RuntimeError, match="Cannot override final"):

            class Child(Parent):
                # Note: This is NOT a proper way to override a property
                # but it's a valid way to test the _Guard metaclass
                @property  # type: ignore[misc]
                def final_property(self) -> str:
                    return "child_final"
