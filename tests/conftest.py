from collections.abc import Generator

import pytest


@pytest.fixture(scope="function")
def clear_registry() -> Generator:
    """Clear the registry before a test."""
    from ether import Registry

    _old_spec = Registry._SPEC_REGISTRY.copy()
    _old_adapter = Registry._ADAPTER_REGISTRY.copy()
    Registry.clear_spec(force=True, sure=True)
    Registry.clear_adapter(force=True, sure=True)
    yield
    Registry._SPEC_REGISTRY = _old_spec
    Registry._ADAPTER_REGISTRY = _old_adapter
