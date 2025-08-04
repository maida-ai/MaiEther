from typing import Any


class Singleton(type):
    __INSTANCES: dict[type, Any] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls.__INSTANCES:
            cls.__INSTANCES[cls] = super().__call__(*args, **kwargs)
        return cls.__INSTANCES[cls]
