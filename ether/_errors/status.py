from dataclasses import dataclass, field
from typing import Any


@dataclass
class ErrorStatus:
    success: bool = False
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.success

    def __str__(self) -> str:
        return self.message
