from typing import Any, ClassVar, Self

def Field(
    default: Any = ...,
    *,
    default_factory: Any = ...,
    ge: Any = ...,
    gt: Any = ...,
    le: Any = ...,
    lt: Any = ...,
) -> Any: ...

class BaseModel:
    model_config: ClassVar[Any]

    def __init__(self, **data: Any) -> None: ...

    @classmethod
    def model_validate(cls, obj: Any) -> Self: ...

    def model_dump(self, *, mode: str = ...) -> dict[str, Any]: ...
