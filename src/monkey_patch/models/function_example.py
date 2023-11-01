import typing
from dataclasses import dataclass


@dataclass(frozen=True)
class FunctionExample:
    args: tuple
    kwargs: dict
    output: typing.Any