import enum


class FunctionType(str, enum.Enum):
    SYMBOLIC = "symbolic"
    EMBEDDABLE = "embeddable"
