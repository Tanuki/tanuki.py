from dataclasses import dataclass
from typing import List, Tuple, Set, Dict, Mapping, MutableMapping, OrderedDict, ChainMap, Counter, DefaultDict, Deque, \
    MutableSequence, Sequence, Union, Literal

from validator import Validator


def test_validate_output():
    print("test_validate_output")
    validator = Validator()
    assert validator.validate_output("1", int)
    assert validator.validate_output("1.0", float)
    assert validator.validate_output('"1"', str)
    assert validator.validate_output("true", bool)
    assert validator.validate_output("null", None)
    assert not validator.validate_output("1", float)
    assert not validator.validate_output("1.0", int)
    assert not validator.validate_output('"1"', int)
    #assert not validator.validate_output("true", int)
    assert not validator.validate_output("null", int)
    assert not validator.validate_output("1", str)
    assert not validator.validate_output("1.0", str)
    assert not validator.validate_output("true", str)


if __name__ == "__main__":
    test_validate_output()