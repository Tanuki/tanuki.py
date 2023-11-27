from dataclasses import dataclass

from pydantic import BaseModel

from tanuki.validator import Validator


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

def test_validate_output_dataclass():
    print("test_validate_output_object")
    input_str = '{"name": "John", "age": 20, "height": 1.8, "is_cool": true}'

    @dataclass
    class Person:
        name: str
        age: int
        height: float
        is_cool: bool

        def __eq__(self, other):
            return self.dict() == other.dict()

        def __hash__(self):
            return hash(str(self.__dict__))

    validator = Validator()
    assert validator.validate_output(input_str, Person)

def test_validate_output_pydantic():
    print("test_validate_output_pydantic")
    class PersonPydantic(BaseModel):
        name: str
        age: int
        height: float
        is_cool: bool

    input_str = '{"name": "John", "age": 20, "height": 1.8, "is_cool": true}'
    validator = Validator()
    assert validator.validate_output(input_str, PersonPydantic)

if __name__ == "__main__":
    test_validate_output_dataclass()
    test_validate_output_pydantic()
    test_validate_output()