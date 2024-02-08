from dataclasses import dataclass
from pydantic import BaseModel

from tanuki.validator import Validator
import datetime
from tanuki.utils import prepare_object_for_saving
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

def test_validate_datetimes():
    print("test_validate_datetimes")
    time_1 = datetime.datetime.now()
    time_2 = datetime.date(2021, 1, 1)
    time_3 = datetime.time(12, 0)
    str_time_1 = prepare_object_for_saving(time_1)
    str_time_2 = prepare_object_for_saving(time_2)
    str_time_3 = prepare_object_for_saving(time_3)
    assert str_time_1 == {"year": time_1.year, "month": time_1.month, "day": time_1.day, "hour": time_1.hour, "minute": time_1.minute, "second": time_1.second, "microsecond": time_1.microsecond}
    assert str_time_2 == {"year": time_2.year, "month": time_2.month, "day": time_2.day}
    assert str_time_3 == {"hour": time_3.hour, "minute": time_3.minute, "second": time_3.second, "microsecond": time_3.microsecond}
    validator = Validator()
    assert validator.check_type(str_time_1, datetime.datetime)
    assert validator.check_type(str_time_2, datetime.date)
    assert validator.check_type(str_time_3, datetime.time)

if __name__ == "__main__":
    test_validate_output_dataclass()
    test_validate_output_pydantic()
    test_validate_output()
    test_validate_datetimes()
    print("All tests passed!")