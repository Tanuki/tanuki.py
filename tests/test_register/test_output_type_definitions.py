
from tanuki.register import Register 
from tanuki.models.function_description import FunctionDescription
from pydantic import BaseModel
from typing import List, Dict, Union, Optional
import json

def test_output_base_classes():
    def output_int(input: str) -> int:
        """
        Does something random
        """
    function_description: FunctionDescription = Register.load_function_description(output_int)
    assert function_description.output_class_definition == "int"
    assert function_description.output_type_hint is int 

    def output_float(input: str) -> float:
        """
        Does something random
        """
    function_description: FunctionDescription = Register.load_function_description(output_float)
    assert function_description.output_class_definition == "float"
    assert function_description.output_type_hint is float 

    def output_str(input: str) -> str:
        """
        Does something random
        """
    function_description: FunctionDescription = Register.load_function_description(output_str)
    assert function_description.output_class_definition == "str"
    assert function_description.output_type_hint is str 


    def output_bool(input: str) -> bool:
        """
        Does something random
        """
    function_description: FunctionDescription = Register.load_function_description(output_bool)
    assert function_description.output_class_definition == "bool"
    assert function_description.output_type_hint is bool

    def output_optional_bool(input: str) -> Optional[bool]:
        """
        Does something random
        """
    function_description: FunctionDescription = Register.load_function_description(output_optional_bool)
    assert function_description.output_class_definition == f"Union of following classes {json.dumps({'bool': 'bool', 'NoneType': 'None'})}"


    def output_union_bool(input: str) -> Union[bool, int]:
        """
        Does something random
        """
    function_description: FunctionDescription = Register.load_function_description(output_union_bool)
    assert function_description.output_class_definition == f"Union of following classes {json.dumps({'bool': 'bool', 'int': 'int'})}"

def test_output_pydantic_classes():
    class Person(BaseModel):
        name: str
        age: int
        height: float
        is_cool: bool
        favourite_numbers: List[int]
        even_more_favourite_numbers: tuple[int, ...]
        favourite_dict: Dict[str, int]

    def output_person(input: str) -> Person:
        """
        Does something random
        """
    person_output_description = '    class Person(BaseModel):\n        name: str\n        age: int\n        height: float\n        is_cool: bool\n        favourite_numbers: List[int]\n        even_more_favourite_numbers: tuple[int, ...]\n        favourite_dict: Dict[str, int]\n'
    function_description: FunctionDescription = Register.load_function_description(output_person)
    assert function_description.output_class_definition == person_output_description

    def output_optional_person(input: str) -> Optional[Person]:
        """
        Does something random
        """
    optional_person_description = f"Union of following classes {json.dumps({'Person': person_output_description, 'NoneType': 'None'})}"
    function_description: FunctionDescription = Register.load_function_description(output_optional_person)
    assert function_description.output_class_definition == optional_person_description

    def output_union_person(input: str) -> Union[Person, int]:
        """
        Does something random
        """
    union_person_description = f"Union of following classes {json.dumps({'Person': person_output_description, 'int': 'int'})}"
    function_description: FunctionDescription = Register.load_function_description(output_union_person)
    assert function_description.output_class_definition == union_person_description



if __name__ == '__main__':
    test_output_base_classes()
    test_output_pydantic_classes()
