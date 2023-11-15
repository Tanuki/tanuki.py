import os
import unittest
import sys
sys.path.append("src")
from typing import Literal, Optional, List, Union
from pydantic import BaseModel, Field
import openai
from dotenv import load_dotenv

from monkey_patch.monkey import Monkey as monkey

load_dotenv()
from monkey_patch.register import Register
openai.api_key = os.getenv("OPENAI_API_KEY")

class Person(BaseModel):
    age: int = Field(..., ge=0, le=155)
    name: str
    favourite_colours: List[str]




@monkey.patch
def classify_sentiment_2(input: str, input_2: str) -> Optional[Literal['Good', 'Bad']]:
    """
    Determine if the inputs are positive or negative sentiment, or None
    """
@monkey.patch
def classify_sentiment(input: str) -> Optional[Literal['Good', 'Bad']]:
    """
    Determine if the input is positive or negative sentiment
    """
@monkey.align
def test_align_classify_sentiment():
    """We can test the function as normal using Pytest or Unittest"""
    i_love_you = "I love you"
    print(classify_sentiment_2(i_love_you, "I love woo"))
    assert classify_sentiment_2(i_love_you, "I love woo") == 'Good'
    assert classify_sentiment("I love you") == 'Good'
    assert classify_sentiment("I hate you") == 'Bad'
    assert not classify_sentiment("Wednesdays are in the middle of the week")


@monkey.patch
def classify_list_generic(input: List[str]) -> str:
    """
    Summarise the input
    """
@monkey.patch
def classify_list_typing(input: List[str]) -> str:
    """
    Summarise the input
    """
@monkey.patch
def classify_list_pydantic(input: List[Person]) -> str:
    """
    Summarise the input
    """
@monkey.patch
def classify_list_dict(input: List[dict]) -> str:
    """
    Summarise the input
    """

@monkey.patch
def classify_list_int(input: List[int]) -> str:
    """
    Summarise the input
    """

@monkey.patch
def classify_list_Union(input: List[Union[int, float]]) -> str:
    """
    Summarise the input
    """

@monkey.align
def align_list_generic():
    assert classify_list_generic(["Thats awesome", "Thats cool"]) == 'They found it awesome and cool'
    assert classify_list_generic(["Thats neat", "Thats ok"]) == 'They found it neat and ok'
@monkey.align
def align_list_typing():
    assert classify_list_typing(["Thats awesome", "Thats cool"]) == 'They found it awesome and cool'
    assert classify_list_typing(["Thats neat", "Thats ok"]) == 'They found it neat and ok'

@monkey.align
def align_list_pydantic():
    person_1 = Person(name="Jeff", age=25, favourite_colours=["Red", "Blue"])
    person_2 = Person(name="Thomas", age=33, favourite_colours=["Green", "Gray"])
    input = [person_1, person_2]
    assert classify_list_pydantic(input) == "Jeff is 25 years old and likes Red and Blue. Thomas is 33 years old and likes Green and Gray."

@monkey.align
def align_list_dict():
    input = [{"name": "Jeff", "age": 25, "favourite_colours": ["Red", "Blue"]}, {"name": "Thomas", "age": 33, "favourite_colours": ["Green", "Gray"]}]
    assert classify_list_dict(input) == "Jeff is 25 years old and likes Red and Blue. Thomas is 33 years old and likes Green and Gray."

@monkey.align
def align_list_int():
    assert classify_list_int([1, 2]) == "The sum of the list is 3"
    assert classify_list_int([1, 2, 3]) == "The sum of the list is 6"

@monkey.align
def align_list_Union():
    assert classify_list_Union([1, 2]) == "The sum of the list is 3"
    assert classify_list_Union([1.0, 2.0, 3.0]) == "The sum of the list is 6.0"

def test_list():
    # This tests all the list aligns
    # can be called by pytest or unittest
    align_list_generic()
    align_list_typing()
    align_list_pydantic()
    align_list_dict()
    align_list_int()
    align_list_Union()
    print("All list aligns passed!")


@monkey.patch
def classify_str(input: str) -> str:
    """
    Summarise the input
    """

@monkey.patch
def classify_pydantic(input: Person) -> str:
    """
    Summarise the input
    """
@monkey.patch
def classify_dict(input: dict) -> str:
    """
    Summarise the input
    """

@monkey.patch
def classify_int(input: int) -> str:
    """
    Summarise the input
    """

@monkey.patch
def classify_Union(input: Union[int, float]) -> str:
    """
    Summarise the input
    """

@monkey.align
def align_string():
    assert classify_str("Thats awesome") == 'They found it awesome and cool'
    assert classify_str("Thats neat") == 'They found it neat and ok'


@monkey.align
def align_pydantic():
    person_1 = Person(name="Jeff", age=25, favourite_colours=["Red", "Blue"])
    input = person_1
    assert classify_pydantic(input) == "Jeff is 25 years old and likes Red and Blue."

@monkey.align
def align_dict():
    input = {"name": "Jeff", "age": 25, "favourite_colours": ["Red", "Blue"]}
    assert classify_dict(input) == "Jeff is 25 years old and likes Red and Blue."

@monkey.align
def align_list_int():
    assert classify_int(1) == "This is number 1"

@monkey.align
def align_list_Union():
    assert classify_Union(1) == "This is number 1"
    assert classify_Union(2.0) == "This is number 2"

def test_single():
    # This tests all the single aligns
    # can be called by pytest or unittest
    align_string()
    align_pydantic()
    align_dict()
    align_list_int()
    align_list_Union()
    print("All single aligns passed!")

def _parse_examples(test_func):
    # check that all the examples are correctly readable
    function_description = Register.load_function_description(test_func)
    function_modeler = monkey.function_modeler
    align_examples = function_modeler.get_alignments(function_description.__hash__())
    examples = "\n".join([f"Inputs:\nArgs: {align['args']}\nKwargs: {align['kwargs']}\nOutput: {align['output']}" for align in align_examples])

def test_parse_align_datasets():
    # Test that all the examples that are aligned are correctly parsable into the prompt format we have defined
    _parse_examples(classify_list_generic)
    _parse_examples(classify_list_typing)
    _parse_examples(classify_list_pydantic)
    _parse_examples(classify_list_dict)
    _parse_examples(classify_list_int)
    _parse_examples(classify_list_Union)
    _parse_examples(classify_str)
    _parse_examples(classify_pydantic)
    _parse_examples(classify_dict)
    _parse_examples(classify_int)
    _parse_examples(classify_Union)
    print("All examples parsed correctly!")
