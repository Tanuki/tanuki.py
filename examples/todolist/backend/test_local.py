
import datetime
import os
from contextlib import asynccontextmanager
import sys
sys.path.append("src")
import openai
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from typing import List

from pydantic import BaseModel

from monkey_patch.monkey import Monkey as monkey
from todo_item import TodoItem


@monkey.patch
def create_todolist_item_new913(input: str) -> TodoItem:
    """
    Convert the input string into a TodoItem object
    :param input: The user-supplied text of things they have to do
    :return: TodoItem object
    """


@monkey.align
def define_behavior():
    """
    We define 2 input/output pairs for the LLM to learn from.
    """

    assert create_todolist_item_new913("I would like to go to the store and buy some milk") \
           == TodoItem(goal="Go to the store and buy some milk", people=["Me"])

    assert create_todolist_item_new913("I need to go and visit Jeff at 3pm tomorrow") == TodoItem(goal="Go and visit Jeff", people=["Me"], deadline=datetime.datetime(2021, 1, 1, 18, 0))


if __name__ == "__main__":
    #output_value = {"deadline": {'year': 2022, 'month': 1, 'day': 1, 'hour': 15, 'minute': 0, 'second': 0, 'microsecond': 0, 'tzinfo': None}, "goal": 'Water the plants', "people": ["Me"]}
    #for key, value in output_value.items():
    #    # get the key attribute type of TODOItem
    #    attribute_type = TodoItem.__annotations__[key]
    #test_item = TodoItem(**output_value)
    #define_behavior()
    #from pydantic import BaseModel
    #import typing
    #from typing import Optional, List, get_args, Union
    #from datetime import datetime
#
    #class TodoItem(BaseModel):
    #    deadline: Optional[datetime] = None
    #    goal: str
    #    people: List[str]
    #    other: Union[int, str]  # This is a required field
#
    #required_fields = [field for field, field_type in TodoItem.__annotations__.items() if not (typing.get_origin(field_type) is Union and type(None) in typing.get_args(field_type))]
    ##required_args = [name, field for name, field in TodoItem.__annotations__.items() if Optional not in get_args(field)]
    #print(required_fields)  # Output: ['goal', 'people']
    define_behavior()
    #print(create_todolist_item_new913("I need to water the plants at 3pm tomorrow"))
    #print(create_todolist_item_new913("I need to go to the store now"))
    #print(create_todolist_item_new913("My mom needs to see the mechanic next wednesday"))
    #print(create_todolist_item_new913("I need to take out the dog"))
    #print(create_todolist_item_new913("I must frive the car to mechanic"))
    #print(create_todolist_item_new913("My sister needs tobuild a new closet"))
    #print(create_todolist_item_new913("I have a dinner with my wife in the evening"))
    #print(create_todolist_item_new913("I need to go to to the swim later"))
    print(create_todolist_item_new913("I need to go to the theatre"))
#