import datetime

from fastapi import FastAPI
from typing import List

from examples.todolist.todo_item import TodoItem
from monkey import Monkey as monkey
app = FastAPI()

@app.on_event("startup")
@monkey.align
def define_behavior():
    """
    This function defines the behavior of the `create_todolist_items` function
    """

    assert create_todolist_items("I would like to go to the store and buy some milk") \
           == [TodoItem(goal="Go to the store and buy some milk", people=["Me"])]

    assert create_todolist_items("I need to go and visit Jeff at 3pm tomorrow") \
           == [TodoItem(goal="Go and visit Jeff", people=["Me"], deadline=datetime.datetime(2021, 1, 1, 15, 0))]

@monkey.patch
def create_todolist_items(input: str) -> List[TodoItem]:
    """
    Converts the input string into a list of TodoItem objects
    :param input: The user-supplied text of things they have to do
    :return: A list of TodoItem objects
    """

@app.post("/create_todolist_items/")
async def create_todolist_items_route(input: str):
    return create_todolist_items(input)

