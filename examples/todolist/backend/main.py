import datetime
import os
from contextlib import asynccontextmanager

import openai
from dotenv import load_dotenv
from fastapi import FastAPI
from typing import List

from pydantic import BaseModel

import tanuki
from todo_item import TodoItem

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class Query(BaseModel):
    input: str


@tanuki.align
async def define_behavior():
    """
    We define 2 input/output pairs for the LLM to learn from.
    """

    assert create_todolist_item("I would like to go to the store and buy some milk") \
           == TodoItem(goal="Go to the store and buy some milk",
                       people=["Me"])

    assert create_todolist_item("I need to go and visit Jeff at 3pm tomorrow") \
           == TodoItem(goal="Go and visit Jeff",
                       people=["Me"],
                       deadline=datetime.datetime(2021, 1, 1, 15, 0))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This function allows us to run code before the FastAPI server starts (and after it stops)
    In this case, we use it to run the align statements on the `create_todolist_items` function.
    :param app:
    :return:
    """

    await define_behavior()
    yield

app = FastAPI(lifespan=lifespan)


@tanuki.patch
def create_todolist_items(input: str) -> List[TodoItem]:
    """
    Converts the input string into a list of TodoItem objects
    :param input: The user-supplied text of things they have to do
    :return: A list of TodoItem objects
    """


@tanuki.patch
def create_todolist_item(input: str) -> TodoItem:
    """
    Converts the input string into a TodoItem object
    :param input: The user-supplied text of things they have to do
    :return: TodoItem object
    """


@app.post("/create_todolist_items/")
async def create_todolist_items_route(input: Query):
    return create_todolist_items(input.input)


@app.post("/create_todolist_item/")
async def create_todolist_item_route(input: Query):
    return create_todolist_item(input.input)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)