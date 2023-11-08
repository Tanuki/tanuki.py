import inspect
import os

import openai
from dotenv import load_dotenv

from monkey_patch.monkey import Monkey as monkey
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

config = {"workspace_id": 0}
monkey.configure(**config)

@monkey.patch
def clean_language(statement: str) -> str:
    """
    Replace offensive or insensitive language with asterisks and return the cleaned statement.
    In case it's the content itself that is offensive (rather than a specific word), redact that part of the statement.
    """

@monkey.align
def test_clean_language():
    """We can test the function as normal using Pytest or Unittest"""
    assert clean_language("I think you should kill yourself") == 'I think you should [*redacted*]'
    assert clean_language("I want you to take a flying fuck at a rolling donut") == "I want you to take a [*redacted*]"


if __name__ == '__main__':
    test_clean_language()
    print(clean_language("I think you should jump off a bridge"))
    print(clean_language("Jesus fucking christ, how can you say that?"))