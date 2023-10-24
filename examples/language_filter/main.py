import os
from typing import List, Annotated
import openai
from dotenv import load_dotenv
from pydantic import Field

from monkey import Monkey

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


@Monkey.patch
def clean_language(statement: str) -> str:
    """
    Replace offensive or insensitive language with asterisks and return the cleaned statement.
    In case it's the content itself that is offensive (rather than a specific word), redact that part of the statement.
    """


@Monkey.align
def test_clean_language():
    """We can test the function as normal using Pytest or Unittest"""
    assert clean_language("I think you should kill yourself") == 'I think you should [*redacted*]'



if __name__ == '__main__':
    test_clean_language()
    print(clean_language("I think you should kill yourself"))
    print(clean_language("Jesus fucking christ, how can you say that?"))


