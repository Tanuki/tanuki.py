from pydantic import Field
from typing import Annotated
from dotenv import load_dotenv
load_dotenv()
import tanuki

@tanuki.patch
def score_sentiment(input: str) -> Annotated[int, Field(gt=0, lt=10)]:
    """
    Scores the input between 0-10
    """


@tanuki.align
def align_score_sentiment():
    """Register several examples to align your function"""

    assert score_sentiment("I love you") == 10
    assert score_sentiment("I hate you") == 0
    assert score_sentiment("You're okay I guess") == 5


# This is a normal test that can be invoked
def test_score_sentiment():
    """We can test the function as normal using Pytest or Unittest"""
    assert score_sentiment("I like you") == 7
