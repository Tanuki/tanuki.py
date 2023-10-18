from typing import Optional, Literal

from monkey import Monkey as monkey


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


def test_classify_sentiment():
    """We can test the function as normal using Pytest or Unittest"""

    i_love_you = "I love you"
    print(classify_sentiment_2(i_love_you, "I love woo"))
    assert classify_sentiment_2(i_love_you, "I love woo") == 'Good'

    print(classify_sentiment("I love you"))
    assert classify_sentiment("I love you") == 'Good'

    assert classify_sentiment("I hate you") == 'Bad'
    assert classify_sentiment("I hate you") != 'Good'
    assert not classify_sentiment("Wednesdays are in the middle of the week")
