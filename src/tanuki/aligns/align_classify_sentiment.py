import os

import openai
from dotenv import load_dotenv

import tanuki

import unittest
from typing import Literal, Optional

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class TestClassifySentiment(unittest.TestCase):

    @tanuki.patch
    def classify_sentiment_2(self, input: str, input_2: str) -> Optional[Literal['Good', 'Bad']]:
        """
        Determine if the inputs are positive or negative sentiment, or None
        """

    @tanuki.patch
    def classify_sentiment(self, input: str) -> Optional[Literal['Good', 'Bad']]:
        """
        Determine if the input is positive or negative sentiment
        """

    @tanuki.align
    def test_align_classify_sentiment(self):
        """We can test the function as normal using Pytest or Unittest"""

        i_love_you = "I love you"
        print(self.classify_sentiment_2(i_love_you, "I love woo"))
        assert self.classify_sentiment_2(i_love_you, "I love woo") == 'Good'
        self.assertEqual(self.classify_sentiment_2(i_love_you, "I love woo"), 'Good')

        print(self.classify_sentiment("I love you"))
        assert self.classify_sentiment("I love you") == 'Good'

        assert self.classify_sentiment("I hate you") == 'Bad'
        assert self.classify_sentiment("I hate you") != 'Good'
        assert not self.classify_sentiment("Wednesdays are in the middle of the week")


if __name__ == '__main__':
    unittest.main()
