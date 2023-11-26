from dotenv import load_dotenv
load_dotenv()
import sys
sys.path.append("src")
from monkey_patch.monkey import Monkey as monkey
from monkey_patch.models.embedding import Embedding
import numpy as np


@monkey.patch
def score_sentiment(input: str) -> Embedding[np.ndarray]:
    """
    Scores the input between 0-10
    """

@monkey.align
def align_embed_sentiment() -> None:
    # We push these embeddings apart by declaring them to be different with the '!=' operator
    assert score_sentiment("I love this movie") != score_sentiment("I hate this movie")

    # And push these embeddings together by declaring equality
    assert score_sentiment("I love this movie") == score_sentiment("I love this film")
    assert score_sentiment("I love this movie") == score_sentiment("I loved watching the movie")

# This is a normal test that can be invoked
def test_score_sentiment():
    """We can test the function as normal using Pytest or Unittest"""
    align_embed_sentiment()
    print(score_sentiment("I like you"))
    print(score_sentiment("I do not like you"))
    print(score_sentiment("I am neutral"))

align_embed_sentiment()
#test_score_sentiment()

if __name__ == "__main__":
    test_score_sentiment()