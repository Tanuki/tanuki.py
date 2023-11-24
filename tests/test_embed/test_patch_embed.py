import os
import numpy as np
import openai
from dotenv import load_dotenv
from monkey_patch.models.embedding import Embedding
from monkey_patch.monkey import Monkey as monkey

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

@monkey.patch
def embed_sentiment(input: str, input_2: str) -> Embedding[np.ndarray]:
    """
    Determine if the inputs are positive or negative sentiment, or None
    """


if __name__ == '__main__':
    i_love_you = "I love you"
    sentiment_vector = embed_sentiment(i_love_you, "I love woo")
    print(sentiment_vector)