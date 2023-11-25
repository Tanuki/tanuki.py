import os
import unittest
from typing import List
from unittest import TestCase

import numpy as np
import openai
from dotenv import load_dotenv
from monkey_patch.models.embedding import Embedding
from monkey_patch.monkey import Monkey as monkey

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class TestEmbedding(TestCase):

    @monkey.patch
    def embed_sentiment(self, input: str) -> Embedding[np.ndarray]:
        """
        Determine if the inputs are positive or negative sentiment, or None
        """

    @monkey.align
    def align_embed_sentiment(self) -> None:
        """
        Align some embed_sentiment functions
        """
        # We try to push these embeddings apart
        assert self.embed_sentiment("I love this movie") != self.embed_sentiment("I hate this movie")

        # And push these embeddings together
        assert self.embed_sentiment("I love this movie") == self.embed_sentiment("I love this film")

    @monkey.align
    def broken_align_embed_sentiment(self) -> None:
        # This should break, because the embedding function 'embed_sentiment' must be compared to another embedding
        # function.
        assert self.embed_sentiment("I love this movie") == "I hate this movie"

    def test_data_type(self):
        # Test with np.ndarray
        embedding_array = Embedding[np.ndarray]([0, 2, 4])
        self.assertIsInstance(embedding_array._data, np.ndarray)

        # Test with list[float]
        embedding_list = Embedding[List[float]]([0.0, 2.0, 4.0])
        self.assertIsInstance(embedding_list._data, list)
        self.assertTrue(all(isinstance(item, float) for item in embedding_list._data))

    def test_instantiate_np_embedding(self):
        embedding = Embedding[np.ndarray]([0, 2, 4])
        trans = embedding.T
        assert isinstance(trans, np.ndarray)

    def test_get_embedding(self):
        embedding = self.embed_sentiment("I love this movie")
        transposed = embedding.T
        assert isinstance(embedding, Embedding)
        assert isinstance(transposed, np.ndarray)

    def test_align(self):
        self.align_embed_sentiment()

if __name__ == '__main__':
    unittest.main()