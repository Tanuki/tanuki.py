import os
import unittest
from typing import List
from unittest import TestCase

from dotenv import load_dotenv

load_dotenv()

import numpy as np
from tanuki.models.embedding import Embedding
import tanuki



class TestEmbedding(TestCase):

    @tanuki.patch
    def embed_sentiment(self, input: str) -> Embedding[np.ndarray]:
        """
        Determine if the inputs are positive or negative sentiment, or None
        """

    @tanuki.patch
    def embed_sentiment2(self, input: str) -> Embedding[np.ndarray]:
        """
        Determine if the inputs are positive or negative sentiment, or None
        """

    @tanuki.patch
    def is_positive_sentiment(self, input: str) -> bool:
        """
        Determine if the inputs are positive or negative sentiment
        """

    @tanuki.align
    def align_embed_sentiment(self) -> None:
        """
        Align some embed_sentiment functions
        """
        # We try to push these embeddings apart
        assert self.embed_sentiment("I love this movie") != self.embed_sentiment("I hate this movie")

        # And push these embeddings together
        assert self.embed_sentiment("I love this movie") == self.embed_sentiment("I love this film")
        assert self.embed_sentiment("I love this movie") == self.embed_sentiment("I loved watching the movie")

    @tanuki.align
    def broken_heterogenous_align(self) -> None:
        # This should break, because the embedding function 'embed_sentiment' must be compared to itself with
        # different inputs.
        # TODO: We should investigate supporting alignment between heterogenous functions.
        #  Will functions be trained piecewise?
        assert self.embed_sentiment("I love this movie") == self.embed_sentiment2("I hate this movie")

    @tanuki.align
    def broken_align_embed_sentiment(self) -> None:
        # This should break, because the embedding function 'embed_sentiment' must be compared to another embedding
        # function.
        assert self.embed_sentiment("I love this movie") == "I hate this movie"

    @tanuki.align
    def broken_align_symbolic_with_embeddable(self) -> None:
        # This should break, because the embedding function 'embed_sentiment' must be compared to another embedding
        # function.
        assert self.embed_sentiment("I love this movie") == self.is_positive_sentiment("I hate this movie")

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

    def test_cannot_align_embeddable_with_literal(self):
        with self.assertRaises(ValueError):
            self.broken_align_embed_sentiment()

    def test_cannot_align_symbolic_with_embeddable(self):
        with self.assertRaises(ValueError):
            self.broken_align_symbolic_with_embeddable()

    def test_cannot_align_heterogenous(self):
        with self.assertRaises(ValueError):
            self.broken_heterogenous_align()

if __name__ == '__main__':
    unittest.main()