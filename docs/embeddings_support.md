# Support for embeddings for RAG (Retrieval augmented generation)

Support for getting embeddings for RAG use-cases have been implemented. The Open-AI ada-002 model is currently supported to get embeddings for input data. For embedding output the output typehint needs to be set as  `Embedding[np.ndarray]`. Currently adding align statements to steer embedding model behaviour is not implemented, but is on the roadmap. 


## Example
```python
@tanuki.patch
def score_sentiment(input: str) -> Embedding[np.ndarray]:
    """
    Scores the input between 0-10
    """
```
