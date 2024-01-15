# import abstract base class
import boto3 as boto3
from tanuki.language_models.aws_bedrock_api import Bedrock_API
from typing import List
from tanuki.language_models.llm_configs.abc_base_config import BaseModelConfig
from tanuki.language_models.llm_configs.titan_config import TitanBedrockConfig
from tanuki.models.embedding import Embedding
import json

class Titan_Bedrock_API(Bedrock_API):
    def __init__(self) -> None:
        # initialise the base class
        super().__init__()

    def generate(self, model: BaseModelConfig, system_message: str, prompt: str, **kwargs):
        """
        Generate a response using the Bedrock API for the specified Titan model.

        Args:
            model: The model to use for generation.
            system_message: The system message to use for generation.
            prompt: The prompt to use for generation.
            kwargs: Additional generation parameters.
        
        Returns:
            The generated response.
        """

        raise NotImplementedError("Response generations for Titan Bedrock API have not yet been implemented")

    def embed(self, texts: List[str], model: TitanBedrockConfig, **kwargs) -> List[Embedding]:
        """
        Generate embeddings for the provided texts using the specified OpenAI model.
        Lightweight wrapper over the OpenAI client.

        :param texts: A list of texts to embed.
        :param model: The model to use for embeddings.
        :return: A list of embeddings.
        """

        embeddings = []
        for text in texts:
            body = json.dumps({
                "inputText": text,
                })
        response_body = self.send_api_request(model, body)
        try:
            embedding = response_body.get("embedding")
            embeddings.append(Embedding(embedding))
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
        
        return embeddings
