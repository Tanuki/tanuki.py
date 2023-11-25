from typing import Dict

from monkey_patch.language_models.embedding_api_abc import Embedding_API
from monkey_patch.language_models.openai_api import OpenAI_API
from monkey_patch.models.embedding import Embedding
from monkey_patch.models.function_description import FunctionDescription


class EmbeddingModelManager(object):
    def __init__(self, function_modeler):
        self.function_modeler = function_modeler
        self.api_models: Dict[str, Embedding_API] = {"openai": OpenAI_API()}

    def get_embedding_case(self, args, function_description: FunctionDescription, kwargs, examples=None):
        #example_input = f"Examples:{examples}\n" if examples else ""
        content = f"Name: {function_description.name}\nArgs: {args}\nKwargs: {kwargs}"
        return content

    def __call__(self,
                 args,
                 function_description,
                 kwargs) -> Embedding:
        prompt = self.get_embedding_case(args, function_description, kwargs)
        embedding_response = self.api_models["openai"].embed([prompt])[0]
        embedding = function_description.output_type_hint(embedding_response.embedding)
        return embedding

