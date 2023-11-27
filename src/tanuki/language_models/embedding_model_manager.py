from typing import Dict

from tanuki.language_models.embedding_api_abc import Embedding_API
from tanuki.language_models.openai_api import OpenAI_API
from tanuki.models.embedding import Embedding
from tanuki.models.embedding_model_output import EmbeddingModelOutput
from tanuki.models.function_description import FunctionDescription


class EmbeddingModelManager(object):
    def __init__(self, function_modeler):
        self.function_modeler = function_modeler
        self.api_models: Dict[str, Embedding_API] = {"openai": OpenAI_API()}

    def get_embedding_case(self, args, function_description: FunctionDescription, kwargs, examples=None):
        # example_input = f"Examples:{examples}\n" if examples else ""
        content = f"Name: {function_description.name}\nArgs: {args}\nKwargs: {kwargs}"
        return content

    def __call__(self,
                 args,
                 function_description,
                 kwargs) -> Embedding:
        prompt = self.get_embedding_case(args, function_description, kwargs)
        embedding_response = self.api_models["openai"].embed([prompt])[0]
        embedding: Embedding = function_description.output_type_hint(embedding_response.embedding)


        # We don't need to postprocess the embedding for now, because saving them offers no information that we can use
        # to improve the model.

        #is_distilled_model = False  # TODO: check if distilled model
        #output = EmbeddingModelOutput(embedding, is_distilled_model)
        #if not output.distilled_model:
        #    self.function_modeler.postprocess_embeddable_datapoint(function_description.__hash__(),
        #                                                           function_description,
        #                                                           embedding)

        return embedding
