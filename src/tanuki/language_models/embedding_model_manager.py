from typing import Dict

from tanuki.language_models.embedding_api_abc import Embedding_API
from tanuki.models.embedding import Embedding
from tanuki.models.function_description import FunctionDescription


class EmbeddingModelManager(object):
    def __init__(self, function_modeler, api_providers: Dict[str, Embedding_API] = None):
        self.function_modeler = function_modeler
        self.api_providers = api_providers

    def get_embedding_case(self, args, function_description: FunctionDescription, kwargs, examples=None):
        # example_input = f"Examples:{examples}\n" if examples else ""
        content = f"Name: {function_description.name}\nArgs: {args}\nKwargs: {kwargs}"
        return content

    def __call__(self,
                 args,
                 function_description,
                 kwargs) -> Embedding:
        prompt = self.get_embedding_case(args, function_description, kwargs)
        embedding_response: Embedding = self.api_providers["openai"].embed([prompt])[0]

        # Coerce the embedding into the correct type
        embedding: Embedding = function_description.output_type_hint(embedding_response)


        # We don't need to postprocess the embedding for now, because saving them offers no information that we can use
        # to improve the model.

        #is_distilled_model = False  # TODO: check if distilled model
        #output = EmbeddingModelOutput(embedding, is_distilled_model)
        #if not output.distilled_model:
        #    self.function_modeler.postprocess_embeddable_datapoint(function_description.__hash__(),
        #                                                           function_description,
        #                                                           embedding)

        return embedding
