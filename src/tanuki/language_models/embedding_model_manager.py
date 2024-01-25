from typing import Dict

from tanuki.language_models.embedding_api_abc import Embedding_API
from tanuki.models.embedding import Embedding
from tanuki.models.function_description import FunctionDescription
from tanuki.language_models.llm_configs import DEFAULT_EMBEDDING_MODELS
from tanuki.constants import DEFAULT_EMBEDDING_MODEL_NAME
from tanuki.models.api_manager import APIManager
import logging

class EmbeddingModelManager(object):
    def __init__(self, function_modeler, api_provider: APIManager):
        self.function_modeler = function_modeler
        self.api_provider = api_provider
        self.initialized_functions = {}

    def get_embedding_case(self, args, function_description: FunctionDescription, kwargs, examples=None):
        # example_input = f"Examples:{examples}\n" if examples else ""
        content = f"Name: {function_description.name}\nArgs: {args}\nKwargs: {kwargs}"
        function_hash = function_description.__hash__()
        if function_hash in self.function_modeler.teacher_models_override: # check for overrides
            model = self.function_modeler.teacher_models_override[function_hash][0] # take currently the first model
        else:
            model = DEFAULT_EMBEDDING_MODELS[DEFAULT_EMBEDDING_MODEL_NAME]
        

        # loggings
        if function_hash not in self.initialized_functions:
            logging.info(f"Generating  function embeddings for {function_description.name} with {model.model_name}")
            self.initialized_functions[function_hash] = model.model_name
        elif self.initialized_functions[function_hash] != model.model_name:
            logging.info(f"Switching embeddings generation for {function_description.name} from {self.initialized_functions[function_hash]} to {model.model_name}")
            self.initialized_functions[function_hash] = model.model_name
        
        return content, model

    def __call__(self,
                 args,
                 function_description,
                 kwargs) -> Embedding:
        prompt, model = self.get_embedding_case(args, function_description, kwargs)
        embedding_response: Embedding = self.api_provider[model.provider].embed([prompt], model)[0]

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
