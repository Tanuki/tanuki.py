from tanuki.language_models.llm_configs.abc_base_config import BaseModelConfig
from tanuki.language_models.llm_configs.openai_config import OpenAIConfig
from tanuki.language_models.llm_configs.llama_config import LlamaBedrockConfig
from tanuki.language_models.llm_configs.titan_config import TitanBedrockConfig
from typing import Union
from tanuki.language_models.llm_configs import DEFAULT_TEACHER_MODELS, DEFAULT_STUDENT_MODELS
from tanuki.constants import DEFAULT_DISTILLED_MODEL_NAME, OPENAI_PROVIDER, LLAMA_BEDROCK_PROVIDER, \
    DISTILLED_MODEL, TEACHER_MODEL, TITAN_BEDROCK_PROVIDER

class ModelConfigFactory:
    @staticmethod
    def create_config(input_config: Union[str, dict, BaseModelConfig], type: str) -> BaseModelConfig:
        """
        Creates a model config from a string, dict or BaseModelConfig
        Args:
            input_config: The config to create the model from
            type: The type of the input model
        Returns:
            A model config
        """
        if isinstance(input_config, BaseModelConfig):
            return input_config
        if isinstance(input_config, str):
            # This is purely for backwards compatibility as we used to save the model as a string
            if type == DISTILLED_MODEL:
                config = DEFAULT_STUDENT_MODELS[DEFAULT_DISTILLED_MODEL_NAME]
                config.model_name = input_config
                return config
            elif type == TEACHER_MODEL:
                if input_config not in DEFAULT_TEACHER_MODELS:
                    raise Exception("Error loading the teacher model, saved config model was saved a string but is not a default model")
                model = DEFAULT_TEACHER_MODELS[input_config]
                return model
        else:
            if input_config["provider"] == OPENAI_PROVIDER:
                return OpenAIConfig(**input_config)
            elif input_config["provider"] == LLAMA_BEDROCK_PROVIDER:
                return LlamaBedrockConfig(**input_config)
            elif input_config["provider"] == TITAN_BEDROCK_PROVIDER:
                return TitanBedrockConfig(**input_config)
            else:
                try:
                    return BaseModelConfig(**input_config)
                except:
                    raise Exception("Error loading the model config, saved config model was saved a dict but is not a valid model config")