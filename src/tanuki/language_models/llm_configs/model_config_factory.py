from tanuki.language_models.llm_configs.abc_base_config import BaseModelConfig
from tanuki.language_models.llm_configs.openai_config import OpenAI_Config
from typing import Union
from tanuki.language_models.llm_configs.default_models import DEFAULT_MODELS


class ModelConfigFactory:
    @staticmethod
    def create_config(input_config: Union[str, dict, BaseModelConfig], type: str) -> BaseModelConfig:
        if isinstance(input_config, BaseModelConfig):
            return input_config
        if isinstance(input_config, str):
            # This is purely for backwards compatibility as we used to save the model as a string
            if type == "distillation":
                config = DEFAULT_MODELS["gpt-3.5-finetune"]
                config.model_name = input_config
                return config
            elif type == "teacher":
                if input_config not in DEFAULT_MODELS:
                    raise Exception("Error loading the teacher model, saved config model was saved a string but is not a default model")
                model = DEFAULT_MODELS[model]
                return model
        else:
            if input_config["provider"] == "openai":
                return OpenAI_Config(**input_config)
            else:
                raise ValueError("Unsupported provider in the config")