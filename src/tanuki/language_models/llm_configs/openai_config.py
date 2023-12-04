from tanuki.language_models.llm_configs.abc_base_config import BaseModelConfig


class OpenAI_Config(BaseModelConfig):
    model_name: str
    provider: str = 'openai'
    context_length: int