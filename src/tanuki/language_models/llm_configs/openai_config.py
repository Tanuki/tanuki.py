from tanuki.language_models.llm_configs.abc_base_config import BaseModelConfig
from tanuki.constants import OPENAI_PROVIDER

class OpenAIConfig(BaseModelConfig):
    model_name: str
    provider: str = OPENAI_PROVIDER
    context_length: int