from tanuki.language_models.llm_configs.abc_base_config import BaseModelConfig
from tanuki.constants import HF_PROVIDER

class HFConfig(BaseModelConfig):
    model_name: str
    provider: str = HF_PROVIDER
    context_length: int
    generator_code: dict = {}