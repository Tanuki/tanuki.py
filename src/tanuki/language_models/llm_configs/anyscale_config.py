from tanuki.language_models.llm_configs.abc_base_config import BaseModelConfig
from tanuki.constants import ANYSCALE_PROVIDER

class Anyscaleconfig(BaseModelConfig):
    model_name: str
    provider: str = ANYSCALE_PROVIDER
    context_length: int