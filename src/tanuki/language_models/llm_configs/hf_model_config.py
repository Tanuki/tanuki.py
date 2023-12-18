from tanuki.language_models.llm_configs.abc_base_config import BaseModelConfig


class HFModelConfig(BaseModelConfig):
    model_name: str
    provider: str = "hf_transformers"
    context_length: int
    generator_code: dict = {}