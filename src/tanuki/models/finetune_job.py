from dataclasses import dataclass
from tanuki.language_models.llm_configs.abc_base_config import BaseModelConfig


@dataclass
class FinetuneJob:
    id: str
    status: str
    fine_tuned_model: BaseModelConfig