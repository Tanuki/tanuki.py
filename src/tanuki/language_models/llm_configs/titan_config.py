from tanuki.language_models.llm_configs.abc_base_config import BaseModelConfig
from tanuki.constants import TITAN_BEDROCK_PROVIDER

class TitanBedrockConfig(BaseModelConfig):
    """
    Config for AWS Titan Bedrock models.
    The custom prompting parameters have been left empty
    as LLM generation has not been implemented yet, only embedding
    """
    model_name: str
    provider: str = TITAN_BEDROCK_PROVIDER
    context_length: int = -1
    