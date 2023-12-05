from tanuki.language_models.llm_configs.abc_base_config import BaseModelConfig


class Claude_Config(BaseModelConfig):
    model_name: str
    provider: str = 'bedrock'
    context_length: int
    chat_template: str = "\n\nHuman: {system_prompt}\n\n {user_prompt}\n\nAssistant:\n"