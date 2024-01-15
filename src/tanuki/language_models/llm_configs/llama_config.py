from tanuki.language_models.llm_configs.abc_base_config import BaseModelConfig
from tanuki.constants import LLAMA_BEDROCK_PROVIDER

class LlamaBedrockConfig(BaseModelConfig):
    """
    Config for Llama Bedrock models.
    The custom prompting parameters have been prefilled
    """
    model_name: str
    provider: str = LLAMA_BEDROCK_PROVIDER
    context_length: int
    chat_template: str = "[INST]{system_message}[/INST]\n{user_prompt}"
    instructions :str  = "You are given below a function description and input data. The function description of what the function must carry out can be found in the Function section, with input and output type hints. The input data can be found in Input section. Using the function description, apply the function to the Input and return a valid output type, that is acceptable by the output_class_definition and output_class_hint.\nINCREDIBLY IMPORTANT: Only output a JSON-compatible string in the correct response format. Use the [END] tokens to specify when the output ends."
    parsing_helper_tokens: dict = {"start_token": "[START]\n", "end_token": "\n[END]"}