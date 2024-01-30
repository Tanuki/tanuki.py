from tanuki.language_models.llm_configs.abc_base_config import BaseModelConfig
from tanuki.constants import TOGETHER_AI_PROVIDER

class TogetherAIConfig(BaseModelConfig):
    model_name: str
    provider: str = TOGETHER_AI_PROVIDER
    context_length: int
    instructions : str = "You are given below a function description and input data. The function description of what the function must carry out can be found in the Function section, with input and output type hints. The input data can be found in Input section. Using the function description, apply the function to the Input and return a valid output type, that is acceptable by the output_class_definition and output_class_hint.\nINCREDIBLY IMPORTANT: Only output a JSON-compatible string in the correct response format. The outputs will be between |START| and |END| tokens, the |START| token will be given in the prompt, use the |END| token to specify when the output ends. Only return the output to this input."
    parsing_helper_tokens : dict = {"start_token": "|START|", "end_token": "|END|"}