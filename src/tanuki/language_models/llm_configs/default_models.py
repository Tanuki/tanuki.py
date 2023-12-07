from tanuki.language_models.llm_configs.openai_config import OpenAI_Config
from tanuki.language_models.llm_configs.claude_config import Claude_Config
from tanuki.language_models.llm_configs.llama_config import Llama_Bedrock_Config
DEFAULT_MODELS = {
            "gpt-4-1106-preview": OpenAI_Config(model_name = "gpt-4-1106-preview", context_length = 128000),
            "gpt-4": OpenAI_Config(model_name = "gpt-4", context_length = 8192),
            "gpt-4-32k": OpenAI_Config(model_name = "gpt-4-32k", context_length = 32768),
            "gpt-3.5-finetune": OpenAI_Config(model_name = "", context_length = 3000),
            "anthropic.claude-v2:1": Claude_Config(model_name = "anthropic.claude-v2:1", context_length = 200000),
            "llama_70b_chat_aws": Llama_Bedrock_Config(model_name = "meta.llama2-70b-chat-v1", context_length = 4096),
        }