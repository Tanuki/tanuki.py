from tanuki.language_models.llm_configs.openai_config import OpenAIConfig
from tanuki.language_models.llm_configs.claude_config import ClaudeConfig
from tanuki.language_models.llm_configs.llama_config import LlamaBedrockConfig
DEFAULT_MODELS = {
            "gpt-4-1106-preview": OpenAIConfig(model_name = "gpt-4-1106-preview", context_length = 128000),
            "gpt-4": OpenAIConfig(model_name = "gpt-4", context_length = 8192),
            "gpt-4-32k": OpenAIConfig(model_name = "gpt-4-32k", context_length = 32768),
            "gpt-3.5-finetune": OpenAIConfig(model_name = "", context_length = 3000),
            "anthropic.claude-v2:1": ClaudeConfig(model_name = "anthropic.claude-v2:1", context_length = 200000),
            "llama_70b_chat_aws": LlamaBedrockConfig(model_name = "meta.llama2-70b-chat-v1", context_length = 4096),
            "ada-002": OpenAIConfig(model_name="text-embedding-ada-002", context_length=-1)
        }