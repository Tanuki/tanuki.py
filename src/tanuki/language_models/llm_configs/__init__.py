from tanuki.language_models.llm_configs.openai_config import OpenAIConfig
from tanuki.language_models.llm_configs.claude_config import ClaudeConfig
from tanuki.language_models.llm_configs.llama_config import LlamaBedrockConfig
from tanuki.language_models.llm_configs.titan_config import TitanBedrockConfig
from tanuki.language_models.llm_configs.togetherai_config import TogetherAIConfig
DEFAULT_TEACHER_MODELS = {
            "gpt-4-1106-preview": OpenAIConfig(model_name = "gpt-4-1106-preview", context_length = 128000),
            "gpt-4": OpenAIConfig(model_name = "gpt-4", context_length = 8192),
            "gpt-4-32k": OpenAIConfig(model_name = "gpt-4-32k", context_length = 32768),
            "gpt-4-turbo": OpenAIConfig(model_name = "gpt-4-1106-preview",
                                        context_length = 128000,
                                        instructions="You are given below a function description and input data. The function description of what the function must carry out can be found in the Function section, with input and output type hints. The input data can be found in Input section. Using the function description, apply the function to the Input and return a valid output type, that is acceptable by the output_class_definition and output_class_hint.\nINCREDIBLY IMPORTANT: Only output a JSON-compatible string in the correct response format. Use the [END] tokens to specify when the output ends.",
                                        parsing_helper_tokens={"start_token": "[START]", "end_token": "[END]"}),
            "gpt-4-turbo-0125": OpenAIConfig(model_name = "gpt-4-0125-preview",
                                        context_length = 128000,
                                        instructions="You are given below a function description and input data. The function description of what the function must carry out can be found in the Function section, with input and output type hints. The input data can be found in Input section. Using the function description, apply the function to the Input and return a valid output type, that is acceptable by the output_class_definition and output_class_hint.\nINCREDIBLY IMPORTANT: Only output a JSON-compatible string in the correct response format. Use the [END] tokens to specify when the output ends.",
                                        parsing_helper_tokens={"start_token": "[START]", "end_token": "[END]"}),
            "anthropic.claude-v2:1": ClaudeConfig(model_name = "anthropic.claude-v2:1", context_length = 200000),
            "llama_70b_chat_aws": LlamaBedrockConfig(model_name = "meta.llama2-70b-chat-v1", context_length = 4096),
            "llama_13b_chat_aws": LlamaBedrockConfig(model_name = "meta.llama2-13b-chat-v1", context_length = 4096),
            "Mixtral-8x7B": TogetherAIConfig(model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1",
                                            chat_template = "{user_prompt}", # for some reason this worked better than using their own supplied chat template
                                            context_length = 32768),
            "OpenHermes-2p5-Mistral": TogetherAIConfig(model_name = "teknium/OpenHermes-2p5-Mistral-7B",
                                            context_length = 4096),
            "llama13b-togetherai": TogetherAIConfig(model_name = "togethercomputer/llama-2-13b-chat",
                                            context_length = 4096),
            "openchat-3.5": TogetherAIConfig(model_name = "openchat/openchat-3.5-1210",
                                            context_length = 8192),
            "Mixtral-8x7B-DPO": TogetherAIConfig(model_name = "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
                                            context_length = 32768),
            "Yi-34B-Chat": TogetherAIConfig(model_name = "zero-one-ai/Yi-34B-Chat",
                                            context_length = 4096),
            "Mistral-7B-Instruct-v0.2": TogetherAIConfig(model_name = "mistralai/Mistral-7B-Instruct-v0.2",
                                            context_length = 32768),
        }

DEFAULT_STUDENT_MODELS = {
            "gpt-3.5-turbo-1106": OpenAIConfig(model_name = "", context_length = 14000),
        }

DEFAULT_EMBEDDING_MODELS = {
            "ada-002": OpenAIConfig(model_name="text-embedding-ada-002", context_length=8191),
            "aws_titan_embed_v1": TitanBedrockConfig(model_name="amazon.titan-embed-text-v1", context_length=8000),
        }