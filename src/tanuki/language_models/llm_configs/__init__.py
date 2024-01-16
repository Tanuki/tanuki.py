from tanuki.language_models.llm_configs.openai_config import OpenAIConfig
from tanuki.language_models.llm_configs.claude_config import ClaudeConfig
from tanuki.language_models.llm_configs.llama_config import LlamaBedrockConfig
from tanuki.language_models.llm_configs.titan_config import TitanBedrockConfig
from tanuki.language_models.llm_configs.hf_config import HFConfig
DEFAULT_GENERATIVE_MODELS = {
            "gpt-4-1106-preview": OpenAIConfig(model_name = "gpt-4-1106-preview", context_length = 128000),
            "gpt-4": OpenAIConfig(model_name = "gpt-4", context_length = 8192),
            "gpt-4-32k": OpenAIConfig(model_name = "gpt-4-32k", context_length = 32768),
            "gpt-3.5-finetune": OpenAIConfig(model_name = "", context_length = 3000),
            "anthropic.claude-v2:1": ClaudeConfig(model_name = "anthropic.claude-v2:1", context_length = 200000),
            "llama_70b_chat_aws": LlamaBedrockConfig(model_name = "meta.llama2-70b-chat-v1", context_length = 4096),
            "llama_13b_chat_aws": LlamaBedrockConfig(model_name = "meta.llama2-13b-chat-v1", context_length = 4096),
            "example_pythia_160" : HFConfig(model_name = "EleutherAI/pythia-160m",
                                            context_length = 2048,
                                            chat_template="{user_prompt}",
                                            model_kwargs={"device_map": "auto"}),
            "Zephyr_7b_8bit": HFConfig(model_name = "HuggingFaceH4/zephyr-7b-beta",
                                            context_length = 2048,
                                            prompt_template="{instruction_prompt}\nFunction: {f}\n{example_input}---"\
                                            "\nINSTRUCTIONS: Using the docstring and examples, return the output of the function for this input. Only return the output and nothing else.\n"\
                                            "\n{start_parsing_helper_token}Inputs:\nArgs: {args}\nKwargs: {kwargs}\nOutput:",
                                            model_kwargs={"device_map": "auto",
                                                          "load_in_8bit":True}),
            "Mistral_7b": HFConfig(model_name = "mistralai/Mistral-7B-Instruct-v0.1",
                                            context_length = 8000,
                                            prompt_template="{instruction_prompt}\nFunction: {f}\n{example_input}---"\
                                            "\nINSTRUCTIONS: Using the docstring and examples, return the output of the function for this input. Only return the output and nothing else.\n"\
                                            "\n{start_parsing_helper_token}Inputs:\nArgs: {args}\nKwargs: {kwargs}\nOutput:",
                                            model_kwargs={"device_map": "auto"})
        }


DEFAULT_EMBEDDING_MODELS = {
            "ada-002": OpenAIConfig(model_name="text-embedding-ada-002", context_length=8191),
            "aws_titan_embed_v1": TitanBedrockConfig(model_name="amazon.titan-embed-text-v1", context_length=8000),
        }