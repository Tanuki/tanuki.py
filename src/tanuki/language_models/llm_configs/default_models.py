from tanuki.language_models.llm_configs.openai_config import OpenAI_Config
from tanuki.language_models.llm_configs.claude_config import Claude_Config
from tanuki.language_models.llm_configs.llama_config import Llama_Bedrock_Config
from tanuki.language_models.llm_configs.hf_model_config import HFModelConfig


DEFAULT_MODELS = {
            "gpt-4-1106-preview": OpenAI_Config(model_name = "gpt-4-1106-preview", context_length = 128000),
            "gpt-4": OpenAI_Config(model_name = "gpt-4", context_length = 8192),
            "gpt-4-32k": OpenAI_Config(model_name = "gpt-4-32k", context_length = 32768),
            "gpt-3.5-finetune": OpenAI_Config(model_name = "", context_length = 3000),
            "anthropic.claude-v2:1": Claude_Config(model_name = "anthropic.claude-v2:1", context_length = 200000),
            "llama_70b_chat_aws": Llama_Bedrock_Config(model_name = "meta.llama2-70b-chat-v1", context_length = 4096),
            "example_pythia_160" : HFModelConfig(model_name = "EleutherAI/pythia-160m",
                                            context_length = 2048,
                                            chat_template="{user_prompt}",
                                            model_kwargs={"device_map": "auto"}),
            "Zephyr_7b_8bit": HFModelConfig(model_name = "HuggingFaceH4/zephyr-7b-beta",
                                            context_length = 2048,
                                            prompt_template="{instruction_prompt}\nFunction: {f}\n{example_input}---"\
                                            "\nINSTRUCTIONS: Using the docstring and examples, return the output of the function for this input. Only return the output and nothing else.\n"\
                                            "\n{start_parsing_helper_token}Inputs:\nArgs: {args}\nKwargs: {kwargs}\nOutput:",
                                            model_kwargs={"device_map": "auto",
                                                          "load_in_8bit":True}),
            "Mistral_7b": HFModelConfig(model_name = "mistralai/Mistral-7B-Instruct-v0.1",
                                            context_length = 8000,
                                            prompt_template="{instruction_prompt}\nFunction: {f}\n{example_input}---"\
                                            "\nINSTRUCTIONS: Using the docstring and examples, return the output of the function for this input. Only return the output and nothing else.\n"\
                                            "\n{start_parsing_helper_token}Inputs:\nArgs: {args}\nKwargs: {kwargs}\nOutput:",
                                            model_kwargs={"device_map": "auto"})
        }