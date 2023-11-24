from monkey_patch.exception import MonkeyPatchException
from monkey_patch.language_models.bedrock_api import Bedrock_API
from monkey_patch.language_models.openai_api import Openai_API


class ApiModelFactory:
    @classmethod
    def get_model(cls, api_model_name: str):
        if api_model_name == 'openai':
            return {"openai": Openai_API()}
        elif api_model_name == 'bedrock':
            return {"bedrock": Bedrock_API()}
        else:
            MonkeyPatchException(f"not support {api_model_name}")

    @classmethod
    def get_all_model_info(cls, api_model_name: str, generation_length):
        if api_model_name == 'bedrock':
            return {
                "anthropic.claude-instant-v1": {
                    "token_limit": 100000 - generation_length,
                    "type": "bedrock"
                },
                "anthropic.claude-v2": {
                    "token_limit": 100000 - generation_length,
                    "type": "bedrock"
                }
            }  # models and token counts
        elif api_model_name == 'openai':
            return {
                "gpt-4": {
                    "token_limit": 8192 - generation_length,
                    "type": "openai"
                },
                "gpt-4-32k": {
                    "token_limit": 32768 - generation_length,
                    "type": "openai"
                }
            }  # models and token counts
        else:
            MonkeyPatchException(f"not support {api_model_name}")

    @classmethod
    def get_teacher_model(cls, api_model_name: str):
        if api_model_name == 'bedrock':
            return [
                'anthropic.claude-v2'
            ]
        elif api_model_name == 'openai':
            return [
                "gpt-4",
                "gpt-4-32k"
            ]
        else:
            MonkeyPatchException(f"not support {api_model_name}")

    @classmethod
    def is_finetune_support(cls, api_model_name: str):
        if api_model_name == 'bedrock':
            lo
            return False
        elif api_model_name == 'openai':
            return True
        else:
            MonkeyPatchException(f"not support {api_model_name}")

    @classmethod
    def get_distilled_model(cls, api_model_name: str):
        if api_model_name == 'bedrock':
            return 'anthropic.claude-instant-v1'

        elif api_model_name == 'openai':
            return ""
        else:
            MonkeyPatchException(f"not support {api_model_name}")
