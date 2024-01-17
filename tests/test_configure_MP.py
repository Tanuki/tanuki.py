from typing import List
from tanuki.register import Register

import os
from typing import Optional, Literal, List
import openai
from dotenv import load_dotenv
import tanuki
from tanuki.language_models.llm_configs.openai_config import OpenAIConfig
from tanuki.language_models.llm_configs.llama_config import LlamaBedrockConfig
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


@tanuki.patch
def classify_sentiment_2(input: str, input_2: str) -> Optional[Literal['Good', 'Bad']]:
    """
    Determine if the inputs are positive or negative sentiment, or None
    """


@tanuki.patch(environment_id = 12, ignore_finetune_fetching=True, ignore_finetuning=True, ignore_data_storage=True)
def classify_sentiment(input: str) -> Optional[Literal['Good', 'Bad']]:
    """
    Determine if the input is positive or negative sentiment
    """

@tanuki.align
def align_classify_sentiment():
    """We can test the function as normal using Pytest or Unittest"""

    i_love_you = "I love you"
    assert classify_sentiment_2(i_love_you, "I love woo") == 'Good'
    assert classify_sentiment_2("I hate you", "You're discusting") == 'Bad'
    assert classify_sentiment_2("Today is wednesday", "The dogs are running outside") == None


    assert classify_sentiment("I love you") == 'Good'
    assert classify_sentiment("I hate you") == 'Bad'
    assert classify_sentiment("Wednesdays are in the middle of the week") == None

def test_classify_sentiment():
    align_classify_sentiment()
    bad_input = "I find you awful"
    good_input = "I really really like you"
    good_input_2 = "I adore you"
    assert classify_sentiment("I like you") == 'Good'
    assert classify_sentiment(bad_input) == 'Bad'
    assert classify_sentiment("I am neutral") == None

    assert classify_sentiment_2(good_input, good_input_2) == 'Good'
    assert classify_sentiment_2("I do not like you you", bad_input) == 'Bad'
    assert classify_sentiment_2("I am neutral", "I am neutral too") == None

@tanuki.patch(teacher_models=[OpenAIConfig(model_name = "gpt-4441", context_length = 112)])
def func_full_openai(input: str) -> Optional[Literal['Good', 'Bad']]:
    """
    Determine if the input is positive or negative sentiment
    """

@tanuki.patch(teacher_models=["gpt-4", "gpt-4-32k"])
def func_default_openai(input: str) -> Optional[Literal['Good', 'Bad']]:
    """
    Determine if the input is positive or negative sentiment
    """


@tanuki.patch
def func_default(input: str) -> Optional[Literal['Good', 'Bad']]:
    """
    Determine if the input is positive or negative sentiment
    """

@tanuki.patch(teacher_models=[LlamaBedrockConfig(model_name = "llama778", context_length = 1)])
def func_full_llama_bedrock(input: str) -> Optional[Literal['Good', 'Bad']]:
    """
    Determine if the input is positive or negative sentiment
    """

@tanuki.patch(teacher_models=["llama_70b_chat_aws"])
def func_default_llama_bedrock(input: str) -> Optional[Literal['Good', 'Bad']]:
    """
    Determine if the input is positive or negative sentiment
    """

@tanuki.patch(teacher_models=["llama_70b_chat_aws",
                                OpenAIConfig(model_name = "gpt-4441", context_length = 8192)])
def func_mixed(input: str) -> Optional[Literal['Good', 'Bad']]:
    """
    Determine if the input is positive or negative sentiment
    """


@tanuki.patch(generation_params={"max_new_tokens": 2, "smth": 1})
def func_gen_params_small() -> str:
    """
    Just write me an essay please
    """
@tanuki.patch(teacher_models=["llama_70b_chat_aws"], generation_params={"max_new_tokens": 15, "temperature": 1})
def func_gen_params_mid() -> str:
    """
    Just write me an essay please
    """

def test_configurability():
    classify_sent_description = Register.load_function_description(classify_sentiment)
    classify_sentiment_2_description = Register.load_function_description(classify_sentiment_2)
    sent_func_hash = classify_sent_description.__hash__()
    sent_func_2_hash = classify_sentiment_2_description.__hash__()

    func_modeler = tanuki.function_modeler
    assert func_modeler.environment_id == 12
    assert sent_func_hash in func_modeler.check_finetune_blacklist
    assert sent_func_2_hash not in func_modeler.check_finetune_blacklist
    assert sent_func_hash in func_modeler.execute_finetune_blacklist
    assert sent_func_2_hash not in func_modeler.execute_finetune_blacklist
    assert sent_func_hash in func_modeler.store_data_blacklist
    assert sent_func_2_hash not in func_modeler.store_data_blacklist
    assert len(func_modeler.teacher_models_override[sent_func_hash]) == 0

def test_teacher_model_override():
    """
    Testing all the teacher model overrides, i.e sending in string and modelconfigs and seeing if they are correctly loaded
    Also seeing if with non-openai models the finetuning is correctly disabled
    
    """
    classify_sent_description = Register.load_function_description(classify_sentiment)
    full_openai_description = Register.load_function_description(func_full_openai)
    default_openai_description = Register.load_function_description(func_default_openai)
    full_llama_bedrock_description = Register.load_function_description(func_full_llama_bedrock)
    default_llama_bedrock_description = Register.load_function_description(func_default_llama_bedrock)
    mixed_description = Register.load_function_description(func_mixed)
    sent_func_hash = classify_sent_description.__hash__()
    full_openai_hash = full_openai_description.__hash__()
    default_openai_hash = default_openai_description.__hash__()
    full_llama_bedrock_hash = full_llama_bedrock_description.__hash__()
    mixed_hash = mixed_description.__hash__()
    default_llama_bedrock_hash = default_llama_bedrock_description.__hash__()
    func_modeler = tanuki.function_modeler
    
    assert sent_func_hash not in func_modeler.teacher_models_override
    assert len(func_modeler.teacher_models_override[full_openai_hash]) == 1
    assert len(func_modeler.teacher_models_override[default_openai_hash]) == 2
    assert len(func_modeler.teacher_models_override[full_llama_bedrock_hash]) == 1
    assert len(func_modeler.teacher_models_override[default_llama_bedrock_hash]) == 1
    assert len(func_modeler.teacher_models_override[mixed_hash]) == 2
    assert func_modeler.teacher_models_override[full_openai_hash][0].model_name == "gpt-4441"
    assert func_modeler.teacher_models_override[full_openai_hash][0].context_length == 112
    assert func_modeler.teacher_models_override[default_openai_hash][0].model_name == "gpt-4"
    assert func_modeler.teacher_models_override[default_openai_hash][0].context_length == 8192
    assert func_modeler.teacher_models_override[default_openai_hash][1].model_name == "gpt-4-32k"
    assert func_modeler.teacher_models_override[default_openai_hash][1].context_length == 32768
    assert func_modeler.teacher_models_override[full_llama_bedrock_hash][0].model_name == "llama778"
    assert func_modeler.teacher_models_override[full_llama_bedrock_hash][0].context_length == 1
    assert func_modeler.teacher_models_override[default_llama_bedrock_hash][0].model_name == "meta.llama2-70b-chat-v1"
    assert func_modeler.teacher_models_override[default_llama_bedrock_hash][0].context_length == 4096
    assert func_modeler.teacher_models_override[mixed_hash][0].model_name == "meta.llama2-70b-chat-v1"
    assert func_modeler.teacher_models_override[mixed_hash][0].context_length == 4096
    assert func_modeler.teacher_models_override[mixed_hash][1].model_name == "gpt-4441"
    assert func_modeler.teacher_models_override[mixed_hash][1].context_length == 8192

    assert sent_func_hash in func_modeler.check_finetune_blacklist
    assert full_openai_hash not in func_modeler.check_finetune_blacklist
    assert default_openai_hash not in func_modeler.check_finetune_blacklist
    assert full_llama_bedrock_hash in func_modeler.check_finetune_blacklist
    assert default_llama_bedrock_hash in func_modeler.check_finetune_blacklist
    assert mixed_hash in func_modeler.check_finetune_blacklist


def test_gen_params():
    small_output = func_gen_params_small() # this should also give a warning
    assert len(small_output.split()) <3
    mid_output = func_gen_params_mid()
    assert len(mid_output.split()) < 16

def test_teacher_model_override_error():
    """
    Testing all the teacher model overrides, i.e sending in string and modelconfigs and seeing if they are correctly loaded
    Also seeing if with non-openai models the finetuning is correctly disabled
    
    """
    func_modeler = tanuki.function_modeler
    try:

        func_modeler._configure_teacher_models(["something_random"], "also_random")
        assert False
    except:
        assert True


def test_finetuning():
    func_default_description = Register.load_function_description(func_default)
    func_default_openai_description = Register.load_function_description(func_default_openai)
    func_full_llama_bedrock_description = Register.load_function_description(func_full_llama_bedrock)
    func_mixed_description = Register.load_function_description(func_mixed)
    func_default_hash = func_default_description.__hash__()
    func_default_openai_hash = func_default_openai_description.__hash__()
    func_full_llama_bedrock_hash = func_full_llama_bedrock_description.__hash__()
    func_mixed_hash = func_mixed_description.__hash__()

    func_modeler = tanuki.function_modeler
    assert func_default_hash not in func_modeler.check_finetune_blacklist
    assert func_default_hash not in func_modeler.execute_finetune_blacklist
    assert func_default_openai_hash not in func_modeler.check_finetune_blacklist
    assert func_default_openai_hash not in func_modeler.execute_finetune_blacklist
    assert func_full_llama_bedrock_hash in func_modeler.check_finetune_blacklist
    assert func_full_llama_bedrock_hash in func_modeler.execute_finetune_blacklist
    assert func_mixed_hash in func_modeler.check_finetune_blacklist
    assert func_mixed_hash in func_modeler.execute_finetune_blacklist