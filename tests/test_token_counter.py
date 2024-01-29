from typing import List

from tanuki.function_modeler import FunctionModeler
from tanuki.language_models.language_model_manager import LanguageModelManager
from tanuki.register import Register
from tanuki.trackers.filesystem_buffered_logger import FilesystemBufferedLogger
from tanuki.language_models.llm_configs.openai_config import OpenAIConfig

def dummy_func(input: str) -> List[str]:
    """
    Below you will find an article with stocks analysis. Bring out the stock symbols of companies who are expected to go up or have positive sentiment
    """

def initiate_test(func_modeler, function_description):
    func_hash = function_description.__hash__()
    # initiate the config
    _ = func_modeler.load_function_config(func_hash, function_description)
    for keys, values in func_modeler.function_configs.items():
        if func_hash in keys:
            values.distilled_model.model_name = "test_ft_1"
            values.teacher_models = [OpenAIConfig(model_name = "gpt-4", context_length = 8192),
                                     OpenAIConfig(model_name = "gpt-4-32k", context_length = 32768)] # model and its token limit]
    func_modeler._update_config_file(func_hash)

def test_token_counter_finetunable():
    args = (0,)
    kwargs = {}
    function_description = Register.load_function_description(dummy_func)
    logger = FilesystemBufferedLogger("test")

    func_modeler = FunctionModeler(logger)
    lang_model = LanguageModelManager(func_modeler)

    initiate_test(func_modeler, function_description)

    prompt, distilled_model, suitable_for_distillation, is_distilled_model = lang_model.get_generation_case(args, 
                                                                                                            kwargs, 
                                                                                                            function_description,
                                                                                                            {},
                                                                                                            "")
    assert suitable_for_distillation
    assert is_distilled_model
    assert distilled_model.model_name == "test_ft_1"

def test_token_counter_non_finetunable_1():
    input = "(" * 6997
    args = (input,)
    kwargs = {}
    function_description = Register.load_function_description(dummy_func)
    logger = FilesystemBufferedLogger("test")
    func_modeler = FunctionModeler(logger)
    lang_model = LanguageModelManager(func_modeler)
    initiate_test(func_modeler, function_description)

    prompt, distilled_model, suitable_for_distillation, is_distilled_model = lang_model.get_generation_case(args, 
                                                                                                            kwargs, 
                                                                                                            function_description,
                                                                                                            {},
                                                                                                            "")
    assert not suitable_for_distillation
    assert not is_distilled_model
    assert distilled_model.model_name == "gpt-4"

def test_token_counter_non_finetunable_2():
    input = "(" * 7700
    args = (input,)
    kwargs = {}
    function_description = Register.load_function_description(dummy_func)
    logger = FilesystemBufferedLogger("test")
    func_modeler = FunctionModeler(logger)
    lang_model = LanguageModelManager(func_modeler)
    initiate_test(func_modeler, function_description)

    prompt, distilled_model, suitable_for_distillation, is_distilled_model = lang_model.get_generation_case(args, 
                                                                                                            kwargs, 
                                                                                                            function_description,
                                                                                                            {},
                                                                                                            "")
    assert not suitable_for_distillation
    assert not is_distilled_model
    assert distilled_model.model_name == "gpt-4-32k"

def test_error_raise():
    input = "(" * 32000
    args = (input,)
    kwargs = {}
    function_description = Register.load_function_description(dummy_func)
    #func_hash = function_description.__hash__()
    logger = FilesystemBufferedLogger("test")
    func_modeler = FunctionModeler(logger)
    lang_model = LanguageModelManager(func_modeler)
    initiate_test(func_modeler, function_description)
    error = False
    try:
        prompt, distilled_model, suitable_for_distillation, is_distilled_model = lang_model.get_generation_case(args, 
                                                                                                            kwargs, 
                                                                                                            function_description,
                                                                                                            {},
                                                                                                            "")
    except ValueError:
        error = True
    assert error

if __name__ == '__main__':
    #test_token_counter_finetunable()
    #test_token_counter_non_finetunable_1()
    #test_token_counter_non_finetunable_2()
    test_error_raise()