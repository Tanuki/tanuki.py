from typing import List

from monkey_patch.function_modeler import FunctionModeler
from monkey_patch.language_models.language_modeler import LanguageModel
from monkey_patch.register import Register
from monkey_patch.trackers.buffered_logger import BufferedLogger


def dummy_func(input: str) -> List[str]:
    """
    Below you will find an article with stocks analysis. Bring out the stock symbols of companies who are expected to go up or have positive sentiment
    """

def initiate_test(func_modeler, func_hash, func_description):
    # initiate the config
    _ = func_modeler._load_function_config(func_hash, func_description)
    for keys, values in func_modeler.function_configs.items():
        if func_hash in keys:
            values["distilled_model"] = "test_ft_1"
            values["teacher_models"] = ["gpt-4","gpt-4-32k"] # model and its token limit]
    func_modeler._update_config_file(func_hash)

def test_token_counter_finetunable():
    args = (0,)
    kwargs = {}
    function_description = Register.load_function_description(dummy_func)
    func_hash = function_description.__hash__()
    logger = BufferedLogger("test")
    lang_model = LanguageModel()
    func_modeler = FunctionModeler(logger)

    initiate_test(func_modeler, func_hash, function_description)

    prompt, distilled_model, suitable_for_distillation, is_distilled_model = lang_model.get_generation_case(args, kwargs, func_modeler, function_description)
    assert suitable_for_distillation
    assert is_distilled_model
    assert distilled_model == "test_ft_1"

def test_token_counter_non_finetunable_1():
    input = "(" * 6997
    args = (input,)
    kwargs = {}
    function_description = Register.load_function_description(dummy_func)
    func_hash = function_description.__hash__()
    logger = BufferedLogger("test")
    lang_model = LanguageModel()
    func_modeler = FunctionModeler(logger)
    initiate_test(func_modeler, func_hash, function_description)

    prompt, distilled_model, suitable_for_distillation, is_distilled_model = lang_model.get_generation_case(args, kwargs, func_modeler, function_description)
    assert not suitable_for_distillation
    assert not is_distilled_model
    assert distilled_model == "gpt-4"

def test_token_counter_non_finetunable_2():
    input = "(" * 7700
    args = (input,)
    kwargs = {}
    function_description = Register.load_function_description(dummy_func)
    func_hash = function_description.__hash__()
    logger = BufferedLogger("test")
    lang_model = LanguageModel()
    func_modeler = FunctionModeler(logger)
    initiate_test(func_modeler, func_hash, function_description)

    prompt, distilled_model, suitable_for_distillation, is_distilled_model = lang_model.get_generation_case(args, kwargs, func_modeler, function_description)
    assert not suitable_for_distillation
    assert not is_distilled_model
    assert distilled_model == "gpt-4-32k"

def test_error_raise():
    input = "(" * 32000
    args = (input,)
    kwargs = {}
    function_description = Register.load_function_description(dummy_func)
    func_hash = function_description.__hash__()
    logger = BufferedLogger("test")
    lang_model = LanguageModel()
    func_modeler = FunctionModeler(logger)
    initiate_test(func_modeler, func_hash, function_description)
    error = False
    try:
        prompt, distilled_model, suitable_for_distillation, is_distilled_model = lang_model.get_generation_case(args, kwargs, func_modeler, function_description)
    except ValueError:
        error = True
    assert error

if __name__ == '__main__':
    test_token_counter_finetunable()
    test_token_counter_non_finetunable_1()
    test_token_counter_non_finetunable_2()
    test_error_raise()
    