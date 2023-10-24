import os


import sys
sys.path.append(r"C:\Users\martb\Documents\Monkey_patch_new\monkey-patch\src")


from trackers.buffered_logger import BufferedLogger
from utils import get_model
import json

def initiate_test(logger, func_hash):
    # initiate the config
    _ = logger.get_models(func_hash)
    for keys, values in logger.configs.items():
        if func_hash in keys:
            values["current_model"] = "test_ft_1"
            values["teacher_models"] = [("gpt-77", 7000),("gpt-77-32k", 31000)] # model and its token limit]
    log_directory = logger._get_log_directory()
    log_file_path = os.path.join(log_directory, func_hash)
    logger._update_config_file(log_file_path)

def test_token_counter_finetunable():
    logger = BufferedLogger("test")
    content = "This is a test"
    func_hash = "finetunable_token_test"
    initiate_test(logger, func_hash)

    model, finetunable = get_model(content, logger, func_hash)
    assert finetunable
    assert model == "test_ft_1"

def test_token_counter_non_finetunable_1():
    logger = BufferedLogger("test")
    content = "(" * 6997
    func_hash = "finetunable_token_test"
    initiate_test(logger, func_hash)

    model, finetunable = get_model(content, logger, func_hash)
    assert not finetunable
    assert model == "gpt-77"

def test_token_counter_non_finetunable_2():
    logger = BufferedLogger("test")
    content = "(" * 7001
    func_hash = "finetunable_token_test"
    initiate_test(logger, func_hash)

    model, finetunable = get_model(content, logger, func_hash)
    assert not finetunable
    assert model == "gpt-77-32k"

def test_error_raise():
    logger = BufferedLogger("test")
    content = "(" * 31001
    func_hash = "finetunable_token_test"
    initiate_test(logger, func_hash)
    error = False
    try:
        model, finetunable = get_model(content, logger, func_hash)
    except ValueError:
        error = True
    assert error

if __name__ == '__main__':
    test_token_counter_finetunable()
    test_token_counter_non_finetunable_1()
    test_token_counter_non_finetunable_2()
    test_error_raise()
    