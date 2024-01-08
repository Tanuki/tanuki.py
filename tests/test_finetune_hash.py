from typing import List

from tanuki.function_modeler import FunctionModeler
from tanuki.models.finetune_job import FinetuneJob
from tanuki.register import Register
from tanuki.trackers.filesystem_buffered_logger import FilesystemBufferedLogger
from tanuki.utils import encode_int, decode_int
from tanuki.language_models.llm_configs.openai_config import OpenAIConfig

def dummy_func(input: str) -> List[str]:
    """
    Below you will find an article with stocks analysis. Bring out the stock symbols of companies who are expected to go up or have positive sentiment
    """

def initiate_test(func_modeler, func_hash):
    # initiate the config
    _ = func_modeler.load_function_config(func_hash, )
    for keys, values in func_modeler.function_configs.items():
        if func_hash in keys:
            values["distilled_model"] = "test_ft_1"
            values["teacher_models"] = ["gpt-4","gpt-4-32k"] # model and its token limit]
    func_modeler._update_config_file(func_hash)

def test_encoding():
    ints = []
    characters = []
    for i in range(37):
        character = encode_int(i)
        assert character not in characters
        characters.append(character)
        integer = decode_int(character)
        assert integer not in ints
        ints.append(integer)
        assert i == integer


def test_encode_decode_hash():
    nr_of_training_runs = 5
    workspace_id = 12
    function_description = function_description = Register.load_function_description(dummy_func)
    logger = FilesystemBufferedLogger("test")
    func_modeler = FunctionModeler(logger, environment_id=workspace_id)
    finetune_hash = function_description.__hash__(purpose = "finetune") + encode_int(func_modeler.environment_id) + encode_int(nr_of_training_runs)
    finetune = FinetuneJob(id="", status="", fine_tuned_model=OpenAIConfig(model_name = f"Test_model:__{finetune_hash}:asd[]asd"
                                                                            , context_length= 1200))

    config = func_modeler._construct_config_from_finetune(finetune_hash[:-1], finetune)
    assert config.distilled_model.model_name == f"Test_model:__{finetune_hash}:asd[]asd"
    assert config.current_model_stats["trained_on_datapoints"] == 6400
    assert config.last_training_run["trained_on_datapoints"] == 6400
    assert len(config.teacher_models) == 2 and  ["gpt-4","gpt-4-32k"]
    assert isinstance(config.teacher_models[0], OpenAIConfig) and isinstance(config.teacher_models[1], OpenAIConfig)
    assert config.teacher_models[0].model_name ==  "gpt-4"
    assert config.teacher_models[1].model_name ==  "gpt-4-32k"
    assert config.nr_of_training_runs == nr_of_training_runs + 1


if __name__ == '__main__':
    #test_token_counter_finetunable()
    #test_token_counter_non_finetunable_1()
    #test_token_counter_non_finetunable_2()
    #test_error_raise()
    test_encoding()
    test_encode_decode_hash()