from typing import List
from tanuki.function_modeler import FunctionModeler
from tanuki.language_models.language_model_manager import LanguageModelManager
from tanuki.models.function_config import FunctionConfig
from tanuki.register import Register
from tanuki.trackers.filesystem_buffered_logger import FilesystemBufferedLogger
from tanuki.language_models.llm_configs.openai_config import OpenAIConfig
from tanuki.language_models.llm_configs.llama_config import LlamaBedrockConfig
from tanuki.models.finetune_job import FinetuneJob
import random
import string
from tanuki.language_models.llm_configs.abc_base_config import BaseModelConfig
from tanuki.constants import OPENAI_PROVIDER, LLAMA_BEDROCK_PROVIDER

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


def test_load_save_config():
    logger = FilesystemBufferedLogger("test")
    function_description = Register.load_function_description(dummy_func)
    func_modeler = FunctionModeler(logger)
    func_hash = function_description.__hash__()
    # initiate the config
    _ = func_modeler.load_function_config(func_hash, function_description)
    random_string_1 = ''.join(random.choices(string.ascii_uppercase + string.digits, k=12))
    func_modeler.function_configs[func_hash].distilled_model.model_name = random_string_1
    random_string_2 = ''.join(random.choices(string.ascii_uppercase + string.digits, k=12))
    func_modeler.function_configs[func_hash].teacher_models = [LlamaBedrockConfig(model_name = random_string_2, context_length = 8192),
                             OpenAIConfig(model_name = "gpt-4-32k", context_length = 32768)] # model and its token limit]
    func_modeler._update_config_file(func_hash)

    # load the config
    config = func_modeler.load_function_config(func_hash, function_description)
    assert config.distilled_model.model_name == random_string_1
    assert config.teacher_models[0].model_name == random_string_2
    assert isinstance(config.teacher_models[0], LlamaBedrockConfig)
    assert config.teacher_models[1].model_name == "gpt-4-32k"
    assert isinstance(config.teacher_models[1], OpenAIConfig)


def test_default_config():
    config = FunctionConfig()
    assert config.distilled_model.model_name == ""
    assert isinstance(config.distilled_model, OpenAIConfig)
    assert config.teacher_models[0].model_name == "gpt-4"
    assert isinstance(config.teacher_models[0], OpenAIConfig)
    assert config.teacher_models[1].model_name == "gpt-4-32k"
    assert isinstance(config.teacher_models[1], OpenAIConfig)
    assert config.current_model_stats["trained_on_datapoints"] == 0
    assert config.current_model_stats["running_faults"] == []
    assert config.last_training_run["trained_on_datapoints"] == 0
    assert config.current_training_run == {}
    assert config.nr_of_training_runs == 0

def test_update_config_full():
    config = FunctionConfig()
    json = {"distilled_model": {"model_name": "test_ft_1", "context_length": 8192, "provider": OPENAI_PROVIDER},
            "current_model_stats": {"trained_on_datapoints": 11, "running_faults": [12,1]},
            "last_training_run": {"trained_on_datapoints": 12},
            "current_training_run": {"asd": 8},
            "teacher_models": [{"model_name": "gpt-88", "context_length": 221, "provider": OPENAI_PROVIDER},
                               {"model_name": "gpt-4-3222k", "context_length": 991, "provider": OPENAI_PROVIDER}],
            "nr_of_training_runs": 15}
    config.load_from_dict(json)
    assert config.distilled_model.model_name == "test_ft_1"
    assert isinstance(config.distilled_model, OpenAIConfig)
    assert config.teacher_models[0].model_name == "gpt-88"
    assert config.teacher_models[0].context_length == 221
    assert isinstance(config.teacher_models[0], OpenAIConfig)
    assert config.teacher_models[1].model_name == "gpt-4-3222k"
    assert config.teacher_models[1].context_length == 991
    assert isinstance(config.teacher_models[1], OpenAIConfig)
    assert config.current_model_stats["trained_on_datapoints"] == 11
    assert config.current_model_stats["running_faults"] == [12,1]
    assert config.last_training_run["trained_on_datapoints"] == 12
    assert config.current_training_run["asd"] == 8
    assert config.nr_of_training_runs == 15


def test_update_config_various():
    config = FunctionConfig()
    json = {"distilled_model": {"model_name": "test_ft_1", "context_length": 8192, "provider": "new"},
            "current_model_stats": {"trained_on_datapoints": 11, "running_faults": [12,1]},
            "last_training_run": {"trained_on_datapoints": 12},
            "current_training_run": {"asd": 8},
            "teacher_models": [{"model_name": "gpt-88", "context_length": 221, "provider": "definitely_new"},
                               {"model_name": "gpt-4-3222k", "context_length": 991, "provider": LLAMA_BEDROCK_PROVIDER}],
            "nr_of_training_runs": 15}
    config.load_from_dict(json)
    assert config.distilled_model.model_name == "test_ft_1"
    assert isinstance(config.distilled_model, BaseModelConfig)
    assert config.distilled_model.model_name.context_length == 8192
    assert config.distilled_model.model_name.provider == "new"
    assert config.teacher_models[0].model_name == "gpt-88"
    assert isinstance(config.teacher_models[0], BaseModelConfig)
    assert config.teacher_models[0].context_length == 221
    assert config.teacher_models[0].provider == "definitely_new"
    assert config.teacher_models[1].model_name == "gpt-4-3222k"
    assert config.teacher_models[1].context_length == 991
    assert isinstance(config.teacher_models[1], LlamaBedrockConfig)
    assert config.teacher_models[1].provider == LLAMA_BEDROCK_PROVIDER


def test_update_config_teachers():
    config = FunctionConfig()
    json = {"distilled_model": {"model_name": "test_ft_1", "context_length": 8192, "provider": "new"},
            "current_model_stats": {"trained_on_datapoints": 11, "running_faults": [12,1]},
            "last_training_run": {"trained_on_datapoints": 12},
            "current_training_run": {"asd": 8},
            "teacher_models": [{"model_name": "gpt-88", "context_length": 221, "provider": "definitely_new"},
                               {"model_name": "gpt-4-3222k", "context_length": 991, "provider": LLAMA_BEDROCK_PROVIDER}],
            "nr_of_training_runs": 15}
    config.load_from_dict(json)
    teacher_models_1 = config.teacher_models

    json["teacher_models"] = []
    config.load_from_dict(json)
    teacher_models_2 = config.teacher_models
    assert teacher_models_1 == teacher_models_2

    json["teacher_models"] = [{"model_name": "gpt-2k", "context_length": 9912, "provider": OPENAI_PROVIDER}]
    config.load_from_dict(json)
    teacher_models_3 = config.teacher_models
    assert len(teacher_models_3) == 1
    assert teacher_models_3[0].model_name == "gpt-2k"
    assert teacher_models_3[0].context_length == 9912
    assert isinstance(teacher_models_3[0], OpenAIConfig)

    


def test_update_config_from_string():
    config = FunctionConfig()
    json = {"distilled_model": "",
            "current_model_stats": {"trained_on_datapoints": 11, "running_faults": [12,1]},
            "last_training_run": {"trained_on_datapoints": 12},
            "current_training_run": {"asd": 8},
            "teacher_models": [{"model_name": "gpt-88", "context_length": 221, "provider": "definitely_new"},
                               {"model_name": "gpt-4-3222k", "context_length": 991, "provider": LLAMA_BEDROCK_PROVIDER}],
            "nr_of_training_runs": 15}
    config.load_from_dict(json)
    assert config.distilled_model.model_name == ""
    assert isinstance(config.distilled_model, OpenAIConfig)
    assert config.distilled_model.context_length == 3000

    json["distilled_model"] = "test_ft_1"
    config.load_from_dict(json)
    assert config.distilled_model.model_name == "test_ft_1"
    assert isinstance(config.distilled_model, OpenAIConfig)
    assert config.distilled_model.context_length == 3000

    json["teacher_models"] = ["gpt-4-32k", "llama_70b_chat_aws"]
    config.load_from_dict(json)
    assert len(config.teacher_models) == 2
    assert config.teacher_models[0].model_name == "gpt-4-32k"
    assert config.teacher_models[0].provider == OPENAI_PROVIDER
    assert isinstance(config.teacher_models[0], OpenAIConfig)
    assert config.teacher_models[0].context_length == 32768
    assert config.teacher_models[1].model_name == "meta.llama2-70b-chat-v1"
    assert config.teacher_models[1].provider == LLAMA_BEDROCK_PROVIDER
    assert isinstance(config.teacher_models[1], LlamaBedrockConfig)
    assert config.teacher_models[1].context_length == 4096

    # finally try something that should fail
    json["teacher_models"] = ["something_random"]
    try:
        config.load_from_dict(json)
        assert False
    except:
        assert True


def test_update_finetune_config():
    finetune_response = FinetuneJob(id = "aas",
                                    status = "success", 
                                    fine_tuned_model=OpenAIConfig(model_name = "ayyoo-finetune", context_length = 32768))
    config = FunctionConfig()
    config.current_training_run = {"trained_on_datapoints": 770}
    config.nr_of_training_runs = 9

    # check that the config is updated if the finetune is successful
    config.update_with_finetuned_response(finetune_response)
    assert config.distilled_model.model_name == "ayyoo-finetune"
    assert isinstance(config.distilled_model, OpenAIConfig)
    assert config.distilled_model.context_length == 32768
    assert config.current_model_stats["trained_on_datapoints"] == 770
    assert config.current_model_stats["running_faults"] == []
    assert config.last_training_run["trained_on_datapoints"] == 770
    assert config.current_training_run == {}
    assert config.nr_of_training_runs == 10

    # check that the config is not updated if the finetune fails
    config.current_training_run = {"trained_on_datapoints": 1100}
    failed_finetune_response = FinetuneJob(id = "aas", status = "failed", 
                                           fine_tuned_model=BaseModelConfig(model_name = "ayyoo-finetune", provider = OPENAI_PROVIDER, context_length = 32768))
    config.update_with_finetuned_response(failed_finetune_response)
    assert config.distilled_model.model_name == "ayyoo-finetune"
    assert isinstance(config.distilled_model, OpenAIConfig)
    assert config.distilled_model.context_length == 32768
    assert config.current_model_stats["trained_on_datapoints"] == 770
    assert config.current_model_stats["running_faults"] == []
    assert config.last_training_run["trained_on_datapoints"] == 770
    assert config.current_training_run == {}
    assert config.nr_of_training_runs == 10












if __name__ == '__main__':
    #test_load_save_config()
    #test_default_config()
    #test_update_config_full()
    #test_update_config_teachers()
    #test_update_config_from_string()
    test_update_finetune_config()