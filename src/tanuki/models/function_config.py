from pydantic import BaseModel
from typing import Dict, List
from tanuki.language_models.llm_configs.abc_base_config import BaseModelConfig
from tanuki.language_models.llm_configs import DEFAULT_TEACHER_MODELS, DEFAULT_STUDENT_MODELS
from tanuki.constants import DEFAULT_TEACHER_MODEL_NAMES, DEFAULT_DISTILLED_MODEL_NAME, \
                            DISTILLED_MODEL, TEACHER_MODEL
from tanuki.language_models.llm_configs.model_config_factory import ModelConfigFactory
config_factory = ModelConfigFactory()


class FunctionConfig(BaseModel):
    """
    The function config to execute the inference for the function and distillation.

    Parameters
    ----------
    distilled_model : BaseModelConfig -- the distilled model config
    current_model_stats : Dict -- the current model stats
    last_training_run : Dict -- the last training run
    current_training_run : Dict -- the current training run
    teacher_models : List[BaseModelConfig] -- the teacher models
    nr_of_training_runs : int -- the number of training runs
    
    """
    distilled_model: BaseModelConfig = DEFAULT_STUDENT_MODELS[DEFAULT_DISTILLED_MODEL_NAME]
    current_model_stats : Dict = {
        "trained_on_datapoints": 0,
        "running_faults": []}
    last_training_run : Dict =  {"trained_on_datapoints": 0}
    current_training_run : Dict =  {}
    teacher_models : List[BaseModelConfig] =  [DEFAULT_TEACHER_MODELS[teacher_model_name] for teacher_model_name in DEFAULT_TEACHER_MODEL_NAMES]
    nr_of_training_runs : int = 0

    def load_from_dict(self, json_dict):
        """
        Load the function config from a dict
        Args:
            json_dict: The dict to load the function config from
        Returns:
            The function config
        """
        self.distilled_model = config_factory.create_config(json_dict["distilled_model"], DISTILLED_MODEL)
        self.current_model_stats = json_dict["current_model_stats"]
        self.last_training_run = json_dict["last_training_run"]
        self.current_training_run = json_dict["current_training_run"]
        self.nr_of_training_runs = json_dict["nr_of_training_runs"]
        if "teacher_models" in json_dict and len(json_dict["teacher_models"]) > 0:
            self.teacher_models = [config_factory.create_config(teacher_model, TEACHER_MODEL) for teacher_model in json_dict["teacher_models"]]
        return self
    
    def to_dict(self):
        """
        Convert the function config to a dict
        Returns:
            The dict
        """
        try:
            config_dictionary = self.model_dump()
        except AttributeError as e:
            config_dictionary = self.dict()

        return config_dictionary
    def update_with_finetuned_response(self, response):
        """
        Update the function config with the finetuned response
        Args:
            response: The finetuned response
        """
        if response.status == "failed":
            self.current_training_run = {}
        else:
            self.distilled_model = response.fine_tuned_model
            self.last_training_run = self.current_training_run
            self.current_model_stats = {
                "trained_on_datapoints": self.current_training_run[
                    "trained_on_datapoints"],
                "running_faults": []}
            self.nr_of_training_runs += 1
            self.current_training_run = {}
    
