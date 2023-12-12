from pydantic import BaseModel
from typing import Dict, List
from tanuki.language_models.llm_configs.abc_base_config import BaseModelConfig
from tanuki.language_models.llm_configs.default_models import DEFAULT_MODELS
from tanuki.language_models.llm_configs.model_config_factory import ModelConfigFactory
config_factory = ModelConfigFactory()


class FunctionConfig(BaseModel):
    distilled_model: BaseModelConfig = DEFAULT_MODELS["gpt-3.5-finetune"]
    current_model_stats : Dict = {
        "trained_on_datapoints": 0,
        "running_faults": []}
    last_training_run : Dict =  {"trained_on_datapoints": 0}
    current_training_run : Dict =  {}
    teacher_models : List[BaseModelConfig] =  [DEFAULT_MODELS["gpt-4"], DEFAULT_MODELS["gpt-4-32k"]]
    nr_of_training_runs : int = 0

    def load_from_dict(self, json_dict):
        self.distilled_model = config_factory.create_config(json_dict["distilled_model"], "distillation")
        self.current_model_stats = json_dict["current_model_stats"]
        self.last_training_run = json_dict["last_training_run"]
        self.current_training_run = json_dict["current_training_run"]
        self.nr_of_training_runs = json_dict["nr_of_training_runs"]
        if len(json_dict["teacher_models"]) > 0:
            self.teacher_models = [config_factory.create_config(teacher_model, "teacher") for teacher_model in json_dict["teacher_models"]]
        return self
    
    def to_dict(self):
        return self.model_dump()
    
    def update_with_finetuned_response(self, response):
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
    
