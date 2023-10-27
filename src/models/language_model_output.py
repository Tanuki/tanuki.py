from dataclasses import dataclass


@dataclass()
class LanguageModelOutput:
    generated_response: str
    suitable_for_finetuning: bool
    distilled_model: bool
