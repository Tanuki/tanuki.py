from dataclasses import dataclass

@dataclass
class FinetuneJob:
    id: str
    status: str
    fine_tuned_model: str