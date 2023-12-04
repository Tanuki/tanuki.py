Class: LLM_Finetune_API
  finetune(self, **kwargs) -> tanuki.models.finetune_job.FinetuneJob
  get_finetuned(self, job_id: str, **kwargs) -> tanuki.models.finetune_job.FinetuneJob
  list_finetuned(self, limit=100, **kwargs) -> List[tanuki.models.finetune_job.FinetuneJob]

Class: LLM_API
  generate(self, model, system_message, prompt, **kwargs)

Class: LLM_Finetune_API
  finetune(self, **kwargs) -> tanuki.models.finetune_job.FinetuneJob
  get_finetuned(self, job_id: str, **kwargs) -> tanuki.models.finetune_job.FinetuneJob
  list_finetuned(self, limit=100, **kwargs) -> List[tanuki.models.finetune_job.FinetuneJob]

Class: LLM_API
  generate(self, model, system_message, prompt, **kwargs)

Class: Embedding_API
  embed(self, texts: List[str], model: str = None, **kwargs) -> List[tanuki.models.embedding.Embedding]

Class: LLM_Finetune_API
  Class Docstring: Helper class that provides a standard way to create an ABC using
inheritance.
  finetune(self, **kwargs) -> tanuki.models.finetune_job.FinetuneJob
Docstring: Creates a fine-tuning run
Args:
    **kwargs: 

Returns:
  get_finetuned(self, job_id: str, **kwargs) -> tanuki.models.finetune_job.FinetuneJob
Docstring: Gets a fine-tuning run by id
  list_finetuned(self, limit=100, **kwargs) -> List[tanuki.models.finetune_job.FinetuneJob]
Docstring: Gets the last N fine-tuning runs

Class: Embedding_API
  Class Docstring: Helper class that provides a standard way to create an ABC using
inheritance.
  embed(self, texts: List[str], model: str = None, **kwargs) -> List[tanuki.models.embedding.Embedding]
Docstring: The main embedding function, given the model and prompt, return a vector representation

Class: Embedding_API
  Class Docstring: Helper class that provides a standard way to create an ABC using
inheritance.
  embed(self, texts: List[str], model: str = None, **kwargs) -> List[tanuki.models.embedding.Embedding]
Docstring: The main embedding function, given the model and prompt, return a vector representation

Class: LLM_API
  Class Docstring: Helper class that provides a standard way to create an ABC using
inheritance.
  generate(self, model, system_message, prompt, **kwargs)
Docstring: The main generation function, given the args, kwargs, function_modeler, function description and model type, generate a response and check if the datapoint can be saved to the finetune dataset

Class: LLM_Finetune_API
  Class Docstring: Helper class that provides a standard way to create an ABC using
inheritance.
  finetune(self, **kwargs) -> tanuki.models.finetune_job.FinetuneJob
Docstring: Creates a fine-tuning run
Args:
    **kwargs: 

Returns:
  get_finetuned(self, job_id: str, **kwargs) -> tanuki.models.finetune_job.FinetuneJob
Docstring: Gets a fine-tuning run by id
  list_finetuned(self, limit=100, **kwargs) -> List[tanuki.models.finetune_job.FinetuneJob]
Docstring: Gets the last N fine-tuning runs

Class: LLM_API
  Class Docstring: Helper class that provides a standard way to create an ABC using
inheritance.
  generate(self, model, system_message, prompt, **kwargs)
Docstring: The main generation function, given the args, kwargs, function_modeler, function description and model type, generate a response and check if the datapoint can be saved to the finetune dataset

Class: LLM_Finetune_API
  Class Docstring: Helper class that provides a standard way to create an ABC using
inheritance.
  finetune(self, **kwargs) -> tanuki.models.finetune_job.FinetuneJob
Docstring: Creates a fine-tuning run
Args:
    **kwargs: 

Returns:
  get_finetuned(self, job_id: str, **kwargs) -> tanuki.models.finetune_job.FinetuneJob
Docstring: Gets a fine-tuning run by id
  list_finetuned(self, limit=100, **kwargs) -> List[tanuki.models.finetune_job.FinetuneJob]
Docstring: Gets the last N fine-tuning runs

Class: LLM_API
  generate(self, model, system_message, prompt, **kwargs) ->
    """
    The main generation function, given the args, kwargs, function_modeler, function description and model type,
    generate a response and check if the datapoint can be saved to the finetune dataset
    """

Class: Embedding_API
  Class Docstring: Helper class that provides a standard way to create an ABC using
inheritance.
  embed(self, texts: List[str], model: str = None, **kwargs) -> List[tanuki.models.embedding.Embedding]
Docstring: The main embedding function, given the model and prompt, return a vector representation

