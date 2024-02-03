# Together AI models

Tanuki now supports all models accessible by the Together AI API. Currently out of the box we support the following hosted models (more to be added soon)
* teknium/OpenHermes-2p5-Mistral-7B
* togethercomputer/llama-2-13b-chat
* openchat/openchat-3.5-1210
* NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO
* zero-one-ai/Yi-34B-Chat
* mistralai/Mistral-7B-Instruct-v0.2
* mistralai/Mixtral-8x7B-Instruct-v0.1


To use Together AI models, firstly the Together AI extra package needs to be installed by `pip install tanuki.py[together_ai]`. When the package has been installed, a configuration flag for the teacher model needs to be sent to the `@tanuki.patch` decorator like shown below at the examples section.

**NB** Currently model distillation is turned off for Together AI models. Model alignment, inference and saving datapoints to local datasets are still being carried out as expected.

## Examples

### Using the mistralai/Mixtral-8x7B-Instruct-v0.1
```python
@tanuki.patch(teacher_models = ["Mixtral-8x7B"])
def example_function(input: TypedInput) -> TypedOutput:
    """(Optional) Include the description of how your function will be used."""

@tanuki.align
def test_example_function():

    assert example_function(example_typed_input) == example_typed_output

```

To use the other pre-implemented models, the following configuration should be sent in to the teacher_models attribute at the `@tanuki.patch` decorator
* To use teknium/OpenHermes-2p5-Mistral-7B, teacher_models = ["OpenHermes-2p5-Mistral"]
* To use togethercomputer/llama-2-13b-chat, teacher_models = ["llama13b-togetherai"]
* To use openchat/openchat-3.5-1210, teacher_models = ["openchat-3.5"]
* To use NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO, teacher_models = ["Mixtral-8x7B-DPO"]
* To use zero-one-ai/Yi-34B-Chat, teacher_models = ["Yi-34B-Chat"]
* To use mistralai/Mistral-7B-Instruct-v0.2, teacher_models = ["Mistral-7B-Instruct-v0.2"]

### Using another TogetherAI model that is not in the pre-implemented model list 
```python
from tanuki.language_models.llm_configs import TogetherAIConfig
model_config = TogetherAIConfig(model_name = "Open-Orca/Mistral-7B-OpenOrca", context_length = 8192)

@tanuki.patch(teacher_models = [model_config])
def example_function(input: TypedInput) -> TypedOutput:
    """(Optional) Include the description of how your function will be used."""

@tanuki.align
def test_example_function():

    assert example_function(example_typed_input) == example_typed_output

```