# AWS Bedrock models

Tanuki now supports AWS Bedrock supported models. Currently out of the box we support the following AWS Bedrock hosted models (more to be added soon)
* llama-70-B (base and chat)
* llama-13-B (base and chat)
* aws_titan_embed_v1 (embeddings)

To use AWS Bedrock models, firstly the AWS extra package needs to be installed by `pip install tanuki.py[aws_bedrock]`. When the package has been installed, a configuration flag needs to be sent to the `@tanuki.patch` decorator like shown below.

 If you want to implement any additional AWS models, feel free to open an issue or implement it yourself and open a PR. To add a newly configured model, have a look at the [llm_configs](https://github.com/Tanuki/tanuki.py/tree/master/src/tanuki/language_models/llm_configs) folder to see how model configurations are addressed and to add a new model configuration, add it to the [default_models](https://github.com/Tanuki/tanuki.py/tree/master/src/tanuki/language_models/llm_configs/__init__.py). If the request template is the same as the LLama Bedrock request, then you just need to add the provider as `llama_bedrock` to the config (import LLAMA_BEDROCK_PROVIDER from the [constants file](https://github.com/Tanuki/tanuki.py/tree/master/src/tanuki/constants.py)), otherwise you need to also add a new API template (have a look at how the [llama_bedrock_api](https://github.com/Tanuki/tanuki.py/tree/master/src/tanuki/language_models/llama_bedrock_api.py) is implemented) and update the [api_manager](https://github.com/Tanuki/tanuki.py/tree/master/src/tanuki/models/api_manager.py) and the [constants file](https://github.com/Tanuki/tanuki.py/tree/master/src/tanuki/constants.py) with the new provider and api template. First try out the prompting configurations with a couple of examples to ensure the outputs are performing well!

**NB** Currently model distillation is turned off for Bedrock llama models. Model alignment, inference and saving datapoints to local datasets are still being carried out as expected.

## Examples

### Using the llama 70B chat model
```python
@tanuki.patch(teacher_models = ["llama_70b_chat_aws"])
def example_function(input: TypedInput) -> TypedOutput:
    """(Optional) Include the description of how your function will be used."""

@tanuki.align
def test_example_function():

    assert example_function(example_typed_input) == example_typed_output

```

### Using the llama 13B chat model
```python
@tanuki.patch(teacher_models = ["llama_13b_chat_aws"])
def example_function(input: TypedInput) -> TypedOutput:
    """(Optional) Include the description of how your function will be used."""

@tanuki.align
def test_example_function():

    assert example_function(example_typed_input) == example_typed_output

```

### Using the aws_titan_embed_v1 for embeddings
```python
@tanuki.patch(teacher_models = ["aws_titan_embed_v1"])
def example_function(input: TypedInput) -> Embedding[np.ndarray]:
    """(Optional) Include the description of how your function will be used."""
```