# AWS Bedrock models

Currently out of the box we support the following AWS Bedrock hosted models
* llama-70-B (base and chat)
* llama-13-B (base and chat)

To use aws bedrock models, a configuration flag needs to be sent to the `@tanuki.patch` decorator like shown below. If you want to implement any additional AWS models, feel free to open an issue or implement it yourself and open a PR. To add a newly configured model, have a look at the [llm_configs](https://github.com/Tanuki/tanuki.py/tree/master/src/tanuki/language_models/llm_configs) folder to see how model configurations are addressed and to add another model configuration, add it to the [default_models](https://github.com/Tanuki/tanuki.py/tree/master/src/tanuki/language_models/llm_configs/default_models.py). If the request template is the same as the LLama Bedrock request, then you just need to add the provider as `llama_bedrock`, otherwise you need to also add a new API template (have a look at how the [Llama_bedrock_api](https://github.com/Tanuki/tanuki.py/tree/master/src/tanuki/language_models/Llama_bedrock_api.py.py) is implemented). First try out the prompting configurations with a couple of examples to ensure the outputs are performing well!

**NB** Currently model distillation is turned off for Bedrock llama models. Model alignment, inference and saving datapoints to local datasets are still being carried out as expected.

## Examples

### Using the llama 70B chat model
```python
@tanuki.patch(teacher_models = ["llama_70b_chat_aws"])
def some_function(input: TypedInput) -> TypedOutput:
    """(Optional) Include the description of how your function will be used."""

@tanuki.align
def test_some_function(example_typed_input: TypedInput, 
                       example_typed_output: TypedOutput):

    assert some_function(example_typed_input) == example_typed_output

```

### Using the llama 13B chat model
```python
@tanuki.patch(teacher_models = ["llama_13b_chat_aws"])
def some_function(input: TypedInput) -> TypedOutput:
    """(Optional) Include the description of how your function will be used."""

@tanuki.align
def test_some_function(example_typed_input: TypedInput, 
                       example_typed_output: TypedOutput):

    assert some_function(example_typed_input) == example_typed_output

```