# Function configurability

The following optional arguments are currently supported for funcion configurability:
* environment_id (int, default = 0): The environment id. Used for fetching correct finetuned models.
* ignore_finetune_fetching (boolean, default = False): Whether to ignore fetching finetuned models. If set to True, during the first call Open-Ai will not be queried for finetuned models, which reduces initial startup latency.
* ignore_finetuning (boolean, default = False): Whether to ignore finetuning the models altogether. If set to True the teacher model will always be used. The data is still saved however if in future the data would need to be used for finetuning.
* ignore_data_storage (boolean, default = False): Whether to ignore storing the data. If set to True, the data will not be stored in the finetune dataset and the align statements will not be saved (align statements are still used for aligning outputs so model performance is not affected). This improves latency as communications with data storage is minimised.
* configuring teacher models (list): Now supporting multiple teacher models to carry out inference, please see respective docs to how to use them ([llama_bedrock](https://github.com/Tanuki/tanuki.py/tree/master/docs/aws_bedrock.md)). If custom teacher models are not specified, OpenAI teacher models (gpt-4, gpt-32k) are used as default.

**NB** - Configurations can be sent in only to `@tanuki.patch` decorator using keyword arguments. If you have any additional configurability needs, feel free to open an issue or implement it yourself and open a PR

## Examples

### Default function
```python
@tanuki.patch
def some_function(input: TypedInput) -> TypedOutput:
    """(Optional) Include the description of how your function will be used."""

@tanuki.align
def test_some_function(example_typed_input: TypedInput, 
                       example_typed_output: TypedOutput):

    assert some_function(example_typed_input) == example_typed_output

```
### Function with configurations (fastest inferece latency)
```python
@tanuki.patch(environment_id = 1,
              ignore_finetune_fetching = True,
              ignore_finetuning = True,
              ignore_data_storage = True)
def some_function(input: TypedInput) -> TypedOutput:
    """(Optional) Include the description of how your function will be used."""

@tanuki.align
def test_some_function(example_typed_input: TypedInput, 
                       example_typed_output: TypedOutput):

    assert some_function(example_typed_input) == example_typed_output

```