# Together AI models

Tanuki now supports the finetuning for all models accessible by the Anyscale API as student models. Currently out of the box we support the following hosted models for finetuning
* Llama-2-7b-chat-hf
* Llama-2-13b-chat-hf
* Llama-2-70b-chat-hf
* Mistral-7B-Instruct-v0.1


Anyscale models use the OpenAI pacakge so there is no need to install any extra packages. To specify custom student models, a configuration flag for the student_model needs to be sent to the `@tanuki.patch` decorator like shown below at the examples section. If no student_model is specified, OpenAIs gpt-3.5-turbo-1106 is used as the default student model.

## Setup

Set your Anyscale API key using:

```
export ANYSCALE_API_KEY=...
```

## Examples

### Using the Llama-2-7b-chat-hf as the student model
```python
@tanuki.patch(student_model = "Llama-2-7b-chat-hf")
def example_function(input: TypedInput) -> TypedOutput:
    """(Optional) Include the description of how your function will be used."""

@tanuki.align
def test_example_function():

    assert example_function(example_typed_input) == example_typed_output

```

To use the other supported student models, the following text handler should be sent in to the student_model attribute at the `@tanuki.patch` decorator
* To use meta-llama/Llama-2-7b-chat-hf as a student model, student_model = "Llama-2-7b-chat-hf"
* To use meta-llama/Llama-2-13b-chat-hf as a student model, student_model = "Llama-2-13b-chat-hf"
* To use meta-llama/Llama-2-70b-chat-hf as a student model, student_model = "Llama-2-70b-chat-hf"
* To use mistralai/Mistral-7B-Instruct-v0.1 as a student model, student_model = "Mistral-7B-Instruct-v0.1"