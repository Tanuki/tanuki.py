## Quickstart

To get started:
1. Create a python function stub decorated with `@tanuki.patch` including type hints and a docstring.
2. Add the API and authentication keys. To set the API key for the default OpenAI models 
```
export OPENAI_API_KEY=sk-...
```

3. (Optional) Create another function decorated with `@tanuki.align` containing normal `assert` statements declaring the expected behaviour of your patched function with different inputs. When executing the function, the function annotated with `align` must also be called
4. (Optional) Configure the model you want to use the function for. By default GPT-4 is used but if you want to use any other models supported in our stack, then configure them in the  `@tanuki.patch` operator. You can find out exactly how to configure OpenAI, Amazon Bedrock and Together AI models in the [models](placeholder_url) section.
The patched function can now be called as normal in the rest of your code. 


Here is what the whole script for a a simple classification function would look like:

```python
import tanuki
from typing import Optional
@tanuki.patch
def classify_sentiment(msg: str) -> Optional[Literal['Good', 'Bad']]:
    """Classifies a message from the user into Good, Bad or None."""

@tanuki.align
def align_classify_sentiment():
    assert classify_sentiment("I love you") == 'Good'
    assert classify_sentiment("I hate you") == 'Bad'
    assert not classify_sentiment("People from Phoenix are called Phoenicians")

if __name__ == "__main__":
    align_classify_sentiment()
    print(classify_sentiment("I like you")) # Good
    print(classify_sentiment("Apples might be red")) # None
```


## Creating typed functions

A core concept of Tanuki is the support for typed parameters and outputs. Supporting typed outputs of patched functions allows you to declare *rules about what kind of data the patched function is allowed to pass back* for use in the rest of your program. This will guard against the verbose or inconsistent outputs of the LLMs that are trained to be as “helpful as possible”.

To set the allowed outputs, you have to specify the output typehint as is normally done in functions. You can use base classes such as ints, strings, lists or Literals or create custom types in Pydantic to express very complex rules about what the patched function can return. These act as guard-rails for the model preventing a patched function breaking the code or downstream workflows, and means you can avoid having to write custom validation logic in your application. You can see a couple of examples below or go to the [Examples](placeholder_url) section to see some more input and output examples

Example with int output
```python
@tanuki.patch
def score_sentiment(input: str) -> Annotated[int, Field(gt=0, lt=10)]:
    """
    Scores the input between 0-10
    """

score_sentiment("I like you") # 7
```

Example with custom Pydantic class output
```python
@dataclass
class ActionItem:
    goal: str = Field(description="What task must be completed")
    deadline: datetime = Field(description="The date the goal needs to be achieved")
    
@tanuki.patch
def action_items(input: str) -> List[ActionItem]:
    """Generate a list of Action Items"""

@tanuki.align
def align_action_items():
    goal = "Can you please get the presentation to me by Tuesday?"
    next_tuesday = (datetime.now() + timedelta((1 - datetime.now().weekday() + 7) % 7)).replace(hour=0, minute=0, second=0, microsecond=0)

    assert action_items(goal) == ActionItem(goal="Prepare the presentation", deadline=next_tuesday)
```

By constraining the types of data that can pass through your patched function, you are declaring the potential outputs that the model can return and specifying the world where the program exists in.

You can add integer constraints to the outputs for Pydantic field values, and generics if you wish.

```python
@tanuki.patch
def score_sentiment(input: str) -> Optional[Annotated[int, Field(gt=0, lt=10)]]:
    """Scores the input between 0-10"""

@tanuki.align
def align_score_sentiment():
    """Register several examples to align your function"""
    assert score_sentiment("I love you") == 10
    assert score_sentiment("I hate you") == 0
    assert score_sentiment("You're okay I guess") == 5

# This is a normal test that can be invoked with pytest or unittest
def test_score_sentiment():
    """We can test the function as normal using Pytest or Unittest"""
    score = score_sentiment("I like you") 
    assert score >= 7

if __name__ == "__main__":
    align_score_sentiment()
    print(score_sentiment("I like you")) # 7
    print(score_sentiment("Apples might be red")) # None
```

## Aligning functions
In classic [test-driven development (TDD)](https://en.wikipedia.org/wiki/Test-driven_development), the standard practice is to write a failing test before writing the code that makes it pass. 

Test-Driven Alignment (TDA) adapts this concept to align the behavior of a patched function with an expectation defined by a test.

To align the behaviour of your patched function to your needs, decorate a function with `@align` and assert the outputs of the function with the ‘assert’ statement as is done with standard tests.

```python
@tanuki.align 
def align_classify_sentiment(): 
    assert classify_sentiment("I love this!") == 'Good' 
    assert classify_sentiment("I hate this.") == 'Bad'
   
@tanuki.align
def align_score_sentiment():
    assert score_sentiment("I love you") == 10
    assert score_sentiment("I hate you") == 0
    assert score_sentiment("You're okay I guess") == 5
```

By writing a test that encapsulates the expected behaviour of the tanuki-patched function, you declare the contract that the function must fulfill. This enables you to:

1. **Verify Expectations:** Confirm that the function adheres to the desired output. 
2. **Capture Behavioural Nuances:** Make sure that the LLM respects the edge cases and nuances stipulated by your test.
3. **Develop Iteratively:** Refine and update the behavior of the tanuki-patched function by declaring the desired behaviour as tests.

TDA offers a lean yet robust methodology for grafting machine learning onto existing or new Python codebases. It combines the preventive virtues of TDD while addressing the specific challenges posed by the dynamism of LLMs. The align statements existence and the form they take are sufficient for LLMs to align themselves with the expected behavior.

It is easy to also make complex align statements with pydantic objects

```python
class Tweet(BaseModel):
    """
    Tweet object
    The name is the account of the user
    The text is the tweet they sent
    id is a unique classifier
    """
    name: str
    text: str
    id: str

class Response(BaseModel):
    """
    Response object, where the response attribute is the response sent to the customer
    requires_ticket is a boolean indicating whether the incoming tweet was a question or a direct issue
    that would require human intervention and action
    """
    requires_ticket: bool 
    response: str 

@tanuki.patch
def classify_and_respond(tweet: Tweet) -> Response:
    """
    Respond to the customer support tweet text empathetically and nicely. 
    Convey that you care about the issue and if the problem was a direct issue that the support team should fix or a question, the team will respond to it. 
    """

@tanuki.align
def align_respond():
    input_tweet_1 = Tweet(name = "Laia Johnson",
                          text = "I really like the new shovel but the handle broke after 2 days of use. Can I get a replacement?",
                          id = "123")
    assert classify_and_respond(input_tweet_1) == Response(
                                                            requires_ticket=True, 
                                                            response="Hi, we are sorry to hear that. We will get back to you with a replacement as soon as possible, can you send us your order nr?"
                                                            )
    input_tweet_2 = Tweet(name = "Keira Townsend",
                          text = "I hate the new design of the iphone. It is so ugly. I am switching to Samsung",
                          id = "10pa")
    assert classify_and_respond(input_tweet_2) == Response(
                                                            requires_ticket=False, 
                                                            response="Hi, we are sorry to hear that. We will take this into consideration and let the product team know of the feedback"
                                                            )
    input_tweet_3 = Tweet(name = "Thomas Bell",
                          text = "@Amazonsupport. I have a question about ordering, do you deliver to Finland?",
                          id = "test")
    assert classify_and_respond(input_tweet_3) == Response(
                                                            requires_ticket=True, 
                                                            response="Hi, thanks for reaching out. The question will be sent to our support team and they will get back to you as soon as possible"
                                                            )
    input_tweet_4 = Tweet(name = "Jillian Murphy",
                          text = "Just bought the new goodybox and so far I'm loving it!",
                          id = "009")
    assert classify_and_respond(input_tweet_4) == Response(
                                                            requires_ticket=False, 
                                                            response="Hi, thanks for reaching out. We are happy to hear that you are enjoying the product"
            
                                                            )

```
## Hosted offering
TBD

## Finetuning
An advantage of using Tanuki in your workflow is the cost and latency benefits that will be provided as the number of datapoints increases. 

Successful executions of your patched function suitable for finetuning will be persisted to a training dataset, which will be used to distil smaller models for each patched function. Model distillation and pseudo-labelling is a verified way how to cut down on model sizes and gain improvements in latency and memory footprints while incurring insignificant and minor cost to performance (https://arxiv.org/pdf/2305.02301.pdf, https://arxiv.org/pdf/2306.13649.pdf, https://arxiv.org/pdf/2311.00430.pdf, etc).

Training smaller function-specific models and deploying them is handled by the Tanuki library, so the user will get the benefits without any additional MLOps or DataOps effort. Currently only OpenAI GPT style models are supported (Teacher - GPT4, Student GPT-3.5-turbo) 

We ran Tanuki on some public datasets like Squad2, Spider and IMDB Movie Reviews. Using the default setting, our preliminary tests show that using less than 600 datapoints in the training data are enough to get gpt 3.5 turbo to perform essentialy equivalent (less than 1.5% of performance difference on held-out dev sets) to GPT-4 while achieving up to 12 times lower cost and over 6 times lower latency (cost and latency reduction are very dependent on task specific characteristics like input-output token sizes and align statement token sizes). These tests show the potential in model-distillation in this form for intelligently cutting costs and lowering latency without sacrificing performance.<br><br>

![Example distillation results](https://github.com/monkeypatch/tanuki.py/assets/113173969/2ac4c2fd-7ba6-4598-891d-6aa2c85827c9)

## Supported Model Providers
We support OpenAI, TogetherAi and AWS Bedrock models for inference and currently support only OpenAI models for finetuning. The default teacher models are GPT-4 and GPT4-32K. To use non-default supported models, a 'teacher_models' parameter needs to be specified in the `@tanuki.patch` operators as seen below in the respective model provider sections. The full list of out-of-the-box models is in the table and the setup and how to use other models from the same provider can be seen in the sections below
### All supported models
Supported teacher models for language generation

| Model Name                                    |  Model Handle              | Provider              | Context length|
| ----------------------------------------------| -------------------------- |-----------------------|-------------- |
|GPT-4-1106-preview                             | gpt-4-1106-preview         | OpenAI                |128000         |
|GPT-4                                          | gpt-4                      | OpenAI                |8192           |
|GPT-4-32k                                      | gpt-4-32k                  | OpenAI                |32768          |
|GPT-4-Turbo (latest)                           | gpt-4-turbo                | OpenAI                |128000         |
|GPT-4-Turbo-0125                               | gpt-4-turbo-0125           | OpenAI                |128000         |
|Meta llama2-70b-chat-v1                        | llama_70b_chat_aws         | AWS Bedrock           |4096           |
|Meta llama2-13b-chat-v1                        | llama_13b_chat_aws         | AWS Bedrock           |4096           |
|Mistralai Mixtral-8x7B-Instruct-v0.1           | Mixtral-8x7B               | TogetherAI            |32768          |
|NousResearch Nous-Hermes-2-Mixtral-8x7B-DPO    | Mixtral-8x7B-DPO           | TogetherAI            |32768          |
|Zero-one-ai Yi-34B-Chat                        | Yi-34B-Chat                | TogetherAI            |4096           |
|Meta llama-2-13b-chat                          | llama13b-togetherai        | TogetherAI            |4096           |
|MistralAI Mistral-7B-Instruct-v0.2             | Mistral-7B-Instruct-v0.2   | TogetherAI            |32768          |
|Openchat-3.5-1210                              | openchat-3.5               | TogetherAI            |8192           |
|Teknium OpenHermes-2p5-Mistral-7B              | OpenHermes-2p5-Mistral     | TogetherAI            |4096           |


Supported embedding models

| Model Name                                    |  Model Handle              | Provider              | Context length|
| ----------------------------------------------| -------------------------- |-----------------------|-------------- |
|Text-embedding-ada-002                         | ada-002                    | OpenAI                |8191           |
|Amazon Titan-embed-text-v1                     | aws_titan_embed_v1         | AWS Bedrock           |8000           |


Supported student models for language generation

| Model Name                                    |  Model Handle              | Provider              | Context length|
| ----------------------------------------------| -------------------------- |-----------------------|-------------- |
|GPT-3.5-turbo                                  | gpt-3.5-turbo-1106         | OpenAI                |16385          |
|Meta Llama-2-7b-chat-hf                        | Llama-2-7b-chat-hf         | Anyscale              |8192           |
|Meta Llama-2-13b-chat-hf                       | Llama-2-13b-chat-hf        | Anyscale              |32768          |
|Meta Llama-2-70b-chat-hf                       | Llama-2-70b-chat-hf        | Anyscale              |128000         |
|MistralAI Mistral-7B-Instruct-v0.1             | Mistral-7B-Instruct-v0.1   | Anyscale              |128000         |


### OpenAI models
Openai models are the default teacher models. Without specifying the teacher models to be used, GPT-4 and GPT4-32K will be used to carry out the function on the inputs. 

To set up OpenAI models, the API key needs to be set as 
```
export OPENAI_API_KEY=sk-...
```
To use supported non-default OpenAI models, specify the "model_handle" of the supported model in the teacher_models parameter in the `@tanuki.patch` decorator

```python
@tanuki.patch(teacher_models = ["gpt-4-turbo-0125"])
def score_sentiment(input: str) -> Optional[Annotated[int, Field(gt=0, lt=10)]]:
    """Scores the input between 0-10"""
```

To use non-supported  OpenAI models, create a OpenAIConfig object with the model parameters and specify it in in the teacher_models parameter in the `@tanuki.patch` decorator

```python
from tanuki.language_models.llm_configs.openai_config import OpenAIConfig
model = OpenAIConfig(model_name = "gpt-3.5-turbo-0125", context_length = 16385 )
@tanuki.patch(teacher_models = [model])
def score_sentiment(input: str) -> Optional[Annotated[int, Field(gt=0, lt=10)]]:
    """Scores the input between 0-10"""
```

### TogetherAI models
If you're using the open-source library, to use Together AI models, firstly the Together AI extra package needs to be installed by `pip install tanuki.py[together_ai]`. When the package has been installed, a configuration flag for the teacher model needs to be sent to the `@tanuki.patch` decorator like shown below in examples

First to set up the API key 
```
export TOGETHER_API_KEY=...
```
To use supported TogetherAI models, specify the "model_handle" of the supported in the teacher_models parameter in the `@tanuki.patch` decorator

```python
@tanuki.patch(teacher_models = ["Mixtral-8x7B"])
def score_sentiment(input: str) -> Optional[Annotated[int, Field(gt=0, lt=10)]]:
    """Scores the input between 0-10"""
```

To use non-supported TogetherAI models, create a TogetherAIConfig object with the model parameters and specify it in in the teacher_models parameter in the `@tanuki.patch` decorator

```python
from tanuki.language_models.llm_configs import TogetherAIConfig
model = TogetherAIConfig(model_name = "Open-Orca/Mistral-7B-OpenOrca", context_length = 8192)

@tanuki.patch(teacher_models = [model])
def score_sentiment(input: str) -> Optional[Annotated[int, Field(gt=0, lt=10)]]:
    """Scores the input between 0-10"""
```

### AWS Bedrock models
If you're using the open-source library, to use AWS Bedrock models, firstly the AWS Bedrock extra package needs to be installed by `pip install tanuki.py[aws_bedrock]`. When the package has been installed, a configuration flag for the teacher model needs to be sent to the `@tanuki.patch` decorator like shown below in examples. Make sure the workspace is correctly authenticated with AWS. We currently support LLama2 chat models and Titan embedding models with AWS Bedrock . 

To use supported AWS Bedrock generation models, specify the "model_handle" of the supported in the teacher_models parameter in the `@tanuki.patch` decorator

```python
@tanuki.patch(teacher_models = ["llama_70b_chat_aws"])
def score_sentiment(input: str) -> Optional[Annotated[int, Field(gt=0, lt=10)]]:
    """Scores the input between 0-10"""
```

Similarly to generation models, to use embedding models the model_handle needs to be specified in the teacher_models parameter in the `@tanuki.patch` decorator

```python
@tanuki.patch(teacher_models = ["aws_titan_embed_v1"])
def example_function(input: TypedInput) -> Embedding[np.ndarray]:
    """(Optional) Include the description of how your function will be used."""
```

As different model provider in the AWS Bedrock stack have different API request templates, then its more difficult to use and implement any non-supported AWS Bedrock models, that your account may have access to. 
If you want to implement any additional AWS models, feel free to open an issue or implement it yourself and open a PR. To add a newly configured model, have a look at the [llm_configs](https://github.com/Tanuki/tanuki.py/tree/master/src/tanuki/language_models/llm_configs) folder to see how model configurations are addressed and to add a new model configuration, add it to the [default_models](https://github.com/Tanuki/tanuki.py/tree/master/src/tanuki/language_models/llm_configs/__init__.py). If the request template is the same as the LLama Bedrock request, then you just need to add the provider as `llama_bedrock` to the config (import LLAMA_BEDROCK_PROVIDER from the [constants file](https://github.com/Tanuki/tanuki.py/tree/master/src/tanuki/constants.py)), otherwise you need to also add a new API template (have a look at how the [llama_bedrock_api](https://github.com/Tanuki/tanuki.py/tree/master/src/tanuki/language_models/llama_bedrock_api.py) is implemented) and update the [api_manager](https://github.com/Tanuki/tanuki.py/tree/master/src/tanuki/models/api_manager.py) and the [constants file](https://github.com/Tanuki/tanuki.py/tree/master/src/tanuki/constants.py) with the new provider and api template. First try out the prompting configurations with a couple of examples to ensure the outputs are performing well!.

## RAG
Support for getting embeddings for RAG use-cases have been implemented. The Open-AI ada-002 (default) and amazon.titan-embed-text-v1 (see [here](https://github.com/Tanuki/tanuki.py/tree/master/docs/aws_bedrock.md) how to configure to use) models are currently supported out of the box to get embeddings for input data. For embedding output the output typehint needs to be set as  `Embedding[np.ndarray]`. Currently adding align statements to steer embedding model behaviour is not implemented, but is on the roadmap. 


## Example with OpenAI Ada model
```python
@tanuki.patch
def score_sentiment(input: str) -> Embedding[np.ndarray]:
    """
    Scores the input between 0-10
    """
```

## Example with AWS Titan model
```python
@tanuki.patch(teacher_models = ["aws_titan_embed_v1"])
def score_sentiment(input: str) -> Embedding[np.ndarray]:
    """
    Scores the input between 0-10
    """
```

## Patterns and Examples

### Sentiment scorer

```python
from pydantic import Field
from typing import Annotated
@tanuki.patch
def score_sentiment(input: str) -> Annotated[int, Field(gt=0, lt=10)]:
    """
    Scores the input between 0-10
    """

@tanuki.align
def align_score_sentiment():
    """Register several examples to align your function"""

    assert score_sentiment("I love you") == 10
    assert score_sentiment("I hate you") == 0
    assert score_sentiment("You're okay I guess") == 5

```


### Chatbot

```python
from pydantic import BaseModel
import tanuki
class Tweet(BaseModel):
    """
    Tweet object
    The name is the account of the user
    The text is the tweet they sent
    id is a unique classifier
    """
    name: str
    text: str
    id: str

@tanuki.patch
def classify_and_respond(tweet: Tweet) -> str:
    """
    Respond to the customer support tweet text empathetically and nicely. 
    Convey that you care about the issue and if the problem was a direct issue that the support team should fix or a question, the team will respond to it. 
    """

@tanuki.align
def align_respond():
    input_tweet_1 = Tweet(name = "Laia Johnson",
                          text = "I really like the new shovel but the handle broke after 2 days of use. Can I get a replacement?",
                          id = "123")
    assert classify_and_respond(input_tweet_1) == "Hi, we are sorry to hear that. We will get back to you with a replacement as soon as possible, can you send us your order nr?"

    input_tweet_2 = Tweet(name = "Keira Townsend",
                          text = "I hate the new design of the iphone. It is so ugly. I am switching to Samsung",
                          id = "10pa")
    assert classify_and_respond(input_tweet_2) == "Hi, we are sorry to hear that. We will take this into consideration and let the product team know of the feedback"

```

### ToDolist formatter

```python
import tanuki
from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional


class TodoItem(BaseModel):
    deadline: Optional[datetime] = None
    goal: str
    people: List[str]

@tanuki.patch
def create_todolist_item(input: str) -> TodoItem:
    """
    Converts the input string into a TodoItem object
    :param input: The user-supplied text of things they have to do
    :return: TodoItem object
    """

@tanuki.align
def align_todolist():
    """
    We define 2 input/output pairs for the LLM to learn from.
    """

    assert create_todolist_item("I would like to go to the store and buy some milk") \
           == TodoItem(goal="Go to the store and buy some milk",
                       people=["Me"])

    assert create_todolist_item("I need to go and visit Jeff at 3pm tomorrow") \
           == TodoItem(goal="Go and visit Jeff",
                       people=["Me"],
                       deadline=datetime.datetime(2021, 1, 1, 15, 0))

```

### ToDolist formatter

```python
import tanuki
from typing import Literal



@tanuki.patch
def classify_email(email: str) -> Literal["Real", "Fake"]:
    """
    Classify the email addresses as Fake or Real. The usual signs of an email being fake is the following:
    1) Using generic email addresses like yahoo, google, etc
    2) Misspellings in the email address
    3) Irregular name in email addresses
    """

@tanuki.align
def align_classify():
    assert classify_email("jeffrey.sieker@gmail.com") == "Fake"
    assert classify_email("jeffrey.sieker@apple.com") == "Real"
    assert classify_email("jon123121@apple.com") == "Fake"
    assert classify_email("jon@apple.com") == "Real"
    assert classify_email("jon.lorna@apple.com") == "Real"
    assert classify_email("jon.lorna@mircosoft.com") == "Fake"
    assert classify_email("jon.lorna@jklstarkka.com") == "Fake"
    assert classify_email("unicorn_rider123@yahoo.com") == "Fake"

```

### Information Extractor

```python
import tanuki
from typing import List

@tanuki.patch
def extract_stock_winner(input: str) -> List[str]:
    """
    Below you will find an article with stocks analysis. Bring out the stock symbols of companies who are expected to go up or have positive sentiment
    """


@tanuki.align
def test_stock():
    """We can test the function as normal using Pytest or Unittest"""

    input_1 = "Consumer spending makes up a huge fraction of the overall economy. Investors are therefore always looking at consumers to try to gauge whether their financial condition remains healthy. That's a big part of why the stock market saw a bear market in 2022, as some feared that a consumer-led recession would result in much weaker business performance across the sector.\nHowever, that much-anticipated recession hasn't happened yet, and there's still plenty of uncertainty about the future direction of consumer-facing stocks. A pair of earnings reports early Wednesday didn't do much to resolve the debate, as household products giant Procter & Gamble (PG 0.13%) saw its stock rise even as recreational vehicle manufacturer Winnebago Industries (WGO 0.58%) declined."
    assert extract_stock_winner(input_1) == ['Procter & Gamble', 'Winnebago Industries']

```
