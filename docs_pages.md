## Quickstart

To get started:
1. Create a python function stub decorated with `@tanuki.patch` including type hints and a docstring.
2. Add the API and authentication keys. To set the API key for the default OpenAI models 
```
export OPENAI_API_KEY=sk-...
```

3. (Optional) Create another function decorated with `@tanuki.align` containing normal `assert` statements declaring the expected behaviour of your patched function with different inputs.
4. (Optional) Configure the model you want to use the function for. By default GPT-4 is used but if you want to use any other models supported in our stack, then configure them in the  `@tanuki.patch` operator. You can find out exactly how to configure OpenAI, Amazon Bedrock and Together AI models in the [models](placeholder_url) section.
The patched function can now be called as normal in the rest of your code. 

To add functional alignment, the functions annotated with `align` must also be called if:
- It is the first time calling the patched function (including any updates to the function signature, i.e docstring, input arguments, input type hints, naming or the output type hint)
- You have made changes to your assert statements.

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
    assert score_sentiment("I like you") == 7
```

By writing a test that encapsulates the expected behaviour of the tanuki-patched function, you declare the contract that the function must fulfill. This enables you to:

1. **Verify Expectations:** Confirm that the function adheres to the desired output. 
2. **Capture Behavioural Nuances:** Make sure that the LLM respects the edge cases and nuances stipulated by your test.
3. **Develop Iteratively:** Refine and update the behavior of the tanuki-patched function by declaring the desired behaviour as tests.

Unlike traditional TDD, where the objective is to write code that passes the test, TDA flips the script: **tests do not fail**. Their existence and the form they take are sufficient for LLMs to align themselves with the expected behavior.

TDA offers a lean yet robust methodology for grafting machine learning onto existing or new Python codebases. It combines the preventive virtues of TDD while addressing the specific challenges posed by the dynamism of LLMs.

You can also make complex align statements with pydantic objects

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

## Finetuning
An advantage of using Tanuki in your workflow is the cost and latency benefits that will be provided as the number of datapoints increases. 

Successful executions of your patched function suitable for finetuning will be persisted to a training dataset, which will be used to distil smaller models for each patched function. Model distillation and pseudo-labelling is a verified way how to cut down on model sizes and gain improvements in latency and memory footprints while incurring insignificant and minor cost to performance (https://arxiv.org/pdf/2305.02301.pdf, https://arxiv.org/pdf/2306.13649.pdf, https://arxiv.org/pdf/2311.00430.pdf, etc).

Training smaller function-specific models and deploying them is handled by the Tanuki library, so the user will get the benefits without any additional MLOps or DataOps effort. Currently only OpenAI GPT style models are supported (Teacher - GPT4, Student GPT-3.5) 

We tested out model distillation using Tanuki using OpenAI models on Squad2, Spider and IMDB Movie Reviews datasets. We finetuned the gpt-3.5-turbo model (student) using few-shot responses of gpt-4 (teacher) and our preliminary tests show that using less than 600 datapoints in the training data we were able to get gpt 3.5 turbo to perform essentialy equivalent (less than 1.5% of performance difference on held-out dev sets) to gpt4 while achieving up to 12 times lower cost and over 6 times lower latency (cost and latency reduction are very dependent on task specific characteristics like input-output token sizes and align statement token sizes). These tests show the potential in model-distillation in this form for intelligently cutting costs and lowering latency without sacrificing performance.<br><br>

![Example distillation results](https://github.com/monkeypatch/tanuki.py/assets/113173969/2ac4c2fd-7ba6-4598-891d-6aa2c85827c9)

## Supported Model Providers
We support OpenAI, TogetherAi and AWS Bedrock models. The default models are GPT-4 and GPT4-32K. To use custom models, a 'teacher_models' parameter needs to be specified in the `@tanuki.patch` operators as seen below. The full list of out-of-the-box models is in the table and the setup and how to use other models from the same provider can be seen in the sections below
### All supported models
table TBD
### OpenAI models
TBD
### TogetherAI models
TBD
### AWS Bedrock models
TBD

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
TBD