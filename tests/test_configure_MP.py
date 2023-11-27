from typing import List
from tanuki.register import Register

import os
from typing import Optional, Literal, List
import openai
from dotenv import load_dotenv
import tanuki

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


@tanuki.patch
def classify_sentiment_2(input: str, input_2: str) -> Optional[Literal['Good', 'Bad']]:
    """
    Determine if the inputs are positive or negative sentiment, or None
    """


@tanuki.patch(environment_id = 12, ignore_finetune_fetching=True, ignore_finetuning=True, ignore_data_storage=True)
def classify_sentiment(input: str) -> Optional[Literal['Good', 'Bad']]:
    """
    Determine if the input is positive or negative sentiment
    """

@tanuki.align
def align_classify_sentiment():
    """We can test the function as normal using Pytest or Unittest"""

    i_love_you = "I love you"
    assert classify_sentiment_2(i_love_you, "I love woo") == 'Good'
    assert classify_sentiment_2("I hate you", "You're discusting") == 'Bad'
    assert classify_sentiment_2("Today is wednesday", "The dogs are running outside") == None


    assert classify_sentiment("I love you") == 'Good'
    assert classify_sentiment("I hate you") == 'Bad'
    assert classify_sentiment("Wednesdays are in the middle of the week") == None

def test_classify_sentiment():
    align_classify_sentiment()
    bad_input = "I find you awful"
    good_input = "I really really like you"
    good_input_2 = "I adore you"
    assert classify_sentiment("I like you") == 'Good'
    assert classify_sentiment(bad_input) == 'Bad'
    assert classify_sentiment("I am neutral") == None

    assert classify_sentiment_2(good_input, good_input_2) == 'Good'
    assert classify_sentiment_2("I do not like you you", bad_input) == 'Bad'
    assert classify_sentiment_2("I am neutral", "I am neutral too") == None

def test_configurability():
    classify_sent_description = Register.load_function_description(classify_sentiment)
    classify_sentiment_2_description = Register.load_function_description(classify_sentiment_2)
    sent_func_hash = classify_sent_description.__hash__()
    sent_func_2_hash = classify_sentiment_2_description.__hash__()

    func_modeler = tanuki.function_modeler
    assert func_modeler.environment_id == 12
    assert sent_func_hash in func_modeler.check_finetune_blacklist
    assert sent_func_2_hash not in func_modeler.check_finetune_blacklist
    assert sent_func_hash in func_modeler.execute_finetune_blacklist
    assert sent_func_2_hash not in func_modeler.execute_finetune_blacklist
    assert sent_func_hash in func_modeler.store_data_blacklist
    assert sent_func_2_hash not in func_modeler.store_data_blacklist




