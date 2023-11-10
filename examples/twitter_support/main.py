from dotenv import load_dotenv
load_dotenv()
import sys
sys.path.append("src")
from monkey_patch.monkey import Monkey as monkey
from pydantic import BaseModel
from typing import Literal

class Response(BaseModel):
    """
    Response object, where the response attribute is the response sent to the customer
    requires_ticket is a boolean indicating whether the incoming tweet was a question or a direct issue
    """
    requires_ticket: bool 
    response: str 

class Support_Ticket(BaseModel):
    """
    Support ticket object, where the tweet_text is the text of the tweet
    """
    name: str
    issue: str
    urgency: Literal["low", "medium", "high"]

@monkey.patch
def classify_and_respond(tweet_text: str) -> Response:
    """
    Respont to the customer support tweet text empathetically and nicely. 
    Convey that you care about the issue and if the problem was a direct issue that the support team should fix or a question, the team will respond to it. 
    """

@monkey.align
def align_respond():
    input_tweet_1 = "Laia Johnson: I really like the new shovel but the handle broke after 2 days of use. Can I get a replacement?"
    assert classify_and_respond(input_tweet_1) == Response(requires_ticket=True, response="Hi, we are sorry to hear that. We will get back to you with a replacement as soon as possible, can you send us your order nr?")
    input_tweet_2 = "Keira Townsend: I hate the new design of the iphone. It is so ugly. I am switching to Samsung"
    assert classify_and_respond(input_tweet_2) == Response(requires_ticket=False, response="Hi, we are sorry to hear that. We will take this into consideration and let the product team know of the feedback")
    input_tweet_3 = "Thomas Bell: @Amazonsupport. I have a question about ordering, do you deliver to Finland?"
    assert classify_and_respond(input_tweet_3) == Response(requires_ticket=True, response="Hi, thanks for reaching out. The question will be sent to our support team and they will get back to you as soon as possible")
    input_tweet_4 = "Jillian Murphy: Just bought the new goodybox and so far I'm loving it!"
    assert classify_and_respond(input_tweet_4) == Response(requires_ticket=False, response="Hi, thanks for reaching out. We are happy to hear that you are enjoying the product")


def main():
    """
    Run through the workflow of the email cleaner
    First get data from the data_path
    Then call aligns for both MP functions
    Then classify emails and if real, extract personas
    Finally save personas to a excel file

    Args:
        data_path (str): the path to the data
        save_path (str): the path to save the personas to
    """
    align_respond()
    input_tweet = "Jack Bell: WTF why did my order not arrive? I ordered it 2 weeks ago. Horrible service"
    response = classify_and_respond(input_tweet)
    print(response)
    if response.requires_ticket:
        print("The tweet was a direct issue, the support team will get back to you")
if __name__ == '__main__':
    main()