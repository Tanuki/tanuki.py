from dotenv import load_dotenv
load_dotenv()
import tanuki
from pydantic import BaseModel
from typing import Literal

class Response(BaseModel):
    """
    Response object, where the response attribute is the response sent to the customer
    requires_ticket is a boolean indicating whether the incoming tweet was a question or a direct issue
    """
    requires_ticket: bool 
    response: str 

class SupportTicket(BaseModel):
    """
    Support ticket object, where the name attribute is customers name,
    issue is a brief description of the issue customer had
    urgency conveys how urgently the team should respond to the issue
    """
    name: str
    issue: str
    urgency: Literal["low", "medium", "high"]

@tanuki.patch
def classify_and_respond(tweet_text: str) -> Response:
    """
    Respond to the customer support tweet text empathetically and nicely. 
    Convey that you care about the issue and if the problem was a direct issue that the support team should fix or a question, the team will respond to it. 
    """

@tanuki.align
def align_respond():
    input_tweet_1 = "Laia Johnson: I really like the new shovel but the handle broke after 2 days of use. Can I get a replacement?"
    assert classify_and_respond(input_tweet_1) == Response(
                                                            requires_ticket=True, 
                                                            response="Hi, we are sorry to hear that. We will get back to you with a replacement as soon as possible, can you send us your order nr?"
                                                            )
    input_tweet_2 = "Keira Townsend: I hate the new design of the iphone. It is so ugly. I am switching to Samsung"
    assert classify_and_respond(input_tweet_2) == Response(
                                                            requires_ticket=False, 
                                                            response="Hi, we are sorry to hear that. We will take this into consideration and let the product team know of the feedback"
                                                            )
    input_tweet_3 = "Thomas Bell: @Amazonsupport. I have a question about ordering, do you deliver to Finland?"
    assert classify_and_respond(input_tweet_3) == Response(
                                                            requires_ticket=True, 
                                                            response="Hi, thanks for reaching out. The question will be sent to our support team and they will get back to you as soon as possible"
                                                            )
    input_tweet_4 = "Jillian Murphy: Just bought the new goodybox and so far I'm loving it!"
    assert classify_and_respond(input_tweet_4) == Response(
                                                            requires_ticket=False, 
                                                            response="Hi, thanks for reaching out. We are happy to hear that you are enjoying the product"
                                                            )

@tanuki.patch
def create_support_ticket(tweet_text: str) -> SupportTicket:
    """
    Using the tweet text create a support ticket for saving to the internal database
    """

@tanuki.align
def align_supportticket():
    input_tweet_1 = "Laia Johnson: I really like the new shovel but the handle broke after 2 days of use. Can I get a replacement?"
    assert create_support_ticket(input_tweet_1) == SupportTicket(
                                                                name = "Laia Johnson", 
                                                                issue="Needs a replacement product because the handle broke", 
                                                                urgency = "high"
                                                                )
    input_tweet_2 = "Thomas Bell: @Amazonsupport. I have a question about ordering, do you deliver to Finland?"
    assert create_support_ticket(input_tweet_2) == SupportTicket(
                                                                name="Thomas Bell", 
                                                                issue="Answer whether we deliver to Finland", 
                                                                urgency="low"
                                                                )
    input_tweet_3 = "Jillian Murphy: Just bought the new goodybox and so far I'm loving it! The cream package was slightly damaged however, would need that to be replaced"
    assert classify_and_respond(input_tweet_3) == SupportTicket(
                                                                name="Jillian Murphy", 
                                                                issue="Needs a new cream as package was slightly damaged", 
                                                                urgency="medium"
                                                                )

def main():
    """
    Run through the workflow of twitter support bot.
    Example usecase uses 3 tweets, where first two require a support ticket and final one does not 
    """
    # start with calling aligns
    align_respond()
    align_supportticket()

    input_tweet_1 = "Jack Bell: Bro @Argos why did my order not arrive? I ordered it 2 weeks ago. Horrible service"
    response = classify_and_respond(input_tweet_1)
    # requires_ticket=True 
    # response="Hi Jack, we're really sorry to hear about this. We'll look into it right away and get back to you as soon as possible. Could you please provide us with your order number?"
    
    if response.requires_ticket:
        ticket = create_support_ticket(input_tweet_1)
        # name='Jack Bell' 
        # issue='Order did not arrive' 
        # urgency='high'
    
    input_tweet_2 = "Casey Montgomery: @Argos The delivery time was 3 weeks but was promised 1. Not a fan. "
    response = classify_and_respond(input_tweet_2)
    # requires_ticket=True 
    # response='Hi, we are sorry to hear about the delay. We will look into this issue and get back to you as soon as possible. Can you provide us with your order number?
    if response.requires_ticket:
        ticket = create_support_ticket(input_tweet_2)
        # name='Casey Montgomery' 
        # issue='Complaint about delivery time' 
        # urgency='medium'
    
    input_tweet_3 = "Jacks Parrow: @Argos The new logo looks quite ugly, wonder why they changed it"
    response = classify_and_respond(input_tweet_3)
    # requires_ticket=False 
    # response='Hi, we are sorry to hear that. We will take this into consideration and let the product team know of the feedback'

    if response.requires_ticket:
        ticket = create_support_ticket(input_tweet_3)
        # No ticket

if __name__ == '__main__':
    main()