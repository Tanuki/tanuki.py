from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel
from typing import Literal, Optional
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

class Response(BaseModel):
    """
    Response object, where the response attribute is the response sent to the customer
    requires_ticket is a boolean indicating whether the incoming tweet was a question or a direct issue
    that would require human intervention and action
    """
    requires_ticket: bool 
    response: str 

class SupportTicket(BaseModel):
    """
    Support ticket object, where 
    issue is a brief description of the issue customer had
    urgency conveys how urgently the team should respond to the issue
    """
    issue: str
    urgency: Literal["low", "medium", "high"]

# response creation
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

# support ticket creation
@tanuki.patch
def create_support_ticket(tweet_text: str) -> SupportTicket:
    """
    Using the tweet text create a support ticket for saving to the internal database
    Create a short summary of action that needs to be taken and the urgency of the issue
    """

@tanuki.align
def align_supportticket():
    input_tweet_1 = "I really like the new shovel but the handle broke after 2 days of use. Can I get a replacement?"
    assert create_support_ticket(input_tweet_1) == SupportTicket(
                                                                issue="Needs a replacement product because the handle broke", 
                                                                urgency = "high"
                                                                )
    input_tweet_2 = "@Amazonsupport. I have a question about ordering, do you deliver to Finland?"
    assert create_support_ticket(input_tweet_2) == SupportTicket(
                                                                issue="Find out and answer whether we currently deliver to Finland",
                                                                urgency="low"
                                                                )
    input_tweet_3 = "Just bought the new goodybox and so far I'm loving it! The cream package was slightly damaged however, would need that to be replaced"
    assert create_support_ticket(input_tweet_3) == SupportTicket(
                                                                issue="Needs a new cream as package was slightly damaged", 
                                                                urgency="medium"
                                                                )
    
# final function for the workflow
def analyse_and_respond(tweet: Tweet) -> tuple[Optional[SupportTicket], Response]:
    # get the response
    response =  classify_and_respond(tweet)
    # if the response requires a ticket, create a ticket
    if response.requires_ticket:
        support_ticket = create_support_ticket(tweet.text)
        return response, support_ticket
    return response, None


def main():
    """
    This function analyses the incoming tweet and returns a response output and if needed a ticket output
    """
    # start with calling aligns to register the align statements
    align_respond()
    align_supportticket()

    input_tweet_1 = Tweet(name = "Jack Bell",
                          text = "Bro @Argos why did my order not arrive? I ordered it 2 weeks ago. Horrible service",
                          id = "1")
    response, ticket = analyse_and_respond(input_tweet_1)
    
    print(response)
    # requires_ticket=True 
    # response="Hi Jack, we're really sorry to hear about this. We'll look into it right away and get back to you as soon as possible."
    
    print(ticket)
    # issue="Customer's order did not arrive after 2 weeks"
    # urgency='high'

    input_tweet_2 = Tweet(name = "Casey Montgomery",
                          text = "@Argos The delivery time was 3 weeks but was promised 1. Not a fan. ",
                          id = "12")
    response, ticket = analyse_and_respond(input_tweet_2)
    
    print(response)
    # requires_ticket=True 
    # response="Hi Casey, we're really sorry to hear about the delay in your delivery. We'll look into this issue and get back to you as soon as possible."
    
    print(ticket)
    # issue='Delivery time was longer than promised'
    # urgency='medium'

    input_tweet_3 = Tweet(name = "Jacks Parrow",
                          text = "@Argos The new logo looks quite ugly, wonder why they changed it",
                          id = "1123")
    response, ticket = analyse_and_respond(input_tweet_3)

    print(response)
    # requires_ticket=False 
    # response="Hi Jacks Parrow, we're sorry to hear that you're not a fan of the new logo. We'll pass your feedback on to the relevant team. Thanks for letting us know."
    
    print(ticket)
    # None


if __name__ == "__main__":
    main()