from typing import List

from dotenv import load_dotenv

import tanuki
import unittest
load_dotenv()
report = """
        Project Scale: Managing 39-home development in London.
        Funding Approach: Utilizing internal cash flow.
        Lead Source: Came through a mutual contact.
        Previous Work: Handled a major project in Lincoln.
        Our Relationship: Just starting, but looks promising.
        Future Business: Client seems keen on continuing with us.
        Notes: While everything's looking up, we're keeping an eye on balancing the project's ambitious scope with our cash flow strategy and nurturing this new relationship carefully. Excited to see where this partnership goes!
        """

example_1 = \
    """
    Development: Engaged in a large 39-home project in London.
    Financing: Self-financed, keeping tabs on cash flow.
    Lead Source: Connection from a friend.
    Experience: Previous success with a significant Lincoln project.
    Our Connection: New but promising relationship.
    Future Plans: The client is positive about ongoing business.
    """

example_2 = \
    """
    Size of the Customer’s Development Project: 21 homes in London, worth about $5m
    How the Project is Being Financed: Through company cash flow
    How the Customer Came Across the Project Lead: Via a friend of a friend,
    Past Projects of the Customer: Large project in Sleaford
    Relationship Between the Company and Customer/Related Parties: New
    Future Purchases by the Customer: Optimistic
    Risk Factors from the Customer:Financial stability concerns, unverified lead sources, and the nascent nature of our relationship.
    Anything else relevant about the Customer: None
    Customer’s Expected Purchases and Reasons: 16 tons of bricks
    """


@tanuki.patch
def highlight_potential_risks(report: str) -> List[str]:
    """
    This is a report provided by a sales representative about a customer that is trading with us.
    Extract any potential risky phrases from the report, especially those that could negatively affect the company's cash flow or likelihood to default on payments.
    Don't include any line headers.
    :return:
    """

@tanuki.align
def read_risk_factors():


    example1 = "\n".join([part.strip() for part in example_1.split("\n")])
    example2 = "\n".join([part.strip() for part in example_2.split("\n")])

    actual_patched_output = highlight_potential_risks(example1)
    print(actual_patched_output)
    output_1 = [
        "keeping tabs on cash flow",
        "New"
    ]

    output_2 = [
        "Financial stability concerns",
        "unverified lead sources",
        "nascent nature of our relationship",
        "Through company cash flow",
        "New"
    ]

    try:
        assert highlight_potential_risks(example1) == output_1
    except AssertionError as e:
        print(f"Expected: {output_1}")
        print(f"Got: {highlight_potential_risks(example1)}")
        raise e

    assert highlight_potential_risks(example2) == output_2

read_risk_factors()
#
# class TestSMS(unittest.TestCase):
#
#     def setUp(self):
#         pass
#         #self.indexer = InferenceService()
#         #self.read_risk_factors()
#
#
#
# if __name__ == '__main__':
#     unittest.main()
