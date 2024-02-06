from typing import List

from dotenv import load_dotenv

import tanuki
import unittest
load_dotenv()

test_input = """
    summary: Based on the scraped website data provided, here are the details extracted per location:

**Midtown NYC:**
- Address Information:
  - Address: 956 Second Avenue
  - City: New York
  - State: NY
  - Zipcode: Not explicitly provided
- Contact Information:
  - Phone Number: 212-644-2700
  - Fax Number: Not provided
  - Email Address: Not explicitly provided
  - URL to location page/website: Not explicitly provided
- URLs to location on other platforms: Not provided
- List of delivery services they use: Not provided

**River North Chicago:**
- Address Information:
  - Address: 400 N Clark Street
  - City: Chicago
  - State: IL
  - Zipcode: Not explicitly provided
- Contact Information:
  - Phone Number: 312-312-5100
  - Fax Number: Not provided
  - Email Address: Not explicitly provided
  - URL to location page/website: Not explicitly provided
- URLs to location on other platforms: Not provided
- List of delivery services they use: Not provided

**East Village NYC:**
- Address Information:
  - Address: 55 Third Avenue
  - City: New York
  - State: NY
  - Zipcode: Not explicitly provided
- Contact Information:
  - Phone Number: 212-420-9800
  - Fax Number: Not provided
  - Email Address: Not explicitly provided
  - URL to location page/website: Not explicitly provided
- URLs to location on other platforms: Not provided
- List of delivery services they use: Not provided

**Lincoln Square NYC:**
- Address Information:
  - Address: 1900 Broadway
  - City: New York
  - State: NY
  - Zipcode: Not explicitly provided
- Contact Information:
  - Phone Number: 212-496-5700
  - Fax Number: Not provided
  - Email Address: Not explicitly provided
  - URL to location page/website: Not explicitly provided
- URLs to location on other platforms: Not provided
- List of delivery services they use: Not provided

**NoMad NYC:**
- Address Information:
  - Address: 1150 Broadway
  - City: New York
  - State: NY
  - Zipcode: Not explicitly provided
- Contact Information:
  - Phone Number: 212-685-4500
  - Fax Number: Not provided
  - Email Address: Not explicitly provided
  - URL to location page/website: Not explicitly provided
- URLs to location on other platforms: Not provided
- List of delivery services they use: Not provided

**Penn Quarter Washington DC:**
- Address Information:
  - Address: 901 F Street NW
  - City: Washington
  - State: DC
  - Zipcode: Not explicitly provided
- Contact Information:
  - Phone Number: 202-868-4900
  - Fax Number: Not provided
  - Email Address: Not explicitly provided
  - URL to location page/website: Not explicitly provided
- URLs to location on other platforms: Not provided
- List of delivery services they use: Not provided

Missing Information:
- Zipcodes for all locations are not explicitly provided.
- Fax numbers are not provided for any location.
- Email addresses specific to each location are not provided, although a general email address info@thesmithrestaurant.com is mentioned.
- URLs to location pages or websites specific to each location are not provided.
- URLs to location on other platforms such as Google Maps, Yelp, Grubhub, Seamless, UberEats, DoorDash, Postmates, and Caviar are not provided.
- A list of delivery services they use is not provided.

Please note that while some general contact information is available, such as a general email address and a phone number for reservations (CTREVENTS@THE
SMITHRESTAURANT.COM), specific details per location are limited to addresses and phone numbers. The rest of the requested information is not present in the scraped text.
    """

test_input_with_tabs = """
summary: Based on the scraped website data provided, here are the details extracted per location:

**Midtown NYC:**
- Address Information:
  - Address: 956 Second Avenue
  - City: New York
  - State: NY
  - Zipcode: Not explicitly provided
- Contact Information:
  - Phone Number: 212-644-2700
  - Fax Number: Not provided
  - Email Address: Not explicitly provided
  - URL to location page/website: Not explicitly provided
- URLs to location on other platforms: Not provided
- List of delivery services they use: Not provided

**River North Chicago:**
- Address Information:
  - Address: 400 N Clark Street
  - City: Chicago
  - State: IL
  - Zipcode: Not explicitly provided
- Contact Information:
  - Phone Number: 312-312-5100
  - Fax Number: Not provided
  - Email Address: Not explicitly provided
  - URL to location page/website: Not explicitly provided
- URLs to location on other platforms: Not provided
- List of delivery services they use: Not provided

**East Village NYC:**
- Address Information:
  - Address: 55 Third Avenue
  - City: New York
  - State: NY
  - Zipcode: Not explicitly provided
- Contact Information:
  - Phone Number: 212-420-9800
  - Fax Number: Not provided
  - Email Address: Not explicitly provided
  - URL to location page/website: Not explicitly provided
- URLs to location on other platforms: Not provided
- List of delivery services they use: Not provided

**Lincoln Square NYC:**
- Address Information:
  - Address: 1900 Broadway
  - City: New York
  - State: NY
  - Zipcode: Not explicitly provided
- Contact Information:
  - Phone Number: 212-496-5700
  - Fax Number: Not provided
  - Email Address: Not explicitly provided
  - URL to location page/website: Not explicitly provided
- URLs to location on other platforms: Not provided
- List of delivery services they use: Not provided

**NoMad NYC:**
- Address Information:
  - Address: 1150 Broadway
  - City: New York
  - State: NY
  - Zipcode: Not explicitly provided
- Contact Information:
  - Phone Number: 212-685-4500
  - Fax Number: Not provided
  - Email Address: Not explicitly provided
  - URL to location page/website: Not explicitly provided
- URLs to location on other platforms: Not provided
- List of delivery services they use: Not provided

**Penn Quarter Washington DC:**
- Address Information:
  - Address: 901 F Street NW
  - City: Washington
  - State: DC
  - Zipcode: Not explicitly provided
- Contact Information:
  - Phone Number: 202-868-4900
  - Fax Number: Not provided
  - Email Address: Not explicitly provided
  - URL to location page/website: Not explicitly provided
- URLs to location on other platforms: Not provided
- List of delivery services they use: Not provided

Missing Information:
- Zipcodes for all locations are not explicitly provided.
- Fax numbers are not provided for any location.
- Email addresses specific to each location are not provided, although a general email address info@thesmithrestaurant.com is mentioned.
- URLs to location pages or websites specific to each location are not provided.
- URLs to location on other platforms such as Google Maps, Yelp, Grubhub, Seamless, UberEats, DoorDash, Postmates, and Caviar are not provided.
- A list of delivery services they use is not provided.

Please note that while some general contact information is available, such as a general email address and a phone number for reservations (CTREVENTS@THE
SMITHRESTAURANT.COM), specific details per location are limited to addresses and phone numbers. The rest of the requested information is not present in the scraped text.
"""

@tanuki.align
def get_location_details():
    """
    Get the location details
    """

    assert get_summary(test_input) == "Based on the scraped website data provided, here are the details extracted per location:"
    assert get_summary(test_input_with_tabs) == "Based on the scraped website data provided, here are the details extracted per location:"


@tanuki.patch
def get_summary(text: str) -> str:
    """
    Summarise the data
    """

get_location_details()
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
