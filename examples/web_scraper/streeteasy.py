from numpy import square
import openai
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional

load_dotenv()

import tanuki
from utils import scrape_url


openai.api_key = os.getenv("OPENAI_API_KEY")


class Property(BaseModel):
    neighborhood: str
    address: str
    price: float
    fee: bool
    beds: float
    bath: float
    listed_by: str


@tanuki.patch
def extract_property(content: str) -> Optional[Property]:
    """
    Examine the content string and extract the rental property details for the neighborhood, address,
    price, number of beds, number of bathrooms, square footage, and company that is listing the property.
    """

@tanuki.align
def align_extract_property() -> None:
    print("Aligning...")
    unit_one = "Rental Unit in Lincoln Square\n      \n\n\n229 West 60th Street #7H\n\n\n\n$7,250\nNO FEE\n\n\n\n\n\n\n\n\n2 Beds\n\n\n\n\n2 Baths\n\n\n\n\n\n                1,386\n                square feet\nsq_ft\n\n\n\n\n\n        Listing by Algin Management"
    assert extract_property(unit_one) == Property(
        neighborhood="Lincoln Square",
        address="229 West 60th Street #7H",
        price=7250.0,
        fee=False,
        beds=2.0,
        bath=2.0,
        listed_by="Algin Management",
    )


if __name__ == '__main__':

    # Align the function
    align_extract_property()

    # Web scrape the url and extract the rental property details
    url = "https://streeteasy.com/2-bedroom-apartments-for-rent/manhattan?page=2"
    contents = scrape_url(url=url, class_name="listingCardBottom")
    print(contents)

    # Process the rental property block using Tanuki
    units = []
    for content in contents[1:3]:
        units.append(extract_property(content))
    print(units)
