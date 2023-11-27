import openai
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional

load_dotenv()

import tanuki
from utils import scrape_url


openai.api_key = os.getenv("OPENAI_API_KEY")


class Country(BaseModel):
    name: str
    capital: str
    population: int
    area: float


@tanuki.patch
def extract_country(content: str) -> Optional[Country]:
    """
    Examine the content string and extract the country information pertaining to it's
    name, capital, population, and area.
    """


@tanuki.align
def align_extract_country() -> None:
    print("Aligning...")
    country = "\n\n\n                            U.S. Virgin Islands\n                        \n\nCapital: Charlotte Amalie\nPopulation: 108708\nArea (km2): 352.0\n\n"
    assert extract_country(country) == Country(
        name="U.S. Virgin Islands",
        capital="Charlotte Amalie",
        population=108708,
        area=352.0,
    )


if __name__ == '__main__':

    # Align the function
    align_extract_country()

    # Web scrape the url and extract the list of countries
    url = "https://www.scrapethissite.com/pages/simple/"
    contents = scrape_url(url=url, class_name="country")

    # Process the country blocks using Tanuki (only sampling a couple for demo purposes)
    countries = []
    for content in contents[10:12]:
        countries.append(extract_country(content))
    print(countries)
