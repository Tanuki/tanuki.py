from pydantic import Field, BaseModel
from typing import List, Annotated, Dict

from monkey import Monkey
import openai
import requests
from bs4 import BeautifulSoup

# from dotenv import load_dotenv
import os

from extenders.yelp_scraper import yelp_scraper

# load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


from zenrows import ZenRowsClient

client = ZenRowsClient(os.getenv("ZENROWS_API_KEY"))


class RatingModel(BaseModel):
    food: int = Annotated[
        Field(..., ge=1, le=10), "The food-only rating based on the provided reviews"
    ]
    service: int = Annotated[
        Field(..., ge=1, le=10), "The service-only rating based on the provided reviews"
    ]
    atmosphere: int = Annotated[
        Field(..., ge=1, le=10),
        "The atmosphere-only rating based on the provided reviews",
    ]
    location: int = Annotated[
        Field(..., ge=1, le=10),
        "The location-only rating based on the provided reviews",
    ]


@Monkey.patch
def specific_ratings(reviews: List[str]) -> RatingModel:
    """
    based on the reviews, separately rate (from 1 to 10) the following aspects:
    - food
    - service
    - atmosphere
    - location
    - overall
    """


@Monkey.align
def test_specific_ratings():
    """We can test the function as normal using Pytest or Unittest"""


@Monkey.patch
def recommended_dishes(reviews: List[str]) -> List[str]:
    """
    List the top 5 (or fewer) best dishes based on the given reviews
    """


@Monkey.align
def test_recommended_dishes():
    """We can test the function as normal using Pytest or Unittest"""


def get_yelp_reviews(business_id_or_alias: str) -> List[str]:
    url = f"https://www.yelp.com/biz/{business_id_or_alias}"
    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9",
        "Host": "httpbin.org",
        "Sec-Ch-Ua": '"Google Chrome";v="117", "Not;A=Brand";v="8", "Chromium";v="117"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "cross-site",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
    }
    params = {"autoparse": "true"}
    response = client.get(url, params=params)
    res_json = response.json()["review"]
    reviews = [review["description"] for review in res_json]

    return reviews
