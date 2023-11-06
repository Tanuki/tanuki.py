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


def get_yelp_reviews(url: str) -> List[str]:
    # url = f"https://api.yelp.com/v3/businesses/{business_id_or_alias}/reviews?offset=100&limit=20&sort_by=yelp_sort"
    # headers = {
    #     "accept": "application/json",
    #     "Authorization": f"Bearer {os.getenv('YELP_API_KEY')}",
    # }
    # response = requests.get(url, headers=headers)
    # data = response.json()
    # reviews = [review["text"] for review in data["reviews"]]
    # url = f"https://api.yelp.com/v3/businesses/{business_id_or_alias}/review_highlights"
    # response = requests.get(url, headers=headers)
    # data = response.json()

    data = yelp_scraper(url)
    reviews = [entry["reviews"] for entry in data if "reviews" in entry.keys()]
    reviews = [item["text"] for item in reviews[0]]

    return reviews
