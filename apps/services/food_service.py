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


class RatingItem(BaseModel):
    rating: int = Field(int, ge=1, le=10)
    confidence: int = Field(
        int, ge=1, le=10, description="How confident are you in your answer?"
    )


class RatingModel(BaseModel):
    food: RatingItem = Field(..., description="The food-only rating")
    service: RatingItem = Field(..., description="The service-only rating")
    atmosphere: RatingItem = Field(..., description="The atmosphere-only rating")
    location: RatingItem = Field(..., description="The location-only rating")


@Monkey.patch
def specific_ratings(reviews: List[str]) -> RatingModel:
    """
    based on the reviews, separately rate (from 1 to 10) the following aspects:
    - food
        rating: 1-10
        confidence: 1-10
    - service
        rating: 1-10
        confidence: 1-10
    - atmosphere
        rating: 1-10
        confidence: 1-10
    - location
        rating: 1-10
        confidence: 1-10
    """


@Monkey.align
def test_specific_ratings():
    """We can test the function as normal using Pytest or Unittest"""
    assert specific_ratings(["The food was great"]) == RatingModel(
        food=RatingItem(rating=10, confidence=10),
        service=RatingItem(rating=5, confidence=1),
        atmosphere=RatingItem(rating=5, confidence=1),
        location=RatingItem(rating=5, confidence=1),
    )
    assert specific_ratings(["The service was great"]) == RatingModel(
        food=RatingItem(rating=5, confidence=1),
        service=RatingItem(rating=10, confidence=10),
        atmosphere=RatingItem(rating=5, confidence=1),
        location=RatingItem(rating=5, confidence=1),
    )
    assert specific_ratings(
        [
            "The atmosphere was kinda average, the location was really nice, and the service was shitty"
        ]
    ) == RatingModel(
        food=RatingItem(rating=5, confidence=1),
        service=RatingItem(rating=1, confidence=10),
        atmosphere=RatingItem(rating=5, confidence=10),
        location=RatingItem(rating=10, confidence=10),
    )
    assert specific_ratings(
        [
            "The food was the worst I've ever had, the service is a little slow but not bad, the atmosphere was really nice but kinda loud, and the location was perfect except that it was a little hard to find parking"
        ]
    ) == RatingModel(
        food=RatingItem(rating=1, confidence=10),
        service=RatingItem(rating=4, confidence=10),
        atmosphere=RatingItem(rating=7, confidence=10),
        location=RatingItem(rating=9, confidence=10),
    )
    assert specific_ratings(
        [
            "The food was the best I've ever had",
            "The food was the worst I've ever had",
        ]
    ) == RatingModel(
        food=RatingItem(rating=5, confidence=5),
        service=RatingItem(rating=5, confidence=1),
        atmosphere=RatingItem(rating=5, confidence=1),
        location=RatingItem(rating=5, confidence=1),
    )


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
    if "review" not in response.json():
        return []
    res_json = response.json()["review"]
    reviews = [review["description"] for review in res_json]

    return reviews
