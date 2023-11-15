# standard / third party
import json

from fastapi import APIRouter

# local
from controllers.food_controller import get_best_dishes, get_ratings

from services.food_service import get_yelp_reviews

router = APIRouter(
    prefix="/food",
    tags=["food"],
)


@router.get("")
async def analyze_reviews(url: str):
    max_attempts = 3
    print("Got ALIAS:", url)
    print("Fetching reviews")
    reviews = get_yelp_reviews(url)
    print("Got reviews")

    max_reviews = 20
    while len(reviews) == 0 and max_attempts > 0:
        reviews = get_yelp_reviews(url)
        max_attempts -= 1
    print("Here is the first review:")
    print(reviews[0])
    if len(reviews) > max_reviews:
        reviews = reviews[:max_reviews]
    print("Getting ratings")
    ratings = get_ratings(reviews)
    print("Got ratings")
    # best_dishes = get_best_dishes(reviews)
    return {
        "message": "success",
        "ratings": ratings,
        # "best_dishes": best_dishes,
    }
