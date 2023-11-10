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


@router.get("/")
async def analyze_reviews(url: str):
    reviews = get_yelp_reviews(url)
    max_reviews = 20
    if len(reviews) == 0:
        return {
            "message": "error",
        }
    if len(reviews) > max_reviews:
        reviews = reviews[:max_reviews]

    ratings = get_ratings(reviews)
    # best_dishes = get_best_dishes(reviews)
    return {
        "message": "success",
        "ratings": ratings,
        # "best_dishes": best_dishes,
    }
