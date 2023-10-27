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
    ratings = get_ratings(reviews)
    best_dishes = get_best_dishes(reviews)
    return {
        "message": "success",
        "ratings": ratings,
        "best_dishes": best_dishes,
    }
