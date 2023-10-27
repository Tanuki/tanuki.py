# standard / third party
from fastapi import APIRouter

# local
from routers import (
    youtube_router as youtube,
    food_router as food,
)


router = APIRouter()
router.include_router(youtube.router)
router.include_router(food.router)
