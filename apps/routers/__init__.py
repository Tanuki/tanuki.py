# standard / third party
from fastapi import APIRouter

# local
from routers import (
    youtube_router as youtube,
)


router = APIRouter()
router.include_router(youtube.router)
