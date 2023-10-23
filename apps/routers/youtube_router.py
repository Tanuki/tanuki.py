# standard / third party
from fastapi import APIRouter

# local
from controllers.youtube_controller import analyze_video


router = APIRouter(
    prefix="/youtube",
    tags=["youtube"],
)


@router.get("/")
async def summarize_youtube_video(url: str):
    analyze_video(url=url)
    return "HELLO HELLO"
