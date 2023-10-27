# standard / third party
import openai
import os
import re
from youtube_transcript_api import YouTubeTranscriptApi

# local
from monkey import Monkey

openai.api_key = os.getenv("OPENAI_API_KEY")


@Monkey.patch
def summarize_text_extra(input: str, instructions: str) -> str:
    """
    Use the input argument and apply the instructions argument for how to assemble into response.
    """


def get_youtube_transcript(video_id: str) -> str:
    payload = YouTubeTranscriptApi.get_transcript(video_id)
    text = ""
    for i in payload:
        text += f"{i['text']} "
    return text
    # summ = summarize_text_extra(text, "return the most interesting and important takeaways.")
    # print(f"{summ=}")


def analyze_video(url: str, prompt: str) -> str:
    # Search for the pattern in the URL
    pattern = r"v=([a-zA-Z0-9_-]+)"
    match = re.search(pattern, url)

    if match:
        # Extract the video ID from the match
        video_id = match.group(1)
        print("YouTube Video ID:", video_id)

        # Get the transcript
        transcript = get_youtube_transcript(video_id=video_id)
        # print(f"{transcript=}")

        # Call OpenAI
        reasoning = summarize_text_extra(transcript, prompt)
        return reasoning

    else:
        return "YouTube Video ID not found in the URL."
