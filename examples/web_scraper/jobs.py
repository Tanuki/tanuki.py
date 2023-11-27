import openai
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional

load_dotenv()

import tanuki
from utils import scrape_url


openai.api_key = os.getenv("OPENAI_API_KEY")


class Job(BaseModel):
    position: str
    company: str
    location: str


@tanuki.patch
def extract_job(content: str) -> Optional[Job]:
    """
    Examine the content string and extract the job details for the position title, company, and location.
    """


@tanuki.align
def align_extract_job() -> None:
    print("Aligning...")
    job = "\n\n\n\n\n\n\n\n\nShip broker\nFuentes, Walls and Castro\n\n\n\n\n        Michelleville, AP\n      \n\n2021-04-08\n\n\n\nLearn\nApply\n\n\n"
    assert extract_job(job) == Job(
        position="Ship broker",
        company="Fuentes, Walls and Castro",
        location="Michelleville, AP",
    )


if __name__ == '__main__':

    # Align the function
    align_extract_job()

    # Web scrape the url and extract the list of jobs
    url = "https://realpython.github.io/fake-jobs/"
    contents = scrape_url(url=url, class_name="card")

    # Process the job blocks using Tanuki (only sampling a couple for demo purposes)
    jobs = []
    for content in contents[1:3]:
        jobs.append(extract_job(content))
    print(jobs)
