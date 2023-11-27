import openai
import os
from pydantic import BaseModel
import sys
import wikipedia

sys.path.append("../../src")
import tanuki

from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


# -- Simple Summary Example --
@tanuki.patch
def explain_simple(summary: str) -> str:
    """Explain the summary in simple terms."""


def ask_wikipedia(topic: str) -> str:
    summary = wikipedia.summary(topic)
    return explain_simple(summary)


def simplify_example(topic: str) -> None:
    print("Wikipedia Summary:\n")
    print(wikipedia.summary(topic))
    print("Simplify Summary:\n")
    print(ask_wikipedia(topic))


# -- Classify Example --
class Dinosaur(BaseModel):
    name: str
    nickname: str
    height: int
    weight: int


@tanuki.patch
def dinosaur_classifer(summary: str) -> Dinosaur:
    """Convert the input summary into a Dinosaur object."""

def dinopedia(dinosaur: str) -> Dinosaur:
    summary = wikipedia.summary(dinosaur)
    print(summary)
    print(dinosaur_classifer(summary))


if __name__ == "__main__":
    # simplify_example("Nuclear fission")

    # dino = "Tyrannosaurus"
    # dino = "Triceratops"
    # dino = "Stegosaurus"
    dino = "Velociraptor"
    # dino = "Spinosaurus"
    dinopedia(dino)
