from dotenv import load_dotenv
load_dotenv()
import tanuki
from pydantic import BaseModel
from typing import Literal


class Persona(BaseModel):
    email: str 
    name: str 
    company : str = None

@tanuki.patch
def classify_email(email: str) -> Literal["Real", "Fake"]:
    """
    Classify the email addresses as Fake or Real. The usual signs of an email being fake is the following:
    1) Using generic email addresses like yahoo, google, etc
    2) Misspellings in the email address
    3) Irregular name in email addresses
    """

@tanuki.align
def align_classify():
    assert classify_email("jeffrey.sieker@gmail.com") == "Fake"
    assert classify_email("jeffrey.sieker@apple.com") == "Real"
    assert classify_email("jon123121@apple.com") == "Fake"
    assert classify_email("jon@apple.com") == "Real"
    assert classify_email("jon.lorna@apple.com") == "Real"
    assert classify_email("jon.lorna@mircosoft.com") == "Fake"
    assert classify_email("jon.lorna@jklstarkka.com") == "Fake"
    assert classify_email("unicorn_rider123@yahoo.com") == "Fake"


@tanuki.patch
def extract_persona(email: str) -> Persona:
    """
    Using the email and email handler, extract the persona from the email
    The persona must have the email of the user,
    company (either the company name or None if generic Google, Yahoo etc email)
    name of the user to the best of the ability
    """

@tanuki.align
def align_extract():
    assert extract_persona("jeffrey.sieker@apple.com") == Persona(email="jeffrey.sieker@apple.com", name="Jeffrey Sieker", company="Apple")
    assert extract_persona("jon@amazon.com") == Persona(email="jon@apple.com", name="Jon", company="Amazon")
    assert extract_persona("jon.lorna@Lionmunch.com") == Persona(email="jon.lorna@apple.com", name="Jon Lorna", company="Lionmunch")
    assert extract_persona("jon.lorna@gmail.com") == Persona(email="jon.lorna@gmail.com", name="Jon Lorna")

def main(data_path, save_path):
    """
    Run through the workflow of the email cleaner
    First get data from the data_path
    Then call aligns for both MP functions
    Then classify emails and if real, extract personas
    Finally save personas to a excel file

    Args:
        data_path (str): the path to the data
        save_path (str): the path to save the personas to
    """
    # get data
    with open(data_path, "r") as f:
        emails = f.readlines()
        emails = [email.strip() for email in emails]

    # aligns
    align_classify()
    align_extract()

    personas = []
    # classify and extract
    for email in emails:
        output = classify_email(email)
        print(f"Checked {email} and classified as {output}")
        if output == "Real":
            personas.append(extract_persona(email))
    # save to excel
    import pandas as pd
    df = pd.DataFrame([persona.dict() for persona in personas])
    df.to_excel(save_path)

if __name__ == '__main__':
    data_path = r"examples\email_cleaner\data\test_emails.txt"
    save_path = r"examples\email_cleaner\data\personas.xlsx"
    main(data_path, save_path)