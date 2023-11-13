# Email-to-Name Matching

This example replaces regex for matching an email to a list of names.

Example use case is where a company has an email list of thousands of emails and list of names in their database that aren't mapped.

## Configuration

You need to set the following environment variables:
```
OPENAI_API_KEY=sk-XXX
```

Ensure you have an account with OpenAI to access their underlying models.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

To align and test, run following command(s):
```
python main.py
ptest test_email_matching.py
```