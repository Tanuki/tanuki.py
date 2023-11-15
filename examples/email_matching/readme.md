# Email-to-Name Matching

This example replaces regex for matching an email to a list of names.

Example use case is where a company has an email list of thousands of emails and list of names in their database that aren't mapped.

## Configuration

Set the following environment variables in your `.env` file:
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
pytest -vv -s test_email_matching.py
```