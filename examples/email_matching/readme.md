# Email-to-Name Matching

This example replaces regex for matching an email to a list of names.

For example, a company has an email list of thousands of emails and list of names in their database that aren't mapped

## Configuration

You need to set the following environment variables:
```
OPENAI_API_KEY=sk-XXX
```

Ensure you have an account with AWS for their email sending service, and OpenAI to access their underlying models.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Ensure that `run.sh` has sufficient permissions
```bash
chmod +x run_main.sh
```
