# Monkey Patch Email Matcher

Monkey Patch Email Matcher is a Python utility for matching email addresses to names using Large Language Models (LLMs). This tool is designed to handle the complexity and edge cases where regular expressions fall short, offering a more dynamic and robust solution for email-to-name matching.

## Installation

Before installing, ensure you have Python and `pip` installed on your system.

```bash
pip install monkey-patch.py openai python-dotenv
```

Set your OpenAI API key in an .env file:
```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```