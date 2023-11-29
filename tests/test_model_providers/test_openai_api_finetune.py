import os
import unittest
from unittest.mock import patch, Mock

from dotenv import load_dotenv

from tanuki.language_models.openai_api import OpenAI_API

load_dotenv()


class TestOpenAI_API(unittest.TestCase):

    @patch('os.getenv')
    def test_missing_api_key(self, mock_getenv):
        mock_getenv.return_value = None
        api = OpenAI_API()
        with self.assertRaises(ValueError):
            api.generate("model_name", "system_message", "prompt")

    @patch('requests.post')
    @patch('os.getenv')
    def test_invalid_api_key(self, mock_getenv, mock_post):
        mock_getenv.return_value = "invalid_key"
        mock_response = Mock()
        mock_response.json.return_value = {
            "error": {"code": 'invalid_api_key'}
        }
        mock_post.return_value = mock_response

        api = OpenAI_API()
        with self.assertRaises(Exception) as context:
            api.generate("model_name", "system_message", "prompt")
        self.assertIn("invalid", str(context.exception))

    @patch('requests.post')
    @patch('os.getenv')
    def test_successful_generation(self, mock_getenv, mock_post):
        mock_getenv.return_value = os.getenv("OPENAI_API_KEY")
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Generated response"}}]
        }
        mock_post.return_value = mock_response

        api = OpenAI_API()
        result = api.generate("model_name", "system_message", "prompt")
        self.assertEqual(result, "Generated response")


if __name__ == '__main__':
    unittest.main()