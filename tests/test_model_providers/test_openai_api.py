import os
import unittest
from unittest.mock import patch, Mock

from dotenv import load_dotenv
load_dotenv()
from openai.types.fine_tuning import FineTuningJob

from tanuki.language_models.openai_api import OpenAI_API
from tanuki.language_models.llm_configs.openai_config import OpenAIConfig



class TestOpenAI_API(unittest.TestCase):

    @patch('os.getenv')
    def test_missing_api_key(self, mock_getenv):
        mock_getenv.return_value = ""
        api = OpenAI_API()
        with self.assertRaises(ValueError):
            api.generate(OpenAIConfig(model_name="test_model", context_length=112), "system_message", "prompt")

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
            api.generate(OpenAIConfig(model_name="test_model", context_length=112), "system_message", "prompt")
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
        result = api.generate(OpenAIConfig(model_name="test_model", context_length=112), "system_message", "prompt")
        self.assertEqual(result, "Generated response")

    @patch('openai.fine_tuning.jobs.list')
    @patch('os.getenv')
    def test_list_finetuned(self, mock_getenv, mock_list):
        mock_getenv.return_value = os.getenv("OPENAI_API_KEY")
        mock_list.return_value = [FineTuningJob(id="b",
                                                created_at=1,
                                                fine_tuned_model="bla",
                                                finished_at=1,
                                                error=None,
                                                hyperparameters={"n_epochs": 1},
                                                object="fine_tuning.job",
                                                result_files=[],
                                                status="succeeded",
                                                trained_tokens=1,
                                                training_file="bla",
                                                validation_file="bla",
                                                model="bla",
                                                organization_id="b",
                                                )]  # Mock response

        api = OpenAI_API()
        result = api.list_finetuned(limit=2)
        self.assertEqual(len(result), 2)


if __name__ == '__main__':
    unittest.main()