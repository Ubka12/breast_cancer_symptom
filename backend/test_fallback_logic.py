import unittest
from unittest.mock import patch
from app import app

class FallbackLogicTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_rule_based_triggered(self):
        response = self.client.post('/check', json={'symptoms': 'nipple discharge'})
        json_data = response.get_json()
        self.assertEqual(json_data['method'], 'rule-based')

    @patch('bert_symptom_checker.bert_symptom_score')
    def test_bert_triggered(self, mock_bert):
        mock_bert.return_value = {
            'risk': 'MEDIUM',
            'similarity_score': 0.75,
            'matched_reference': 'swollen breast'
        }
        response = self.client.post('/check', json={'symptoms': 'swelling under nipple'})
        json_data = response.get_json()
        self.assertEqual(json_data['method'], 'bert')

    @patch('bert_symptom_checker.bert_symptom_score')
    @patch('app.openai_client', None)
    def test_llm_triggered_no_key(self, mock_bert):
        mock_bert.return_value = {
            'risk': 'LOW',
            'similarity_score': 0.3,
            'matched_reference': None
        }
        response = self.client.post('/check', json={'symptoms': 'itchiness and tingling'})
        json_data = response.get_json()
        self.assertEqual(json_data['method'], 'llm')
        self.assertIn('fallback', json_data['advice'].lower())

if __name__ == '__main__':
    unittest.main()
