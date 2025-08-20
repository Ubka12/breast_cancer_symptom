import unittest
from symptom_rules import rule_based_score
from bert_symptom_checker import bert_symptom_score
from app import llm_symptom_score

class TestSymptomChecker(unittest.TestCase):
    def test_rule_based_score(self):
        score, matches = rule_based_score("There is a hard lump and nipple discharge.")
        self.assertGreater(score, 0)
        self.assertIn("hard lump", matches)

    def test_bert_symptom_score(self):
        result = bert_symptom_score("My breast is red and swollen.")
        self.assertIn(result["risk"], ["LOW", "MEDIUM", "HIGH"])
        self.assertGreaterEqual(result["similarity_score"], 0)

    def test_llm_symptom_score(self):
        result = llm_symptom_score("My nipple has changed shape and skin is dimpling.")
        self.assertIn(result["risk"], ["LOW", "MEDIUM", "HIGH"])
        self.assertTrue("advice" in result)

if __name__ == '__main__':
    unittest.main()
