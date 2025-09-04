# tests/test_unit.py
import os
import unittest

# Ensure SBERT threshold for tests
os.environ.setdefault("SBERT_TAU", "0.75")

# Import the Flask app object, NOT the module
try:
    from backend.app import app as flask_app  # backend/app.py defines `app = Flask(__name__, ...)`
except ImportError:
    # Fallback if PYTHONPATH points differently
    from app import app as flask_app


class TestAPIContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        flask_app.testing = True
        cls.client = flask_app.test_client()

    def _post(self, text: str):
        return self.client.post("/check", json={"symptoms": text})

    # 1) Empty input returns LOW and schema present
    def test_empty_input_returns_low(self):
        res = self._post("")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("risk"), "LOW")
        self.assertIn("method", data)
        self.assertIn("advice", data)

    # 2) Oversize payload (> 4 KB) is rejected
    def test_oversize_payload_guard(self):
        big = "x" * (5 * 1024)  # > 4 KB
        res = self._post(big)
        self.assertIn(res.status_code, (400, 413))

    # 3) Rule-based red-flag override
    def test_rule_redflag(self):
        res = self._post("bloody discharge from the nipple")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("method"), "rule-based")
        self.assertEqual(data.get("risk"), "HIGH")

    # 4) SBERT acceptance at τ = 0.75
    def test_sbert_accepts_paraphrase(self):
        res = self._post("a small groove appears near the nipple on my left breast")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("method"), "bert")
        self.assertGreaterEqual(float(data.get("similarity_score", 0.0)), 0.75)
        self.assertTrue(data.get("matched_reference", ""))

    # 5) LLM fallback on ambiguous input
    def test_llm_fallback_ambiguous(self):
        res = self._post("I feel strange")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("method"), "llm")
        self.assertEqual(data.get("risk"), "LOW")


if __name__ == "__main__":
    unittest.main(verbosity=2)
