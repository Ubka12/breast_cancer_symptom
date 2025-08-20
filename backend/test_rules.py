# backend/test_rules.py

import pytest
from backend.symptom_rules import score_symptoms, classify_risk

@pytest.mark.parametrize('text, expected_score', [
    ('I have a lump in my breast', 3),
    ('There is a hard lump and redness', 5),  # 4 + 1
    ('Just mild tenderness', 1),
    ('No symptoms here', 0),
])
def test_scoring(text, expected_score):
    score, matches = score_symptoms(text)
    assert score == expected_score

@pytest.mark.parametrize('score, expected', [
    (0, 'LOW'),
    (1, 'LOW'),
    (2, 'MEDIUM'),
    (3, 'HIGH'),
    (5, 'HIGH'),
])
def test_classification(score, expected):
    assert classify_risk(score) == expected
