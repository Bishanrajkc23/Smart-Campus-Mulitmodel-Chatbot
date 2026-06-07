import pandas as pd

from src.evaluate import create_testing_scenarios
from src.utils import RESULTS_DIR


def test_testing_scenarios_file_contains_10_cases():
    scenarios = create_testing_scenarios()
    output = RESULTS_DIR / "testing_results.csv"
    assert output.exists()
    saved = pd.read_csv(output)
    assert len(saved) == 10
    assert len(scenarios["expected_intent"].unique()) >= 5
