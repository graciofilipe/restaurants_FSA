import pytest
import google.auth
from scripts.train_bqml_model import train_model

def test_sql_injection_model_name():
    _, default_project = google.auth.default()
    project = default_project or "filipegracio-genai"
    malicious_model_name = "test_model`; SELECT 1; --"
    try:
        train_model(
            project_id=project,
            dataset_id="filipegracio_fsa_restaurants",
            table_id="fsa_master",
            model_name=malicious_model_name,
            dry_run=True
        )
        pytest.fail("SQL Injection should not pass dry run if it parses, or we should at least catch it. Wait, if it parses as multiple statements, BigQuery might execute it.")
    except Exception as e:
        print(f"Exception caught: {e}")
