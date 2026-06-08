import pytest
from scripts.train_bqml_model import train_model

def test_bqml_training_dry_run():
    # Perform a dry run against the live schema
    # This validates the SQL syntax and schema mathematically.
    try:
        train_model(
            project_id="filipegracio-ai-learning",
            dataset_id="filipegracio_fsa_restaurants",
            table_id="fsa_master",
            model_name="restaurant_preference_model",
            dry_run=True
        )
    except Exception as e:
        pytest.fail(f"Dry run failed: {e}")

