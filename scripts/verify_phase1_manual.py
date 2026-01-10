import sys
import os
import logging

# Add app to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.bq_utils import load_specific_agent_insights

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify():
    project_id = "filipegracio-ai-learning"
    dataset_id = "filipegracio_fsa_restaurants"
    fhrsids = ["0"] # Dummy ID that likely doesn't exist

    print(f"Attempting to call load_specific_agent_insights with project={project_id}, dataset={dataset_id}, fhrsids={fhrsids}")
    
    try:
        results = load_specific_agent_insights(project_id, dataset_id, fhrsids)
        print(f"Success! Result: {results}")
        if isinstance(results, list):
             print("Verification passed: Function returned a list (as expected).")
        else:
             print("Verification failed: Function did not return a list.")
             
    except Exception as e:
        print(f"Function call failed with error: {e}")
        # Check if it is a credential error
        if "DefaultCredentialsError" in str(e) or "Could not automatically determine credentials" in str(e):
             print("NOTE: This failure is expected if running without local GCP credentials. The code path was executed.")
        else:
             print("Verification failed with unexpected error.")

if __name__ == "__main__":
    verify()
