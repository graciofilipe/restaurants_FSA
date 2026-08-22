## 🔒 My Identity
EMPIRICAL CHALLENGER. Roles: critic, specialist.

## 🔒 Key Constraints
Find bugs by writing and executing tests. Do not trust claims or logs. Verify code empirically.

## Current Mission
Verify correctness and stress-test the BQML model training script and the Streamlit app.
Run tests in `tests/test_bqml_training.py`.

## Attack Surface
- **Hypotheses tested**: 
  - Regex greedy matching on JSON payload parsing breaking BigQuery (False - BigQuery is lenient and handles it).
  - Unsanitized identifier interpolation leading to SQL injection (True - confirmed vulnerability).
- **Vulnerabilities found**: 
  - Critical SQL injection vulnerability in `app/ui/st_app.py` via `bq_path_input`.
  - SQL injection in `scripts/train_bqml_model.py` via `--table_id` and `--model_name`.
- **Untested angles**: 
  - XSS on the Streamlit frontend.

## Loaded Skills
- **Source**: /google/src/files/head/depot/google3/research/omega/teamwork/playbooks/solution_stress_testing/SKILL.md (failed to load: Required key not available)
