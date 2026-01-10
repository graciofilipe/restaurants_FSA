# Specification: Code Cleanup and Simplification

## Overview
This track aims to improve the maintainability and readability of the codebase by removing obsolete artifacts, streamlining tests, and modularizing complex code segments. The focus is on the `scripts/` directory and test files, with a secondary goal of identifying and executing simplification opportunities within the main application code.

## Functional Requirements

### 1. Script Cleanup
- **Goal:** Remove temporary, one-off, or obsolete scripts from the `scripts/` directory.
- **Action:** Identify scripts used for past verifications (e.g., `verify_*.py`, `test_update_search_config.py` if it was temporary) and delete them if they are no longer needed for ongoing maintenance or CI/CD.
- **Retention:** Keep essential scripts like `create_cloud_run_job.sh`, `update_search_config.py`, and `bq_scripts.py`.

### 2. Test Suite Rationalization
- **Goal:** Remove redundant or obsolete tests to ensure a clean, fast, and relevant test suite.
- **Action:** Review all files matching `test_*.py` or `*_test.py`.
- **Criteria:** Delete tests that:
    - Were created for specific, completed tracks and are now covered by main regression tests.
    - Test deprecated functionality.
    - Are manual verification scripts masquerading as tests.
- **Consolidation:** Ensure valuable test cases from deleted files are migrated to the core test suite (e.g., `app/services/test_*.py`) if not already present.

### 3. Code Modularization & Simplification
- **Goal:** Identify and refactor complex or repetitive code.
- **Target Areas (to be investigated):**
    - `app/services/bq_utils.py`: Check for redundant query construction or overlapping functions.
    - `app/ui/st_app.py`: Look for opportunities to extract UI components into separate modules/functions (e.g., "Agent Research" tab logic).
    - `scripts/bq_scripts.py`: Ensure SQL strings are managed cleanly.
- **Action:** Perform refactoring to reduce function length and improve separation of concerns.

## Non-Functional Requirements
- **Safety:** Ensure no core functionality is lost. Run the full test suite before and after changes.
- **Documentation:** Update docstrings where code is modified.

## Acceptance Criteria
- [ ] `scripts/` directory contains only active, necessary scripts.
- [ ] Test suite passes (`pytest`) and contains no obsolete files in the root or `app/` directories.
- [ ] `app/ui/st_app.py` is reviewed and refactored if sections exceed ~100 lines or handle multiple distinct responsibilities.
- [ ] `app/services/bq_utils.py` is reviewed for redundancy.
