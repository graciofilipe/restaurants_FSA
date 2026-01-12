# Plan: Code Cleanup and Simplification

## Phase 1: Analysis & Inventory
- [x] Task: List all files in `scripts/` and root directory `test_*.py` files.
- [x] Task: Analyze each file to determine if it should be kept, deleted, or merged.
- [x] Task: Analyze `app/ui/st_app.py` and `app/services/bq_utils.py` to identify specific refactoring candidates (e.g., extract `handle_insight_generation` to a service or UI module, consolidate BQ query building).
- [x] Task: Conductor - User Manual Verification 'Phase 1: Analysis & Inventory' (Protocol in workflow.md)

## Phase 2: Cleanup (Scripts & Tests)
- [x] Task: Delete identified obsolete scripts in `scripts/`.
- [x] Task: Delete identified obsolete test files in root and `app/`.
- [x] Task: Run `pytest` to ensure no regressions in the remaining suite.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Cleanup (Scripts & Tests)' (Protocol in workflow.md)

## Phase 3: Refactoring & Simplification
- [x] Task: Refactor `app/ui/st_app.py`: Extract the "Agent Research" tab logic and "Gemini Analysis" tab logic into separate functions or a new UI module (e.g., `app/ui/tabs.py`).
- [x] Task: Refactor `app/services/bq_utils.py`: Review `execute_gemini_enrichment` and other large functions for simplification.
- [x] Task: Verify refactoring by running the application (manual check) and passing all tests.
- [x] Task: Conductor - User Manual Verification 'Phase 3: Refactoring & Simplification' (Protocol in workflow.md)
