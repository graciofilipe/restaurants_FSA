# Plan: Code Cleanup and Simplification

## Phase 1: Analysis & Inventory
- [ ] Task: List all files in `scripts/` and root directory `test_*.py` files.
- [ ] Task: Analyze each file to determine if it should be kept, deleted, or merged.
- [ ] Task: Analyze `app/ui/st_app.py` and `app/services/bq_utils.py` to identify specific refactoring candidates (e.g., extract `handle_insight_generation` to a service or UI module, consolidate BQ query building).
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Analysis & Inventory' (Protocol in workflow.md)

## Phase 2: Cleanup (Scripts & Tests)
- [ ] Task: Delete identified obsolete scripts in `scripts/`.
- [ ] Task: Delete identified obsolete test files in root and `app/`.
- [ ] Task: Run `pytest` to ensure no regressions in the remaining suite.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Cleanup (Scripts & Tests)' (Protocol in workflow.md)

## Phase 3: Refactoring & Simplification
- [ ] Task: Refactor `app/ui/st_app.py`: Extract the "Agent Research" tab logic and "Gemini Analysis" tab logic into separate functions or a new UI module (e.g., `app/ui/tabs.py`).
- [ ] Task: Refactor `app/services/bq_utils.py`: Review `execute_gemini_enrichment` and other large functions for simplification.
- [ ] Task: Verify refactoring by running the application (manual check) and passing all tests.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Refactoring & Simplification' (Protocol in workflow.md)
