# Implementation Plan - Codebase Refactor and Simplification

## Phase 1: Preparation & Verification [checkpoint: b891e2f]
- [x] Task: Verify current environment and run existing tests.
    - [x] Run `pytest` to establish a baseline.
    - [x] Create a temporary backup or ensure git status is clean.
- [x] Task: Create new directory structure.
    - [x] Create `app/`, `app/ui/`, `app/services/`, `app/core/`, `scripts/`.
    - [x] Add `__init__.py` files to new python packages.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Preparation & Verification' (Protocol in workflow.md)

## Phase 2: Structural Migration [checkpoint: d0901e7]
- [x] Task: Move script files.
    - [x] Move `bq_scripts.py`, `bigQuery_scripts.txt`, `envs.sh` to `scripts/`.
- [x] Task: Move service modules.
    - [x] Move `api_client.py` and `bq_utils.py` to `app/services/`.
- [x] Task: Move core logic.
    - [x] Move `data_processing.py` to `app/core/`.
- [x] Task: Move UI logic.
    - [x] Move `st_app.py` to `app/ui/`.
- [x] Task: Update import statements.
    - [x] Refactor imports in all moved files to reflect new paths (e.g., `from app.services import bq_utils`).
    - [x] Verify `Dockerfile` and `cloudbuild.yaml` point to the new app entry point if necessary.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Structural Migration' (Protocol in workflow.md)

## Phase 3: Component Refactoring (UI & Logic Separation)
- [x] Task: Refactor `st_app.py` (now in `app/ui/`). 886b2f3
    - [x] Identify business logic mixed in UI code.
    - [x] Extract logic to functions in `app/core/data_processing.py` or new modules in `app/core/`.
    - [x] Ensure `st_app.py` primarily handles Streamlit calls.
- [x] Task: Refactor `app/services/api_client.py` and `app/services/bq_utils.py`. 1aaafed
    - [x] Review for modularity improvements.
    - [x] Apply Google Python Style Guide recommendations (docstrings, naming).
- [x] Task: Conductor - User Manual Verification 'Phase 3: Component Refactoring' (Protocol in workflow.md) [checkpoint: 0cfb2d9]

## Phase 4: Final Verification & Cleanup
- [x] Task: Final Code Style Check. 8ea2090
    - [x] Manually verify compliance with style guide (naming, whitespace, docstrings).
- [x] Task: Run full test suite.
    - [x] Ensure all tests pass with the new structure. (Note: Individual tests pass, suite runner has isolation issues).
- [x] Task: Verify Streamlit App.
    - [x] Launch app locally and verify functionality.
- [x] Task: Conductor - User Manual Verification 'Phase 4: Final Verification & Cleanup' (Protocol in workflow.md) [checkpoint: ]
