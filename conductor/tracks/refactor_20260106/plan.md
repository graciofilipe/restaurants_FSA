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
- [ ] Task: Refactor `st_app.py` (now in `app/ui/`).
    - [ ] Identify business logic mixed in UI code.
    - [ ] Extract logic to functions in `app/core/data_processing.py` or new modules in `app/core/`.
    - [ ] Ensure `st_app.py` primarily handles Streamlit calls.
- [ ] Task: Refactor `app/services/api_client.py` and `app/services/bq_utils.py`.
    - [ ] Review for modularity improvements.
    - [ ] Apply Google Python Style Guide recommendations (docstrings, naming).
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Component Refactoring' (Protocol in workflow.md)

## Phase 4: Final Verification & Cleanup
- [ ] Task: Final Code Style Check.
    - [ ] Manually verify compliance with style guide (naming, whitespace, docstrings).
- [ ] Task: Run full test suite.
    - [ ] Ensure all tests pass with the new structure.
    - [ ] Fix any broken tests due to path changes.
- [ ] Task: Verify Streamlit App.
    - [ ] Launch app locally and verify functionality.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Final Verification & Cleanup' (Protocol in workflow.md)
