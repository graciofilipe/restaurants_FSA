# Track Plan: Remove Firebase Authentication

## Phase 1: Codebase Cleanup
- [x] Task: Remove Firebase dependencies.
    - [x] Subtask: Remove `firebase-admin` and `extra-streamlit-components` from `requirements.txt`.
    - [x] Subtask: Update `Dockerfile` if necessary (usually just a rebuild, but check for specific install commands).
- [x] Task: Remove Authentication Logic Modules.
    - [x] Subtask: Delete `auth/` directory.
    - [x] Subtask: Delete `login.py`.
- [x] Task: Conductor - User Manual Verification (Protocol in workflow.md).

## Phase 2: Application Logic Reversion
- [x] Task: Update `st_app.py` to remove authentication.
    - [x] Subtask: Remove imports `AuthManager` and `login_page`.
    - [x] Subtask: Remove `auth_popup_handler`.
    - [x] Subtask: Remove `main_ui` auth checks and sidebar login info.
    - [x] Subtask: Verify `main_ui` allows direct access to functionality.
- [x] Task: Clean up Configuration.
    - [x] Subtask: Remove `[firebase]` section from `.streamlit/secrets.toml`.
- [x] Task: Conductor - User Manual Verification (Protocol in workflow.md).

## Phase 3: Test Cleanup and Verification
- [x] Task: Remove Auth Tests.
    - [x] Subtask: Delete `test_app_auth.py`, `test_auth_fix.py`, `test_firebase_auth.py`.
    - [x] Subtask: Update `test_st_app.py` if it mocks auth.
- [x] Task: Final Verification.
    - [x] Subtask: Run full test suite.
    - [x] Subtask: Run app locally to confirm no startup errors.
- [ ] Task: Conductor - User Manual Verification (Protocol in workflow.md).

## Phase 3: Test Cleanup and Verification
- [ ] Task: Remove Auth Tests.
    - [ ] Subtask: Delete `test_app_auth.py`, `test_auth_fix.py`, `test_firebase_auth.py`.
    - [ ] Subtask: Update `test_st_app.py` if it mocks auth.
- [ ] Task: Final Verification.
    - [ ] Subtask: Run full test suite.
    - [ ] Subtask: Run app locally to confirm no startup errors.
- [ ] Task: Conductor - User Manual Verification (Protocol in workflow.md).