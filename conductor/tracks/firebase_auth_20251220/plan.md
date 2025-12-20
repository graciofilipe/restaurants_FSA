# Track Plan: Firebase Authentication Integration

## Phase 1: Firebase Project Setup and Configuration
- [x] Task: Initialize Firebase project and obtain configuration.
    - [x] Subtask: Create a Firebase project in the console (Manual step for user).
    - [x] Subtask: Enable Google Sign-In as an authentication provider.
    - [x] Subtask: Add a Web App to the project and copy the `firebaseConfig`.
- [x] Task: Securely store Firebase configuration in Streamlit.
    - [x] Subtask: Create/Update `.streamlit/secrets.toml` with Firebase configuration keys.
    - [x] Subtask: Verify configuration is accessible from Python.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Firebase Project Setup' (Protocol in workflow.md) [checkpoint: c529e18]

## Phase 2: Core Authentication Logic
- [x] Task: Create `auth/firebase_auth.py` and implement authentication primitives with TDD.
    - [x] Subtask: Write failing tests for user state detection (e.g., `is_user_authenticated`).
    - [x] Subtask: Implement Google Sign-In flow logic (using Firebase JS SDK via Streamlit components or a Python-friendly wrapper).
    - [x] Subtask: Implement session persistence logic ("Remember Me").
    - [x] Subtask: Verify tests pass and coverage >80%.
- [x] Task: Create a dedicated Login Page in Streamlit.
    - [x] Subtask: Write tests for the login page redirection logic.
    - [x] Subtask: Implement `login.py` (or a similar mechanism) to handle the Google Sign-In button and UI.
    - [x] Subtask: Verify tests pass.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Core Authentication Logic' (Protocol in workflow.md) [checkpoint: 509bba0]

## Phase 3: Application Integration and Access Control
- [x] Task: Implement global authentication check in `st_app.py`.
    - [x] Subtask: Update `st_app.py` to check authentication status at the start of `main_ui`.
    - [x] Subtask: Implement redirect to login page for unauthenticated users.
    - [x] Subtask: Update sidebar to display the authenticated user's email.
- [x] Task: Refactor existing IAP email logic.
    - [x] Subtask: Remove or conditionalize `get_iap_user_email` to prefer Firebase Authentication.
    - [x] Subtask: Verify that all protected subheaders (Fetch, Gemini, Export) remain inaccessible until login.
- [x] Task: Conductor - User Manual Verification 'Phase 3: Application Integration' (Protocol in workflow.md) [checkpoint: fdd59f7]

## Phase 4: Final Verification and Cleanup
- [x] Task: End-to-end testing of the authentication flow.
    - [x] Subtask: Verify sign-in, session persistence, and sign-out (if implemented).
    - [x] Subtask: Perform final code review and linting.
- [x] Task: Conductor - User Manual Verification 'Phase 4: Final Verification' (Protocol in workflow.md) [checkpoint: 31f2f11]

## Phase 5: Post-Implementation Fixes
- [x] Task: Fix Project ID mismatch in token verification. 4a4f9c6
    - [x] Subtask: Investigate `auth/firebase_auth.py` and `test_firebase_auth.py` to reproduce the issue.
    - [x] Subtask: Update token verification logic to support Firebase Project ID validation.
    - [x] Subtask: Verify tests pass.
