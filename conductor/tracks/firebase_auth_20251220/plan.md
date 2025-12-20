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
- [ ] Task: Create `auth/firebase_auth.py` and implement authentication primitives with TDD.
    - [ ] Subtask: Write failing tests for user state detection (e.g., `is_user_authenticated`).
    - [ ] Subtask: Implement Google Sign-In flow logic (using Firebase JS SDK via Streamlit components or a Python-friendly wrapper).
    - [ ] Subtask: Implement session persistence logic ("Remember Me").
    - [ ] Subtask: Verify tests pass and coverage >80%.
- [ ] Task: Create a dedicated Login Page in Streamlit.
    - [ ] Subtask: Write tests for the login page redirection logic.
    - [ ] Subtask: Implement `login.py` (or a similar mechanism) to handle the Google Sign-In button and UI.
    - [ ] Subtask: Verify tests pass.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Core Authentication Logic' (Protocol in workflow.md)

## Phase 3: Application Integration and Access Control
- [ ] Task: Implement global authentication check in `st_app.py`.
    - [ ] Subtask: Update `st_app.py` to check authentication status at the start of `main_ui`.
    - [ ] Subtask: Implement redirect to login page for unauthenticated users.
    - [ ] Subtask: Update sidebar to display the authenticated user's email.
- [ ] Task: Refactor existing IAP email logic.
    - [ ] Subtask: Remove or conditionalize `get_iap_user_email` to prefer Firebase Authentication.
    - [ ] Subtask: Verify that all protected subheaders (Fetch, Gemini, Export) remain inaccessible until login.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Application Integration' (Protocol in workflow.md)

## Phase 4: Final Verification and Cleanup
- [ ] Task: End-to-end testing of the authentication flow.
    - [ ] Subtask: Verify sign-in, session persistence, and sign-out (if implemented).
    - [ ] Subtask: Perform final code review and linting.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Final Verification' (Protocol in workflow.md)
