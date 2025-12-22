# Track Plan: Firebase Authentication from Scratch

## Phase 1: Firebase & GCP Project Initialization [checkpoint: 332bc78]
- [x] Task: Interactive Setup Guide for Firebase Console. [d2f52b9]
    - [x] Subtask: Guide user to create/select a Firebase project.
    - [x] Subtask: Guide user to enable Google Sign-In as a provider.
    - [x] Subtask: Guide user to create a Firebase Web App and retrieve the configuration.
- [x] Task: Secure Configuration Storage. [d2f52b9]
    - [x] Subtask: Update `.streamlit/secrets.toml` with the new Firebase configuration.
    - [x] Subtask: Verify the app can read the configuration without errors.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Project Initialization' (Protocol in workflow.md) [checkpoint: 332bc78]

## Phase 2: Authentication Primitives (TDD)
- [~] Task: Implement Core Authentication Logic.
    - [ ] Subtask: Write failing tests for token verification and session state.
    - [ ] Subtask: Implement the backend logic to verify Firebase ID tokens.
    - [ ] Subtask: Implement session persistence and logout logic.
    - [ ] Subtask: Verify all authentication tests pass with >80% coverage.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Authentication Primitives' (Protocol in workflow.md)

## Phase 2: Authentication Primitives (TDD)
- [ ] Task: Implement Core Authentication Logic.
    - [ ] Subtask: Write failing tests for token verification and session state.
    - [ ] Subtask: Implement the backend logic to verify Firebase ID tokens.
    - [ ] Subtask: Implement session persistence and logout logic.
    - [ ] Subtask: Verify all authentication tests pass with >80% coverage.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Authentication Primitives' (Protocol in workflow.md)

## Phase 3: UI Integration and Access Control
- [ ] Task: Implement the Login Experience.
    - [ ] Subtask: Create a dedicated login page with a "Sign in with Google" button.
    - [ ] Subtask: Integrate the Firebase JS SDK for the frontend authentication flow.
- [ ] Task: Enforce Global Access Control.
    - [ ] Subtask: Update `st_app.py` to redirect unauthenticated users to the login page.
    - [ ] Subtask: Update the sidebar to display the user's email and a "Sign Out" button.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: UI Integration' (Protocol in workflow.md)

## Phase 4: Cloud Run Deployment & Final Verification
- [ ] Task: Production Deployment.
    - [ ] Subtask: Build and deploy the updated container to Cloud Run.
    - [ ] Subtask: Guide the user to add the Cloud Run URL to the Firebase "Authorized Domains" list.
- [ ] Task: End-to-End Verification.
    - [ ] Subtask: Perform a final manual test of the entire login/logout flow on the live URL.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Cloud Run Deployment' (Protocol in workflow.md)
