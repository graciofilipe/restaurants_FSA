# Plan: Remove Firebase Authentication

This plan outlines the steps to decouple Firebase Authentication from the FSA API Explorer, transitioning it to an open-access application with a simplified landing page.

## Phase 1: Preparation and Environment Cleanup
- [x] Task: Conductor - Remove authentication dependencies from `requirements.txt` d5cbb6c
- [ ] Task: Conductor - Clean up environment variables in `envs.sh` and `.streamlit/secrets.toml`
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Preparation and Environment Cleanup' (Protocol in workflow.md)

## Phase 2: Backend Logic Decoupling
- [ ] Task: Conductor - Remove `@login_required` decorators and auth-gating logic in `st_app.py`
- [ ] Task: Conductor - Remove auth verification logic and session state management related to Firebase
- [ ] Task: Conductor - Delete or deprecate the `auth/` directory and `login.py`
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Backend Logic Decoupling' (Protocol in workflow.md)

## Phase 3: UI Transformation
- [ ] Task: Conductor - Modify landing page to replace "Login" with "Enter App" button
- [ ] Task: Conductor - Implement state transition from landing page to main dashboard via "Enter App"
- [ ] Task: Conductor - Remove "Sign Out" and user profile elements from the UI
- [ ] Task: Conductor - User Manual Verification 'Phase 3: UI Transformation' (Protocol in workflow.md)

## Phase 4: Verification and Finalization
- [ ] Task: Conductor - Run full test suite to ensure core functionality (data fetching, BQ) is intact
- [ ] Task: Conductor - Verify local run without Firebase credentials
- [ ] Task: Conductor - Verify `cloudbuild.yaml` and `Dockerfile` successfully build
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Verification and Finalization' (Protocol in workflow.md)
