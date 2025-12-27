# Specification: Remove Firebase Authentication

## 1. Overview
The goal of this track is to completely remove the Firebase Authentication layer from the FSA API Explorer application. The application will transition from a secure, login-gated tool to an open-access application. The user experience will be simplified to allow direct entry into the application via a "Enter App" button on the landing page, removing the need for credentials.

## 2. Functional Requirements

### 2.1 Authentication Removal
- **Dependencies:** Remove `firebase-admin`, `google-auth`, and any other auth-specific libraries from `requirements.txt`.
- **Environment:** Clean up `envs.sh` and `.streamlit/secrets.toml` by removing Firebase-related configuration keys (e.g., `FIREBASE_CREDENTIALS`, `FIREBASE_API_KEY`).
- **Application Logic:**
    -   Remove all `@login_required` decorators (or equivalent checks) from the codebase.
    -   Remove any middleware or functions responsible for verifying authentication tokens or session state.
    -   Remove `auth/` directory or specific auth-handling modules if they serve no other purpose.

### 2.2 User Interface Updates
- **Landing Page:**
    -   Retain the existing "Landing Page" concept (i.e., the app does not immediately show data upon load).
    -   **Change:** Replace the "Login with Google" (or similar) button/form with a single, clear "Enter App" button.
    -   **Behavior:** Clicking "Enter App" should immediately transition the user to the main application dashboard (`st_app.py` core functionality).
- **Navigation:** Remove any "Sign Out" or "Logout" buttons from the sidebar or main interface.

## 3. Non-Functional Requirements
- **Deployment Stability:** The removal of authentication must **not** break the existing deployment pipeline (`Dockerfile`, `cloudbuild.yaml`) or the core application functionality (fetching data, querying BigQuery).
- **Performance:** The application load time should theoretically decrease slightly due to the removal of auth checks.

## 4. Acceptance Criteria
- [ ] Application starts successfully locally without requiring Firebase credentials.
- [ ] No `ImportError` or runtime errors related to missing auth libraries.
- [ ] Visiting the root URL displays a landing page with an "Enter App" button.
- [ ] Clicking "Enter App" grants full access to the restaurant discovery tools and BigQuery data.
- [ ] No "Login" or "Sign Out" UI elements remain visible.
- [ ] `cloudbuild.yaml` and `Dockerfile` successfully build and deploy the app in its new state.

## 5. Out of Scope
- Major redesign of the main dashboard.
- Changes to BigQuery schemas or data fetching logic (unless strictly coupled to user ID, which is not expected based on current context).
