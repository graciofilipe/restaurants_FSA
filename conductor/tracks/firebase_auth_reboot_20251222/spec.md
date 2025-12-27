# Specification: Firebase Authentication from Scratch

## Overview
Implement a robust Google Sign-In authentication flow using Firebase for the Streamlit application. This track aims to replace the previous problematic implementation with a clean, reliable, and secure system that restricts all application content to authenticated users.

## Functional Requirements
1. **Immediate Redirection:** Any unauthenticated access attempt must be immediately redirected to a dedicated login page. No app content should be visible to unauthenticated users.
2. **Google Sign-In:** Users must authenticate using their Google accounts via Firebase.
3. **Session Management:** The application must maintain user session state.
4. **Access Control:** All core features (Fetch Data, Gemini Analysis, Export, Bulk Update) must be protected and only visible/executable after successful login.
5. **User Identity:** The authenticated user's email should be displayed in the sidebar, along with a "Sign Out" option.

## Non-Functional Requirements
1. **Security:** Securely handle configuration via Streamlit secrets. Use Firebase ID tokens for backend verification.
2. **User Experience:** Provide a clean, branded login experience.
3. **Robustness:** Implement clear error handling for failed authentication attempts.

## Acceptance Criteria
1. Application starts and immediately shows a login page for unauthenticated users.
2. Clicking "Login with Google" initiates the Firebase authentication flow.
3. Upon successful login, the user is redirected to the main application UI.
4. The sidebar correctly identifies the logged-in user by email.
5. "Sign Out" successfully clears the session and returns the user to the login page.
6. The application is successfully deployed to and functional on Cloud Run.

## Out of Scope
- Implementing other providers (Email/Password, GitHub, etc.).
- Persistent user database in Firestore (unless required for session sync).
