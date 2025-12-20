# Specification: Firebase Authentication (Google Sign-In)

## Overview
Implement Firebase Authentication using Google Sign-In to secure the FSA API Explorer application. This ensures that only authorized users can access the restaurant discovery features and BigQuery data integration.

## Functional Requirements
1.  **Google Sign-In Integration:**
    -   Integrate the Firebase SDK into the Streamlit frontend.
    -   Provide a "Sign in with Google" option.
2.  **Access Control:**
    -   Implement a redirect flow that forces unauthenticated users to a dedicated login page.
    -   Secure all main application features (Fetching data, Gemini analysis, Export, Bulk updates) behind the authentication wall.
3.  **Session Management:**
    -   Support session persistence with an optional "Remember Me" configuration to allow users to stay logged in across browser sessions.
4.  **User Identity:**
    -   Display the authenticated user's email in the sidebar (replacing the current IAP-based email display logic where appropriate).

## Non-Functional Requirements
-   **Security:** Ensure Firebase configuration (API keys, etc.) is handled securely (e.g., via environment variables or Streamlit secrets).
-   **Performance:** The authentication check should be fast and not significantly delay the application startup.

## Acceptance Criteria
-   The application redirect to a login page if no user is signed in.
-   A user can successfully sign in using their Google account.
-   Once signed in, the user has full access to the application's features.
-   The user's email is correctly displayed in the sidebar.
-   The user can choose to stay logged in or require re-authentication for future sessions.

## Out of Scope
-   Implementing other authentication providers (Email, Phone, etc.) beyond Google Sign-In.
-   Fine-grained Role-Based Access Control (RBAC) beyond simple authentication.
