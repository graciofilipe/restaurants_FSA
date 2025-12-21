# Track Specification: Firebase Auth Fixes & Hardening

## Context
Two critical issues have been identified in the current Firebase Authentication implementation:

1.  **"auth/unauthorized-domain" Error:**
    *   **Symptom:** Users attempting to sign in via the Google Sign-In popup receive an `auth/unauthorized-domain` error in the browser console/alert.
    *   **Cause:** The Cloud Run domain (e.g., `podcast-recommender-web-*.run.app`) is not whitelisted in the Firebase Console's "Authorized Domains" list.
    *   **Resolution:** This requires a manual configuration step in the Firebase Console, but we can improve the error handling in the application to guide the user.

2.  **Token Verification Failure (Project ID Mismatch):**
    *   **Symptom:** After a successful frontend sign-in, the backend `verify_id_token` call fails, likely with an "Audience mismatch" or similar error.
    *   **Cause:** The application runs on Google Cloud Project A (`filipegracio-ai-learning`), but the Firebase Authentication is configured for Project B (`podcast-ce10c`). The `firebase_admin` SDK might be auto-discovering the GCP environment's project ID (A) instead of using the configured Firebase project ID (B) for token validation.
    *   **Resolution:** Ensure `firebase_admin` is strictly initialized with the credential/project ID from `st.secrets` and that `verify_id_token` validates against the correct audience.

## Goals
1.  Resolve the "unauthorized-domain" error by documenting the fix and/or improving UI feedback.
2.  Fix the backend token verification logic to support the cross-project scenario (Cloud Run on Project A, Firebase on Project B).
3.  Add robustness to the `AuthManager` to handle and log these errors more clearly.

## Technical Details
*   **File:** `auth/firebase_auth.py`
*   **Library:** `firebase-admin` python SDK, Firebase JS SDK (v9).
