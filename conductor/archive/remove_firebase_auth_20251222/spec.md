# Specification: Remove Firebase Authentication

## Goal
To remove all traces of Firebase Authentication from the application, reverting it to a state where access is unrestricted (or relies on IAP if previously configured, but primarily "unauthenticated" from an app-code perspective).

## Scope
- **Frontend (`st_app.py`):** Remove all authentication checks, login page redirection, and UI elements related to user identity (except potential IAP fallback if simple).
- **Backend/Logic (`auth/`):** Delete the entire `auth` directory and `login.py`.
- **Configuration:** Remove Firebase keys from `.streamlit/secrets.toml`.
- **Dependencies:** Remove `firebase-admin` and `extra-streamlit-components` (if used only for auth) from `requirements.txt` and `Dockerfile`.
- **Testing:** Remove all tests related to authentication.

## Success Criteria
1. Application starts without errors.
2. No "Login" page is presented; users go straight to the main app.
3. No lingering `firebase-admin` imports or initialization.
4. `requirements.txt` is clean.
5. All tests pass (after removing auth tests).