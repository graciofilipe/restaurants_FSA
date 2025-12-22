# Track Plan: Firebase Auth Fixes & Hardening

## Phase 1: Diagnostics and Error Handling
- [x] Task: Improve Frontend Error Handling.
    - [x] Subtask: Update `login_button` JS code in `auth/firebase_auth.py` to catch `auth/unauthorized-domain` specifically and show a helpful alert (e.g., "Domain not authorized. Add to Firebase Console.").
- [x] Task: Improve Backend Logging.
    - [x] Subtask: Wrap `firebase_auth.verify_id_token` in a try/catch block that logs the *exact* error message (including audience mismatch details) to the Streamlit UI or console for debugging.

## Phase 2: Fix Backend Token Verification
- [x] Task: Enforce explicit Project ID in Admin SDK.
    - [x] Subtask: Verify `firebase_admin.initialize_app` logic. Ensure it's not being overridden by ADC (Application Default Credentials) environmental discovery.
    - [x] Subtask: If necessary, pass the `serviceAccountId` or explicit credentials if `projectId` alone isn't sufficient (though `projectId` in `options` *should* work for `verify_id_token` audience checks).
    - [x] Subtask: Test by simulating a token (or using the real flow if possible).

## Phase 3: Documentation and Verification
- [x] Task: Update Documentation.
    - [x] Subtask: Add a section to `README.md` or a new `setup_guide.md` about adding the Cloud Run domain to Firebase Authorized Domains.
- [ ] Task: Conductor - User Manual Verification (Protocol in workflow.md).