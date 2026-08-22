# Original User Request

## Initial Request — 2026-06-08T09:04:00Z

# Teamwork Project Prompt — Draft

> Status: Step 2 — Identifying ambiguity
> Goal: Craft prompt → get user approval → delegate to teamwork_preview

Refactor the restaurant profiling system to feature modular, decomposable pipelines for Gemini profiles, Google Maps data, and ML score predictions. Create a unified prediction trigger that automatically executes missing upstream enrichments ("just in time" fetching) and supports a user-defined option to forcefully regenerate existing profiles before generating the final ML prediction.

Working directory: /usr/local/google/home/filipegracio/restaurants/restaurants_FSA

## Requirements

### R1. Modular Pipelines
Refactor the codebase to ensure that the Gemini profiling, Google Maps enrichment, and ML prediction workflows exist as distinct, independently callable modular functions or classes.

### R2. Unified "Just-In-Time" Prediction Workflow
Implement a unified ML prediction workflow that accepts a group of restaurants. Before executing the ML prediction, it must intelligently evaluate the features of each restaurant. If a restaurant lacks a Google Maps profile or a Gemini insights profile, the workflow must automatically trigger the respective modular pipeline to generate it.

### R3. Granular Force Regeneration Option
Expose granular UI options (e.g., checkboxes for "Force Regenerate Maps Data" and "Force Regenerate Gemini Profiles") that allow the user to explicitly force the re-generation and overwriting of existing profiles before the ML prediction occurs.

## Acceptance Criteria

### Workflow Verification
- [ ] A programmatic `pytest` test suite exists that verifies the "Just-In-Time" orchestration logic.
- [ ] The test suite uses mocking (e.g., `unittest.mock`) to prove that upstream pipelines are correctly skipped when data exists, correctly triggered when data is missing, and explicitly triggered when the force regeneration flags are passed, all without making live API/network calls.
- [ ] The Streamlit UI correctly renders the granular force regeneration checkboxes alongside the prediction action.

<latest_turn_user_message>
proceed
</latest_turn_user_message>
