# Plan: Remove Maps URL Generation

This plan outlines the steps to remove the Google Maps URL generation logic and UI components from the FSA API Explorer.

## Phase 1: Logic Removal and Code Cleanup [checkpoint: 40b5f96]
- [x] Task: Conductor - Remove `generate_maps_url` usage in `data_processing.py` 22912d7
- [x] Task: Conductor - Delete `utils/url_generator.py` and its directory if empty ae75d52
- [x] Task: Conductor - Delete `test_url_generator.py` ae75d52
- [x] Task: Conductor - User Manual Verification 'Phase 1: Logic Removal and Code Cleanup' (Protocol in workflow.md) ae75d52

## Phase 2: UI Transformation
- [ ] Task: Conductor - Remove "Maps Link" column configuration in `st_app.py`
- [ ] Task: Conductor - Remove "Maps Link" from any display logic in `st_app.py`
- [ ] Task: Conductor - User Manual Verification 'Phase 2: UI Transformation' (Protocol in workflow.md)

## Phase 3: Verification and Finalization
- [ ] Task: Conductor - Run full test suite to ensure no regressions
- [ ] Task: Conductor - Verify application starts and fetches data locally
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Verification and Finalization' (Protocol in workflow.md)
