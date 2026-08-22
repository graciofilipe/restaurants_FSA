Last visited: 2026-07-22T09:35:00Z
- Codebase Simplification & Bloat Elimination: Refactored and streamlined `app/services/bq_utils.py` (from 691 to 345 lines), `app/core/data_processing.py` (from 317 to 191 lines), `scripts/enrich_maps_data.py` (from 202 to 96 lines), and migration scripts. Removed 8 orphaned scratch scripts from the root directory, achieving a net reduction of ~993 lines of duplicate boilerplate.
- AI Model Standards: Aligned all ADK agents and test assertions to `gemini-3.7-flash`.
- Resilient Telemetry & Test Isolation: Refactored `fast_api_app.py` with graceful Google Cloud Logging and Auth fallbacks, standardized integration test routes, and mocked BQ client in unit tests.
- 100% Test Pass Rate: Verified all 97 tests across both `app/` (80/80 passed) and `tests/` (17/17 passed) suites.
