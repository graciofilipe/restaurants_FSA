# Scope: Model Training Trigger UI

## Architecture
- Streamlit frontend (app/ui/st_app.py) adds a button in sidebar/settings.
- `scripts/train_bqml_model.py` needs to support an async fire-and-forget execution (returning the BQ job ID or similar without blocking).

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | BQML Async Trigger | Update `scripts/train_bqml_model.py` and `app/ui/st_app.py` | none | PLANNED |
