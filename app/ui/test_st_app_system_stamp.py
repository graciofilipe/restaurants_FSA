"""Unit tests for the live system status bar, sidebar diagnostics, and model feature importance UI."""
import datetime
import unittest
from unittest.mock import MagicMock, patch


class TestSystemStampAndDiagnosticsUi(unittest.TestCase):

    def test_render_system_status_bar_writes_caption_with_deploy_and_model(self):
        from app.ui import st_app

        diag = {
            "model_trained_at": datetime.datetime(2026, 10, 4, 6, 14, tzinfo=datetime.timezone.utc),
            "vertex_version": "18",
            "mae": 0.16,
            "r_squared": 0.98,
            "current_predictions": 50,
            "stale_predictions": 10,
            "unscored_in_scope": 5,
            "latest_first_seen": "2026-09-28",
        }
        with patch.object(st_app, "st") as mock_st:
            st_app.render_system_status_bar(diag)

        mock_st.caption.assert_called_once()
        caption_text = mock_st.caption.call_args.args[0]
        self.assertIn("Deploy:", caption_text)
        self.assertIn("`v18`", caption_text)
        self.assertIn("50 current / 10 stale / 5 unscored", caption_text)

    def test_render_sidebar_diagnostics_refresh_button_clears_cache_and_reruns(self):
        from app.ui import st_app

        diag = {
            "model_trained_at": datetime.datetime(2026, 10, 4, 6, 14, tzinfo=datetime.timezone.utc),
            "vertex_version": "18",
            "mae": 0.16,
            "r_squared": 0.98,
            "in_scope_rows": 2000,
            "current_predictions": 100,
            "stale_predictions": 1900,
            "unscored_in_scope": 0,
            "labeled_rows": 444,
            "table_modified_at": datetime.datetime(2026, 10, 4, 6, 15, tzinfo=datetime.timezone.utc),
            "latest_first_seen": "2026-09-28",
            "gemini_profiled_in_scope": 1950,
            "oldest_gemini_at": datetime.datetime(2026, 9, 23, 13, 0, tzinfo=datetime.timezone.utc),
            "newest_gemini_at": datetime.datetime(2026, 10, 3, 19, 0, tzinfo=datetime.timezone.utc),
            "maps_checked_in_scope": 1980,
            "oldest_maps_at": datetime.datetime(2026, 10, 3, 19, 0, tzinfo=datetime.timezone.utc),
            "newest_maps_at": datetime.datetime(2026, 10, 4, 6, 3, tzinfo=datetime.timezone.utc),
            "feature_importance": [
                {"feature": "maps_types_array", "gain_pct": 43.9, "importance_gain": 82.55},
                {"feature": "match_score", "gain_pct": 25.8, "importance_gain": 48.57},
            ],
        }
        with patch.object(st_app, "st") as mock_st, \
             patch.object(st_app, "clear_diagnostics_cache") as mock_clear:
            mock_st.button.return_value = True
            st_app.render_sidebar_diagnostics("p", "d", "t", "m", diagnostics=diag)

        mock_clear.assert_called_once()
        mock_st.rerun.assert_called_once()
        captions = [c.args[0] for c in mock_st.caption.call_args_list if c.args]
        self.assertTrue(any("maps_types_array" in c for c in captions))
        self.assertTrue(any("444 rated" in c for c in captions))

    def test_render_feature_importance_section_displays_dataframe_in_training_tab(self):
        from app.ui import st_app

        diag = {
            "model_trained_at": datetime.datetime(2026, 10, 4, 6, 14, tzinfo=datetime.timezone.utc),
            "vertex_version": "18",
            "feature_importance": [
                {
                    "feature": "maps_types_array",
                    "gain_pct": 43.9,
                    "importance_gain": 82.55,
                    "importance_weight": 123,
                    "importance_cover": 6952.4,
                },
                {
                    "feature": "match_score",
                    "gain_pct": 25.8,
                    "importance_gain": 48.57,
                    "importance_weight": 40,
                    "importance_cover": 222.6,
                },
            ],
        }
        with patch.object(st_app, "st") as mock_st:
            st_app.render_feature_importance_section(diagnostics=diag)

        mock_st.subheader.assert_called_once()
        self.assertIn("Model Feature Importance", mock_st.subheader.call_args.args[0])
        mock_st.dataframe.assert_called_once()
        df_rendered = mock_st.dataframe.call_args.args[0]
        self.assertEqual(list(df_rendered["feature"]), ["maps_types_array", "match_score"])

    def test_batch_size_slider_allows_up_to_1000(self):
        with open("app/ui/st_app.py", "r") as f:
            content = f.read()
        self.assertIn(
            'st.slider("Batch Size (Budget of Restaurants to Score)", min_value=5, max_value=1000, value=25, step=5, key="batch_pred_limit")',
            content,
        )


if __name__ == "__main__":
    unittest.main()

