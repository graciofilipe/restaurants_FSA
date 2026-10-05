"""Unit tests for app/core/system_stamp.py."""
import datetime
from app.core.system_stamp import (
    format_relative_age,
    format_timestamp_utc,
    format_top_status_bar,
    get_runtime_build_stamp,
)


class TestRuntimeBuildStamp:

    def test_reads_cloud_run_and_build_env_variables(self):
        env = {
            "APP_COMMIT_SHA": "01715bd1d1f50868b0889da70afabcaee7b9a54a",
            "APP_BUILD_TIMESTAMP": "2026-10-04 05:51 UTC",
            "K_REVISION": "restaurants-fsa-00240-jsd",
            "K_SERVICE": "restaurants-fsa",
        }
        stamp = get_runtime_build_stamp(env=env)
        assert stamp["commit_sha"] == "01715bd"
        assert stamp["build_timestamp"] == "2026-10-04 05:51 UTC"
        assert stamp["revision"] == "restaurants-fsa-00240-jsd"
        assert stamp["service"] == "restaurants-fsa"

    def test_falls_back_cleanly_in_local_dev(self):
        stamp = get_runtime_build_stamp(env={})
        assert len(stamp["commit_sha"]) >= 4
        assert "UTC" in stamp["build_timestamp"]
        assert stamp["revision"] == "local-dev"


class TestTimestampAndRelativeAgeFormatting:

    def test_format_timestamp_utc(self):
        dt = datetime.datetime(2026, 10, 4, 6, 14, 29, tzinfo=datetime.timezone.utc)
        assert format_timestamp_utc(dt) == "2026-10-04 06:14 UTC"
        assert format_timestamp_utc("2026-10-04T06:14:29Z") == "2026-10-04 06:14 UTC"
        assert format_timestamp_utc(None) == "unknown"

    def test_format_relative_age_units(self):
        now = datetime.datetime(2026, 10, 4, 12, 0, 0, tzinfo=datetime.timezone.utc)
        assert format_relative_age(now - datetime.timedelta(seconds=20), now=now) == "just now"
        assert format_relative_age(now - datetime.timedelta(minutes=25), now=now) == "25m ago"
        assert format_relative_age(now - datetime.timedelta(hours=5), now=now) == "5h ago"
        assert format_relative_age(now - datetime.timedelta(days=11), now=now) == "11d ago"


class TestFormatTopStatusBar:

    def test_includes_deploy_model_drift_and_fsa_newest(self):
        now = datetime.datetime(2026, 10, 4, 6, 40, 0, tzinfo=datetime.timezone.utc)
        trained_at = datetime.datetime(2026, 10, 4, 6, 14, 0, tzinfo=datetime.timezone.utc)
        runtime_stamp = {
            "build_timestamp": "2026-10-04 05:51 UTC",
            "commit_sha": "01715bd",
            "revision": "restaurants-fsa-00240-jsd",
        }
        diag = {
            "model_trained_at": trained_at,
            "vertex_version": "18",
            "mae": 0.1601,
            "r_squared": 0.9835,
            "current_predictions": 120,
            "stale_predictions": 1821,
            "unscored_in_scope": 45,
            "latest_first_seen": "2026-09-28",
        }

        bar = format_top_status_bar(runtime_stamp, diag, now=now)
        assert "2026-10-04 05:51 UTC (`01715bd` · `restaurants-fsa-00240-jsd`)" in bar
        assert "`v18` 2026-10-04 06:14 UTC (26m ago · MAE 0.16 · R² 0.98)" in bar
        assert "120 current / 1,821 stale / 45 unscored" in bar
        assert "2026-09-28" in bar

    def test_includes_companion_linear_model_metrics_when_present(self):
        now = datetime.datetime(2026, 10, 5, 8, 31, 0, tzinfo=datetime.timezone.utc)
        trained_at = datetime.datetime(2026, 10, 5, 7, 31, 0, tzinfo=datetime.timezone.utc)
        runtime_stamp = {
            "build_timestamp": "2026-10-05 08:00 UTC",
            "commit_sha": "4fdc481",
            "revision": "restaurants-fsa-00245-p24",
        }
        diag = {
            "model_trained_at": trained_at,
            "vertex_version": "21",
            "mae": 0.96,
            "r_squared": 0.42,
            "linear_mae": 0.70,
            "linear_r_squared": 0.56,
            "current_predictions": 3333,
            "stale_predictions": 0,
            "unscored_in_scope": 0,
        }

        bar = format_top_status_bar(runtime_stamp, diag, now=now)
        assert "`v21` 2026-10-05 07:31 UTC (1h ago · MAE 0.96 · R² 0.42 · Lin MAE 0.70 · Lin R² 0.56)" in bar
