"""Pure runtime build stamp and live telemetry formatting helpers."""
import datetime
import os
import subprocess
from typing import Any, Dict, Mapping, Optional

_PROCESS_START_UTC = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _detect_local_git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=7", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=2.0,
            text=True,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def get_runtime_build_stamp(env: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """Return the container/process build and Cloud Run revision stamp.

    In Cloud Run + Cloud Build:
      - `APP_BUILD_TIMESTAMP` is baked into the Docker image at build time.
      - `APP_COMMIT_SHA` is the Git commit SHA baked at build time.
      - `K_REVISION` and `K_SERVICE` are injected automatically by Cloud Run.
    In local development, falls back to `git rev-parse --short=7 HEAD` and the
    local process start time.
    """
    source = env if env is not None else os.environ
    raw_sha = (source.get("APP_COMMIT_SHA") or "").strip()
    if raw_sha:
        commit_sha = raw_sha[:7]
    else:
        commit_sha = _detect_local_git_sha()

    build_ts = (source.get("APP_BUILD_TIMESTAMP") or "").strip() or f"{_PROCESS_START_UTC} (local)"
    revision = (source.get("K_REVISION") or "").strip() or "local-dev"
    service = (source.get("K_SERVICE") or "").strip() or "restaurants-fsa"

    return {
        "build_timestamp": build_ts,
        "commit_sha": commit_sha,
        "revision": revision,
        "service": service,
    }


def _coerce_utc_datetime(value: Any) -> Optional[datetime.datetime]:
    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=datetime.timezone.utc)
        return value.astimezone(datetime.timezone.utc)
    if isinstance(value, datetime.date):
        return datetime.datetime(value.year, value.month, value.day, tzinfo=datetime.timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=datetime.timezone.utc)
            return parsed.astimezone(datetime.timezone.utc)
        except ValueError:
            return None
    return None


def format_timestamp_utc(value: Any) -> str:
    """Format a timestamp/datetime as 'YYYY-MM-DD HH:MM UTC' or 'unknown'."""
    dt = _coerce_utc_datetime(value)
    if dt is None:
        return "unknown"
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def format_relative_age(
    value: Any,
    now: Optional[datetime.datetime] = None,
) -> str:
    """Return a concise relative age string like '14m ago', '3h ago', or '11d ago'."""
    dt = _coerce_utc_datetime(value)
    if dt is None:
        return "unknown"
    ref = _coerce_utc_datetime(now) or datetime.datetime.now(datetime.timezone.utc)
    delta_seconds = max(0.0, (ref - dt).total_seconds())
    minutes = int(delta_seconds // 60)
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 48:
        return f"{hours}h ago"
    days = hours // 24
    return f"{days}d ago"


def format_top_status_bar(
    runtime_stamp: Mapping[str, str],
    diagnostics: Optional[Mapping[str, Any]] = None,
    now: Optional[datetime.datetime] = None,
) -> str:
    """Render a compact 1-line status bar with deploy, model, drift, and FSA freshness."""
    build_ts = runtime_stamp.get("build_timestamp", "unknown")
    commit_sha = runtime_stamp.get("commit_sha", "unknown")
    revision = runtime_stamp.get("revision", "local-dev")
    parts = [f"🚀 **Deploy:** {build_ts} (`{commit_sha}` · `{revision}`)"]

    diag = diagnostics or {}
    model_ts = diag.get("model_trained_at")
    if model_ts is not None:
        ts_str = format_timestamp_utc(model_ts)
        age_str = format_relative_age(model_ts, now=now)
        ver = diag.get("vertex_version")
        ver_badge = f"`v{ver}` " if ver else ""
        metrics_bits = [age_str]
        mae = diag.get("mae")
        r2 = diag.get("r_squared")
        if isinstance(mae, (int, float)):
            metrics_bits.append(f"MAE {mae:.2f}")
        if isinstance(r2, (int, float)):
            metrics_bits.append(f"R² {r2:.2f}")
        parts.append(f"🧠 **Model:** {ver_badge}{ts_str} ({' · '.join(metrics_bits)})")
    else:
        parts.append("🧠 **Model:** unknown")

    current_preds = diag.get("current_predictions")
    stale_preds = diag.get("stale_predictions")
    unscored = diag.get("unscored_in_scope")
    if current_preds is not None and stale_preds is not None:
        drift_str = f"🎯 **Predictions:** {int(current_preds):,} current / {int(stale_preds):,} stale"
        if unscored is not None:
            drift_str += f" / {int(unscored):,} unscored"
        parts.append(drift_str)

    latest_fsa = diag.get("latest_first_seen")
    if latest_fsa:
        parts.append(f"📥 **FSA Newest:** {latest_fsa}")

    return "  |  ".join(parts)
