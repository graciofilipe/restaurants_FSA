"""Guards on the one predicate that decides whether Gemini gets paid again.

Two surfaces answer "does this restaurant need a profile?": the UI's
"Estimated New Gemini Calls" and the enrichment the Predict button actually
triggers. They used to answer it separately, and the estimate was wrong --
that is D1. These tests pin both to one function.
"""
import datetime
import pathlib
import unittest

import pandas as pd

from app.core.profile_freshness import (
    GEMINI_PROFILE_MAX_AGE_DAYS,
    count_needing_gemini_profile,
    needs_gemini_profile,
)

NOW = datetime.datetime(2026, 9, 23, 12, 0, tzinfo=datetime.timezone.utc)
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def days_ago(n):
    return NOW - datetime.timedelta(days=n)


class TestNeedsGeminiProfile(unittest.TestCase):

    def test_a_never_profiled_row_needs_one(self):
        self.assertTrue(needs_gemini_profile(False, None, now=NOW))

    def test_a_fresh_profile_is_left_alone(self):
        self.assertFalse(needs_gemini_profile(True, days_ago(1), now=NOW))

    def test_a_profile_past_the_threshold_is_stale(self):
        self.assertTrue(
            needs_gemini_profile(True, days_ago(GEMINI_PROFILE_MAX_AGE_DAYS + 1), now=NOW))

    def test_a_profile_just_inside_the_threshold_is_not(self):
        self.assertFalse(
            needs_gemini_profile(True, days_ago(GEMINI_PROFILE_MAX_AGE_DAYS - 1), now=NOW))

    def test_force_overrides_a_fresh_profile(self):
        self.assertTrue(needs_gemini_profile(True, days_ago(1), force=True, now=NOW))

    def test_a_stamp_without_a_profile_still_needs_one(self):
        """The D-16 shape: AI.GENERATE returned NULL but the merge stamped the
        row anyway. Those 7 rows were cleared, and the merge is now conditional,
        but the predicate must not read a bare timestamp as a profile."""
        self.assertTrue(needs_gemini_profile(False, days_ago(1), now=NOW))

    def test_a_profile_of_unknown_age_counts_as_fresh(self):
        """The money-safe default. Phase 5 stamped every existing profile, so
        this combination should not occur; if it ever does, re-profiling the
        whole table off the back of a missing timestamp is the expensive way to
        be wrong."""
        self.assertFalse(needs_gemini_profile(True, None, now=NOW))

    def test_max_age_none_means_never_stale(self):
        """How the training pre-flight asks the question. Retraining is cheap
        and re-profiling is not, so a scheduled training run fills gaps only --
        it never refreshes a profile it already has."""
        self.assertFalse(
            needs_gemini_profile(True, days_ago(10_000), max_age_days=None, now=NOW))
        self.assertTrue(needs_gemini_profile(False, None, max_age_days=None, now=NOW))

    def test_a_naive_timestamp_is_read_as_utc(self):
        self.assertFalse(needs_gemini_profile(True, datetime.datetime(2026, 9, 22, 12), now=NOW))

    def test_a_pandas_timestamp_is_accepted(self):
        """What a DataFrame column holds after BigQuery hands back a TIMESTAMP."""
        self.assertFalse(needs_gemini_profile(True, pd.Timestamp(days_ago(1)), now=NOW))
        self.assertTrue(needs_gemini_profile(True, pd.Timestamp(days_ago(400)), now=NOW))

    def test_a_string_timestamp_is_accepted(self):
        self.assertFalse(needs_gemini_profile(True, '2026-09-22 12:00:00+00:00', now=NOW))
        self.assertTrue(needs_gemini_profile(True, '2020-01-01 00:00:00+00:00', now=NOW))

    def test_an_unreadable_timestamp_does_not_trigger_a_refresh(self):
        """Same reasoning as the unknown-age case: a parse failure must not
        translate into a bill."""
        for value in ('not a date', float('nan'), pd.NaT, 0):
            self.assertFalse(needs_gemini_profile(True, value, now=NOW), value)

    def test_it_defaults_to_the_current_time(self):
        self.assertFalse(needs_gemini_profile(True, datetime.datetime.now(datetime.timezone.utc)))


class TestCountingWhatTheEstimateShows(unittest.TestCase):

    ROWS = [
        {'gemini_insights_structured': '{"match_score": 90}', 'gemini_profiled_at': days_ago(1)},
        {'gemini_insights_structured': '{"match_score": 90}', 'gemini_profiled_at': days_ago(400)},
        {'gemini_insights_structured': None, 'gemini_profiled_at': None},
        {'gemini_insights_structured': '   ', 'gemini_profiled_at': None},
        {},
    ]

    def test_it_counts_exactly_what_the_scalar_predicate_selects(self):
        """The estimate is only honest if it is the same rule, applied row by
        row, that the executor will apply."""
        expected = sum(
            1 for row in self.ROWS
            if needs_gemini_profile(
                bool((row.get('gemini_insights_structured') or '').strip()),
                row.get('gemini_profiled_at'), now=NOW)
        )
        self.assertEqual(count_needing_gemini_profile(self.ROWS, now=NOW), expected)
        self.assertEqual(expected, 4)  # fresh one cached; stale, absent, blank, missing

    def test_force_counts_every_row(self):
        self.assertEqual(
            count_needing_gemini_profile(self.ROWS, force=True, now=NOW), len(self.ROWS))

    def test_it_accepts_a_dataframe(self):
        """What the UI has in hand. Iterating a DataFrame directly yields column
        names, so the frame case has to be handled rather than assumed."""
        self.assertEqual(count_needing_gemini_profile(pd.DataFrame(self.ROWS), now=NOW), 4)

    def test_an_empty_frame_estimates_nothing(self):
        self.assertEqual(count_needing_gemini_profile(pd.DataFrame(), now=NOW), 0)


class TestBothSurfacesUseIt(unittest.TestCase):
    """D1 was not a wrong number, it was a second implementation of the number.
    A behavioural test covers the executor (`test_ml_prediction.py`); the UI's
    estimate is inline Streamlit, so the guard on it is that the hand-rolled
    count is gone and the shared one is imported."""

    def _source(self, relative):
        return (REPO_ROOT / relative).read_text()

    def test_the_ui_estimate_calls_the_shared_counter(self):
        source = self._source('app/ui/st_app.py')
        self.assertIn('count_needing_gemini_profile', source)
        self.assertNotIn('gem_missing += 1', source)

    def test_the_executor_calls_the_shared_predicate(self):
        source = self._source('app/services/ml_prediction.py')
        self.assertIn('needs_gemini_profile', source)
        self.assertNotIn('row.gemini_insights_structured is None', source)

    def test_the_training_preflight_calls_it_too(self):
        source = self._source('scripts/train_bqml_model.py')
        self.assertIn('needs_gemini_profile', source)
        self.assertNotIn('row.gemini_insights_structured is None', source)


if __name__ == '__main__':
    unittest.main()
