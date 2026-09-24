"""A frozen fixture pinning `calculate_restaurant_priority`'s exact output.

D8 vectorises that function. Vectorising is a rewrite of every branch at once,
and the branches are the interesting part: a NaN latitude, a postcode that
resolves only by prefix, a `predicted_at` that arrived as a string, an
`in_scope` of `"true"` rather than `True`. The behaviour tests in
`test_scoring_priority.py` say what each rule *means*; this one says what the
function *returned* on 2026-09-24, to the tenth, so that a refactor which is
supposed to change nothing can be shown to have changed nothing.

The expected values below were captured from the pre-refactor implementation.
They are not independently derived, and that is deliberate -- re-deriving them
by hand would just be the same arithmetic with a second chance to be wrong.
If a future change to the scoring *rules* makes this fail, the fix is to
re-capture the numbers in the same commit that changes the rule, not to relax
the assertion.
"""
import datetime
import math
import unittest

import numpy as np
import pandas as pd

from app.core.data_processing import calculate_restaurant_priority

# Frozen so `curr_date` cannot drift the staleness tiers out from under us.
TODAY = datetime.date(2026, 9, 24)

# One row per branch, named in `case` so a failure says which one broke.
FIXTURE_ROWS = [
    # -- proximity ---------------------------------------------------------
    dict(case='exact coords, on the anchor', latitude=51.4212, longitude=-0.1292),
    dict(case='exact coords, 5km out', latitude=51.4662, longitude=-0.1292),
    dict(case='exact coords, far from the anchor', latitude=53.4808, longitude=-2.2426),
    dict(case='coords outside the UK box, postcode rescues it', latitude=0.0, longitude=0.0,
         postcode='SW16 1AA'),
    dict(case='coords outside the UK box, nothing rescues it', latitude=40.7128,
         longitude=-74.0060),
    dict(case='NaN coords, exact outcode', latitude=float('nan'), longitude=float('nan'),
         postcode='E1 6AN'),
    dict(case='NaN coords, four-character outcode', postcode='EC2A 3AY'),
    # The branch whose 310-key `sorted()` D8 hoists: no direct hit, resolved by
    # longest prefix. Junk like this is what the live table actually holds.
    # Deliberately not a prefix of SW16: resolving to the anchor would give a
    # distance of 0, indistinguishable from the on-anchor row above.
    dict(case='no coords, outcode resolved only by prefix', postcode='EC1YYY'),
    dict(case='no coords, junk postcode', postcode='WATERLOOVI'),
    dict(case='no coords, outside the London dictionary', postcode='M1 4AB'),
    dict(case='no coords, no postcode at all'),
    dict(case='no coords, empty postcode', postcode=''),
    dict(case='no coords, NaN postcode', postcode=float('nan')),
    dict(case='no coords, lowercase postcode', postcode='sw2 1aa'),
    dict(case='no coords, untrimmed postcode', postcode='  E1 6AN  '),
    dict(case='PascalCase PostCode key', PostCode='SW2 1AA'),
    dict(case='latitude present, longitude missing', latitude=51.5),
    dict(case='coords as strings', latitude='51.5074', longitude='-0.1278'),
    dict(case='coords as unparseable strings', latitude='north', longitude='west',
         postcode='N1 9GU'),
    # -- staleness ---------------------------------------------------------
    dict(case='never predicted', postcode='SW16 1AA',
         gemini_profiled_at=pd.Timestamp('2026-09-01')),
    dict(case='never profiled', postcode='SW16 1AA', predicted_user_rating=7.0),
    dict(case='predicted 0 days ago', postcode='SW16 1AA', predicted_user_rating=7.0,
         gemini_profiled_at=pd.Timestamp('2026-09-01'),
         predicted_at=pd.Timestamp('2026-09-24')),
    dict(case='predicted 13 days ago', postcode='SW16 1AA', predicted_user_rating=7.0,
         gemini_profiled_at=pd.Timestamp('2026-09-01'),
         predicted_at=pd.Timestamp('2026-09-11')),
    dict(case='predicted 14 days ago', postcode='SW16 1AA', predicted_user_rating=7.0,
         gemini_profiled_at=pd.Timestamp('2026-09-01'),
         predicted_at=pd.Timestamp('2026-09-10')),
    dict(case='predicted 30 days ago', postcode='SW16 1AA', predicted_user_rating=7.0,
         gemini_profiled_at=pd.Timestamp('2026-09-01'),
         predicted_at=pd.Timestamp('2026-08-25')),
    dict(case='predicted 60 days ago', postcode='SW16 1AA', predicted_user_rating=7.0,
         gemini_profiled_at=pd.Timestamp('2026-09-01'),
         predicted_at=pd.Timestamp('2026-07-26')),
    dict(case='predicted 400 days ago', postcode='SW16 1AA', predicted_user_rating=7.0,
         gemini_profiled_at=pd.Timestamp('2026-09-01'),
         predicted_at=pd.Timestamp('2025-08-20')),
    dict(case='predicted_at missing, falls back to first_seen', postcode='SW16 1AA',
         predicted_user_rating=7.0, gemini_profiled_at=pd.Timestamp('2026-09-01'),
         first_seen=pd.Timestamp('2026-06-01')),
    dict(case='no timestamp at all, default 45 days', postcode='SW16 1AA',
         predicted_user_rating=7.0, gemini_profiled_at=pd.Timestamp('2026-09-01')),
    dict(case='predicted_at as a string', postcode='SW16 1AA', predicted_user_rating=7.0,
         gemini_profiled_at=pd.Timestamp('2026-09-01'), predicted_at='2026-08-01T12:30:00'),
    dict(case='predicted_at as an unparseable string', postcode='SW16 1AA',
         predicted_user_rating=7.0, gemini_profiled_at=pd.Timestamp('2026-09-01'),
         predicted_at='not a date'),
    dict(case='predicted_at as a date', postcode='SW16 1AA', predicted_user_rating=7.0,
         gemini_profiled_at=pd.Timestamp('2026-09-01'), predicted_at=datetime.date(2026, 8, 1)),
    dict(case='predicted_at in the future', postcode='SW16 1AA', predicted_user_rating=7.0,
         gemini_profiled_at=pd.Timestamp('2026-09-01'),
         predicted_at=pd.Timestamp('2026-12-25')),
    # -- maps prior --------------------------------------------------------
    dict(case='no maps rating', postcode='SW16 1AA'),
    dict(case='maps 3.0, no reviews', postcode='SW16 1AA', maps_rating=3.0),
    dict(case='maps 2.0 floors at zero', postcode='SW16 1AA', maps_rating=2.0, maps_reviews=500),
    dict(case='maps 4.5, 200 reviews', postcode='SW16 1AA', maps_rating=4.5, maps_reviews=200),
    dict(case='maps 5.0, 10000 reviews caps at 100', postcode='SW16 1AA', maps_rating=5.0,
         maps_reviews=10000),
    dict(case='maps rating as a string', postcode='SW16 1AA', maps_rating='4.2',
         maps_reviews='55'),
    dict(case='maps rating unparseable', postcode='SW16 1AA', maps_rating='no idea'),
    # 4.6 rather than 4.0 on purpose: a 4.0 with no reviews scores exactly 50,
    # which is also the no-rating default, so the assertion could not tell the
    # two apart.
    dict(case='maps rating present, reviews NaN', postcode='SW16 1AA', maps_rating=4.6,
         maps_reviews=float('nan')),
    dict(case='maps reviews zero', postcode='SW16 1AA', maps_rating=4.6, maps_reviews=0),
    # -- scope -------------------------------------------------------------
    dict(case='in scope, bool', postcode='SW16 1AA', in_scope=True),
    dict(case='out of scope, bool', postcode='SW16 1AA', in_scope=False),
    dict(case='in scope, int 1', postcode='SW16 1AA', in_scope=1),
    dict(case='out of scope, int 0', postcode='SW16 1AA', in_scope=0),
    dict(case='in scope, string true', postcode='SW16 1AA', in_scope='TRUE'),
    dict(case='out of scope, string false', postcode='SW16 1AA', in_scope='false'),
    dict(case='untriaged, None', postcode='SW16 1AA', in_scope=None),
    dict(case='untriaged, NaN', postcode='SW16 1AA', in_scope=float('nan')),
    dict(case='in_scope garbage string', postcode='SW16 1AA', in_scope='maybe'),
    # -- the user-rating discount, and its interaction with scope ----------
    dict(case='already rated', postcode='SW16 1AA', in_scope=True, user_rating=8),
    dict(case='already rated and out of scope', postcode='SW16 1AA', in_scope=False,
         user_rating=8),
    dict(case='user_rating is an empty string', postcode='SW16 1AA', in_scope=True,
         user_rating=''),
    dict(case='user_rating is whitespace', postcode='SW16 1AA', in_scope=True, user_rating='   '),
    dict(case='user_rating is NaN', postcode='SW16 1AA', in_scope=True,
         user_rating=float('nan')),
    dict(case='user_rating is zero', postcode='SW16 1AA', in_scope=True, user_rating=0),
    # -- everything at once ------------------------------------------------
    dict(case='fully populated', latitude=51.5074, longitude=-0.1278, postcode='WC2N 5DU',
         predicted_user_rating=6.5, gemini_profiled_at=pd.Timestamp('2026-08-01'),
         predicted_at=pd.Timestamp('2026-08-20'), first_seen=pd.Timestamp('2026-01-01'),
         maps_rating=4.4, maps_reviews=1200, in_scope=True),
    dict(case='fully empty'),
]

# Captured from the implementation as it stood before D8. Columns:
# (distance_km, proximity_score, staleness_score, maps_prior_score, priority_score).
EXPECTED = [
    ('exact coords, on the anchor', 0.0, 100.0, 100.0, 50.0, 85.0),
    ('exact coords, 5km out', 5.0, 36.8, 100.0, 50.0, 62.9),
    ('exact coords, far from the anchor', 270.08, 0.0, 100.0, 50.0, 50.0),
    ('coords outside the UK box, postcode rescues it', 0.0, 100.0, 100.0, 50.0, 85.0),
    ('coords outside the UK box, nothing rescues it', float('nan'), 10.0, 100.0, 50.0, 53.5),
    ('NaN coords, exact outcode', 11.74, 9.6, 100.0, 50.0, 53.4),
    ('NaN coords, four-character outcode', 11.79, 9.5, 100.0, 50.0, 53.3),
    ('no coords, outcode resolved only by prefix', 11.61, 9.8, 100.0, 50.0, 53.4),
    ('no coords, junk postcode', float('nan'), 10.0, 100.0, 50.0, 53.5),
    ('no coords, outside the London dictionary', float('nan'), 10.0, 100.0, 50.0, 53.5),
    ('no coords, no postcode at all', float('nan'), 10.0, 100.0, 50.0, 53.5),
    ('no coords, empty postcode', float('nan'), 10.0, 100.0, 50.0, 53.5),
    ('no coords, NaN postcode', float('nan'), 10.0, 100.0, 50.0, 53.5),
    ('no coords, lowercase postcode', 3.19, 52.8, 100.0, 50.0, 68.5),
    ('no coords, untrimmed postcode', 11.74, 9.6, 100.0, 50.0, 53.4),
    ('PascalCase PostCode key', 3.19, 52.8, 100.0, 50.0, 68.5),
    ('latitude present, longitude missing', float('nan'), 10.0, 100.0, 50.0, 53.5),
    ('coords as strings', 9.59, 14.7, 100.0, 50.0, 55.1),
    ('coords as unparseable strings', 13.16, 7.2, 100.0, 50.0, 52.5),
    ('never predicted', 0.0, 100.0, 100.0, 50.0, 85.0),
    ('never profiled', 0.0, 100.0, 100.0, 50.0, 85.0),
    ('predicted 0 days ago', 0.0, 100.0, 15.0, 50.0, 55.2),
    ('predicted 13 days ago', 0.0, 100.0, 15.0, 50.0, 55.2),
    ('predicted 14 days ago', 0.0, 100.0, 40.0, 50.0, 64.0),
    ('predicted 30 days ago', 0.0, 100.0, 60.0, 50.0, 71.0),
    ('predicted 60 days ago', 0.0, 100.0, 80.0, 50.0, 78.0),
    ('predicted 400 days ago', 0.0, 100.0, 80.0, 50.0, 78.0),
    ('predicted_at missing, falls back to first_seen', 0.0, 100.0, 80.0, 50.0, 78.0),
    ('no timestamp at all, default 45 days', 0.0, 100.0, 60.0, 50.0, 71.0),
    ('predicted_at as a string', 0.0, 100.0, 60.0, 50.0, 71.0),
    ('predicted_at as an unparseable string', 0.0, 100.0, 60.0, 50.0, 71.0),
    ('predicted_at as a date', 0.0, 100.0, 60.0, 50.0, 71.0),
    ('predicted_at in the future', 0.0, 100.0, 15.0, 50.0, 55.2),
    ('no maps rating', 0.0, 100.0, 100.0, 50.0, 85.0),
    ('maps 3.0, no reviews', 0.0, 100.0, 100.0, 0.0, 75.0),
    ('maps 2.0 floors at zero', 0.0, 100.0, 100.0, 13.5, 77.7),
    ('maps 4.5, 200 reviews', 0.0, 100.0, 100.0, 86.5, 92.3),
    ('maps 5.0, 10000 reviews caps at 100', 0.0, 100.0, 100.0, 100.0, 95.0),
    ('maps rating as a string', 0.0, 100.0, 100.0, 68.7, 88.7),
    ('maps rating unparseable', 0.0, 100.0, 100.0, 50.0, 85.0),
    ('maps rating present, reviews NaN', 0.0, 100.0, 100.0, 80.0, 91.0),
    ('maps reviews zero', 0.0, 100.0, 100.0, 80.0, 91.0),
    ('in scope, bool', 0.0, 100.0, 100.0, 50.0, 90.0),
    ('out of scope, bool', 0.0, 100.0, 100.0, 50.0, 0.0),
    ('in scope, int 1', 0.0, 100.0, 100.0, 50.0, 90.0),
    ('out of scope, int 0', 0.0, 100.0, 100.0, 50.0, 0.0),
    ('in scope, string true', 0.0, 100.0, 100.0, 50.0, 90.0),
    ('out of scope, string false', 0.0, 100.0, 100.0, 50.0, 0.0),
    ('untriaged, None', 0.0, 100.0, 100.0, 50.0, 85.0),
    ('untriaged, NaN', 0.0, 100.0, 100.0, 50.0, 85.0),
    ('in_scope garbage string', 0.0, 100.0, 100.0, 50.0, 85.0),
    ('already rated', 0.0, 100.0, 100.0, 50.0, 9.0),
    ('already rated and out of scope', 0.0, 100.0, 100.0, 50.0, 0.0),
    ('user_rating is an empty string', 0.0, 100.0, 100.0, 50.0, 90.0),
    ('user_rating is whitespace', 0.0, 100.0, 100.0, 50.0, 90.0),
    ('user_rating is NaN', 0.0, 100.0, 100.0, 50.0, 90.0),
    ('user_rating is zero', 0.0, 100.0, 100.0, 50.0, 9.0),
    ('fully populated', 9.59, 14.7, 60.0, 85.0, 53.1),
    ('fully empty', float('nan'), 10.0, 100.0, 50.0, 53.5),
]

SCORE_COLUMNS = ('distance_km', 'proximity_score', 'staleness_score',
                 'maps_prior_score', 'priority_score')


def build_fixture() -> pd.DataFrame:
    """The fixture as a DataFrame, with every column present on every row.

    Built through `pd.DataFrame` rather than row dicts so the missing keys
    become real NaNs in a typed column -- which is how they arrive from
    BigQuery, and the reason D-18 was possible.
    """
    return pd.DataFrame(FIXTURE_ROWS)


def score_fixture(df: pd.DataFrame = None) -> pd.DataFrame:
    return calculate_restaurant_priority(
        build_fixture() if df is None else df,
        anchor_lat=51.4212, anchor_lon=-0.1292, today_date=TODAY)


class TestTheFixtureStillScoresTheSame(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.scored = score_fixture()

    def test_every_case_is_covered_by_an_expectation(self):
        """Adding a fixture row without capturing its output would let the row
        pass silently, which defeats the purpose of having it."""
        self.assertEqual(len(FIXTURE_ROWS), len(EXPECTED))
        self.assertEqual([r['case'] for r in FIXTURE_ROWS], [e[0] for e in EXPECTED])

    def test_the_scores_are_identical_to_the_captured_output(self):
        for i, (case, *expected) in enumerate(EXPECTED):
            row = self.scored.iloc[i]
            for col, want in zip(SCORE_COLUMNS, expected):
                got = row[col]
                with self.subTest(case=case, column=col):
                    if isinstance(want, float) and math.isnan(want):
                        self.assertTrue(
                            pd.isna(got),
                            f"{case}/{col}: expected NaN, got {got!r}")
                    else:
                        self.assertAlmostEqual(
                            float(got), want, places=6,
                            msg=f"{case}/{col}: expected {want}, got {got!r}")

    def test_the_input_columns_survive_untouched(self):
        """The function returns an augmented copy. A vectorised rewrite that
        reindexed or reordered would break every caller downstream of it."""
        original = build_fixture()
        for col in original.columns:
            self.assertIn(col, self.scored.columns)
        pd.testing.assert_index_equal(original.index, self.scored.index)
        pd.testing.assert_series_equal(original['case'], self.scored['case'])

    def test_the_input_frame_is_not_mutated(self):
        original = build_fixture()
        calculate_restaurant_priority(original, today_date=TODAY)
        for col in SCORE_COLUMNS:
            self.assertNotIn(col, original.columns)

    def test_the_scores_are_plain_floats_not_numpy_objects(self):
        """`st.dataframe` and the BigQuery writers both see these columns. A
        vectorised implementation naturally yields numpy dtypes; that is fine,
        but the column must stay numeric rather than becoming `object`."""
        for col in SCORE_COLUMNS:
            self.assertTrue(
                pd.api.types.is_numeric_dtype(self.scored[col]),
                f"{col} is {self.scored[col].dtype}")


class TestTheWeightsStillApplyTheSameWay(unittest.TestCase):
    """The weight vector is the UI's strategy presets, so it is as much a part
    of the contract as the fixture is."""

    def _totals(self, weights):
        scored = calculate_restaurant_priority(
            build_fixture(), anchor_lat=51.4212, anchor_lon=-0.1292,
            weights=weights, today_date=TODAY)
        return [round(float(v), 1) for v in scored['priority_score']]

    def test_unnormalised_weights_are_normalised(self):
        """`{"prox": 7, ...}` and `{"prox": 0.35, ...}` are the same strategy."""
        self.assertEqual(
            self._totals({"prox": 35, "stale": 35, "prior": 20, "scope": 10}),
            self._totals({"prox": 0.35, "stale": 0.35, "prior": 0.20, "scope": 0.10}))

    def test_all_zero_weights_do_not_divide_by_zero(self):
        totals = self._totals({"prox": 0, "stale": 0, "prior": 0, "scope": 0})
        self.assertEqual(totals[0], 0.0)

    def test_a_missing_weight_key_falls_back_to_its_default(self):
        self.assertEqual(
            self._totals({"prox": 0.35}),
            self._totals({"prox": 0.35, "stale": 0.35, "prior": 0.20, "scope": 0.10}))

    def test_proximity_only_scores_the_proximity_component(self):
        scored = calculate_restaurant_priority(
            build_fixture(), anchor_lat=51.4212, anchor_lon=-0.1292,
            weights={"prox": 1.0, "stale": 0.0, "prior": 0.0, "scope": 0.0},
            today_date=TODAY)
        on_anchor = scored.iloc[0]
        self.assertAlmostEqual(float(on_anchor['priority_score']), 100.0, places=6)


class TestTheAnchorStillBehaves(unittest.TestCase):

    def test_a_bad_anchor_falls_back_to_sw16(self):
        bad = calculate_restaurant_priority(
            build_fixture(), anchor_lat='north', anchor_lon='west', today_date=TODAY)
        sw16 = calculate_restaurant_priority(
            build_fixture(), anchor_lat=51.4212, anchor_lon=-0.1292, today_date=TODAY)
        pd.testing.assert_series_equal(bad['priority_score'], sw16['priority_score'])

    def test_no_anchor_at_all_is_sw16(self):
        none = calculate_restaurant_priority(build_fixture(), today_date=TODAY)
        sw16 = calculate_restaurant_priority(
            build_fixture(), anchor_lat=51.4212, anchor_lon=-0.1292, today_date=TODAY)
        pd.testing.assert_series_equal(none['priority_score'], sw16['priority_score'])

    def test_moving_the_anchor_moves_the_distances(self):
        far = calculate_restaurant_priority(
            build_fixture(), anchor_lat=53.4808, anchor_lon=-2.2426, today_date=TODAY)
        self.assertGreater(float(far.iloc[0]['distance_km']), 200.0)


class TestTheDegenerateInputsStillReturnEarly(unittest.TestCase):

    def test_none_is_returned_unchanged(self):
        self.assertIsNone(calculate_restaurant_priority(None))

    def test_an_empty_frame_is_returned_unchanged(self):
        empty = pd.DataFrame()
        self.assertTrue(calculate_restaurant_priority(empty).empty)

    def test_an_empty_frame_with_columns_keeps_its_columns(self):
        empty = pd.DataFrame(columns=['fhrsid', 'latitude', 'longitude'])
        out = calculate_restaurant_priority(empty)
        self.assertEqual(list(out.columns), ['fhrsid', 'latitude', 'longitude'])

    def test_a_frame_missing_every_optional_column_still_scores(self):
        """The realistic minimum: a freshly ingested row with an FHRSID and
        nothing else yet."""
        minimal = pd.DataFrame([{'fhrsid': '1'}, {'fhrsid': '2'}])
        out = calculate_restaurant_priority(minimal, today_date=TODAY)
        self.assertEqual(list(out['priority_score']), [53.5, 53.5])

    def test_a_non_default_index_is_preserved(self):
        """`filter_and_sort_restaurants` hands this function sliced frames, so
        the index is routinely neither unique-from-zero nor sorted."""
        df = build_fixture().iloc[[5, 2, 40]]
        out = calculate_restaurant_priority(
            df, anchor_lat=51.4212, anchor_lon=-0.1292, today_date=TODAY)
        self.assertEqual(list(out.index), [5, 2, 40])
        self.assertEqual(list(out['case']), [FIXTURE_ROWS[i]['case'] for i in (5, 2, 40)])
        for pos, src in enumerate((5, 2, 40)):
            self.assertAlmostEqual(
                float(out.iloc[pos]['priority_score']), EXPECTED[src][5], places=6)


if __name__ == '__main__':
    unittest.main()
