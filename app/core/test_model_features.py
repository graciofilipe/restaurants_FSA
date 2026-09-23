"""Tests for the one definition of the model's feature set (D3).

The training `SELECT` and the `ML.PREDICT` subquery were hand-copied 20-line
blocks. Nothing checked they matched, and CLAUDE.md warns that editing one
without the other makes predictions "silently skew". These tests are the check
that was missing.
"""
import re
import unittest

from app.core.model_features import (
    FEATURE_ALIASES,
    PILLAR_FEATURE_ALIASES,
    feature_select_list,
)
from app.core.pillar_schema import FEATURE_COLUMNS, PILLAR_FIELDS


class TestFeatureList(unittest.TestCase):

    def test_every_canonical_feature_is_present(self):
        """`is_feature` on a PillarField is what puts it in the model. If the
        two lists can disagree, the flag means nothing."""
        self.assertEqual(PILLAR_FEATURE_ALIASES, FEATURE_COLUMNS)

    def test_the_free_text_fields_stay_out(self):
        """Unbounded model prose is not a feature. `summary_reasoning` as a
        categorical would be one level per row."""
        sql = feature_select_list()
        for field in PILLAR_FIELDS:
            if not field.is_feature:
                self.assertNotIn(f"AS {field.column}", sql, field.column)

    def test_the_non_pillar_features_are_still_there(self):
        """The Gemini pillars are an addition, not a replacement."""
        for alias in ('postcode', 'maps_rating', 'maps_reviews', 'latitude', 'longitude',
                      'price_level', 'ratingvalue', 'business_status', 'localauthorityname',
                      'maps_types_array', 'lsoa', 'msoa', 'imd_rank'):
            self.assertIn(alias, FEATURE_ALIASES, alias)

    def test_the_label_is_selected(self):
        """BQML reads `user_rating` as input_label_cols; without it, training
        fails outright."""
        self.assertIn('user_rating', FEATURE_ALIASES)

    def test_missing_stays_missing(self):
        """No IFNULL(..., 0). Defaulting a missing score to zero is what made
        D2 invisible: five features read as constant 0 and nothing complained.
        BQML handles NULL natively, so the default bought nothing and hid
        everything."""
        sql = feature_select_list()
        self.assertNotIn('IFNULL', sql)
        self.assertNotIn('COALESCE', sql)

    def test_it_reads_the_nested_paths(self):
        """D2 itself: the old SQL read `$.1_value_and_volume_rating`, which
        resolves on zero of 2,766 rows."""
        sql = feature_select_list()
        self.assertIn('$.1_value_and_volume.rating', sql)
        self.assertNotIn('1_value_and_volume_rating', sql)

    def test_the_enum_pillars_are_not_cast_to_int(self):
        """`CAST('GENERIC_NATIONAL' AS INT64)` is NULL, and with the old IFNULL
        it was 0. Pillar 4 has three real levels; as an integer it had one."""
        sql = feature_select_list()
        for line in sql.splitlines():
            if 'pillar_geo_specificity' in line or 'pillar_establishment_type' in line:
                self.assertNotIn('AS INT64', line, line)

    def test_the_typed_pillars_are_safe_cast(self):
        """One profile returning "N/A" for a score must not fail the query for
        the other 2,765. Matched with a lookbehind because `SAFE_CAST(` contains
        `CAST(`."""
        sql = feature_select_list()
        self.assertIn('SAFE_CAST', sql)
        self.assertIsNone(re.search(r'(?<!SAFE_)CAST\(JSON_EXTRACT_SCALAR', sql))

    def test_aliases_are_unique(self):
        self.assertEqual(len(FEATURE_ALIASES), len(set(FEATURE_ALIASES)))

    def test_the_table_aliases_are_configurable_but_default_to_m_and_d(self):
        self.assertIn('m.postcode', feature_select_list())
        self.assertIn('d.imd_rank', feature_select_list())
        self.assertIn('x.postcode', feature_select_list(master='x'))


class TestFormatTemplateEscaping(unittest.TestCase):
    """The regex contains braces, and one of the two consumers runs the SQL
    through `str.format()`. Getting this wrong yields a regex matching nothing
    and a silent column of NULLs -- D2's failure mode exactly."""

    def test_plain_output_has_single_braces(self):
        self.assertIn("[{].*[}]", feature_select_list())

    def test_template_output_has_doubled_braces(self):
        self.assertIn("[{{].*[}}]", feature_select_list(for_format_template=True))

    def test_the_template_form_survives_a_format_call(self):
        formatted = feature_select_list(for_format_template=True).format()
        self.assertEqual(formatted, feature_select_list())


class TestTrainServeParity(unittest.TestCase):
    """The test CLAUDE.md says does not exist."""

    def test_training_and_prediction_select_the_same_features(self):
        from app.services.ml_prediction import build_prediction_input_select
        from scripts.train_bqml_model import build_training_select

        training = build_training_select('p', 'd', 'p.d.t')
        prediction = build_prediction_input_select('p', 'd', 'p.d.t', "'1'")

        shared = feature_select_list()
        self.assertIn(shared, training)
        self.assertIn(shared, prediction)

    def test_prediction_additionally_selects_the_join_key(self):
        from app.services.ml_prediction import build_prediction_input_select
        self.assertIn('m.fhrsid', build_prediction_input_select('p', 'd', 'p.d.t', "'1'"))

    def test_neither_carries_a_hand_written_pillar_path(self):
        """A second copy reintroduced by hand would pass the parity test above
        only if it were added to both -- this catches it in either."""
        from app.services.ml_prediction import build_prediction_input_select
        from scripts.train_bqml_model import build_training_select

        for sql in (build_training_select('p', 'd', 'p.d.t'),
                    build_prediction_input_select('p', 'd', 'p.d.t', "'1'")):
            self.assertEqual(sql.count('JSON_EXTRACT_SCALAR'), len(PILLAR_FEATURE_ALIASES))


if __name__ == '__main__':
    unittest.main()
