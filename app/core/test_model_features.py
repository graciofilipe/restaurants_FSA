"""Tests for the one definition of the model's feature set (D3).

The training `SELECT` and the `ML.PREDICT` subquery were hand-copied 20-line
blocks. Nothing checked they matched, and CLAUDE.md warns that editing one
without the other makes predictions "silently skew". These tests are the check
that was missing.
"""
import re
import unittest

from app.core.model_features import (
    DEMOGRAPHIC_COLUMNS,
    FEATURE_ALIASES,
    PILLAR_FEATURE_ALIASES,
    feature_select_list,
    feature_source_clause,
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


class TestTheDemographicsJoinCannotFanOut(unittest.TestCase):
    """D-32. The join key is a normalised postcode, and it is not unique:
    `uk_postcode_demographics` holds 15 normalised postcodes more than once,
    one of them three times, because the enrichment inserts one row per *raw*
    spelling. 103 rows of `fsa_master` carry one of them.

    Joining the table raw therefore returns those restaurants two or three
    times, and `ML.PREDICT` carries the duplicates into the MERGE's source,
    where BigQuery rejects the whole statement:

        UPDATE/MERGE must match at most one source row for each target row

    Which is the serious part. It is not a skewed number, it is a batch that
    dies -- *after* the Places and `AI.GENERATE` pre-flight has been paid for.
    Observed twice in production on 2026-09-24, at 14:45:58 and 16:10:33.

    Training reads the same clause, so those 103 rows were also duplicated in
    the training set. Only one of them is labelled, so the model skew is one
    double-weighted example out of 411 -- real, but not why this is a fix.

    The dedupe is in the query rather than only in the table because this join
    must be safe against a reference table it does not own.
    """

    def test_the_demographics_table_is_not_joined_raw(self):
        clause = feature_source_clause('p', 'd', 'p.d.t')

        self.assertNotIn('uk_postcode_demographics` AS d', clause,
                         "joining the table directly fans out on its duplicate keys")

    def test_one_row_survives_per_normalised_postcode(self):
        """`ROW_NUMBER() = 1` over the same expression the join matches on.
        Partitioning by anything else -- the raw postcode, say -- would leave
        exactly the duplicates that cause this."""
        clause = feature_source_clause('p', 'd', 'p.d.t')

        self.assertIn("PARTITION BY REPLACE(UPPER(postcode), ' ', '')", clause)
        self.assertIn('= 1', clause)

    def test_the_surviving_row_is_chosen_deterministically(self):
        """One of the 15 has two genuinely different payloads, so the choice is
        a real one. `ANY_VALUE` would let a training run and a prediction run
        pick differently, which is the train/serve skew this module exists to
        prevent -- and it can also mix columns from different source rows."""
        clause = feature_source_clause('p', 'd', 'p.d.t')

        self.assertNotIn('ANY_VALUE', clause)
        self.assertIn('ORDER BY ' + ', '.join(f'{c} NULLS LAST' for c in DEMOGRAPHIC_COLUMNS),
                      clause)

    def test_a_populated_row_beats_an_empty_one(self):
        """The disagreeing pair is `SW3 5 UH` (a stray space, answered with an
        all-NULL row) against `SW3 5UH`. BigQuery sorts nulls first ascending,
        so without `NULLS LAST` the deterministic choice is the blank."""
        clause = feature_source_clause('p', 'd', 'p.d.t')

        self.assertEqual(clause.count('NULLS LAST'), len(DEMOGRAPHIC_COLUMNS))

    def test_the_demographic_columns_are_still_reachable_as_d(self):
        """The feature list is unchanged and still says `d.lsoa`; the subquery
        has to keep answering to that alias or every demographic goes NULL."""
        clause = feature_source_clause('p', 'd', 'p.d.t')

        for column in DEMOGRAPHIC_COLUMNS:
            self.assertIn(f'd.{column}', feature_select_list())
            self.assertIn(column, clause)

    def test_the_join_still_matches_on_the_normalised_postcode(self):
        clause = feature_source_clause('p', 'd', 'p.d.t')

        self.assertIn("REPLACE(UPPER(m.postcode), ' ', '')", clause)

    def test_it_changes_row_counts_and_not_the_feature_schema(self):
        """Deduplicating rows is not a feature change, so `ML.PREDICT` against
        the already-trained model keeps working and no retrain is forced. The
        selected aliases are what the model's input schema is made of, so
        pinning them here is what makes that claim checkable."""
        aliases = []
        for line in feature_select_list().splitlines():
            expression = line.strip().rstrip(',')
            aliases.append(expression.rsplit(' AS ', 1)[-1].rsplit('.', 1)[-1])

        self.assertEqual(tuple(aliases), FEATURE_ALIASES)


if __name__ == '__main__':
    unittest.main()
