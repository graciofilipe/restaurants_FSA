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
    DEMOGRAPHIC_FEATURE_COLUMNS,
    ENGINEERED_FEATURE_ALIASES,
    FEATURE_ALIASES,
    LOCATION_FEATURE_ALIASES,
    PILLAR_FEATURE_ALIASES,
    TREE_FEATURE_ALIASES,
    counterweight_linear_replication_sql,
    counterweight_linear_where_clause,
    feature_select_list,
    feature_source_clause,
    is_stage1_gated_sql,
    normalized_brand_key_sql,
    stage1_deterministic_score_sql,
    stage2_training_where_clause,
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
        """Low-cardinality structural and numeric features remain in the model."""
        for alias in ('maps_rating', 'maps_reviews', 'price_level', 'ratingvalue',
                      'business_status', 'localauthorityname', 'imd_rank'):
            self.assertIn(alias, FEATURE_ALIASES, alias)

    def test_location_blind_feature_list_omits_borough_and_imd_rank(self):
        """Course 1b location-blind Boosted Tree omits `localauthorityname` and
        `imd_rank` while keeping all other 21 features."""
        sql_no_loc = feature_select_list(include_location=False)
        self.assertEqual(LOCATION_FEATURE_ALIASES, ('localauthorityname', 'imd_rank'))
        self.assertNotIn('localauthorityname', sql_no_loc)
        self.assertNotIn('imd_rank', sql_no_loc)
        aliases = [
            line.strip().rstrip(',').rsplit(' AS ', 1)[-1].rsplit('.', 1)[-1]
            for line in sql_no_loc.splitlines()
        ]
        self.assertEqual(tuple(aliases), TREE_FEATURE_ALIASES)

    def test_high_cardinality_memorization_features_are_pruned(self):
        """`postcode`, `lsoa`, `msoa`, `latitude`, `longitude`, and
        `maps_types_array` allowed trees to memorize postcodes/coordinates
        rather than learning culinary quality."""
        sql = feature_select_list()
        for pruned in ('postcode', 'lsoa', 'msoa', 'latitude', 'longitude', 'maps_types_array'):
            self.assertNotIn(pruned, FEATURE_ALIASES, pruned)
            self.assertNotIn(f"m.{pruned}", sql)
            self.assertNotIn(f"d.{pruned}", sql)

    def test_engineered_interaction_features_are_present(self):
        """Engineered continuous/interaction features are present in both
        `FEATURE_ALIASES` and `feature_select_list()`."""
        sql = feature_select_list()
        for alias in ENGINEERED_FEATURE_ALIASES:
            self.assertIn(alias, FEATURE_ALIASES, alias)
            self.assertIn(f"AS {alias}", sql, alias)

    def test_the_label_is_selected(self):
        """BQML reads `user_rating` as input_label_cols; without it, training
        fails outright."""
        self.assertIn('user_rating', FEATURE_ALIASES)

    def test_missing_stays_missing_for_required_core_pillars(self):
        """No IFNULL(..., 0) anywhere, and required core V2 pillars are never
        wrapped in COALESCE. Optional Stage-2 pillars (`required=False`) use
        neutral COALESCE(..., 3) fallback on rows profiled before Stage-2."""
        sql = feature_select_list()
        self.assertNotIn('IFNULL', sql)
        lines_by_alias = {
            line.strip().rstrip(',').rsplit(' AS ', 1)[-1]: line
            for line in sql.splitlines()
            if ' AS ' in line
        }
        for field in PILLAR_FIELDS:
            if not field.is_feature:
                continue
            line = lines_by_alias[field.column]
            if field.required:
                self.assertNotIn('COALESCE', line, field.column)
            else:
                self.assertIn('COALESCE(', line, field.column)
                self.assertIn(', 3)', line, field.column)

    def test_it_reads_the_nested_paths(self):
        """D2 itself: the old SQL read `$.1_value_and_volume_rating`, which
        resolves on zero of 2,766 rows."""
        sql = feature_select_list()
        self.assertIn('$.1_value_and_volume.rating', sql)
        self.assertNotIn('1_value_and_volume_rating', sql)
        self.assertIn('$.7_plausible_discriminators.dining_pace_score', sql)
        self.assertIn('$.7_plausible_discriminators.cooking_quality_score', sql)
        self.assertIn('$.7_plausible_discriminators.anti_hype_score', sql)

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
        self.assertIn('m.localauthorityname', feature_select_list())
        self.assertIn('d.imd_rank', feature_select_list())
        self.assertIn('b.branch_count_in_fsa', feature_select_list())
        self.assertIn('x.localauthorityname', feature_select_list(master='x'))


class TestStage1AndStage2SqlHelpers(unittest.TestCase):

    def test_normalized_brand_key_strips_non_alphanumeric(self):
        sql = normalized_brand_key_sql('m')
        self.assertIn('REGEXP_REPLACE(LOWER(TRIM(COALESCE(m.businessname', sql)
        self.assertIn("r'[^a-z0-9]'", sql)

    def test_stage1_gate_checks_scope_sit_down_type_and_branches(self):
        sql = is_stage1_gated_sql('m', 'b')
        self.assertIn('m.in_scope', sql)
        self.assertIn('m.pillar_is_sit_down', sql)
        self.assertIn("!= 'RESTAURANT_DINING'", sql)
        self.assertIn('b.branch_count_in_fsa, 1) >= 5', sql)

    def test_stage1_deterministic_score_is_bounded_between_1_and_2(self):
        sql = stage1_deterministic_score_sql('m')
        self.assertIn('LEAST(2.0, GREATEST(1.0', sql)
        self.assertIn('m.match_score', sql)

    def test_stage2_training_where_clause_filters_to_plausible_sit_down(self):
        sql = stage2_training_where_clause('m', 'b')
        self.assertIn('m.user_rating IS NOT NULL', sql)
        self.assertIn('m.in_scope IS TRUE', sql)
        self.assertIn('m.pillar_is_sit_down IS TRUE', sql)
        self.assertIn("m.pillar_establishment_type = 'RESTAURANT_DINING'", sql)
        self.assertIn('COALESCE(b.branch_count_in_fsa, 1) < 5', sql)

    def test_counterweight_linear_helpers_use_all_in_scope_and_4_2_1_replication(self):
        where_sql = counterweight_linear_where_clause('m')
        self.assertIn('m.user_rating IS NOT NULL', where_sql)
        self.assertIn('(m.in_scope = TRUE OR m.in_scope IS NULL)', where_sql)

        rep_sql = counterweight_linear_replication_sql('m', 'b')
        self.assertIn("m.rating_source = 'visited'", rep_sql)
        self.assertIn(', 4, ', rep_sql)
        self.assertIn(', 2, 1))', rep_sql)


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

        training_tree = build_training_select('p', 'd', 'p.d.t', model_family='boosted_tree')
        training_linear = build_training_select('p', 'd', 'p.d.t', model_family='linear_reg')
        prediction = build_prediction_input_select('p', 'd', 'p.d.t', "'1'")

        self.assertIn(feature_select_list(include_location=False), training_tree)
        self.assertIn(feature_select_list(include_location=True), training_linear)
        self.assertIn(feature_select_list(include_location=True), prediction)

    def test_prediction_additionally_selects_the_join_key(self):
        from app.services.ml_prediction import build_prediction_input_select
        self.assertIn('m.fhrsid', build_prediction_input_select('p', 'd', 'p.d.t', "'1'"))

    def test_neither_carries_a_hand_written_pillar_path(self):
        """A second copy reintroduced by hand would pass the parity test above
        only if it were added to both -- this catches it in either."""
        from app.services.ml_prediction import build_prediction_input_select
        from scripts.train_bqml_model import build_training_select

        for sql in (build_training_select('p', 'd', 'p.d.t', model_family='boosted_tree'),
                    build_training_select('p', 'd', 'p.d.t', model_family='linear_reg'),
                    build_prediction_input_select('p', 'd', 'p.d.t', "'1'")):
            self.assertEqual(sql.count('JSON_EXTRACT_SCALAR'), len(PILLAR_FEATURE_ALIASES))


class TestTheDemographicsJoinCannotFanOut(unittest.TestCase):
    """D-32. The join key is a normalised postcode, and it is not unique:
    `uk_postcode_demographics` holds 15 normalised postcodes more than once,
    one of them three times, because the enrichment inserts one row per *raw*
    spelling. 103 rows of `fsa_master` carry one of them.
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
        """The deduplicating subquery still selects all DEMOGRAPHIC_COLUMNS,
        while the feature list selects only DEMOGRAPHIC_FEATURE_COLUMNS (`imd_rank`)."""
        clause = feature_source_clause('p', 'd', 'p.d.t')

        for column in DEMOGRAPHIC_COLUMNS:
            self.assertIn(column, clause)
        for column in DEMOGRAPHIC_FEATURE_COLUMNS:
            self.assertIn(f'd.{column}', feature_select_list())

    def test_the_join_still_matches_on_the_normalised_postcode(self):
        clause = feature_source_clause('p', 'd', 'p.d.t')

        self.assertIn("REPLACE(UPPER(m.postcode), ' ', '')", clause)

    def test_branch_count_subquery_groups_by_brand_key_and_cannot_fan_out(self):
        clause = feature_source_clause('p', 'd', 'p.d.t')

        self.assertIn('COUNT(DISTINCT b_src.fhrsid) AS branch_count_in_fsa', clause)
        self.assertIn('GROUP BY brand_key', clause)
        self.assertIn(f"ON {normalized_brand_key_sql('m')} = b.brand_key", clause)

    def test_avoids_qualify_for_bqml_create_model_compatibility(self):
        """BigQuery ML's `CREATE MODEL ... AS SELECT` validator rejects `QUALIFY`
        with `400: QUALIFY is not supported`. The `ROW_NUMBER()` filter must use
        an outer `WHERE _rn = 1` subquery instead."""
        clause = feature_source_clause('p', 'd', 'p.d.t')

        self.assertNotIn('QUALIFY', clause.upper())
        self.assertIn('WHERE _rn = 1', clause)

    def test_it_changes_row_counts_and_not_the_feature_schema(self):
        """The selected aliases are what the model's input schema is made of."""
        aliases = []
        for line in feature_select_list().splitlines():
            expression = line.strip().rstrip(',')
            aliases.append(expression.rsplit(' AS ', 1)[-1].rsplit('.', 1)[-1])

        self.assertEqual(tuple(aliases), FEATURE_ALIASES)


if __name__ == '__main__':
    unittest.main()
