"""Guards on the SQL templates.

These are format strings executed against production. A placeholder that is
renamed on one side only, or a stray brace in the prompt text, fails at
`.format()` time inside a BigQuery call -- not here, where it is cheap.
"""
import json
import string

import pytest

from app.core.pillar_schema import PILLAR_FIELDS, sql_json_object_regex
from scripts.bq_scripts import (
    MODEL_PARAMS_JSON,
    SCRIPT_BULK_UPDATE_MERGE,
    SCRIPT_GENERATE_INSIGHTS,
    SCRIPT_IDENTIFY_RECENTS,
    SCRIPT_MERGE_INSIGHTS,
)


def _placeholders(template):
    return {name for _, name, _, _ in string.Formatter().parse(template) if name}


EXPECTED_PLACEHOLDERS = {
    SCRIPT_IDENTIFY_RECENTS: {
        'project_id', 'dataset_id', 'source_table', 'target_table_recents', 'filter_condition',
    },
    SCRIPT_GENERATE_INSIGHTS: {
        'project_id', 'dataset_id', 'source_table_recents', 'target_table_insights',
        'connection_id', 'model_endpoint', 'model_params_json',
    },
    SCRIPT_MERGE_INSIGHTS: {
        'project_id', 'dataset_id', 'source_table_insights', 'target_table_master',
    },
    SCRIPT_BULK_UPDATE_MERGE: {
        'project_id', 'dataset_id', 'target_table', 'source_table_temp', 'update_set_clause',
    },
}


@pytest.mark.parametrize(
    "template,expected", EXPECTED_PLACEHOLDERS.items(), ids=range(len(EXPECTED_PLACEHOLDERS))
)
def test_templates_take_exactly_the_documented_placeholders(template, expected):
    assert _placeholders(template) == expected


@pytest.mark.parametrize("template", EXPECTED_PLACEHOLDERS)
def test_templates_render(template):
    """An unescaped brace in the prompt would raise here rather than mid-query.

    The JSON-unwrap regex is the one place a brace is meant to survive
    rendering, so it is stripped before the check rather than exempting the
    whole template -- a stray brace elsewhere in the merge still fails.
    """
    rendered = template.format(**{name: 'x' for name in _placeholders(template)})
    assert '{' not in rendered.replace("'''", "").replace(sql_json_object_regex(), "")


def test_the_prompt_survives_a_null_field():
    """Concatenating a NULL in BigQuery yields NULL, so one missing column makes
    the whole prompt NULL and AI.GENERATE returns nothing. The address lines
    were already guarded; `postcode` and `businessname` were not, which is why
    7 labelled rows had never been profiled despite being retried on every
    training and prediction run. See D-16."""
    for column in ('businessname', 'postcode', 'addressline1', 'addressline2', 'addressline3'):
        assert f"COALESCE({column}, '')" in SCRIPT_GENERATE_INSIGHTS, column
    # ...and no bare reference survives alongside the guarded one.
    for column in ('businessname', 'postcode'):
        assert f",{column}," not in SCRIPT_GENERATE_INSIGHTS.replace(' ', ''), column


def test_scratch_tables_expire():
    """Both scratch tables live in the production dataset; neither may outlive its run."""
    for template in (SCRIPT_IDENTIFY_RECENTS, SCRIPT_GENERATE_INSIGHTS):
        assert 'expiration_timestamp' in template


def test_model_params_are_valid_json():
    """The string is interpolated into a raw BigQuery literal, so nothing validates it downstream."""
    params = json.loads(MODEL_PARAMS_JSON)
    assert params['generationConfig']['maxOutputTokens'] > 0


def test_model_params_carry_no_triple_quote():
    """It is embedded in r'''...''' -- a triple quote inside would terminate the literal early."""
    assert "'''" not in MODEL_PARAMS_JSON


class TestMergeWritesTypedColumns:
    """Phase 6 dual-write. Every new profile must land in the typed columns as
    well as the JSON blob, or the Phase 5 backfill starts decaying the moment
    the next enrichment run finishes."""

    def test_every_canonical_column_is_assigned(self):
        for field in PILLAR_FIELDS:
            assert f"T.{field.column} =" in SCRIPT_MERGE_INSIGHTS, field.column

    def test_it_reads_the_scratch_table_not_the_master(self):
        """The raw payload is on S; T's own column is not written until this
        same statement, so extracting from T would read the previous run."""
        assert 'T.gemini_insights_structured, ' not in SCRIPT_MERGE_INSIGHTS
        assert SCRIPT_MERGE_INSIGHTS.count('S.gemini_insights') >= len(PILLAR_FIELDS)

    def test_it_no_longer_nulls_the_retired_v1_column(self):
        """`T.gemini_insights = NULL` was how the V2 merge cleared the V1 text
        as it superseded it. The column is dropped, so the assignment is now a
        reference to a column that does not exist -- and a MERGE that fails is
        a MERGE that loses a profile already paid for.
        """
        assert 'T.gemini_insights = NULL' not in SCRIPT_MERGE_INSIGHTS

    def test_the_raw_payload_is_still_kept(self):
        """`gemini_insights_structured` stays the audit trail: it is the only
        way to re-derive a column after a schema change, and the one row whose
        profile is unparseable exists only there."""
        assert 'T.gemini_insights_structured = S.gemini_insights' in SCRIPT_MERGE_INSIGHTS

    def test_it_stamps_the_profile_time(self):
        """Without this, Phase 7's staleness sweep re-profiles every row the
        merge just paid for."""
        assert 'T.gemini_profiled_at = CURRENT_TIMESTAMP()' in SCRIPT_MERGE_INSIGHTS

    def test_missing_scores_are_not_defaulted(self):
        assert 'IFNULL' not in SCRIPT_MERGE_INSIGHTS
        assert 'COALESCE' not in SCRIPT_MERGE_INSIGHTS

    def test_it_reads_the_nested_paths(self):
        assert '$.1_value_and_volume.rating' in SCRIPT_MERGE_INSIGHTS
        assert '1_value_and_volume_rating' not in SCRIPT_MERGE_INSIGHTS

    def test_a_failed_generation_is_not_recorded_as_a_profile(self):
        """AI.GENERATE returns NULL when its prompt is NULL. Merging that would
        stamp gemini_profiled_at on a row with no profile, which reads as fresh
        to a staleness sweep and as missing to the JIT guard -- the row is then
        retried forever and never refreshed. Observed live on 7 rows."""
        assert 'WHEN MATCHED AND S.gemini_insights IS NOT NULL THEN' in SCRIPT_MERGE_INSIGHTS

    def test_the_braces_in_the_unwrap_regex_are_escaped(self):
        """The template goes through `.format()`. A single brace here raises
        KeyError mid-enrichment, after the AI.GENERATE call has been paid for."""
        assert sql_json_object_regex(for_format_template=True) in SCRIPT_MERGE_INSIGHTS
        assert sql_json_object_regex() not in SCRIPT_MERGE_INSIGHTS
