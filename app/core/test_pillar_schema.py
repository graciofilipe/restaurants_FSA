"""Contract tests for the canonical pillar schema.

These run the real extraction against real payloads captured from production
in Phase 0. That is the whole point: every existing test in this repo feeds the
extraction its own hand-written literal, so all of them passed throughout the
period when all five pillar features were silently zero.

`tests/fixtures/gemini_profiles/` holds three shapes that actually occur:
plain JSON, markdown-fenced JSON, and a leaked reasoning trace with no JSON.
"""
import json
import pathlib

import pytest

from app.core.pillar_schema import (
    ALL_COLUMNS,
    CONFORMANCE_TOTAL,
    CONFORMANCE_UNPARSEABLE,
    FEATURE_COLUMNS,
    NON_JSON_COLUMNS,
    PILLAR_FIELDS,
    extract,
    missing_paths,
    sql_conformance_check,
    sql_extract,
    sql_json_object_regex,
    sql_select_list,
    summarise_conformance,
    unwrap_json,
)

FIXTURE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / 'tests/fixtures/gemini_profiles'
GOOD_FIXTURES = sorted(p for p in FIXTURE_DIR.glob('*.json') if '.' not in p.stem)
UNPARSEABLE = FIXTURE_DIR / '1855447.unparseable.json'
MARKDOWN_FENCED = FIXTURE_DIR / '1040595.markdown_fenced.json'


def test_the_fixtures_are_actually_present():
    """A silently empty fixture glob would make every test below vacuous."""
    assert len(GOOD_FIXTURES) >= 8
    assert UNPARSEABLE.exists() and MARKDOWN_FENCED.exists()


@pytest.mark.parametrize("fixture", GOOD_FIXTURES, ids=lambda p: p.stem)
def test_every_real_payload_conforms(fixture):
    """The contract. If a future profiler drifts, this is what fails."""
    assert missing_paths(unwrap_json(fixture.read_text())) == []


@pytest.mark.parametrize("fixture", GOOD_FIXTURES, ids=lambda p: p.stem)
def test_extraction_yields_every_column_from_real_payloads(fixture):
    values = extract(unwrap_json(fixture.read_text()))
    assert set(values) == set(ALL_COLUMNS)
    assert all(v is not None for v in values.values())


@pytest.mark.parametrize("fixture", GOOD_FIXTURES, ids=lambda p: p.stem)
def test_scores_are_numbers_and_the_sit_down_flag_is_boolean(fixture):
    """Types the BigQuery columns depend on, checked against real data rather
    than against the prompt's promise."""
    values = extract(unwrap_json(fixture.read_text()))
    for field in PILLAR_FIELDS:
        value = values[field.column]
        if field.bq_type == 'INT64':
            assert isinstance(value, int) and not isinstance(value, bool), field.column
        elif field.bq_type == 'BOOL':
            assert isinstance(value, bool), field.column
        else:
            assert isinstance(value, str), field.column


def test_markdown_fenced_payloads_still_parse():
    """22 rows in production wrap the JSON in ``` fences. The unwrap must cope,
    or the backfill loses them."""
    payload = unwrap_json(MARKDOWN_FENCED.read_text())
    assert payload is not None
    assert missing_paths(payload) == []


def test_a_leaked_reasoning_trace_is_missing_everything_not_zero():
    """The one production row with no JSON object. It must read as absent, so
    the backfill writes NULL. Reading it as 0 is the D2 failure mode exactly."""
    payload = unwrap_json(UNPARSEABLE.read_text())
    assert payload is None
    assert missing_paths(payload) == [f.json_path for f in PILLAR_FIELDS]
    assert set(extract(payload).values()) == {None}


def test_a_drifted_payload_is_detected_rather_than_absorbed():
    """The convention-D shape from the ADK recordings: no numeric prefixes and
    different leaf names. Nothing in the repo previously noticed this."""
    drifted = json.dumps({
        "value_and_volume": {"score": 7, "description": "..."},
        "match_score": 55,
    })
    missing = missing_paths(unwrap_json(drifted))
    assert '$.1_value_and_volume.rating' in missing
    assert '$.match_score' not in missing


def test_a_partially_drifted_payload_reports_only_what_is_missing():
    good = json.loads(GOOD_FIXTURES[0].read_text())
    del good['3_linguistic_signal']['score']
    assert missing_paths(good) == ['$.3_linguistic_signal.score']


def test_free_text_fields_are_not_model_features():
    """Unbounded model prose as a training feature is memorisation, not signal."""
    for column in ('pillar_value_verdict', 'pillar_community_evidence',
                   'pillar_culinary_pander_check', 'summary_reasoning', 'pillar_geo_region'):
        assert column not in FEATURE_COLUMNS


def test_the_feature_list_is_what_the_spec_says():
    assert set(FEATURE_COLUMNS) == {
        'match_score', 'pillar_value_rating', 'pillar_community_score',
        'pillar_linguistic_score', 'pillar_culinary_score',
        'pillar_geo_specificity', 'pillar_is_sit_down', 'pillar_establishment_type',
    }


def test_the_categorical_pillars_stay_strings():
    """CAST(specificity_level AS INT64) is what zeroes pillar 4 today. The
    schema has to make that impossible rather than merely discouraged."""
    by_column = {f.column: f for f in PILLAR_FIELDS}
    assert by_column['pillar_geo_specificity'].bq_type == 'STRING'
    assert by_column['pillar_establishment_type'].bq_type == 'STRING'


def test_column_names_are_unique():
    assert len(ALL_COLUMNS) == len(set(ALL_COLUMNS))
    assert not set(ALL_COLUMNS) & {c for c, _ in NON_JSON_COLUMNS}


def test_json_paths_are_unique():
    paths = [f.json_path for f in PILLAR_FIELDS]
    assert len(paths) == len(set(paths))


def test_generated_sql_reads_the_nested_paths_not_the_flat_ones():
    """The regression guard for D2. The flat spellings must never reappear."""
    sql = sql_select_list()
    assert '$.1_value_and_volume.rating' in sql
    assert '1_value_and_volume_rating' not in sql
    assert '6_establishment_integrity_is_sit_down_restaurant' not in sql


def test_generated_sql_never_hides_a_missing_value_behind_a_default():
    """IFNULL(..., 0) is how five dead features looked healthy for months."""
    sql = sql_select_list()
    assert 'IFNULL' not in sql
    assert 'COALESCE' not in sql


def test_generated_sql_uses_safe_cast_for_typed_fields():
    by_column = {f.column: f for f in PILLAR_FIELDS}
    assert 'SAFE_CAST' in sql_extract(by_column['pillar_value_rating'])
    assert 'SAFE_CAST' in sql_extract(by_column['pillar_is_sit_down'])
    assert 'CAST' not in sql_extract(by_column['summary_reasoning'])


def test_the_format_template_variant_doubles_its_braces():
    """A single-braced regex inside a .format() template raises KeyError at
    render time; a wrongly doubled one in plain SQL matches nothing and yields
    a column of silent NULLs. Both have happened in this repo."""
    assert sql_json_object_regex(for_format_template=True) == r"r'(?s)[{{].*[}}]'"
    assert sql_json_object_regex() == r"r'(?s)[{].*[}]'"
    assert sql_select_list(for_format_template=True).format() == sql_select_list()


def test_unwrap_handles_the_empty_and_null_cases():
    assert unwrap_json(None) is None
    assert unwrap_json('') is None
    assert unwrap_json('not json at all') is None
    assert unwrap_json('[1, 2, 3]') is None


def test_the_conformance_check_counts_every_canonical_path():
    sql = sql_conformance_check('p.d.t')
    for field in PILLAR_FIELDS:
        assert f"missing_{field.column}" in sql
        assert f"'{field.json_path}'" in sql


def test_the_conformance_check_only_reads():
    """It runs inside the enrichment path, so it must not be able to mutate."""
    sql = sql_conformance_check('p.d.t').upper()
    for keyword in ('INSERT', 'UPDATE', 'DELETE', 'MERGE', 'CREATE', 'DROP', 'ALTER'):
        assert keyword not in sql


def test_the_conformance_check_defaults_to_the_scratch_column_unfiltered():
    """A NULL AI.GENERATE result in the scratch table is a failed profile and
    must be counted, so the default must not filter NULLs away."""
    sql = sql_conformance_check('p.d.t')
    assert 'gemini_insights' in sql
    assert 'WHERE' not in sql


def test_the_conformance_check_accepts_a_predicate_for_auditing_master():
    sql = sql_conformance_check('p.d.t', column='gemini_insights_structured',
                                where='gemini_insights_structured IS NOT NULL')
    assert 'WHERE gemini_insights_structured IS NOT NULL' in sql


def test_summarise_reports_only_the_paths_that_actually_failed():
    row = {CONFORMANCE_TOTAL: 50, CONFORMANCE_UNPARSEABLE: 0,
           'missing_match_score': 0, 'missing_pillar_linguistic_score': 3}
    total, offenders = summarise_conformance(row)
    assert total == 50
    assert offenders == {'pillar_linguistic_score': 3}


def test_summarise_surfaces_unparseable_rows_alongside_missing_paths():
    row = {CONFORMANCE_TOTAL: 50, CONFORMANCE_UNPARSEABLE: 1, 'missing_match_score': 1}
    total, offenders = summarise_conformance(row)
    assert total == 50
    assert offenders == {'match_score': 1, CONFORMANCE_UNPARSEABLE: 1}


def test_summarise_of_a_clean_run_is_empty():
    row = {CONFORMANCE_TOTAL: 50, CONFORMANCE_UNPARSEABLE: 0}
    row.update({f"missing_{c}": 0 for c in ALL_COLUMNS})
    assert summarise_conformance(row) == (50, {})
