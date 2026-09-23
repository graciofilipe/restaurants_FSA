"""Guards on the SQL templates.

These are format strings executed against production. A placeholder that is
renamed on one side only, or a stray brace in the prompt text, fails at
`.format()` time inside a BigQuery call -- not here, where it is cheap.
"""
import json
import string

import pytest

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
    """An unescaped brace in the prompt would raise here rather than mid-query."""
    rendered = template.format(**{name: 'x' for name in _placeholders(template)})
    assert '{' not in rendered.replace("'''", "")


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
