"""Guards on the recon script.

Recon exists to measure what the production pipeline actually reads. If its
path list drifts from the SQL it is meant to be measuring, it produces
confident numbers about the wrong thing -- worse than no numbers, because the
schema decisions in Phase 4 rest on them.
"""
import pathlib
import re

import pytest

from scripts.recon_pipeline_state import (
    DEFAULT_BQ_PATH,
    FLAT_SIT_DOWN_PATH,
    JSON_UNWRAP_REGEX,
    PRODUCTION_FEATURE_PATHS,
    PROMPT_NESTED_PATHS,
    _format_cost,
    build_queries,
    fixture_query,
    snapshot_query,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Every JSON path handed to JSON_EXTRACT_SCALAR, in source order. Non-greedy
# because the first argument is itself a REGEXP_EXTRACT call containing a comma.
_PATH_RE = re.compile(r"JSON_EXTRACT_SCALAR\(.+?,\s*'(\$\.[^']+)'\)")


def _paths_in(relative_path):
    return _PATH_RE.findall((REPO_ROOT / relative_path).read_text())


@pytest.mark.parametrize(
    "source", ["scripts/train_bqml_model.py", "app/services/ml_prediction.py"]
)
def test_recon_measures_the_paths_production_actually_reads(source):
    """The whole point of the D2 census. Drift here invalidates the finding."""
    assert _paths_in(source) == PRODUCTION_FEATURE_PATHS


def test_recon_covers_the_path_the_in_scope_migration_gated_on():
    """D13: the flat sit-down path, from migrate_to_in_scope_workflow.py."""
    assert FLAT_SIT_DOWN_PATH in _paths_in("scripts/migrate_to_in_scope_workflow.py")


def test_recon_unwraps_exactly_as_production_does():
    """Production writes the regex inside a .format() template, so its braces are
    doubled. Recon's copy is a plain string and must be the same regex once
    that doubling is undone -- otherwise it parses a different substring."""
    sql = (REPO_ROOT / "scripts/train_bqml_model.py").read_text()
    template_regex = re.search(r"REGEXP_EXTRACT\([^,]+, (r'\(\?s\)[^']+')\)", sql).group(1)
    assert template_regex.replace("{{", "{").replace("}}", "}") == JSON_UNWRAP_REGEX


def test_every_query_unwraps_before_parsing():
    """A query reading gemini_insights_structured raw would report markdown
    fences as unparseable JSON and miss every key."""
    for name, sql in build_queries(DEFAULT_BQ_PATH).items():
        if 'gemini_insights_structured' in sql and 'JSON_' in sql:
            assert 'REGEXP_EXTRACT' in sql, f"{name} parses the column without unwrapping it"


def test_parse_json_is_always_safe():
    """The column holds raw model output; a bare PARSE_JSON fails the whole
    query on the first malformed row, which is exactly the row worth counting."""
    for name, sql in build_queries(DEFAULT_BQ_PATH).items():
        for match in re.finditer(r"(\w*\.?)PARSE_JSON", sql):
            assert match.group(0) == 'SAFE.PARSE_JSON', f"{name} uses a bare PARSE_JSON"


def test_the_census_is_read_only():
    """Only --snapshot writes, and it is a separate function behind its own flag."""
    forbidden = ('INSERT', 'UPDATE', 'DELETE', 'MERGE', 'CREATE', 'DROP', 'ALTER', 'TRUNCATE')
    for name, sql in build_queries(DEFAULT_BQ_PATH).items():
        for keyword in forbidden:
            assert not re.search(rf'\b{keyword}\b', sql.upper()), f"{name} contains {keyword}"


def test_fixture_query_is_deterministic():
    """Fixtures are committed; a random sample would churn the diff every run."""
    assert 'ORDER BY' in fixture_query(DEFAULT_BQ_PATH, 8)


def test_snapshot_will_not_clobber_an_existing_backup():
    """CREATE OR REPLACE here would overwrite the restore point with whatever
    state a later phase had already left in fsa_master."""
    sql = snapshot_query(DEFAULT_BQ_PATH, 'fsa_master_backup_20260923')
    assert 'CREATE TABLE IF NOT EXISTS' in sql
    assert 'OR REPLACE' not in sql


def test_both_path_conventions_cover_the_same_six_pillars():
    """The census is only a comparison if both lists ask about the same things."""
    def pillar_numbers(paths):
        return {p[2] for p in paths if p[2].isdigit()}

    assert pillar_numbers(PRODUCTION_FEATURE_PATHS) == {'1', '2', '3', '4', '5'}
    assert pillar_numbers(PROMPT_NESTED_PATHS) == {'1', '2', '3', '4', '5', '6'}
    assert '$.match_score' in PRODUCTION_FEATURE_PATHS
    assert '$.match_score' in PROMPT_NESTED_PATHS


def test_cost_is_reported_in_pounds():
    assert '£' in _format_cost(1024 ** 4)
