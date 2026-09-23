"""The single definition of the profiler's 6-pillar output.

Before this module the same schema existed in four places that disagreed with
each other -- the prompt's example output, the training SQL, the UI parser, and
a migration script -- and the disagreement was invisible because
`IFNULL(..., 0)` turned every wrong path into a zero. That is D2, and it cost
the model five of its six Gemini features.

Everything downstream is generated from `PILLAR_FIELDS`: the BigQuery columns,
the extraction SQL, the Python parser, the model feature list, and the
conformance check. Adding or renaming a field is a one-line change here.

Shape confirmed against production on 2026-09-23: 2,766 of 2,767 profiled rows
carry every path below. See D-08 in the track's decision log.
"""
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# The profiler returns prose around its JSON often enough that every consumer
# has always stripped it. Two real shapes in production: markdown ``` fences,
# and a leaked reasoning trace with no JSON at all.
JSON_OBJECT_RE = re.compile(r'(?s)[{].*[}]')

# The BigQuery equivalent of the above. Callers embedding this in a `.format()`
# template must double the braces; `sql_json_object_regex()` does that for you.
BQ_JSON_OBJECT_REGEX = r"r'(?s)[{].*[}]'"


@dataclass(frozen=True)
class PillarField:
    """One leaf of the profiler's output.

    `keys` is the same path as `json_path`, split for dict traversal, so the
    Python parser and the SQL cannot disagree about where a value lives.
    """
    column: str
    bq_type: str
    keys: Tuple[str, ...]
    is_feature: bool
    description: str

    @property
    def json_path(self) -> str:
        return '$.' + '.'.join(self.keys)


# Ordered as the prompt emits them. `is_feature` marks what the model trains
# on: the four integer pillar scores, the two categorical pillars, and
# match_score. The free-text fields are for the UI and must stay out of the
# feature list -- they are unbounded model prose, not signal.
PILLAR_FIELDS: Tuple[PillarField, ...] = (
    PillarField('match_score', 'INT64', ('match_score',), True,
                'Overall 0-100 fit against the taste profile.'),
    PillarField('pillar_value_rating', 'INT64', ('1_value_and_volume', 'rating'), True,
                'Pillar 1 score: value and volume.'),
    PillarField('pillar_value_verdict', 'STRING', ('1_value_and_volume', 'verdict'), False,
                'Pillar 1 free text.'),
    PillarField('pillar_community_score', 'INT64', ('2_demographic_community', 'score'), True,
                'Pillar 2 score: demographic community.'),
    PillarField('pillar_community_evidence', 'STRING', ('2_demographic_community', 'evidence'), False,
                'Pillar 2 free text.'),
    PillarField('pillar_linguistic_score', 'INT64', ('3_linguistic_signal', 'score'), True,
                'Pillar 3 score: linguistic signal.'),
    PillarField('pillar_linguistic_menu_type', 'STRING', ('3_linguistic_signal', 'menu_type'), False,
                'Pillar 3 free text.'),
    PillarField('pillar_geo_region', 'STRING', ('4_geographic_precision', 'region_identified'), False,
                'Pillar 4 free text: the cuisine region named.'),
    PillarField('pillar_geo_specificity', 'STRING', ('4_geographic_precision', 'specificity_level'), True,
                'Pillar 4 enum, e.g. GENERIC_NATIONAL. STRING, not INT64 -- '
                'casting this to an integer is what silently zeroes pillar 4 today.'),
    PillarField('pillar_culinary_score', 'INT64', ('5_culinary_uncompromisingness', 'score'), True,
                'Pillar 5 score: culinary uncompromisingness.'),
    PillarField('pillar_culinary_pander_check', 'STRING', ('5_culinary_uncompromisingness', 'pander_check'), False,
                'Pillar 5 free text.'),
    PillarField('pillar_is_sit_down', 'BOOL', ('6_establishment_integrity', 'is_sit_down_restaurant'), True,
                'Pillar 6: the in_scope gate. Mis-derived in production -- see D13.'),
    PillarField('pillar_establishment_type', 'STRING', ('6_establishment_integrity', 'type'), True,
                'Pillar 6 enum, e.g. FAST_FOOD_JOINT.'),
    PillarField('summary_reasoning', 'STRING', ('summary_reasoning',), False,
                'One-paragraph overall verdict, shown in the UI.'),
)

# Additive columns that are not derived from the profile JSON. They live here
# so Phase 4's migration has one list to read rather than two.
NON_JSON_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ('gemini_profiled_at', 'TIMESTAMP'),
    ('maps_lookup_at', 'TIMESTAMP'),
    ('maps_found', 'BOOL'),
)

FEATURE_COLUMNS: Tuple[str, ...] = tuple(f.column for f in PILLAR_FIELDS if f.is_feature)
ALL_COLUMNS: Tuple[str, ...] = tuple(f.column for f in PILLAR_FIELDS)


def unwrap_json(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    """Parse a stored `gemini_insights_structured` value, or None.

    Returns None rather than raising for the two malformed shapes production
    actually contains -- a markdown-fenced payload parses fine after the strip,
    a leaked reasoning trace has no object at all.
    """
    if not raw:
        return None
    match = JSON_OBJECT_RE.search(raw)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _walk(payload: Dict[str, Any], keys: Tuple[str, ...]) -> Any:
    node: Any = payload
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def extract(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Map a parsed profile onto the canonical columns.

    Missing values come back as None, never as 0. Reading a missing score as
    zero is precisely how D2 stayed invisible for the life of the model.
    """
    if not payload:
        return {field.column: None for field in PILLAR_FIELDS}
    return {field.column: _walk(payload, field.keys) for field in PILLAR_FIELDS}


def missing_paths(payload: Optional[Dict[str, Any]]) -> List[str]:
    """Which canonical paths this profile fails to provide.

    The conformance check. An empty list means the profile is usable as-is; a
    non-empty one is the signal that generation has drifted, which is the thing
    nothing in the repo could previously detect.
    """
    if not payload:
        return [field.json_path for field in PILLAR_FIELDS]
    return [f.json_path for f in PILLAR_FIELDS if _walk(payload, f.keys) is None]


def sql_json_object_regex(for_format_template: bool = False) -> str:
    """The unwrap regex as a BigQuery literal.

    `for_format_template` doubles the braces for SQL that is later passed
    through `str.format()`, which is how `scripts/bq_scripts.py` and the
    training query are built. Getting this wrong yields a regex that matches
    nothing and a column of silent NULLs.
    """
    if for_format_template:
        return BQ_JSON_OBJECT_REGEX.replace('{', '{{').replace('}', '}}')
    return BQ_JSON_OBJECT_REGEX


def sql_extract(field: PillarField, column_ref: str = 'm.gemini_insights_structured',
                for_format_template: bool = False) -> str:
    """The BigQuery expression reading one field, typed per the schema.

    SAFE_CAST, not CAST: a profile that returns "N/A" for a score should yield
    NULL rather than failing the whole statement.
    """
    regex = sql_json_object_regex(for_format_template)
    scalar = f"JSON_EXTRACT_SCALAR(REGEXP_EXTRACT({column_ref}, {regex}), '{field.json_path}')"
    if field.bq_type == 'STRING':
        return scalar
    return f"SAFE_CAST({scalar} AS {field.bq_type})"


def sql_select_list(column_ref: str = 'm.gemini_insights_structured',
                    fields: Optional[Tuple[PillarField, ...]] = None,
                    for_format_template: bool = False) -> str:
    """A `SELECT` fragment aliasing every field to its canonical column."""
    chosen = PILLAR_FIELDS if fields is None else fields
    return ",\n".join(
        f"  {sql_extract(f, column_ref, for_format_template)} AS {f.column}" for f in chosen
    )


# The conformance query lives here rather than in `scripts/bq_scripts.py` with
# the other SQL because it is generated from PILLAR_FIELDS. A hand-written copy
# in the templates file would be one more thing that can drift from the schema,
# which is the class of bug this module exists to end.
CONFORMANCE_UNPARSEABLE = 'unparseable'
CONFORMANCE_TOTAL = 'profiles'


def sql_conformance_check(table_ref: str, column: str = 'gemini_insights',
                          where: str = '') -> str:
    """Count, per canonical path, how many profiles fail to provide it.

    Run against the scratch insights table before the merge, so the reading
    describes what this run generated rather than the accumulated table. There
    the default empty `where` is deliberate: a row whose `AI.GENERATE` returned
    NULL is a failed profile and belongs in the unparseable count.

    Auditing the accumulated master table instead needs
    `where='gemini_insights_structured IS NOT NULL'`, or every never-profiled
    row lands in that same count and the reading means nothing.
    """
    regex = sql_json_object_regex()
    counters = ",\n".join(
        f"  COUNTIF(JSON_EXTRACT_SCALAR(u, '{f.json_path}') IS NULL) AS missing_{f.column}"
        for f in PILLAR_FIELDS
    )
    predicate = f"\n  WHERE {where}" if where else ""
    return f"""SELECT
  COUNT(*) AS {CONFORMANCE_TOTAL},
  COUNTIF(u IS NULL) AS {CONFORMANCE_UNPARSEABLE},
{counters}
FROM (
  SELECT REGEXP_EXTRACT({column}, {regex}) AS u
  FROM `{table_ref}`{predicate}
)"""


def summarise_conformance(row: Dict[str, Any]) -> Tuple[int, Dict[str, int]]:
    """Reduce a conformance row to (total, {column: missing}) for non-zero misses."""
    total = int(row.get(CONFORMANCE_TOTAL) or 0)
    offenders = {
        key[len('missing_'):]: int(value)
        for key, value in row.items()
        if key.startswith('missing_') and value
    }
    if row.get(CONFORMANCE_UNPARSEABLE):
        offenders[CONFORMANCE_UNPARSEABLE] = int(row[CONFORMANCE_UNPARSEABLE])
    return total, offenders
