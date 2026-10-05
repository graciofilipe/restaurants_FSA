"""The one definition of what the model looks at (D3).

Two places build this list: the `CREATE MODEL ... AS SELECT` in
`scripts/train_bqml_model.py` and the `ML.PREDICT` subquery in
`app/services/ml_prediction.py`. Until now they were hand-copied twenty-line
blocks, and CLAUDE.md carried a standing warning that editing one without the
other makes predictions "silently skew". Both now call `feature_select_list()`,
and `app/core/test_model_features.py` fails if the strings ever differ.

Two things change versus the copies this replaces, both deliberate:

* **The pillar paths are the canonical nested ones.** The old SQL read
  `$.1_value_and_volume_rating`; production has `$.1_value_and_volume.rating`.
  Five of the model's six Gemini features resolved on zero of 2,766 profiled
  rows. That is D2.
* **No `IFNULL(..., 0)`.** The default is what hid D2 for the life of the
  model: a feature that is always zero looks exactly like a feature the model
  found uninformative. BQML handles NULL natively, so the default bought
  nothing. Missing now reads as missing (R6).

The feature set also grows from six to eight. `pillar_geo_specificity` and
`pillar_establishment_type` are enums the old SQL cast to INT64 -- `CAST(
'GENERIC_NATIONAL' AS INT64)` is NULL, which the IFNULL then turned into 0 --
and `pillar_is_sit_down` was not read at all. As STRING and BOOL they are
categorical features BQML can actually split on.

**Changing this list changes the model's input schema**, so `ML.PREDICT` against
a model trained on the old one will fail or skew. Retrain in the same change.
"""
from typing import Tuple

from app.core.pillar_schema import FEATURE_COLUMNS, PILLAR_FIELDS, sql_extract

# Plain passthrough columns from `fsa_master`. `user_rating` is the label, named
# in `input_label_cols`; BQML needs it present at training and ignores it at
# prediction, so it stays in the shared list rather than being special-cased.
# High-cardinality geographic/array memorization columns (`postcode`, `latitude`,
# `longitude`, `maps_types_array`, `lsoa`, `msoa`) are intentionally excluded so
# boosted trees split on semantic pillars and engineered signals rather than
# memorizing postcodes or raw coordinates.
MASTER_COLUMNS: Tuple[str, ...] = (
    'localauthorityname',
    'ratingvalue',
    'user_rating',
    'price_level',
    'maps_rating',
    'maps_reviews',
    'business_status',
)

# From the `uk_postcode_demographics` join: all three columns order the
# deterministic deduplication subquery, while only `imd_rank` is exposed to the
# model feature vector.
DEMOGRAPHIC_COLUMNS: Tuple[str, ...] = ('lsoa', 'msoa', 'imd_rank')
DEMOGRAPHIC_FEATURE_COLUMNS: Tuple[str, ...] = ('imd_rank',)

ENGINEERED_FEATURE_ALIASES: Tuple[str, ...] = (
    'log_maps_reviews',
    'branch_count_in_fsa',
    'enclave_vs_hype_ratio',
    'dine_in_table_service_score',
)

# Generated, not listed: `is_feature` on a `PillarField` is what decides this.
PILLAR_FEATURE_ALIASES: Tuple[str, ...] = FEATURE_COLUMNS

FEATURE_ALIASES: Tuple[str, ...] = (
    MASTER_COLUMNS
    + ENGINEERED_FEATURE_ALIASES
    + PILLAR_FEATURE_ALIASES
    + DEMOGRAPHIC_FEATURE_COLUMNS
)

# The raw JSON the pillar features are read out of. Phase 7 switches this to the
# typed columns the Phase 5 backfill filled; because both consumers come through
# here, that is a one-line change rather than two hand-edited query bodies.
PROFILE_COLUMN = 'gemini_insights_structured'

_INDENT = '  '


def normalized_brand_key_sql(master: str = 'm') -> str:
    """Alphanumeric-normalised `businessname` key used for branch counting and
    brand-grouped cross-validation splits."""
    raw_lower = f"LOWER(TRIM(COALESCE({master}.businessname, '')))"
    stripped = (
        f"REGEXP_REPLACE(REGEXP_REPLACE({raw_lower}, "
        r"r'\([^)]*\)|\b(at|in)\s+the\b.*|\b(ltd|limited|plc|uk|llp|inc|restaurant|restaurants|cafe|deli|bombay|soho|victoria|waterloo|kingston|battersea)\b', "
        "''), r'[^a-z0-9]', '')"
    )
    fallback = f"REGEXP_REPLACE({raw_lower}, r'[^a-z0-9]', '')"
    return f"COALESCE(NULLIF({stripped}, ''), {fallback})"


def is_stage1_gated_sql(master: str = 'm', branch: str = 'b') -> str:
    """Boolean SQL predicate identifying structural non-candidates (Stage 1 of
    the Two-Stage Hurdle): out-of-scope rows, takeaways/counters, non-restaurant
    establishment types, or chains with >= 5 branches in `fsa_master`."""
    return (
        f"(COALESCE({master}.in_scope, FALSE) IS FALSE "
        f"OR COALESCE({master}.pillar_is_sit_down, FALSE) IS FALSE "
        f"OR COALESCE({master}.pillar_establishment_type, '') != 'RESTAURANT_DINING' "
        f"OR COALESCE({branch}.branch_count_in_fsa, 1) >= 5)"
    )


def stage1_deterministic_score_sql(master: str = 'm') -> str:
    """Deterministic Stage-1 cap in `[1.0, 2.0]` for structural non-candidates."""
    return (
        f"ROUND(LEAST(2.0, GREATEST(1.0, 1.0 + COALESCE({master}.match_score, 0.0) / 100.0)), 2)"
    )


def stage2_training_where_clause(master: str = 'm', branch: str = 'b') -> str:
    """Stage-2 training filter: only sit-down `RESTAURANT_DINING` candidates
    with `< 5` branches in `fsa_master`."""
    return (
        f"{master}.user_rating IS NOT NULL "
        f"AND {master}.in_scope IS TRUE "
        f"AND {master}.pillar_is_sit_down IS TRUE "
        f"AND {master}.pillar_establishment_type = 'RESTAURANT_DINING' "
        f"AND COALESCE({branch}.branch_count_in_fsa, 1) < 5"
    )


def feature_select_list(master: str = 'm', demo: str = 'd', branch: str = 'b',
                        for_format_template: bool = False) -> str:
    """The shared `SELECT` fragment, without a trailing comma.

    Emitted at a fixed indent so both call sites interpolate byte-identical
    text -- the parity test compares the strings, and re-indenting per caller
    would defeat it.

    `for_format_template` doubles the braces in the unwrap regex for SQL that
    later passes through `str.format()`. Get it wrong and the regex matches
    nothing, which is D2's failure mode all over again.
    """
    lines = [f"{_INDENT}{master}.{column}" for column in MASTER_COLUMNS]
    lines.extend([
        f"{_INDENT}LOG10(GREATEST(COALESCE(SAFE_CAST({master}.maps_reviews AS FLOAT64), 0.0), 0.0) + 1.0) AS log_maps_reviews",
        f"{_INDENT}COALESCE({branch}.branch_count_in_fsa, 1) AS branch_count_in_fsa",
        f"{_INDENT}SAFE_DIVIDE(CAST(COALESCE({master}.pillar_community_score, 5) AS FLOAT64), LOG10(GREATEST(COALESCE(SAFE_CAST({master}.maps_reviews AS FLOAT64), 0.0), 0.0) + 10.0)) AS enclave_vs_hype_ratio",
        f"{_INDENT}(COALESCE(SAFE_CAST({master}.pillar_dining_pace AS FLOAT64), 3.0) * IF(COALESCE({master}.pillar_is_sit_down, FALSE), 1.0, 0.0)) AS dine_in_table_service_score",
    ])
    profile_ref = f"{master}.{PROFILE_COLUMN}"
    for field in PILLAR_FIELDS:
        if not field.is_feature:
            continue
        extracted = sql_extract(field, profile_ref, for_format_template)
        if field.required:
            lines.append(f"{_INDENT}{extracted} AS {field.column}")
        else:
            lines.append(f"{_INDENT}COALESCE({extracted}, 3) AS {field.column}")
    lines.extend(f"{_INDENT}{demo}.{column}" for column in DEMOGRAPHIC_FEATURE_COLUMNS)
    return ",\n".join(lines)


def feature_source_clause(project_id: str, dataset_id: str, source_table: str,
                          master: str = 'm', demo: str = 'd', branch: str = 'b') -> str:
    """The `FROM`/`LEFT JOIN` the feature list is selected against.

    Shared for the same reason as the list itself: the demographics join key is
    normalised (`REPLACE(UPPER(...), ' ', '')`), and a training run joining on a
    different rule than the prediction run would skew `imd_rank` on exactly the
    rows whose postcodes are formatted inconsistently.

    The demographics join is against a deduplicated subquery, not the table
    (D-32). That key is normalised but the table is not: it holds one row per
    *raw* spelling, so 15 normalised postcodes appear two or three times and 103
    rows of `fsa_master` match more than one.

    The second `LEFT JOIN` computes `branch_count_in_fsa` per normalised brand
    key (`GROUP BY brand_key`), which is 1:1 per brand key and cannot fan out.
    """
    demographics = ", ".join(DEMOGRAPHIC_COLUMNS)
    best_first = ", ".join(f"{column} NULLS LAST" for column in DEMOGRAPHIC_COLUMNS)
    normalised = "REPLACE(UPPER(postcode), ' ', '')"
    brand_key_inner = normalized_brand_key_sql('b_src')
    brand_key_outer = normalized_brand_key_sql(master)
    return f"""FROM `{source_table}` AS {master}
LEFT JOIN (
  SELECT postcode_key, {demographics}
  FROM (
    SELECT {normalised} AS postcode_key, {demographics},
           ROW_NUMBER() OVER (PARTITION BY {normalised} ORDER BY {best_first}) AS _rn
    FROM `{project_id}.{dataset_id}.uk_postcode_demographics`
  )
  WHERE _rn = 1
) AS {demo}
  ON REPLACE(UPPER({master}.postcode), ' ', '') = {demo}.postcode_key
LEFT JOIN (
  SELECT {brand_key_inner} AS brand_key,
         COUNT(DISTINCT b_src.fhrsid) AS branch_count_in_fsa
  FROM `{source_table}` AS b_src
  WHERE {brand_key_inner} != ''
  GROUP BY brand_key
) AS {branch}
  ON {brand_key_outer} = {branch}.brand_key"""
