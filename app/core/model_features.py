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
MASTER_COLUMNS: Tuple[str, ...] = (
    'postcode',
    'localauthorityname',
    'ratingvalue',
    'user_rating',
    'price_level',
    'maps_rating',
    'maps_reviews',
    'latitude',
    'longitude',
    'business_status',
)

# From the `uk_postcode_demographics` join.
DEMOGRAPHIC_COLUMNS: Tuple[str, ...] = ('lsoa', 'msoa', 'imd_rank')

MAPS_TYPES_ALIAS = 'maps_types_array'

# Generated, not listed: `is_feature` on a `PillarField` is what decides this.
PILLAR_FEATURE_ALIASES: Tuple[str, ...] = FEATURE_COLUMNS

FEATURE_ALIASES: Tuple[str, ...] = (
    MASTER_COLUMNS + (MAPS_TYPES_ALIAS,) + PILLAR_FEATURE_ALIASES + DEMOGRAPHIC_COLUMNS
)

# The raw JSON the pillar features are read out of. Phase 7 switches this to the
# typed columns the Phase 5 backfill filled; because both consumers come through
# here, that is a one-line change rather than two hand-edited query bodies.
PROFILE_COLUMN = 'gemini_insights_structured'

_INDENT = '  '


def feature_select_list(master: str = 'm', demo: str = 'd',
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
    lines.append(
        f"{_INDENT}SPLIT(REPLACE({master}.maps_types, ' ', ''), ',') AS {MAPS_TYPES_ALIAS}"
    )
    profile_ref = f"{master}.{PROFILE_COLUMN}"
    lines.extend(
        f"{_INDENT}{sql_extract(field, profile_ref, for_format_template)} AS {field.column}"
        for field in PILLAR_FIELDS if field.is_feature
    )
    lines.extend(f"{_INDENT}{demo}.{column}" for column in DEMOGRAPHIC_COLUMNS)
    return ",\n".join(lines)


def feature_source_clause(project_id: str, dataset_id: str, source_table: str,
                          master: str = 'm', demo: str = 'd') -> str:
    """The `FROM`/`LEFT JOIN` the feature list is selected against.

    Shared for the same reason as the list itself: the demographics join key is
    normalised (`REPLACE(UPPER(...), ' ', '')`), and a training run joining on a
    different rule than the prediction run would skew `lsoa`/`msoa`/`imd_rank`
    on exactly the rows whose postcodes are formatted inconsistently.

    The join is against a deduplicated subquery, not the table (D-32). That key
    is normalised but the table is not: it holds one row per *raw* spelling, so
    15 normalised postcodes appear two or three times and 103 rows of
    `fsa_master` match more than one. On the prediction side that is fatal
    rather than untidy -- the duplicates reach the MERGE's source and BigQuery
    refuses the whole statement ("must match at most one source row for each
    target row"), after the caller has already paid for the Gemini pre-flight.
    On the training side it double-weights those rows.

    `ROW_NUMBER` and not `ANY_VALUE`: one of the 15 has two genuinely different
    payloads, so something has to choose, and a choice made independently at
    training time and at prediction time is the skew this module exists to
    prevent. Ordering by the payload makes both pick the same row, and picks a
    whole row rather than a column at a time.

    `NULLS LAST` because that one is `SW3 5 UH` -- a stray space in the raw
    postcode, fetched before the correct spelling and answered with an empty
    row. Ascending order defaults to nulls first in BigQuery, so the plain
    ordering would deterministically pick the blank over the real demographics.
    A populated row wins.
    """
    demographics = ", ".join(DEMOGRAPHIC_COLUMNS)
    best_first = ", ".join(f"{column} NULLS LAST" for column in DEMOGRAPHIC_COLUMNS)
    normalised = "REPLACE(UPPER(postcode), ' ', '')"
    return f"""FROM `{source_table}` AS {master}
LEFT JOIN (
  SELECT {normalised} AS postcode_key, {demographics}
  FROM `{project_id}.{dataset_id}.uk_postcode_demographics`
  QUALIFY ROW_NUMBER() OVER (PARTITION BY {normalised} ORDER BY {best_first}) = 1
) AS {demo}
  ON REPLACE(UPPER({master}.postcode), ' ', '') = {demo}.postcode_key"""
