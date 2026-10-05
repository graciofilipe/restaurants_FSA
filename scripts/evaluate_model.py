"""Held-out evaluation of the restaurant preference model.

The production model is trained on every labelled row, so its own
`ML.EVALUATE` scores the data it memorised. Nothing in the repo has ever
measured it on rows it did not see. This does, and it does so *before* the
feature repair, so Phase 9 has a comparable before-number.

Three numbers come out, all on the same held-out rows:

1. `boosted_tree`   -- the current model architecture and current features,
                       retrained on the training split only.
2. `match_score`    -- the no-ML baseline. A one-feature LINEAR_REG on
                       `match_score` alone. It has to be *fitted* rather than
                       compared raw: `match_score` runs 0-100 and `user_rating`
                       runs 1-10, so an unfitted MAE would measure the scale
                       difference, not the signal.
3. `mean`           -- predict the training mean for everything. The floor any
                       model must clear to have earned its existence.

    python -m scripts.evaluate_model                 # print SQL + cost, run nothing
    python -m scripts.evaluate_model --execute       # train the eval models and score

The split is a hash of `fhrsid`, so it is stable across runs without storing
anything. Phase 9 must pass the same `--holdout_modulus` to compare like with
like.
"""
import argparse
import json
import logging

from google.cloud import bigquery

from app.core.model_features import (
    feature_select_list,
    feature_source_clause,
    is_stage1_gated_sql,
    normalized_brand_key_sql,
    stage1_deterministic_score_sql,
    stage2_training_where_clause,
)
from scripts.train_bqml_model import build_create_model_sql, build_training_select

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_PROJECT_ID = "filipegracio-ai-learning"
DEFAULT_DATASET_ID = "filipegracio_fsa_restaurants"
DEFAULT_TABLE_ID = "fsa_master"

# 1 bucket in 5 held out, so ~20% of labelled brands.
DEFAULT_HOLDOUT_MODULUS = 5
HOLDOUT_BUCKET = 0

# Named apart from `restaurant_preference_model` so a harness run can never
# replace the model the app serves predictions from.
MODEL_PREFIX = "eval_holdout"


def split_predicate(modulus: int, bucket: int, holdout: bool, alias: str = "m") -> str:
    """Deterministic brand-grouped train/holdout carve, appended to the WHERE clause.

    Hashes the normalised `businessname` (`normalized_brand_key_sql(alias)`)
    rather than `fhrsid` so multi-branch chains never span both train and
    holdout splits.
    """
    operator = "=" if holdout else "!="
    brand_expr = normalized_brand_key_sql(alias)
    return (
        f"AND MOD(ABS(FARM_FINGERPRINT({brand_expr})), {modulus}) "
        f"{operator} {bucket}"
    )


def build_boosted_tree_model_sql(project_id, dataset_id, source_table, model_name, predicate) -> str:
    """Stage-2 regularized boosted tree on the training split.

    `register_vertex=False` omits `model_registry='vertex_ai'` because these are
    evaluation models rather than production registry artefacts.
    """
    full_model_name = f"{project_id}.{dataset_id}.{model_name}"
    return build_create_model_sql(
        project_id,
        dataset_id,
        source_table,
        full_model_name,
        model_family="boosted_tree",
        extra_predicate=predicate,
        register_vertex=False,
    ).strip()


def build_linear_reg_model_sql(project_id, dataset_id, source_table, model_name, predicate) -> str:
    """Stage-2 regularized linear regression (`l2_reg=1.0`) on the training split."""
    full_model_name = f"{project_id}.{dataset_id}.{model_name}"
    return build_create_model_sql(
        project_id,
        dataset_id,
        source_table,
        full_model_name,
        model_family="linear_reg",
        extra_predicate=predicate,
        register_vertex=False,
    ).strip()


def build_match_score_model_sql(project_id, dataset_id, source_table, model_name, predicate) -> str:
    """The no-ML baseline: `match_score` fitted to the label, nothing else."""
    return f"""CREATE OR REPLACE MODEL `{project_id}.{dataset_id}.{model_name}`
OPTIONS(
  model_type='LINEAR_REG',
  input_label_cols=['user_rating']
) AS
SELECT
  m.user_rating,
  IFNULL(CAST(JSON_EXTRACT_SCALAR(REGEXP_EXTRACT(m.gemini_insights_structured, r'(?s)[{{].*[}}]'), '$.match_score') AS INT64), 0) AS match_score
{feature_source_clause(project_id, dataset_id, source_table)}
WHERE
  {stage2_training_where_clause('m', 'b')}
  {predicate}"""


def build_evaluate_sql(
    project_id,
    dataset_id,
    source_table,
    model_name,
    predicate,
    model_family: str = "boosted_tree",
) -> str:
    """ML.EVALUATE of a trained model against the holdout rows."""
    return f"""SELECT * FROM ML.EVALUATE(
  MODEL `{project_id}.{dataset_id}.{model_name}`,
  ({build_training_select(project_id, dataset_id, source_table, predicate, model_family=model_family)})
)"""


def build_mean_baseline_sql(project_id, dataset_id, source_table, train_predicate, holdout_predicate) -> str:
    """Predict the training mean for every holdout row."""
    return f"""WITH train_mean AS (
  SELECT AVG(user_rating) AS mu
  FROM `{source_table}` AS m
  WHERE (m.in_scope = TRUE OR m.in_scope IS NULL) AND m.user_rating IS NOT NULL
  {train_predicate}
)
SELECT
  COUNT(*) AS holdout_rows,
  (SELECT mu FROM train_mean) AS prediction,
  AVG(ABS(m.user_rating - (SELECT mu FROM train_mean))) AS mean_absolute_error,
  SQRT(AVG(POW(m.user_rating - (SELECT mu FROM train_mean), 2))) AS root_mean_squared_error
FROM `{source_table}` AS m
WHERE (m.in_scope = TRUE OR m.in_scope IS NULL) AND m.user_rating IS NOT NULL
{holdout_predicate}"""


def build_rank_correlation_sql(
    project_id,
    dataset_id,
    source_table,
    model_name,
    predicate,
    model_family: str = "boosted_tree",
) -> str:
    """Spearman correlation between predicted and actual on the holdout."""
    return f"""WITH scored AS (
  SELECT predicted_user_rating, user_rating
  FROM ML.PREDICT(
    MODEL `{project_id}.{dataset_id}.{model_name}`,
    ({build_training_select(project_id, dataset_id, source_table, predicate, model_family=model_family)})
  )
),
ranked AS (
  SELECT
    RANK() OVER (ORDER BY predicted_user_rating) AS rank_pred,
    RANK() OVER (ORDER BY user_rating) AS rank_actual
  FROM scored
)
SELECT CORR(rank_pred, rank_actual) AS spearman FROM ranked"""


def build_match_score_rank_sql(source_table, holdout_predicate) -> str:
    """Spearman of raw `match_score` against the label -- no model at all."""
    return f"""WITH scored AS (
  SELECT
    m.user_rating,
    IFNULL(CAST(JSON_EXTRACT_SCALAR(REGEXP_EXTRACT(m.gemini_insights_structured, r'(?s)[{{].*[}}]'), '$.match_score') AS INT64), 0) AS match_score
  FROM `{source_table}` AS m
  WHERE (m.in_scope = TRUE OR m.in_scope IS NULL) AND m.user_rating IS NOT NULL
  {holdout_predicate}
),
ranked AS (
  SELECT
    RANK() OVER (ORDER BY match_score) AS rank_pred,
    RANK() OVER (ORDER BY user_rating) AS rank_actual
  FROM scored
)
SELECT CORR(rank_pred, rank_actual) AS spearman FROM ranked"""


def build_two_stage_hurdle_benchmark_sql(
    project_id: str,
    dataset_id: str,
    source_table: str,
    tree_model: str,
    linear_reg_model: str,
    holdout_predicate: str,
) -> str:
    """Evaluate `boosted_tree`, `linear_reg`, and their 50/50 ensemble on both
    the Plausible Holdout slice (`NOT _is_stage1_gated`) and the All Holdout
    slice (with Stage-1 deterministic capping)."""
    return f"""WITH holdout_features AS (
  SELECT
    m.fhrsid,
    COALESCE(m.match_score, 0.0) AS raw_match_score,
    {is_stage1_gated_sql('m', 'b')} AS _is_stage1_gated,
    {stage1_deterministic_score_sql('m')} AS _stage1_capped_score,
{feature_select_list()}
{feature_source_clause(project_id, dataset_id, source_table)}
  WHERE (m.in_scope = TRUE OR m.in_scope IS NULL)
    AND m.user_rating IS NOT NULL
    {holdout_predicate}
),
tree_preds AS (
  SELECT fhrsid, user_rating, raw_match_score, _is_stage1_gated, _stage1_capped_score,
         predicted_user_rating AS tree_pred
  FROM ML.PREDICT(MODEL `{project_id}.{dataset_id}.{tree_model}`, TABLE holdout_features)
),
lin_preds AS (
  SELECT fhrsid, predicted_user_rating AS lin_pred
  FROM ML.PREDICT(MODEL `{project_id}.{dataset_id}.{linear_reg_model}`, TABLE holdout_features)
),
combined AS (
  SELECT
    t.fhrsid,
    t.user_rating,
    t.raw_match_score,
    t._is_stage1_gated,
    IF(t._is_stage1_gated, t._stage1_capped_score, ROUND(LEAST(10.0, GREATEST(1.0, t.tree_pred)), 2)) AS pred_tree,
    IF(t._is_stage1_gated, t._stage1_capped_score, ROUND(LEAST(10.0, GREATEST(1.0, l.lin_pred)), 2)) AS pred_lin,
    IF(t._is_stage1_gated, t._stage1_capped_score, ROUND(LEAST(10.0, GREATEST(1.0, 0.5 * t.tree_pred + 0.5 * l.lin_pred)), 2)) AS pred_ens
  FROM tree_preds AS t
  JOIN lin_preds AS l USING (fhrsid)
),
plausible_ranked AS (
  SELECT
    user_rating,
    pred_tree,
    pred_lin,
    pred_ens,
    raw_match_score,
    RANK() OVER (ORDER BY user_rating) AS r_actual,
    RANK() OVER (ORDER BY pred_tree) AS r_tree,
    RANK() OVER (ORDER BY pred_lin) AS r_lin,
    RANK() OVER (ORDER BY pred_ens) AS r_ens,
    RANK() OVER (ORDER BY raw_match_score) AS r_match
  FROM combined
  WHERE NOT _is_stage1_gated
),
all_ranked AS (
  SELECT
    user_rating,
    pred_tree,
    pred_lin,
    pred_ens,
    raw_match_score,
    RANK() OVER (ORDER BY user_rating) AS r_actual,
    RANK() OVER (ORDER BY pred_tree) AS r_tree,
    RANK() OVER (ORDER BY pred_lin) AS r_lin,
    RANK() OVER (ORDER BY pred_ens) AS r_ens,
    RANK() OVER (ORDER BY raw_match_score) AS r_match
  FROM combined
)
SELECT
  'plausible_holdout' AS slice,
  COUNT(*) AS n,
  ROUND(AVG(ABS(user_rating - pred_tree)), 4) AS tree_mae,
  ROUND(CORR(r_tree, r_actual), 4) AS tree_spearman,
  ROUND(AVG(ABS(user_rating - pred_lin)), 4) AS lin_mae,
  ROUND(CORR(r_lin, r_actual), 4) AS lin_spearman,
  ROUND(AVG(ABS(user_rating - pred_ens)), 4) AS ens_mae,
  ROUND(CORR(r_ens, r_actual), 4) AS ens_spearman,
  ROUND(CORR(r_match, r_actual), 4) AS match_score_spearman
FROM plausible_ranked
UNION ALL
SELECT
  'all_holdout' AS slice,
  COUNT(*) AS n,
  ROUND(AVG(ABS(user_rating - pred_tree)), 4) AS tree_mae,
  ROUND(CORR(r_tree, r_actual), 4) AS tree_spearman,
  ROUND(AVG(ABS(user_rating - pred_lin)), 4) AS lin_mae,
  ROUND(CORR(r_lin, r_actual), 4) AS lin_spearman,
  ROUND(AVG(ABS(user_rating - pred_ens)), 4) AS ens_mae,
  ROUND(CORR(r_ens, r_actual), 4) AS ens_spearman,
  ROUND(CORR(r_match, r_actual), 4) AS match_score_spearman
FROM all_ranked"""


def build_split_sizes_sql(source_table, train_predicate, holdout_predicate) -> str:
    """How the split actually landed, and whether the holdout is representative."""
    base = "(m.in_scope = TRUE OR m.in_scope IS NULL) AND m.user_rating IS NOT NULL"
    plausible = (
        f"{base} AND m.pillar_is_sit_down IS TRUE "
        "AND m.pillar_establishment_type = 'RESTAURANT_DINING'"
    )
    return f"""SELECT
  (SELECT COUNT(*) FROM `{source_table}` AS m WHERE {base} {train_predicate}) AS train_rows,
  (SELECT COUNT(*) FROM `{source_table}` AS m WHERE {base} {holdout_predicate}) AS holdout_rows,
  (SELECT COUNT(*) FROM `{source_table}` AS m WHERE {plausible} {train_predicate}) AS plausible_train_rows,
  (SELECT COUNT(*) FROM `{source_table}` AS m WHERE {plausible} {holdout_predicate}) AS plausible_holdout_rows,
  (SELECT ROUND(AVG(m.user_rating), 3) FROM `{source_table}` AS m WHERE {base} {train_predicate}) AS train_mean_rating,
  (SELECT ROUND(AVG(m.user_rating), 3) FROM `{source_table}` AS m WHERE {base} {holdout_predicate}) AS holdout_mean_rating,
  (SELECT COUNT(*) FROM `{source_table}` AS m WHERE m.user_rating IS NOT NULL) AS labelled_rows_total"""


def build_all_statements(project_id, dataset_id, table_id, modulus) -> list:
    """Every statement the harness runs, in order, as (label, sql, writes) tuples."""
    source_table = f"{project_id}.{dataset_id}.{table_id}"
    train = split_predicate(modulus, HOLDOUT_BUCKET, holdout=False)
    holdout = split_predicate(modulus, HOLDOUT_BUCKET, holdout=True)
    tree_model = f"{MODEL_PREFIX}_boosted_tree"
    linear_reg_model = f"{MODEL_PREFIX}_linear_reg"
    match_model = f"{MODEL_PREFIX}_match_score"

    return [
        ("split_sizes",
         build_split_sizes_sql(source_table, train, holdout), False),
        ("train_boosted_tree",
         build_boosted_tree_model_sql(project_id, dataset_id, source_table, tree_model, train), True),
        ("train_linear_reg",
         build_linear_reg_model_sql(project_id, dataset_id, source_table, linear_reg_model, train), True),
        ("train_match_score_baseline",
         build_match_score_model_sql(project_id, dataset_id, source_table, match_model, train), True),
        ("evaluate_boosted_tree",
         build_evaluate_sql(project_id, dataset_id, source_table, tree_model, holdout, model_family="boosted_tree"), False),
        ("evaluate_linear_reg",
         build_evaluate_sql(project_id, dataset_id, source_table, linear_reg_model, holdout, model_family="linear_reg"), False),
        ("evaluate_match_score",
         build_evaluate_sql(project_id, dataset_id, source_table, match_model, holdout), False),
        ("mean_baseline",
         build_mean_baseline_sql(project_id, dataset_id, source_table, train, holdout), False),
        ("rank_correlation_boosted_tree",
         build_rank_correlation_sql(project_id, dataset_id, source_table, tree_model, holdout, model_family="boosted_tree"), False),
        ("rank_correlation_linear_reg",
         build_rank_correlation_sql(project_id, dataset_id, source_table, linear_reg_model, holdout, model_family="linear_reg"), False),
        ("rank_correlation_match_score",
         build_match_score_rank_sql(source_table, holdout), False),
        ("two_stage_hurdle_benchmark",
         build_two_stage_hurdle_benchmark_sql(
             project_id, dataset_id, source_table, tree_model, linear_reg_model, holdout
         ), False),
    ]


def run_evaluation(project_id=DEFAULT_PROJECT_ID, dataset_id=DEFAULT_DATASET_ID,
                   table_id=DEFAULT_TABLE_ID, modulus=DEFAULT_HOLDOUT_MODULUS,
                   execute=False) -> dict:
    client = bigquery.Client(project=project_id)
    results = {}

    for name, sql, writes in build_all_statements(project_id, dataset_id, table_id, modulus):
        marker = " (creates a model)" if writes else ""
        logger.info(f"--- {name}{marker} ---\n{sql}\n")

        if not execute:
            # ML.EVALUATE against a model that does not exist yet cannot be
            # dry-run, so only the statements that can be validated are.
            if writes or name == 'split_sizes' or name == 'mean_baseline':
                try:
                    job = client.query(sql, job_config=bigquery.QueryJobConfig(
                        dry_run=True, use_query_cache=False))
                    logger.info(f"[{name}] valid; {job.total_bytes_processed / (1024 ** 2):.1f} MiB")
                except Exception as e:
                    logger.error(f"[{name}] dry run failed: {e}")
            else:
                logger.info(f"[{name}] not dry-runnable until the model exists")
            continue

        try:
            rows = [dict(r) for r in client.query(sql).result()]
            results[name] = rows
            if rows:
                logger.info(f"[{name}] {rows[0]}")
            else:
                logger.info(f"[{name}] done")
        except Exception as e:
            logger.error(f"[{name}] failed: {e}")
            results[name] = {'error': str(e)}

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--project_id", default=DEFAULT_PROJECT_ID)
    parser.add_argument("--dataset_id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--table_id", default=DEFAULT_TABLE_ID)
    parser.add_argument("--holdout_modulus", type=int, default=DEFAULT_HOLDOUT_MODULUS,
                        help="1 row in N is held out. Phase 9 must reuse this value.")
    parser.add_argument("--execute", action="store_true",
                        help="Train the eval models and score them. Without this, SQL only.")
    parser.add_argument("--report", default="", help="Write results as JSON to this path.")
    args = parser.parse_args()

    out = run_evaluation(args.project_id, args.dataset_id, args.table_id,
                         args.holdout_modulus, args.execute)
    if args.report and out:
        with open(args.report, "w") as fh:
            json.dump(out, fh, indent=2, default=str)
        logger.info(f"Report written to {args.report}")
