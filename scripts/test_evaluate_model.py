"""Guards on the evaluation harness.

The harness exists to produce a before-number that Phase 9 can compare against.
Two things would silently destroy that: a split that leaks training rows into
the holdout, and a feature list that stops matching the one production trains
on. Both are tested here.
"""
import re

import pytest

from scripts.evaluate_model import (
    DEFAULT_HOLDOUT_MODULUS,
    HOLDOUT_BUCKET,
    MODEL_PREFIX,
    build_all_statements,
    build_boosted_tree_model_sql,
    build_evaluate_sql,
    build_match_score_model_sql,
    split_predicate,
)
from scripts.train_bqml_model import build_training_select

PROJECT, DATASET, TABLE = 'p', 'd', 't'
SOURCE = f'{PROJECT}.{DATASET}.{TABLE}'


def test_train_and_holdout_predicates_are_complements():
    """Any row satisfying one must fail the other, or the holdout is contaminated.

    The two must differ in the comparison operator and in nothing else -- a
    different modulus or a different hashed column on one side would silently
    put the same row in both splits.
    """
    train = split_predicate(5, 0, holdout=False)
    holdout = split_predicate(5, 0, holdout=True)
    assert train.count('!=') == 1 and holdout.count('!=') == 0
    assert train.replace('!=', '=') == holdout


def test_the_split_is_deterministic_not_random():
    """RAND() would reshuffle every run and make the Phase 9 comparison meaningless."""
    sql = split_predicate(5, 0, holdout=True)
    assert 'FARM_FINGERPRINT' in sql
    assert 'RAND' not in sql


def test_split_honours_the_modulus_and_bucket():
    assert 'MOD(ABS(FARM_FINGERPRINT(CAST(m.fhrsid AS STRING))), 7) = 3' in \
        split_predicate(7, 3, holdout=True)


def test_eval_models_cannot_overwrite_the_production_model():
    """A harness run must never replace the model the app serves from."""
    for name, sql, _ in build_all_statements(PROJECT, DATASET, TABLE, 5):
        assert 'restaurant_preference_model' not in sql, name
    assert MODEL_PREFIX != 'restaurant_preference_model'


def test_eval_models_stay_out_of_the_vertex_registry():
    """They are throwaway measurement artefacts, not deployable models."""
    sql = build_boosted_tree_model_sql(PROJECT, DATASET, SOURCE, 'm', 'AND 1=1')
    assert 'model_registry' not in sql


def test_the_boosted_tree_trains_on_the_production_feature_set():
    """If the harness drifts from `train_bqml_model.py`, it baselines a model
    that does not exist. Phase 9's delta would then be measuring the drift."""
    features = build_training_select(PROJECT, DATASET, SOURCE)
    sql = build_boosted_tree_model_sql(PROJECT, DATASET, SOURCE, 'm', '')
    assert features.strip() in sql


def test_evaluation_uses_the_same_feature_set_as_training():
    """Train/serve skew inside the harness itself would be undetectable in the output."""
    evaluate = build_evaluate_sql(PROJECT, DATASET, SOURCE, 'm', 'AND 1=1')
    assert build_training_select(PROJECT, DATASET, SOURCE, 'AND 1=1').strip() in evaluate


def test_training_uses_train_split_and_evaluation_uses_holdout():
    statements = dict((name, sql) for name, sql, _ in build_all_statements(PROJECT, DATASET, TABLE, 5))
    for name in ('train_boosted_tree', 'train_match_score_baseline'):
        assert f'!= {HOLDOUT_BUCKET}' in statements[name], name
    for name in ('evaluate_boosted_tree', 'evaluate_match_score', 'rank_correlation_boosted_tree'):
        assert f'= {HOLDOUT_BUCKET}' in statements[name]
        assert f'!= {HOLDOUT_BUCKET}' not in statements[name], name


def test_match_score_baseline_uses_only_match_score():
    """'No-ML baseline' means one feature. Any other column leaking in makes it
    a second model rather than a floor."""
    sql = build_match_score_model_sql(PROJECT, DATASET, SOURCE, 'm', '')
    for leaked in ('maps_rating', 'imd_rank', 'price_level', 'localauthorityname'):
        assert leaked not in sql, leaked
    assert 'match_score' in sql


def test_match_score_baseline_is_fitted_not_compared_raw():
    """`match_score` is 0-100 and `user_rating` is 1-10; an unfitted comparison
    would measure the scale gap. LINEAR_REG puts it on the label's scale."""
    sql = build_match_score_model_sql(PROJECT, DATASET, SOURCE, 'm', '')
    assert "model_type='LINEAR_REG'" in sql
    assert "input_label_cols=['user_rating']" in sql


def test_only_the_two_training_statements_write():
    writers = [name for name, _, writes in build_all_statements(PROJECT, DATASET, TABLE, 5) if writes]
    assert writers == ['train_boosted_tree', 'train_match_score_baseline']

    for name, sql, writes in build_all_statements(PROJECT, DATASET, TABLE, 5):
        if not writes:
            assert not re.search(r'\bCREATE\b', sql.upper()), name


def test_training_statements_precede_the_statements_that_read_them():
    order = [name for name, _, _ in build_all_statements(PROJECT, DATASET, TABLE, 5)]
    assert order.index('train_boosted_tree') < order.index('evaluate_boosted_tree')
    assert order.index('train_match_score_baseline') < order.index('evaluate_match_score')


def test_default_holdout_is_a_usable_fraction():
    """404 labelled rows: 1-in-5 gives ~80 holdout, ~324 train."""
    assert DEFAULT_HOLDOUT_MODULUS == 5
    assert 404 // DEFAULT_HOLDOUT_MODULUS >= 50


@pytest.mark.parametrize("predicate", ["", "AND 1=1"])
def test_training_select_always_filters_to_labelled_in_scope_rows(predicate):
    sql = build_training_select(PROJECT, DATASET, SOURCE, predicate)
    assert 'm.user_rating IS NOT NULL' in sql
    assert '(m.in_scope = TRUE OR m.in_scope IS NULL)' in sql
