"""Tests for the training script's just-in-time enrichment pre-flight.

The training run is the third place that decides "does this row need a Maps
lookup?", and it has always made that decision with its own hand-copied
predicate. These pin it to the same rule as `ml_prediction.py` and
`enrich_maps_data.py`, because a row is either enriched or it is not and three
functions disagreeing about it is how money gets spent twice.
"""
from unittest.mock import MagicMock, patch

from scripts.train_bqml_model import train_model


class DummyRow:
    def __init__(self, fhrsid, maps_rating=4.5, maps_lookup_at='2026-01-01 00:00:00+00:00',
                 gemini_insights_structured='{"match_score": 80}',
                 postcode='SW16 1AA', d_postcode='SW16 1AA',
                 gemini_profiled_at='2026-01-01 00:00:00+00:00'):
        self.fhrsid = fhrsid
        self.maps_rating = maps_rating
        self.maps_lookup_at = maps_lookup_at
        self.gemini_insights_structured = gemini_insights_structured
        self.has_profile = gemini_insights_structured is not None
        self.gemini_profiled_at = gemini_profiled_at
        self.postcode = postcode
        self.d_postcode = d_postcode


def _mock_client(rows):
    client = MagicMock()
    check_job = MagicMock()
    check_job.result.return_value = rows
    client.query.side_effect = [check_job] + [MagicMock() for _ in range(4)]
    return client


@patch('scripts.enrich_maps_data.enrich_restaurants_by_fhrsid')
@patch('scripts.train_bqml_model.bigquery.Client')
def test_a_permanent_maps_miss_does_not_repay_for_places(mock_bq, mock_enrich):
    """Phase 5 nulled the ratings of 243 rows Places has permanently failed on.

    Under the old `maps_rating is None` guard every training run would re-query
    all of them, and the training run is the one thing here that is scheduled.
    """
    mock_bq.return_value = _mock_client([DummyRow('1', maps_rating=None)])

    train_model('p', 'd', 't', 'm', dry_run=True)

    mock_enrich.assert_not_called()


@patch('scripts.enrich_maps_data.enrich_restaurants_by_fhrsid')
@patch('scripts.train_bqml_model.bigquery.Client')
def test_a_never_looked_up_restaurant_still_gets_enriched(mock_bq, mock_enrich):
    """The guard must not become a blanket off switch."""
    mock_bq.return_value = _mock_client(
        [DummyRow('2', maps_rating=None, maps_lookup_at=None)])

    train_model('p', 'd', 't', 'm', dry_run=True)

    mock_enrich.assert_called_once()


@patch('scripts.train_bqml_model.bigquery.Client')
def test_the_preflight_selects_the_columns_its_guards_read(mock_bq):
    """A guard on a column the query does not select raises AttributeError,
    which the bare `except` around the pre-flight would downgrade to a warning
    and silently skip all enrichment."""
    client = _mock_client([])
    mock_bq.return_value = client

    train_model('p', 'd', 't', 'm', dry_run=True)

    check_query = client.query.call_args_list[0].args[0]
    assert 'maps_lookup_at' in check_query
    assert 'AS has_profile' in check_query


@patch('app.services.bq_utils.execute_gemini_enrichment')
@patch('scripts.train_bqml_model.bigquery.Client')
def test_training_never_refreshes_a_profile_it_already_has(mock_bq, mock_gemini):
    """Training is cheap; re-profiling is not. The Predict button applies a
    staleness threshold because a person pressed it — this path is scheduled,
    so it fills gaps only, however old the profile is."""
    mock_bq.return_value = _mock_client(
        [DummyRow('1', gemini_profiled_at='2019-01-01 00:00:00+00:00')])

    train_model('p', 'd', 't', 'm', dry_run=True)

    mock_gemini.assert_not_called()


@patch('app.services.bq_utils.execute_gemini_enrichment')
@patch('scripts.train_bqml_model.bigquery.Client')
def test_an_unprofiled_labelled_row_is_still_filled_in(mock_bq, mock_gemini):
    """The gap-filling half. These are labelled rows: a missing profile means
    the model trains on NULL features for a row a human actually rated."""
    mock_bq.return_value = _mock_client(
        [DummyRow('1', gemini_insights_structured=None, gemini_profiled_at=None)])

    train_model('p', 'd', 't', 'm', dry_run=True)

    mock_gemini.assert_called_once()
    assert mock_gemini.call_args.kwargs['fhrsids'] == ['1']
