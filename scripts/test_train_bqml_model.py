"""Tests for the training script's just-in-time enrichment pre-flight.

The training run is the third place that decides "does this row need a Maps
lookup?", and it has always made that decision with its own hand-copied
predicate. These pin it to the same rule as `ml_prediction.py` and
`enrich_maps_data.py`, because a row is either enriched or it is not and three
functions disagreeing about it is how money gets spent twice.

Every test that exercises the pre-flight passes `dry_run=False`, and that is
load-bearing rather than incidental: a dry run does not reach the pre-flight at
all. `TestADryRunSpendsNothing` is why.
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

    train_model('p', 'd', 't', 'm', dry_run=False)

    mock_enrich.assert_not_called()


@patch('scripts.enrich_maps_data.enrich_restaurants_by_fhrsid')
@patch('scripts.train_bqml_model.bigquery.Client')
def test_a_never_looked_up_restaurant_still_gets_enriched(mock_bq, mock_enrich):
    """The guard must not become a blanket off switch."""
    mock_bq.return_value = _mock_client(
        [DummyRow('2', maps_rating=None, maps_lookup_at=None)])

    train_model('p', 'd', 't', 'm', dry_run=False)

    mock_enrich.assert_called_once()


@patch('scripts.train_bqml_model.bigquery.Client')
def test_the_preflight_selects_the_columns_its_guards_read(mock_bq):
    """A guard on a column the query does not select raises AttributeError,
    which the bare `except` around the pre-flight would downgrade to a warning
    and silently skip all enrichment."""
    client = _mock_client([])
    mock_bq.return_value = client

    train_model('p', 'd', 't', 'm', dry_run=False)

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

    train_model('p', 'd', 't', 'm', dry_run=False)

    mock_gemini.assert_not_called()


@patch('app.services.bq_utils.execute_gemini_enrichment')
@patch('scripts.train_bqml_model.bigquery.Client')
def test_an_unprofiled_labelled_row_is_still_filled_in(mock_bq, mock_gemini):
    """The gap-filling half. These are labelled rows: a missing profile means
    the model trains on NULL features for a row a human actually rated."""
    mock_bq.return_value = _mock_client(
        [DummyRow('1', gemini_insights_structured=None, gemini_profiled_at=None)])

    train_model('p', 'd', 't', 'm', dry_run=False)

    mock_gemini.assert_called_once()
    assert mock_gemini.call_args.kwargs['fhrsids'] == ['1']


class TestADryRunSpendsNothing:
    """`--dry-run` is documented as "validate BQML training SQL without
    spending", and until D15 that was false.

    The pre-flight ran unconditionally, before the flag was ever consulted, so
    a dry run could issue grounded `AI.GENERATE` calls for every labelled row
    with no profile — the expensive half of the pipeline, from the flag whose
    whole purpose is to avoid spending. Worse, it is the flag a person reaches
    for precisely when they are unsure what a run will do.
    """

    _ALL_MISSING = dict(maps_rating=None, maps_lookup_at=None,
                        gemini_insights_structured=None, gemini_profiled_at=None,
                        d_postcode=None)

    @patch('scripts.enrich_postcode_demographics.enrich_postcodes')
    @patch('app.services.bq_utils.execute_gemini_enrichment')
    @patch('scripts.enrich_maps_data.enrich_restaurants_by_fhrsid')
    @patch('scripts.train_bqml_model.bigquery.Client')
    def test_a_dry_run_triggers_no_enrichment_of_any_kind(
        self, mock_bq, mock_maps, mock_gemini, mock_postcodes
    ):
        """The row is missing everything, so a run that reaches the pre-flight
        calls all three enrichers. A dry run must call none."""
        mock_bq.return_value = _mock_client([DummyRow('1', **self._ALL_MISSING)])

        train_model('p', 'd', 't', 'm', dry_run=True)

        mock_maps.assert_not_called()
        mock_gemini.assert_not_called()
        mock_postcodes.assert_not_called()

    @patch('scripts.enrich_postcode_demographics.enrich_postcodes')
    @patch('app.services.bq_utils.execute_gemini_enrichment')
    @patch('scripts.enrich_maps_data.enrich_restaurants_by_fhrsid')
    @patch('scripts.train_bqml_model.bigquery.Client')
    def test_the_same_row_does_get_enriched_on_a_real_run(
        self, mock_bq, mock_maps, mock_gemini, mock_postcodes
    ):
        """The other half: the guard must be the flag, not a blanket off
        switch. Without this the test above passes on a broken pre-flight."""
        mock_bq.return_value = _mock_client([DummyRow('1', **self._ALL_MISSING)])

        train_model('p', 'd', 't', 'm', dry_run=False)

        mock_maps.assert_called_once()
        mock_gemini.assert_called_once()
        mock_postcodes.assert_called_once()

    @patch('scripts.train_bqml_model.bigquery.Client')
    def test_a_dry_run_submits_nothing_to_bigquery_for_real(self, mock_bq):
        """Stronger than "no enrichment", and the same standard
        `migrate_pillar_columns.py` already holds itself to: every query a dry
        run submits carries `dry_run=True`. The pre-flight's own SELECT is
        pennies, but "dry" should mean dry rather than cheap."""
        client = _mock_client([DummyRow('1', **self._ALL_MISSING)])
        mock_bq.return_value = client

        train_model('p', 'd', 't', 'm', dry_run=True)

        assert client.query.called, "dry run issued no query at all"
        for call in client.query.call_args_list:
            config = call.kwargs.get('job_config')
            assert config is not None, f"submitted without a job_config: {call.args[0]}"
            assert config.dry_run, f"submitted for real: {call.args[0]}"

    @patch('scripts.train_bqml_model.bigquery.Client')
    def test_a_dry_run_still_validates_the_training_sql(self, mock_bq):
        """Skipping the pre-flight must not turn the dry run into a no-op.
        Validating the generated SQL is the entire point of the flag, and the
        feature list it interpolates changes often enough to be worth it."""
        client = _mock_client([])
        mock_bq.return_value = client

        train_model('p', 'd', 't', 'm', dry_run=True)

        submitted = [c.args[0] for c in client.query.call_args_list]
        assert any('CREATE OR REPLACE MODEL' in q for q in submitted)


class TestTheDryRunHandsBackItsAnswer:
    """`--dry-run`'s result is the byte estimate, and until D-28 it was logged
    and thrown away. The UI's validate button has nothing else to report."""

    @patch('scripts.train_bqml_model.bigquery.Client')
    def test_it_returns_the_bytes_the_query_would_process(self, mock_bq):
        client = _mock_client([])
        client.query.side_effect = None
        job = MagicMock()
        job.total_bytes_processed = 1234567
        client.query.return_value = job
        mock_bq.return_value = client

        assert train_model('p', 'd', 't', 'm', dry_run=True) == 1234567


class TestReadingBackAnAsyncJob:
    """`run_async=True` returns a job id in a second for a job that takes ten
    to fifteen minutes. Nothing read the other end of that until D-28."""

    def _status(self, state='DONE', error_result=None, location='EU'):
        from scripts.train_bqml_model import training_job_status

        client = MagicMock()
        client.get_dataset.return_value.location = location
        job = MagicMock()
        job.state = state
        job.error_result = error_result
        client.get_job.return_value = job

        with patch('scripts.train_bqml_model.bigquery.Client', return_value=client):
            return training_job_status('p', 'd', 'job-1'), client

    def test_a_running_job_reports_running(self):
        status, _ = self._status(state='RUNNING')

        assert status == {'state': 'RUNNING', 'error': None}

    def test_a_failed_job_is_done_and_carries_its_error(self):
        """The distinction the UI hangs on: BigQuery marks a failed query DONE,
        so "finished" and "succeeded" are different questions."""
        status, _ = self._status(
            error_result={'reason': 'invalidQuery',
                          'message': 'Unrecognized name: pillar_typo'})

        assert status['state'] == 'DONE'
        assert 'pillar_typo' in status['error']

    def test_the_job_is_looked_up_in_the_dataset_location(self):
        """`jobs.get` needs the location for anything outside the US, and this
        dataset is in the EU. Asking without it is a 404 on a job that exists."""
        _, client = self._status(location='EU')

        client.get_dataset.assert_called_once_with('p.d')
        assert client.get_job.call_args.kwargs['location'] == 'EU'
