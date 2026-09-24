import datetime
from unittest.mock import patch, MagicMock

from app.core.profile_freshness import GEMINI_PROFILE_MAX_AGE_DAYS
from app.services.ml_prediction import generate_predictions


def _days_ago(days):
    return datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)


class DummyRow:
    """A row as the find_query returns it.

    There is no `gemini_insights`: the V1 text column was dropped in Phase 10,
    so a row carrying one is a row BigQuery cannot return.

    `has_profile` is computed here exactly as the query computes it, so a test
    cannot set up a row that BigQuery could not return: a profile the column
    does not hold, or the reverse.
    """

    def __init__(self, fhrsid, maps_rating=4.5,
                 gemini_insights_structured=None, postcode='SW16 1AA', d_postcode='SW16 1AA',
                 maps_lookup_at='2026-01-01 00:00:00+00:00', gemini_profiled_at=None):
        self.fhrsid = fhrsid
        self.maps_rating = maps_rating
        self.maps_lookup_at = maps_lookup_at
        self.gemini_insights_structured = gemini_insights_structured
        self.has_profile = gemini_insights_structured is not None
        # Phase 5 stamped every row that already had a profile, so "profiled
        # but no timestamp" is not a state production is in.
        if gemini_profiled_at is None and self.has_profile:
            gemini_profiled_at = _days_ago(1)
        self.gemini_profiled_at = gemini_profiled_at
        self.postcode = postcode
        self.d_postcode = d_postcode


def _mock_client(rows):
    """Wire a BigQuery client whose first query returns `rows` and second is the MERGE."""
    client = MagicMock()
    find_job = MagicMock()
    find_job.result.return_value = rows
    predict_job = MagicMock()
    predict_job.num_dml_affected_rows = len(rows)
    client.query.side_effect = [find_job, predict_job]
    return client


@patch('app.services.ml_prediction.bigquery.Client')
@patch('app.services.ml_prediction.enrich_restaurants_by_fhrsid')
@patch('app.services.ml_prediction.execute_gemini_enrichment')
def test_profiled_restaurant_does_not_repay_for_gemini(mock_gemini, mock_maps, mock_bq):
    """A restaurant that already has a structured profile must not be re-profiled.

    Regression test for the cost defect: the guard used to read `gemini_insights`,
    which the merge sets to NULL every time, so every run re-paid for AI.GENERATE.
    """
    mock_bq.return_value = _mock_client([
        DummyRow('123', gemini_insights_structured='{"match_score": 90}')
    ])

    success, _ = generate_predictions(
        'project', 'dataset', 'table', 'model',
        target_fhrsids=['123'], force_maps=False, force_gemini=False,
    )

    assert success is True
    mock_gemini.assert_not_called()


@patch('app.services.ml_prediction.bigquery.Client')
@patch('app.services.ml_prediction.enrich_restaurants_by_fhrsid')
@patch('app.services.ml_prediction.execute_gemini_enrichment')
def test_unprofiled_restaurant_is_enriched(mock_gemini, mock_maps, mock_bq):
    mock_bq.return_value = _mock_client([
        DummyRow('123', gemini_insights_structured=None)
    ])

    generate_predictions(
        'project', 'dataset', 'table', 'model',
        target_fhrsids=['123'], force_maps=False, force_gemini=False,
    )

    mock_gemini.assert_called_once()
    assert mock_gemini.call_args.kwargs['fhrsids'] == ['123']


@patch('app.services.ml_prediction.bigquery.Client')
@patch('app.services.ml_prediction.enrich_restaurants_by_fhrsid')
@patch('app.services.ml_prediction.execute_gemini_enrichment')
def test_force_gemini_reprofiles_an_already_profiled_restaurant(mock_gemini, mock_maps, mock_bq):
    mock_bq.return_value = _mock_client([
        DummyRow('123', gemini_insights_structured='{"match_score": 90}')
    ])

    generate_predictions(
        'project', 'dataset', 'table', 'model',
        target_fhrsids=['123'], force_maps=False, force_gemini=True,
    )

    mock_gemini.assert_called_once()


@patch('app.services.ml_prediction.bigquery.Client')
@patch('app.services.ml_prediction.enrich_restaurants_by_fhrsid')
@patch('app.services.ml_prediction.execute_gemini_enrichment')
def test_a_profile_past_the_staleness_threshold_is_refreshed(mock_gemini, mock_maps, mock_bq):
    """Phase 7. The executor and the UI's estimate share one predicate, so this
    also pins what the "Estimated New Gemini Calls" figure will say."""
    mock_bq.return_value = _mock_client([
        DummyRow('123', gemini_insights_structured='{"match_score": 90}',
                 gemini_profiled_at=_days_ago(GEMINI_PROFILE_MAX_AGE_DAYS + 1))
    ])

    generate_predictions(
        'project', 'dataset', 'table', 'model',
        target_fhrsids=['123'], force_maps=False, force_gemini=False,
    )

    mock_gemini.assert_called_once()


@patch('app.services.ml_prediction.bigquery.Client')
@patch('app.services.ml_prediction.enrich_restaurants_by_fhrsid')
@patch('app.services.ml_prediction.execute_gemini_enrichment')
def test_a_profile_inside_the_threshold_is_not(mock_gemini, mock_maps, mock_bq):
    """The whole point of the timestamp. Every profiled row was stamped at the
    Phase 5 migration, so a threshold that caught them would re-profile 2,767
    rows the first time anyone pressed Predict."""
    mock_bq.return_value = _mock_client([
        DummyRow('123', gemini_insights_structured='{"match_score": 90}',
                 gemini_profiled_at=_days_ago(GEMINI_PROFILE_MAX_AGE_DAYS - 1))
    ])

    generate_predictions(
        'project', 'dataset', 'table', 'model',
        target_fhrsids=['123'], force_maps=False, force_gemini=False,
    )

    mock_gemini.assert_not_called()


@patch('app.services.ml_prediction.bigquery.Client')
@patch('app.services.ml_prediction.enrich_restaurants_by_fhrsid')
@patch('app.services.ml_prediction.execute_gemini_enrichment')
def test_the_find_query_selects_what_the_guard_reads(mock_gemini, mock_maps, mock_bq):
    """A guard on an unselected column raises AttributeError inside the `try`,
    which returns "Failed to identify target batch" and skips every step."""
    client = _mock_client([DummyRow('123', gemini_insights_structured='{"match_score": 90}')])
    mock_bq.return_value = client

    generate_predictions('project', 'dataset', 'table', 'model', target_fhrsids=['123'])

    find_query = client.query.call_args_list[0].args[0]
    assert 'AS has_profile' in find_query
    assert 'm.gemini_profiled_at' in find_query


@patch('app.services.ml_prediction.bigquery.Client')
@patch('app.services.ml_prediction.enrich_restaurants_by_fhrsid')
@patch('app.services.ml_prediction.execute_gemini_enrichment')
def test_targeted_find_query_reads_the_structured_column(mock_gemini, mock_maps, mock_bq):
    client = _mock_client([DummyRow('123', gemini_insights_structured='{"match_score": 90}')])
    mock_bq.return_value = client

    generate_predictions(
        'project', 'dataset', 'table', 'model', target_fhrsids=['123'],
    )

    find_query = client.query.call_args_list[0].args[0]
    assert 'm.gemini_insights_structured' in find_query
    assert 'm.gemini_insights,' not in find_query


@patch('app.services.ml_prediction.bigquery.Client')
@patch('app.services.ml_prediction.enrich_restaurants_by_fhrsid')
@patch('app.services.ml_prediction.execute_gemini_enrichment')
def test_untargeted_find_query_reads_the_structured_column(mock_gemini, mock_maps, mock_bq):
    """The untargeted branch carries the same guard and must be fixed with it."""
    client = _mock_client([DummyRow('123', gemini_insights_structured='{"match_score": 90}')])
    mock_bq.return_value = client

    generate_predictions('project', 'dataset', 'table', 'model', limit=10)

    find_query = client.query.call_args_list[0].args[0]
    assert 'm.gemini_insights_structured' in find_query
    assert 'm.gemini_insights,' not in find_query


@patch('app.services.ml_prediction.bigquery.Client')
@patch('app.services.ml_prediction.enrich_restaurants_by_fhrsid')
@patch('app.services.ml_prediction.execute_gemini_enrichment')
def test_a_permanent_maps_miss_does_not_repay_for_places(mock_gemini, mock_maps, mock_bq):
    """A row Places has already failed to find must not be looked up again.

    Phase 5 retired the `-1` sentinel and nulled those 243 ratings, so
    `maps_rating is None` no longer distinguishes "never tried" from "tried and
    found nothing". `maps_lookup_at` does.
    """
    row = DummyRow('1', maps_rating=None, maps_lookup_at='2026-01-01 00:00:00+00:00',
                   gemini_insights_structured='{"match_score": 80}')
    mock_bq.return_value = _mock_client([row])

    generate_predictions('p', 'd', 't', 'm', limit=1)

    mock_maps.assert_not_called()


@patch('app.services.ml_prediction.bigquery.Client')
@patch('app.services.ml_prediction.enrich_restaurants_by_fhrsid')
@patch('app.services.ml_prediction.execute_gemini_enrichment')
def test_a_never_looked_up_restaurant_still_gets_enriched(mock_gemini, mock_maps, mock_bq):
    """The other half of the guard: the change must not switch enrichment off."""
    row = DummyRow('2', maps_rating=None, maps_lookup_at=None,
                   gemini_insights_structured='{"match_score": 80}')
    mock_bq.return_value = _mock_client([row])

    generate_predictions('p', 'd', 't', 'm', limit=1)

    mock_maps.assert_called_once()


# --- The enrichment split, shared with the backfill script ---


class TestTheEnrichmentSplitIsOneFunction:
    """`split_enrichment_targets` answers "what does this batch still need?".

    It was inline in `generate_predictions` until the prediction backfill needed
    to ask the same question in order to *refuse* to run when the answer is not
    "nothing". A backfill that re-derives the predicate is a backfill that can
    disagree with the thing it is guarding, which is D1 with the cost moved to a
    different script.
    """

    def test_a_fully_enriched_batch_needs_nothing(self):
        from app.services.ml_prediction import split_enrichment_targets

        split = split_enrichment_targets([DummyRow('1', gemini_insights_structured='{}')])

        assert split['maps'] == []
        assert split['gemini'] == []
        assert split['postcodes'] == []
        assert split['fhrsids'] == ['1']

    def test_a_row_never_looked_up_in_maps_needs_maps(self):
        from app.services.ml_prediction import split_enrichment_targets

        split = split_enrichment_targets([
            DummyRow('1', gemini_insights_structured='{}', maps_lookup_at=None)])

        assert split['maps'] == ['1']

    def test_a_permanent_maps_miss_is_not_re_queried(self):
        """`maps_lookup_at`, not `maps_rating` -- the retired `-1` sentinel."""
        from app.services.ml_prediction import split_enrichment_targets

        split = split_enrichment_targets([
            DummyRow('1', gemini_insights_structured='{}', maps_rating=None)])

        assert split['maps'] == []

    def test_an_unprofiled_row_needs_gemini(self):
        from app.services.ml_prediction import split_enrichment_targets

        split = split_enrichment_targets([DummyRow('1')])

        assert split['gemini'] == ['1']
        assert split['never_profiled'] == 1

    def test_a_stale_profile_needs_gemini_and_is_not_counted_as_never_profiled(self):
        from app.services.ml_prediction import split_enrichment_targets

        split = split_enrichment_targets([
            DummyRow('1', gemini_insights_structured='{}',
                     gemini_profiled_at=_days_ago(GEMINI_PROFILE_MAX_AGE_DAYS + 1))])

        assert split['gemini'] == ['1']
        assert split['never_profiled'] == 0

    def test_an_unresolved_postcode_needs_demographics(self):
        from app.services.ml_prediction import split_enrichment_targets

        split = split_enrichment_targets([
            DummyRow('1', gemini_insights_structured='{}', d_postcode=None)])

        assert split['postcodes'] == ['1']

    def test_a_null_postcode_cannot_be_looked_up_and_is_not_queued(self):
        from app.services.ml_prediction import split_enrichment_targets

        split = split_enrichment_targets([
            DummyRow('1', gemini_insights_structured='{}', postcode=None, d_postcode=None)])

        assert split['postcodes'] == []

    def test_force_flags_queue_everything(self):
        from app.services.ml_prediction import split_enrichment_targets

        rows = [DummyRow('1', gemini_insights_structured='{}')]

        assert split_enrichment_targets(rows, force_maps=True)['maps'] == ['1']
        assert split_enrichment_targets(rows, force_gemini=True)['gemini'] == ['1']
