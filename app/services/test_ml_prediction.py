from unittest.mock import patch, MagicMock

from app.services.ml_prediction import generate_predictions


class DummyRow:
    """A row as the find_query returns it.

    `gemini_insights` is None because SCRIPT_MERGE_INSIGHTS nulls it on every
    successful enrichment — that is the production state for every profiled row.
    """

    def __init__(self, fhrsid, maps_rating=4.5, gemini_insights=None,
                 gemini_insights_structured=None, postcode='SW16 1AA', d_postcode='SW16 1AA',
                 maps_lookup_at='2026-01-01 00:00:00+00:00'):
        self.fhrsid = fhrsid
        self.maps_rating = maps_rating
        self.maps_lookup_at = maps_lookup_at
        self.gemini_insights = gemini_insights
        self.gemini_insights_structured = gemini_insights_structured
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
        DummyRow('123', gemini_insights=None, gemini_insights_structured='{"match_score": 90}')
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
        DummyRow('123', gemini_insights=None, gemini_insights_structured=None)
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
        DummyRow('123', gemini_insights=None, gemini_insights_structured='{"match_score": 90}')
    ])

    generate_predictions(
        'project', 'dataset', 'table', 'model',
        target_fhrsids=['123'], force_maps=False, force_gemini=True,
    )

    mock_gemini.assert_called_once()


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
