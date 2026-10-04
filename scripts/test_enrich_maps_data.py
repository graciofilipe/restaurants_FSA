from unittest.mock import patch, MagicMock

from scripts.enrich_maps_data import enrich_restaurants_by_fhrsid


class DummyRow:
    def __init__(self, fhrsid='123', BusinessName='Taste of Sichuan',
                 PostCode='SW16 1AA', AddressLine1='1 High St'):
        self.fhrsid = fhrsid
        self.BusinessName = BusinessName
        self.PostCode = PostCode
        self.AddressLine1 = AddressLine1


def _mock_client(rows):
    client = MagicMock()
    select_job = MagicMock()
    select_job.result.return_value = rows
    client.query.side_effect = [select_job, MagicMock()]
    return client


def _merge_query(client):
    """The second query issued is the MERGE."""
    return client.query.call_args_list[1].args[0]


def _response(payload, status=200):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload
    return resp


@patch('scripts.enrich_maps_data.time.sleep')
@patch('scripts.enrich_maps_data.requests.post')
@patch('scripts.enrich_maps_data.bigquery.Client')
def test_places_miss_does_not_erase_existing_coordinates(mock_bq, mock_post, _sleep):
    """A Places miss must not clear latitude/longitude.

    The miss payload carries latitude=None, and the MERGE used to assign it
    unconditionally. Harmless while nothing else populated those columns, but
    destructive once FSA coordinates are stored at ingest.
    """
    client = _mock_client([DummyRow()])
    mock_bq.return_value = client
    mock_post.return_value = _response({})

    enrich_restaurants_by_fhrsid(fhrsids=['123'])

    merge = _merge_query(client)
    assert 'latitude=IFNULL(S.latitude, T.latitude)' in merge
    assert 'longitude=IFNULL(S.longitude, T.longitude)' in merge


@patch('scripts.enrich_maps_data.time.sleep')
@patch('scripts.enrich_maps_data.requests.post')
@patch('scripts.enrich_maps_data.bigquery.Client')
def test_places_hit_still_writes_coordinates(mock_bq, mock_post, _sleep):
    client = _mock_client([DummyRow()])
    mock_bq.return_value = client
    mock_post.return_value = _response({
        "places": [{
            "rating": 4.7,
            "userRatingCount": 2723,
            "location": {"latitude": 51.42, "longitude": -0.12},
            "types": ["restaurant"],
        }]
    })

    enrich_restaurants_by_fhrsid(fhrsids=['123'])

    merge = _merge_query(client)
    assert '51.42' in merge
    assert '-0.12' in merge


@patch('scripts.enrich_maps_data.time.sleep')
@patch('scripts.enrich_maps_data.requests.post')
@patch('scripts.enrich_maps_data.bigquery.Client')
def test_hit_without_location_preserves_coordinates(mock_bq, mock_post, _sleep):
    """Places can return a match with no location; that is not a reason to clear ours."""
    client = _mock_client([DummyRow()])
    mock_bq.return_value = client
    mock_post.return_value = _response({"places": [{"rating": 4.1, "userRatingCount": 10}]})

    enrich_restaurants_by_fhrsid(fhrsids=['123'])

    merge = _merge_query(client)
    assert 'latitude=IFNULL(S.latitude, T.latitude)' in merge


@patch('scripts.enrich_maps_data.time.sleep')
@patch('scripts.enrich_maps_data.requests.post')
@patch('scripts.enrich_maps_data.bigquery.Client')
def test_a_permanent_miss_is_not_re_queried(mock_bq, mock_post, _sleep):
    """The do-not-retry guard reads `maps_lookup_at`, not `maps_rating`.

    `maps_rating = -1` used to carry two meanings at once: "Places has nothing
    for this" and "do not spend another lookup on it". Phase 5 retired the
    sentinel and nulled those 243 ratings, so a guard on `maps_rating IS NULL`
    now reads every permanent miss as never-tried and re-queries it forever --
    the paid version of D1.
    """
    client = _mock_client([])
    mock_bq.return_value = client

    enrich_restaurants_by_fhrsid(fhrsids=[])

    select_query = client.query.call_args_list[0].args[0]
    assert 'maps_lookup_at IS NULL' in select_query
    assert 'maps_rating IS NULL' not in select_query


@patch('scripts.enrich_maps_data.time.sleep')
@patch('scripts.enrich_maps_data.requests.post')
@patch('scripts.enrich_maps_data.bigquery.Client')
def test_force_regen_still_overrides_the_guard(mock_bq, mock_post, _sleep):
    """Re-checking a stale lookup on purpose must stay possible."""
    client = _mock_client([])
    mock_bq.return_value = client

    enrich_restaurants_by_fhrsid(fhrsids=[], force_regen=True)

    assert 'maps_lookup_at IS NULL' not in client.query.call_args_list[0].args[0]


@patch('scripts.enrich_maps_data.time.sleep')
@patch('scripts.enrich_maps_data.requests.post')
@patch('scripts.enrich_maps_data.bigquery.Client')
def test_a_miss_records_the_lookup_instead_of_a_sentinel(mock_bq, mock_post, _sleep):
    """The merge writes the flags the guard now reads, and no more `-1`.

    Without this the guard change is a one-way door: Phase 5 cleaned the
    existing sentinels, but the next Places miss would write a fresh one and
    leave `maps_lookup_at` NULL, putting the row straight back in the queue.
    """
    client = _mock_client([DummyRow()])
    mock_bq.return_value = client
    mock_post.return_value = _response({})

    enrich_restaurants_by_fhrsid(fhrsids=['123'])

    merge = _merge_query(client)
    assert 'maps_found' in merge
    assert 'maps_lookup_at' in merge
    assert '-1' not in merge


@patch('scripts.enrich_maps_data.time.sleep')
@patch('scripts.enrich_maps_data.requests.post')
@patch('scripts.enrich_maps_data.bigquery.Client')
def test_places_http_call_passes_explicit_timeout(mock_bq, mock_post, _sleep):
    client = _mock_client([DummyRow()])
    mock_bq.return_value = client
    mock_post.return_value = _response({})

    enrich_restaurants_by_fhrsid(fhrsids=['123'])

    assert mock_post.call_args.kwargs.get('timeout') == 10
    select_job = client.query.side_effect  # already consumed; check mock calls
    assert client.query.call_count == 2


@patch('scripts.enrich_maps_data.time.sleep')
@patch('scripts.enrich_maps_data.requests.post')
@patch('scripts.enrich_maps_data.bigquery.Client')
def test_stalled_bigquery_select_cancels_job_and_raises(mock_bq, mock_post, _sleep):
    import pytest
    from app.core.profile_freshness import PreFlightEnrichmentError

    client = MagicMock()
    stalled_job = MagicMock()
    stalled_job.result.side_effect = TimeoutError("BigQuery SELECT timed out")
    client.query.return_value = stalled_job
    mock_bq.return_value = client

    with pytest.raises(PreFlightEnrichmentError, match="Maps BigQuery SELECT failed or timed out"):
        enrich_restaurants_by_fhrsid(fhrsids=['123'])

    stalled_job.cancel.assert_called_once()
    mock_post.assert_not_called()


@patch('scripts.enrich_maps_data.time.sleep')
@patch('scripts.enrich_maps_data.requests.post')
@patch('scripts.enrich_maps_data.bigquery.Client')
def test_places_http_failures_exceeding_5_percent_raise_preflight_error(mock_bq, mock_post, _sleep):
    import pytest
    from app.core.profile_freshness import PreFlightEnrichmentError

    # 1 row targeted, 1 fails (100% > 5%) -> raises
    client = _mock_client([DummyRow('1')])
    mock_bq.return_value = client
    mock_post.side_effect = TimeoutError("socket hung")

    with pytest.raises(PreFlightEnrichmentError, match="Maps Places API failed"):
        enrich_restaurants_by_fhrsid(fhrsids=['1'])


@patch('scripts.enrich_maps_data.time.sleep')
@patch('scripts.enrich_maps_data.requests.post')
@patch('scripts.enrich_maps_data.bigquery.Client')
def test_places_http_failure_within_5_percent_tolerance_succeeds(mock_bq, mock_post, _sleep):
    # 20 rows targeted, 1 fails (5% <= 5% allowed=1), 19 succeed -> returns 19
    rows = [DummyRow(str(i)) for i in range(20)]
    client = _mock_client(rows)
    mock_bq.return_value = client
    mock_post.side_effect = [TimeoutError("transient")] + [_response({}) for _ in range(19)]

    updated = enrich_restaurants_by_fhrsid(fhrsids=[str(i) for i in range(20)])
    assert updated == 19


@patch('scripts.enrich_maps_data.time.sleep')
@patch('scripts.enrich_maps_data.requests.post')
@patch('scripts.enrich_maps_data.bigquery.Client')
def test_progress_callback_emits_every_10_rows_and_flushes_every_50_rows(mock_bq, mock_post, _sleep):
    rows = [DummyRow(str(i), BusinessName=f"Resto {i}") for i in range(120)]
    client = MagicMock()
    select_job = MagicMock()
    select_job.result.return_value = rows
    # 1 SELECT + 3 MERGE batches (50 + 50 + 20)
    client.query.side_effect = [select_job, MagicMock(), MagicMock(), MagicMock()]
    mock_bq.return_value = client
    mock_post.return_value = _response({"places": [{"rating": 4.6, "userRatingCount": 42}]})

    msgs = []
    updated = enrich_restaurants_by_fhrsid(
        fhrsids=[str(i) for i in range(120)],
        progress_callback=msgs.append,
    )

    assert updated == 120
    # 1 SELECT + 3 MERGE queries
    assert client.query.call_count == 4
    lookup_msgs = [m for m in msgs if "Maps lookup " in m]
    merge_msgs = [m for m in msgs if "Merged Maps batch to BigQuery" in m]
    # 120 / 10 = 12 lookup progress messages
    assert len(lookup_msgs) == 12
    assert "Maps lookup 10/120" in lookup_msgs[0]
    assert "Resto 9 (4.6★)" in lookup_msgs[0]
    assert "Maps lookup 120/120 (100%)" in lookup_msgs[-1]
    # 3 incremental MERGE confirmations (50/120, 100/120, 120/120)
    assert len(merge_msgs) == 3
    assert "(50/120 complete)" in merge_msgs[0]
    assert "(100/120 complete)" in merge_msgs[1]
    assert "(120/120 complete)" in merge_msgs[2]

