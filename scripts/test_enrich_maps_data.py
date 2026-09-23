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
