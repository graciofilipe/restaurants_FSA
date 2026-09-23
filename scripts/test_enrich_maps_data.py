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
